# -*- coding: utf-8 -*-
"""CPU synthetic audit regression and one tiny CUDA epoch (never fullgraph).

Run using the project venv: .venv/Scripts/python.exe -m unittest tests.test_reproducibility -v
All generated graphs/runs live in TemporaryDirectory, not the historical results.
"""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_smoke import make_fake
from scripts.run_enhanced import (compute_metrics, main as runner_main, run_enhanced,
                                  sha256_file, unique_pairs)
from scripts.verify_run import verify_run
from src.data.splits import G2P, split_data
from src.evaluate import binary_metrics
from src.models.hignn import HiGNN
from src.train import build_model, run, set_seed, train_one


def tiny_graph():
    set_seed(314)
    data = make_fake(num_gene=48, num_pheno=4, num_snp=8, num_meth=8)
    # make_fake generates duplicate targets. Deduplicate BEFORE split_data.
    targets = unique_pairs(data[G2P].edge_index)[:, :64]
    data[G2P].edge_index = targets
    data[G2P].edge_attr = torch.zeros(targets.size(1), 3)
    for edge in data.edge_types:
        if edge != G2P:
            data[edge].edge_index = data[edge].edge_index[:, :24].clone()
            data[edge].edge_attr = data[edge].edge_attr[:24].clone()
    return data


def original_training(model, data, epochs):
    """Frozen historical sigmoid loss, clip, validation cadence and strict > selection."""
    store = data[G2P]
    pos = store.edge_index[:, store.train_mask]
    neg = store.neg_train
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
    best_auc, state, best_epoch = -1, None, 0
    for ep in range(epochs):
        model.train()
        scores = model.score_all()
        loss = -torch.log(scores[pos[0], pos[1]] + 1e-7).mean()
        loss += -torch.log(1 - scores[neg[0], neg[1]] + 1e-7).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                auc = binary_metrics(model.score_all(), store.edge_index[:, store.valid_mask],
                                     store.neg_valid)["AUROC"]
            if auc > best_auc:
                best_auc, best_epoch = auc, ep + 1
                state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best_epoch, best_auc


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.historical = {str(path): sha256_file(path) for path in (ROOT / "results").glob("*")
                          if path.is_file()}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        current = {str(path): sha256_file(path) for path in (ROOT / "results").glob("*")
                   if path.is_file()}
        if current != cls.historical:
            raise AssertionError("historical result files changed during tests")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hignn-audit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "graph.pt"
        torch.save(tiny_graph(), self.graph)
        self.output = self.root / "results"
        self.output.mkdir()
        self.legacy = {"runs.jsonl": b'{"historical":27}\n',
                       "score_hignn_seed0.pt": b"old-score",
                       "logits_hignn_seed0.pt": b"old-logits",
                       "explain_hignn_seed0.pt": b"old-explanation"}
        for name, content in self.legacy.items():
            (self.output / name).write_bytes(content)

    def audited(self, name="hignn", seed=0, device="cpu", explain=False):
        return run_enhanced(name, seed, epochs=1, hidden=4, layers=1, dropout=0,
                            device=device, output_root=self.output, graph_path=self.graph,
                            explain=explain)

    def assert_legacy_untouched(self):
        for name, content in self.legacy.items():
            self.assertEqual((self.output / name).read_bytes(), content)

    def test_default_and_uniform_pooling_and_baseline_logits(self):
        data = tiny_graph()
        set_seed(7)
        default = HiGNN(data, hidden=4, layers=1, dropout=0)
        set_seed(7)
        explicit = HiGNN(data, hidden=4, layers=1, dropout=0, task_pooling="attention")
        self.assertTrue(torch.equal(default.score_all(), explicit.score_all()))
        set_seed(7)
        none = HiGNN(data, hidden=4, layers=1, use_task_attn=False, dropout=0)
        none.load_state_dict(default.state_dict())
        h = none.encode()
        expected_logits = none.out_g(h["gene"]) @ none.out_p(h["phenotype"]).t() / 2
        self.assertTrue(torch.equal(none.score_all(), torch.sigmoid(expected_logits)))
        self.assertIsNone(none.gamma)
        uniform = build_model("hignn_uniformtask", data, hidden=4, layers=1, dropout=0)
        uniform.load_state_dict(default.state_dict())
        h = uniform.encode()
        context = uniform.phenotype_context(h)
        self.assertTrue(torch.equal(context, h["gene"].mean(0, keepdim=True) + h["phenotype"]))
        self.assertTrue(torch.equal(uniform.gamma, torch.full((4, 48), 1 / 48)))
        for name in ("mlp", "rgcn", "gatv2", "han"):
            model = build_model(name, data, hidden=4, layers=1, dropout=0)
            scores = model.score_all()
            self.assertTrue(torch.equal(scores, torch.sigmoid(model.logits)), name)
            scores.mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
        with self.assertRaises(ValueError):
            HiGNN(data, task_pooling="bad")

    def test_training_selection_and_legacy_return_unchanged(self):
        data = split_data(tiny_graph(), seed=2)
        set_seed(8)
        reference, ep, auc = original_training(build_model("mlp", data, hidden=4), data, 26)
        set_seed(8)
        audited, selection = train_one(build_model("mlp", data, hidden=4), data, epochs=26,
                                        return_selection=True)
        self.assertEqual(selection["best_epoch"], ep)
        self.assertEqual(selection["best_val_AUROC"], auc)
        self.assertEqual([v["epoch"] for v in selection["validation_history"]], [25, 26])
        self.assertGreater(selection["training_seconds"], 0)
        self.assertEqual(len(selection["gradient_finite_checks"]), 26)
        self.assertTrue(all(v["gradients_finite"] and v["nonzero_gradient"]
                            for v in selection["gradient_finite_checks"]))
        set_seed(8)
        legacy = train_one(build_model("mlp", data, hidden=4), data, epochs=26)
        self.assertIsInstance(legacy, torch.nn.Module)
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, audited.state_dict()[key]), key)
            self.assertTrue(torch.equal(value, legacy.state_dict()[key]), key)
        with patch("src.train.load_graph", return_value=data.clone()):
            result = run("mlp", seed=2, epochs=1, hidden=4, device="cpu", log=False)
        self.assertEqual(result["model"], "mlp")
        self.assertIn("test_NDCG@10", result)
        self.assertIn("test_AUROC", result)
        self.assertNotIn("run_id", result)
        self.assert_legacy_untouched()

    def test_cli_budget_serial_defaults_and_duplicate_seeds(self):
        output = io.StringIO()
        with patch("sys.argv", ["run_enhanced.py"]), redirect_stdout(output), patch(
                "scripts.run_enhanced.run_enhanced", return_value=self.root / "mock") as invoke:
            runner_main()
        self.assertEqual(invoke.call_count, 30)
        self.assertEqual([(call.args[0], call.args[1]) for call in invoke.call_args_list],
                         [(name, seed) for name in ("hignn", "han", "hignn_uniformtask")
                          for seed in range(10)])
        self.assertTrue(all(call.kwargs["epochs"] == 150 for call in invoke.call_args_list))
        self.assertIn('"total_training_epochs": 4500', output.getvalue())
        with patch("sys.argv", ["run_enhanced.py", "--models", "han", "--seeds", "0", "0",
                                "--epochs", "1", "--device", "cpu"]), redirect_stdout(io.StringIO()), patch(
                "scripts.run_enhanced.run_enhanced", return_value=self.root / "mock") as invoke:
            runner_main()
        self.assertEqual(invoke.call_count, 1)

    def test_legacy_export_refuses_overwrite_before_training(self):
        data = split_data(tiny_graph(), seed=0)
        with patch("src.train.RESULTS", str(self.output)), patch(
                "src.train.load_graph", return_value=data), patch("src.train.train_one") as train:
            with self.assertRaisesRegex(FileExistsError, "historical exports"):
                run("hignn", seed=0, epochs=1, hidden=4, device="cpu", save_explain=True)
            train.assert_not_called()
        self.assert_legacy_untouched()

    def test_nonfinite_training_rejected(self):
        data = split_data(tiny_graph(), seed=0)
        model = build_model("mlp", data, hidden=4)
        with torch.no_grad():
            model.gb.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            train_one(model, data, epochs=1, return_selection=True)

    def test_roundtrip_all_models_metrics_runid_and_no_overwrite(self):
        with patch.object(HiGNN, "explain", side_effect=AssertionError("must not recompute explanation")):
            for name in ("hignn", "han", "hignn_uniformtask"):
                with self.subTest(model=name):
                    directory = self.audited(name, explain=True)
                    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                    checkpoint = torch.load(directory / "best_model.pt", weights_only=True)
                    splits = torch.load(directory / "splits.pt", weights_only=True)
                    saved = torch.load(directory / "FP32logits.pt", weights_only=True)
                    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                    for payload in (manifest, checkpoint, splits, saved, metrics):
                        self.assertEqual(payload["run_id"], directory.name)
                    self.assertEqual(saved["logits"].dtype, torch.float32)
                    self.assertTrue(all(v.device.type == "cpu" for v in checkpoint["state_dict"].values()))
                    self.assertTrue(torch.equal(splits["edge_index"], splits["edge_indices"][G2P]))
                    data = tiny_graph()
                    for key in ("train_mask", "valid_mask", "test_mask", "neg_train", "neg_valid", "neg_test"):
                        data[G2P][key] = splits[key]
                    self.assertEqual(metrics, compute_metrics(saved["logits"], torch.sigmoid(saved["logits"]),
                                                               data, directory.name))
                    counts = metrics["standard_metrics"]["head_tail_counts"]
                    self.assertEqual(sum(group["n_test_negatives"] for group in counts.values()),
                                     splits["neg_test"].size(1))
                    if name != "han":
                        ex = torch.load(directory / "explanation.pt", weights_only=True)
                        self.assertEqual(ex["run_id"], directory.name)
                        self.assertEqual(ex["task_gamma"].device.type, "cpu")
                        self.assertEqual(len(ex["rel_beta"]), 1)
                    else:
                        self.assertFalse((directory / "explanation.pt").exists())
                    before = {p.name: sha256_file(p) for p in directory.iterdir()}
                    report = verify_run(directory, trusted_local_files=True)
                    self.assertEqual(report["status"], "passed", report.get("error"))
                    self.assertTrue(report["logits_allclose"] and report["metrics_equal"])
                    self.assertEqual(report["logits_max_absolute_error"], 0)
                    self.assertTrue(all(report["hash_checks"].values()))
                    for filename, digest in before.items():
                        self.assertEqual(sha256_file(directory / filename), digest)
                    with self.assertRaises(FileExistsError):
                        verify_run(directory, trusted_local_files=True)
        first = self.audited()
        first_hash = sha256_file(first / "best_model.pt")
        second = self.audited()
        self.assertNotEqual(first, second)
        self.assertEqual(sha256_file(first / "best_model.pt"), first_hash)
        a, b = [json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in (first, second)]
        self.assertEqual(a["replicate_key"], b["replicate_key"])
        self.assertEqual(a["config_code_graph_fingerprint"], b["config_code_graph_fingerprint"])
        self.assert_legacy_untouched()

    def test_hash_tampering_and_trust_guard(self):
        directory = self.audited("han")
        with self.assertRaisesRegex(ValueError, "trusted"):
            verify_run(directory)
        self.assertFalse((directory / "verification.json").exists())
        with (directory / "FP32logits.pt").open("ab") as stream:
            stream.write(b"tampered")
        with patch("scripts.verify_run.torch.load", side_effect=AssertionError("must check hash first")):
            report = verify_run(directory, trusted_local_files=True)
        self.assertEqual(report["status"], "failed")
        self.assertIn("SHA256 mismatch: FP32logits.pt", report["error"])
        self.assertFalse(report["hash_checks"]["FP32logits.pt"])
        self.assert_legacy_untouched()

    def test_duplicate_targets_rejected_before_split(self):
        data = tiny_graph()
        data[G2P].edge_index = torch.cat([data[G2P].edge_index, data[G2P].edge_index[:, :1]], dim=1)
        torch.save(data, self.graph)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.audited()
        self.assert_legacy_untouched()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_tiny_cuda_one_epoch_roundtrip(self):
        # Only 48 genes, 4 phenotypes, hidden=4, one layer/epoch, <=24 edges/relation.
        directory = self.audited("hignn", device="cuda", explain=True)
        report = verify_run(directory, device="cuda", trusted_local_files=True)
        self.assertEqual(report["status"], "passed", report.get("error"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["config"]["epochs"], 1)
        self.assertGreater(manifest["peak_gpu_allocated_bytes"], 0)
        self.assertLess(manifest["peak_gpu_allocated_bytes"], 64 * 1024 * 1024)
        self.assert_legacy_untouched()
        print("\nTINY CUDA audit:", json.dumps({"gpu": manifest["environment"]["gpu_name"],
                                               "peak_bytes": manifest["peak_gpu_allocated_bytes"],
                                               "training_seconds": manifest["selection"]["training_seconds"],
                                               "logit_max_error": report["logits_max_absolute_error"],
                                               "verification": report["status"]}))


if __name__ == "__main__":
    unittest.main()
