"""Synthetic regression tests for the verified-run summarizer; no real training data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_enhanced import run_enhanced, sha256_file
from scripts.verify_run import verify_run
from scripts.summarize_enhanced import summarize
from tests.test_reproducibility import tiny_graph
import torch


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "graph.pt"
        torch.save(tiny_graph(), self.graph)
        self.output = self.root / "runs"
        self.output.mkdir()

    def run_tiny(self, seed=0):
        return run_enhanced("hignn", seed=seed, epochs=1, hidden=4, layers=1,
                            dropout=0, device="cpu", graph_path=self.graph, output_root=self.output)

    def test_unverified_rejected_then_summary_passes(self):
        directory = self.run_tiny()
        with self.assertRaisesRegex(ValueError, "Unverified"):
            summarize(self.output)
        self.assertEqual(verify_run(directory, trusted_local_files=True)["status"], "passed")
        result = summarize(self.output)
        self.assertEqual(result["groups"][0]["seeds"], [0])
        self.assertEqual(result["groups"][0]["n_runs"], 1)

    def test_manifest_tampering_rejected(self):
        directory = self.run_tiny()
        verify_run(directory, trusted_local_files=True)
        path = directory / "manifest.json"
        value = json.loads(path.read_text())
        value["config"]["seed"] = 99
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            summarize(self.output)
        (directory / "manifest.sha256").write_text(sha256_file(path))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            summarize(self.output)

    def test_duplicate_and_artifact_change_rejected(self):
        first = self.run_tiny()
        verify_run(first, trusted_local_files=True)
        second = self.run_tiny()
        verify_run(second, trusted_local_files=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(self.output)
        with (first / "metrics.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "Artifact changed"):
            summarize(self.output)

    def test_paired_comparison_and_holm(self):
        from unittest.mock import patch
        from scripts import compare_enhanced
        for got, want in zip(compare_enhanced.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)
        for name in ("hignn", "han", "hignn_uniformtask"):
            for seed in (0, 1, 2):
                directory = run_enhanced(name, seed=seed, epochs=1, hidden=4, layers=1, dropout=0,
                                         device="cpu", graph_path=self.graph, output_root=self.output)
                self.assertEqual(verify_run(directory, trusted_local_files=True)["status"], "passed")
        metrics = {"AUROC": "binary_logits.AUROC", "AUPRC": "binary_logits.AUPRC"}
        with patch.object(compare_enhanced, "METRICS", metrics):
            result = compare_enhanced.compare(self.output)
        self.assertEqual(set(map(tuple, result["paired_seeds"].values())), {(0, 1, 2)})
        self.assertEqual(result["failed_verification"], [])
        self.assertEqual(len(result["paired_tests"]), 4)
        for test in result["paired_tests"]:
            self.assertEqual(test["wins"] + test["ties"] + test["losses"], 3)
            self.assertLessEqual(test["wilcoxon_exact_p"], test["holm_p"])
            a, b = test["comparison"].split(" - ")
            expected = [x - y for x, y in zip(result["descriptive"][a][test["metric"]]["per_seed"],
                                              result["descriptive"][b][test["metric"]]["per_seed"])]
            self.assertEqual(test["per_seed_difference"], expected)
        self.assertEqual(set(result["resources"]), {"hignn", "han", "hignn_uniformtask"})

    def test_failed_verification_modes(self):
        from unittest.mock import patch
        directory = self.run_tiny()
        with patch("scripts.verify_run.compute_metrics", return_value={}):
            report = verify_run(directory, trusted_local_files=True)
        self.assertEqual(report["status"], "failed")
        with self.assertRaisesRegex(ValueError, "Failed"):
            summarize(self.output)
        self.assertEqual(summarize(self.output, on_failed="exclude")["groups"], [])
        included = summarize(self.output, on_failed="include")
        self.assertEqual(included["groups"][0]["n_runs"], 1)
        self.assertTrue(included["failed_verification"][0]["included"])
        self.assertIn("SENSITIVITY", included["scope"])


if __name__ == "__main__":
    unittest.main()
