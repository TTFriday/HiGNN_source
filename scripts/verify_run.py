# -*- coding: utf-8 -*-
"""Replay an enhanced run; original artifacts are read-only.

WARNING: torch.load can execute pickle code. ONLY verify your own trusted local
checkpoint/graph/artifacts, NEVER downloaded or untrusted files. SHA256 verifies
integrity, NOT authenticity. CLI requires --trusted-local-files acknowledgement.
Only verification.json is created, exclusively (existing reports are not replaced).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_enhanced import (code_hashes, compute_metrics, environment, json_digest,
                                  numeric_policy, prepare_graph, restore_splits,
                                  runner_hashes, sha256_file, write_json)
from src.train import build_model, set_seed


@contextmanager
def saved_numeric_policy(policy):
    original = numeric_policy()

    def apply(value):
        torch.set_float32_matmul_precision(value["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = value["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = value["cudnn_allow_tf32"]
        torch.backends.cudnn.benchmark = value["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = value["cudnn_deterministic"]
        torch.use_deterministic_algorithms(value["deterministic_algorithms"],
                                          warn_only=value["deterministic_warn_only"])

    try:
        apply(policy)
        yield
    finally:
        apply(original)


def compare_metrics(actual, expected, atol=1e-12):
    """Same structure/counts/protocols; float metrics equal within explicit tolerance."""
    maximum = 0.0

    def walk(a, b, path):
        nonlocal maximum
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                raise ValueError(f"metric structure differs: {path}")
            for key in b:
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(b, (list, tuple)):
            if not isinstance(a, (list, tuple)) or len(a) != len(b):
                raise ValueError(f"metric structure differs: {path}")
            for index, value in enumerate(b):
                walk(a[index], value, f"{path}[{index}]")
        elif isinstance(b, float):
            if not isinstance(a, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"nonfinite/invalid metric: {path}")
            error = abs(a - b)
            maximum = max(maximum, error)
            if error > atol:
                raise ValueError(f"metric differs: {path}; absolute error={error}")
        elif a != b:
            raise ValueError(f"metric differs: {path}")

    walk(actual, expected, "metrics")
    return maximum


def verify_run(run_dir, device=None, *, trusted_local_files=False, graph_path=None,
               atol=1e-6, rtol=1e-5, metrics_atol=1e-12):
    """Load only self-produced trusted files; return passed/failed audit report."""
    if not trusted_local_files:
        raise ValueError("torch.load is allowed only for your own trusted local files")
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol, metrics_atol)):
        raise ValueError("tolerances must be finite and nonnegative")
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError("run directory does not exist")
    if (run_dir / "verification.json").exists():
        raise FileExistsError("verification.json already exists; original report will not be overwritten")
    started = time.perf_counter()
    report = {"status": "failed", "run_id": run_dir.name,
              "tolerance": {"logits_atol": atol, "logits_rtol": rtol, "metrics_atol": metrics_atol},
              "hash_checks": {}, "logits_max_absolute_error": None,
              "logits_max_relative_error": None, "metrics_max_absolute_error": None,
              "logits_allclose": False, "metrics_equal": False,
              "trust": "trusted self-produced local files only; SHA256 is not authentication"}
    try:
        def check_hash(label, path, expected):
            actual = sha256_file(path)
            report["hash_checks"][label] = actual == expected
            if actual != expected:
                raise ValueError(f"SHA256 mismatch: {label}")

        check_hash("manifest.json", run_dir / "manifest.json",
                   (run_dir / "manifest.sha256").read_text(encoding="ascii").strip())
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["status"] != "complete":
            raise ValueError("unsupported/incomplete manifest")
        run_id = manifest["run_id"]
        if run_id != run_dir.name:
            raise ValueError("directory/run ID mismatch")
        report["run_id"] = run_id
        for label, actual, expected in (("src_code", code_hashes(), manifest["code_sha256"]),
                                        ("runner_code", runner_hashes(), manifest["runner_sha256"])):
            report["hash_checks"][label] = actual == expected
            if actual != expected:
                raise ValueError(f"code SHA256 mismatch: {label}")
        artifacts = manifest["artifacts_sha256"]
        required = {"best_model.pt", "splits.pt", "FP32logits.pt", "metrics.json"}
        if set(artifacts) not in (required, required | {"explanation.pt"}):
            raise ValueError("unexpected/missing manifest artifacts")
        for name, expected in artifacts.items():
            check_hash(name, run_dir / name, expected)
        graph_path = Path(graph_path or manifest["graph"]["path"]).resolve()
        check_hash("graph.pt", graph_path, manifest["graph"]["sha256"])
        config = manifest["config"]
        fingerprint = json_digest({"config": config, "code_sha256": manifest["code_sha256"],
                                   "runner_sha256": manifest["runner_sha256"],
                                   "graph_sha256": manifest["graph"]["sha256"]})
        if fingerprint != manifest["config_code_graph_fingerprint"]:
            raise ValueError("config/code/graph fingerprint mismatch")
        if manifest["replicate_key"] != json_digest({"model": config["model_config"]["name"],
                                                     "seed": config["seed"],
                                                     "graph_sha256": manifest["graph"]["sha256"]}):
            raise ValueError("replicate key mismatch")
        # All hashes are checked before deserialization. This still requires trusted inputs.
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        splits = torch.load(run_dir / "splits.pt", map_location="cpu", weights_only=True)
        saved = torch.load(run_dir / "FP32logits.pt", map_location="cpu", weights_only=True)
        expected_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        payloads = [checkpoint, splits, saved, expected_metrics]
        if "explanation.pt" in artifacts:
            payloads.append(torch.load(run_dir / "explanation.pt", map_location="cpu", weights_only=True))
        for payload in payloads:
            if payload["run_id"] != run_id:
                raise ValueError("artifact run ID mismatch")
        if checkpoint["config"] != config or checkpoint["model_config"] != config["model_config"]:
            raise ValueError("checkpoint configuration mismatch")
        if checkpoint["selection"] != manifest["selection"]:
            raise ValueError("checkpoint selection mismatch")
        if splits["seed"] != config["seed"] or splits["protocol"] != config["split"]:
            raise ValueError("split seed/protocol mismatch")
        if saved["logits"].dtype != torch.float32 or not torch.isfinite(saved["logits"]).all():
            raise ValueError("saved logits must be finite FP32")
        device = torch.device(device or config["device"])
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("only cpu/cuda are supported")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; pass --device cpu for cross-device verification")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        report["verification_device"] = str(device)
        report["training_device"] = config["device"]
        with saved_numeric_policy(config["numeric_policy"]):
            report["environment"] = environment(device)
            if numeric_policy()["CUBLAS_WORKSPACE_CONFIG"] != config["numeric_policy"]["CUBLAS_WORKSPACE_CONFIG"]:
                raise ValueError("CUBLAS_WORKSPACE_CONFIG differs; launch verifier with saved environment")
            set_seed(config["seed"])
            data = prepare_graph(torch.load(graph_path, map_location="cpu", weights_only=False))
            data = restore_splits(data, splits).to(device)
            mc = config["model_config"]
            model = build_model(mc["name"], data, mc["hidden"], mc["layers"], mc["dropout"]).float().to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                scores = model.score_all().detach().cpu().float()
                logits = model.logits.detach().cpu().float()
            expected = saved["logits"]
            if logits.shape != expected.shape or not torch.isfinite(logits).all():
                raise ValueError("replayed logits shape/nonfinite mismatch")
            error = (logits - expected).abs()
            report["logits_max_absolute_error"] = float(error.max())
            report["logits_max_relative_error"] = float((error / expected.abs().clamp_min(1e-12)).max())
            report["logits_allclose"] = bool(torch.allclose(logits, expected, atol=atol, rtol=rtol))
            if not report["logits_allclose"]:
                raise ValueError("replayed logits are not allclose")
            actual_metrics = compute_metrics(logits, scores, data, run_id)
            report["metrics_max_absolute_error"] = compare_metrics(actual_metrics, expected_metrics, metrics_atol)
            report["metrics_equal"] = True
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["runtime_seconds"] = time.perf_counter() - started
    write_json(run_dir / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trusted-local-files", action="store_true", required=True,
                        help="acknowledge all inputs are your own trusted local files, not downloads")
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-path", type=Path, default=None,
                        help="relocated trusted graph, must have the exact saved SHA256")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--metrics-atol", type=float, default=1e-12)
    args = parser.parse_args()
    report = verify_run(args.run_dir, args.device, trusted_local_files=args.trusted_local_files,
                        graph_path=args.graph_path, atol=args.atol, rtol=args.rtol,
                        metrics_atol=args.metrics_atol)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
