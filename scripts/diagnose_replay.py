# -*- coding: utf-8 -*-
"""Diagnose runs whose checkpoint replay FAILED the pre-specified logit tolerance.

Read-only: verification.json is never rewritten. Recomputes logits and metrics from the
hashed checkpoint and records where the deviations occur, so a sensitivity analysis can
state whether the failures affect the reported metrics. Trusted local files only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_enhanced import compute_metrics, prepare_graph, restore_splits, sha256_file
from scripts.verify_run import compare_metrics
from src.train import build_model, set_seed


def diagnose(run_dir, device="cuda", repeats=2):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts_sha256"].items():
        if sha256_file(run_dir / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    if sha256_file(manifest["graph"]["path"]) != manifest["graph"]["sha256"]:
        raise ValueError("graph hash mismatch")
    tol = report["tolerance"]
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    splits = torch.load(run_dir / "splits.pt", map_location="cpu", weights_only=True)
    saved = torch.load(run_dir / "FP32logits.pt", map_location="cpu", weights_only=True)["logits"]
    expected = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    set_seed(manifest["config"]["seed"])
    cpu_data = restore_splits(prepare_graph(torch.load(manifest["graph"]["path"], weights_only=False)), splits)
    # HeteroData.to/.cpu move in place, so keep an independent CPU copy for metric computation.
    data = cpu_data.clone().to(device)
    mc = manifest["config"]["model_config"]
    model = build_model(mc["name"], data, mc["hidden"], mc["layers"], mc["dropout"]).float().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    replays, previous = [], None
    for _ in range(repeats):
        with torch.no_grad():
            scores = model.score_all().cpu().float()
            logits = model.logits.cpu().float()
        error = (logits - saved).abs()
        violation = error > tol["logits_atol"] + tol["logits_rtol"] * saved.abs()
        metric_error = compare_metrics(compute_metrics(logits, scores, cpu_data, manifest["run_id"]),
                                       expected, atol=1e-6)
        replays.append({"logits_max_absolute_error": float(error.max()),
                        "n_elements_outside_tolerance": int(violation.sum()),
                        "n_elements": int(saved.numel()),
                        "max_abs_saved_logit_among_violations": float(saved[violation].abs().max())
                        if violation.any() else None,
                        "metrics_max_absolute_error": metric_error,
                        "self_difference_from_previous_replay": float((logits - previous).abs().max())
                        if previous is not None else None})
        previous = logits
    return {"run_id": manifest["run_id"], "original_verification_status": report["status"],
            "original_error": report.get("error"), "tolerance": tol,
            "saved_logit_abs_max": float(saved.abs().max()),
            "saved_logit_abs_median": float(saved.abs().median()),
            "replays": replays,
            "interpretation": "diagnostic only; the run remains FAILED under the pre-specified rule"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trusted-local-files", action="store_true", required=True)
    args = parser.parse_args()
    failed = [d for d in sorted(args.root.iterdir()) if (d / "verification.json").is_file()
              and json.loads((d / "verification.json").read_text(encoding="utf-8"))["status"] != "passed"]
    result = {"n_failed": len(failed), "runs": [diagnose(d, args.device) for d in failed]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for run in result["runs"]:
        print(run["run_id"], [(r["n_elements_outside_tolerance"], r["metrics_max_absolute_error"])
                              for r in run["replays"]])


if __name__ == "__main__":
    main()
