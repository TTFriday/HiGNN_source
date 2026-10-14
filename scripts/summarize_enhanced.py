"""Summarize checkpoint-verified runs without mixing protocols or duplicate seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def summarize(root, on_failed="raise"):
    """on_failed: 'raise' (default), 'exclude' (primary analysis) or 'include' (sensitivity only).

    Failed verification reports are never deleted or rewritten; they are listed explicitly.
    """
    if on_failed not in {"raise", "exclude", "include"}:
        raise ValueError("on_failed must be raise, exclude or include")
    groups, seen = {}, set()
    pending, failed = [], []
    for path in sorted(Path(root).glob("*/manifest.json")):
        directory = path.parent
        expected_hash = (directory / "manifest.sha256").read_text(encoding="ascii").strip()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Manifest changed after run: " + directory.name)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("run_id") != directory.name:
            raise ValueError("Incomplete/mismatched manifest: " + directory.name)
        fingerprint_payload = {"config": manifest["config"], "code_sha256": manifest["code_sha256"],
                               "runner_sha256": manifest["runner_sha256"],
                               "graph_sha256": manifest["graph"]["sha256"]}
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True,
                                                allow_nan=False).encode("utf-8")).hexdigest()
        if fingerprint != manifest["config_code_graph_fingerprint"]:
            raise ValueError("Config/code/graph fingerprint mismatch: " + directory.name)
        verification = directory / "verification.json"
        if not verification.is_file():
            pending.append(directory.name)
            continue
        report = json.loads(verification.read_text(encoding="utf-8"))
        if report.get("run_id") != manifest["run_id"]:
            raise ValueError("Mismatched checkpoint verification: " + directory.name)
        for artifact, expected in manifest["artifacts_sha256"].items():
            if hashlib.sha256((directory / artifact).read_bytes()).hexdigest() != expected:
                raise ValueError("Artifact changed after verification: " + directory.name)
        passed = report.get("status") == "passed" and report.get("logits_allclose") and report.get("metrics_equal")
        if not passed:
            if on_failed == "raise":
                raise ValueError("Failed/incomplete checkpoint verification: " + directory.name)
            failed.append({"run_id": directory.name, "error": report.get("error"),
                           "logits_max_absolute_error": report.get("logits_max_absolute_error"),
                           "included": on_failed == "include"})
            if on_failed == "exclude":
                continue
        key = manifest["replicate_key"]
        if key in seen:
            raise ValueError("Duplicate model/graph/seed: choose one run explicitly; do not count reruns as independent")
        seen.add(key)
        config = manifest["config"]
        protocol = dict(config)
        protocol.pop("seed")
        # Model-specific configurations stay separate; code, graph and metric protocols must match.
        group_key = json.dumps({"protocol": protocol, "code": manifest["code_sha256"],
                                "runner": manifest["runner_sha256"], "graph": manifest["graph"]["sha256"]},
                               sort_keys=True)
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics["run_id"] != manifest["run_id"]:
            raise ValueError("Metrics run ID mismatch")
        row = {"run_id": directory.name, "seed": config["seed"],
               "metrics": flatten(metrics["standard_metrics"])}
        groups.setdefault(group_key, {"model": config["model_config"]["name"],
                                      "protocol": json.loads(group_key), "rows": []})["rows"].append(row)
    if pending:
        raise ValueError("Unverified runs present; verify these first: " + ", ".join(pending))
    output = []
    for group in groups.values():
        rows = sorted(group["rows"], key=lambda row: row["seed"])
        names = set.intersection(*(set(row["metrics"]) for row in rows))
        group["rows"] = rows
        group["n_runs"] = len(rows)
        group["seeds"] = [row["seed"] for row in rows]
        group["summary"] = {name: {"mean": statistics.mean(row["metrics"][name] for row in rows),
                                   "sample_sd": statistics.stdev(row["metrics"][name] for row in rows)
                                   if len(rows) > 1 else None}
                            for name in sorted(names)}
        output.append(group)
    return {"scope": "checkpoint-verified standard protocol only" if on_failed != "include"
            else "SENSITIVITY: includes runs that failed checkpoint verification (listed)",
            "statistics": "descriptive mean and sample SD; no superiority/significance claim",
            "failed_verification": failed, "groups": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--on-failed", choices=("raise", "exclude", "include"), default="raise")
    args = parser.parse_args()
    if not args.root.is_dir() or not args.output.parent.is_dir():
        parser.error("root and output parent must already exist")
    result = summarize(args.root, args.on_failed)
    if not result["groups"]:
        raise SystemExit("No completed verified runs; no summary generated")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print("Verified standard-protocol summary:", args.output)


if __name__ == "__main__":
    main()
