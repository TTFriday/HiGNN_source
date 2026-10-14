# -*- coding: utf-8 -*-
"""Paired seed-level comparison of checkpoint-verified enhanced runs.

Pre-specified analysis (fixed before inspecting the ten-seed results):
  * comparisons: HiGNN - HAN-style and HiGNN - uniform task pooling;
  * metrics: pooled logit AUROC/AUPRC, phenotype-macro AUPRC, tail AUPRC,
    macro NDCG@10, macro Recall@50 and pair MRR (standard protocol only);
  * per comparison and metric: mean paired difference, 95% t interval, Cohen's d_z,
    wins/ties/losses, exact two-sided Wilcoxon signed-rank p and Holm adjustment over
    all 14 tests.
Seeds resample the split and initialisation on ONE fixed graph, so intervals describe
split/initialisation variability, not generalisation to new data sets.
Pairing is checked by comparing the saved split tensors of every model for each seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_enhanced import summarize

METRICS = {
    "AUROC": "binary_logits.AUROC",
    "AUPRC": "binary_logits.AUPRC",
    "Macro AUPRC": "macro_binary_logits.macro_AUPRC",
    "Tail AUPRC": "head_tail_binary_logits.tail.AUPRC",
    "Macro NDCG@10": "ranking.macro_NDCG@10",
    "Macro Recall@50": "ranking.macro_Recall@50",
    "Pair MRR": "ranking.pair_MRR",
}
COMPARISONS = (("hignn", "han"), ("hignn", "hignn_uniformtask"))
SPLIT_KEYS = ("edge_index", "train_mask", "valid_mask", "test_mask", "neg_train", "neg_valid", "neg_test")


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def check_pairing(root, by_model, models):
    seeds = sorted(set.intersection(*(set(by_model[m]) for m in models)))
    for seed in seeds:
        reference = None
        for model in models:
            rows = by_model[model]
            split = torch.load(Path(root) / rows[seed]["run_id"] / "splits.pt", map_location="cpu",
                               weights_only=True)
            tensors = {key: split[key] for key in SPLIT_KEYS}
            if reference is None:
                reference = tensors
            elif any(not torch.equal(reference[key], tensors[key]) for key in SPLIT_KEYS):
                raise ValueError(f"seed {seed}: models do not share the same split; pairing invalid")
    return seeds


def resources(root, rows):
    values = {"runtime_seconds": [], "training_seconds": [], "peak_gpu_GiB": [], "best_epoch": []}
    for row in rows.values():
        manifest = json.loads((Path(root) / row["run_id"] / "manifest.json").read_text(encoding="utf-8"))
        values["runtime_seconds"].append(manifest["runtime_seconds"])
        values["training_seconds"].append(manifest["selection"]["training_seconds"])
        values["peak_gpu_GiB"].append(manifest["peak_gpu_allocated_bytes"] / 2 ** 30)
        values["best_epoch"].append(manifest["selection"]["best_epoch"])
    return {key: {"mean": statistics.mean(v), "sample_sd": statistics.stdev(v) if len(v) > 1 else None,
                  "min": min(v), "max": max(v)} for key, v in values.items()}


def compare(root, on_failed="exclude"):
    """Primary analysis: on_failed='exclude'. Sensitivity analysis: on_failed='include'."""
    summary = summarize(root, on_failed=on_failed)
    by_model = {}
    for group in summary["groups"]:
        if group["model"] in by_model:
            raise ValueError("more than one protocol group per model; compare groups explicitly")
        by_model[group["model"]] = {row["seed"]: row for row in group["rows"]}
    descriptive = {model: {label: {"n": len(rows), "seeds": sorted(rows),
                                   "mean": statistics.mean(rows[s]["metrics"][key] for s in sorted(rows)),
                                   "sample_sd": statistics.stdev(rows[s]["metrics"][key] for s in sorted(rows))
                                   if len(rows) > 1 else None,
                                   "per_seed": [rows[s]["metrics"][key] for s in sorted(rows)]}
                           for label, key in METRICS.items()}
                   for model, rows in by_model.items()}
    tests, paired_seeds = [], {}
    for a, b in COMPARISONS:
        seeds = check_pairing(root, by_model, (a, b))
        paired_seeds[f"{a} - {b}"] = seeds
        for label, key in METRICS.items():
            diff = np.array([by_model[a][s]["metrics"][key] - by_model[b][s]["metrics"][key] for s in seeds])
            n, mean, sd = len(diff), float(diff.mean()), float(diff.std(ddof=1))
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            nonzero = diff[diff != 0]
            p = float(stats.wilcoxon(nonzero, alternative="two-sided", method="exact").pvalue) if len(nonzero) else 1.0
            tests.append({"comparison": f"{a} - {b}", "metric": label, "n_pairs": n, "seeds": seeds,
                          "mean_difference": mean, "sd_difference": sd,
                          "ci95": [mean - half, mean + half], "cohen_dz": mean / sd if sd > 0 else None,
                          "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
                          "losses": int((diff < 0).sum()), "wilcoxon_exact_p": p,
                          "per_seed_difference": diff.tolist()})
    for test, adjusted in zip(tests, holm([t["wilcoxon_exact_p"] for t in tests])):
        test["holm_p"] = adjusted
    return {"scope": summary["scope"] + "; paired by identical saved splits",
            "analysis": "primary (verified runs only)" if on_failed == "exclude" else "sensitivity",
            "failed_verification": summary["failed_verification"],
            "paired_seeds": paired_seeds, "metric_keys": METRICS,
            "caveat": "seeds share one graph; intervals reflect split/initialisation variability only",
            "descriptive": descriptive, "paired_tests": tests,
            "resources": {model: resources(root, rows) for model, rows in by_model.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--on-failed", choices=("exclude", "include"), default="exclude",
                        help="exclude = primary analysis; include = labelled sensitivity analysis")
    args = parser.parse_args()
    result = compare(args.root, args.on_failed)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    for test in result["paired_tests"]:
        print(f'{test["comparison"]:28s} {test["metric"]:16s} diff={test["mean_difference"]:+.4f} '
              f'CI=[{test["ci95"][0]:+.4f},{test["ci95"][1]:+.4f}] W/T/L={test["wins"]}/{test["ties"]}/'
              f'{test["losses"]} p={test["wilcoxon_exact_p"]:.4f} holm={test["holm_p"]:.4f}')


if __name__ == "__main__":
    main()
