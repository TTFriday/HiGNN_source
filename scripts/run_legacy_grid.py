# -*- coding: utf-8 -*-
"""Reproduce the historical grid: 5 methods + 4 ablations x 3 seeds, then write the main and ablation tables."""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from src.train import run

MODELS = ["mlp", "rgcn", "gatv2", "han", "hignn",
          "hignn_nosign", "hignn_noev", "hignn_norelattn", "hignn_notaskattn"]
SEEDS = [0, 1, 2]
EPOCHS = int(os.environ.get("HIGNN_EPOCHS", "150"))

METRICS = ["test_AUROC", "test_AUPRC", "test_MRR", "test_Hits@10", "test_Hits@50", "test_NDCG@10"]


def main():
    results = []
    for m in MODELS:
        per = {m: {k: [] for k in METRICS}}
        for s in SEEDS:
            print(f"== {m} seed={s} ==", flush=True)
            r = run(m, seed=s, epochs=EPOCHS, save_explain=(s == 0 and m == "hignn"))
            for k in METRICS:
                if k in r:
                    per[m][k].append(r[k])
        results.append(per)
    agg = {}
    for per in results:
        for m, d in per.items():
            agg[m] = {k: {"mean": statistics.mean(v) if v else None,
                          "std": statistics.stdev(v) if len(v) > 1 else 0.0}
                      for k, v in d.items()}
    out = os.path.join(HERE, "results", "legacy_main_table.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    # markdown 表
    lines = ["| 方法 | " + " | ".join(METRICS) + " |", "|---| " + "| ".join(["---"] * len(METRICS)) + "|"]
    for m in MODELS:
        row = [m]
        for k in METRICS:
            if m in agg and agg[m][k]["mean"] is not None:
                row.append(f"{agg[m][k]['mean']:.4f}±{agg[m][k]['std']:.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(HERE, "results", "legacy_main_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("已写入", out)


if __name__ == "__main__":
    main()
