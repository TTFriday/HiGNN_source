# HiGNN_source[README.md](https://github.com/user-attachments/files/33411428/README.md)
# HiGNN

Evidence-aware heterogeneous graph network for multi-omics link prediction.
The benchmark is gene–phenotype association completion on AtMAD (Arabidopsis),
formulated as transductive link prediction over a signed, evidence-weighted
multi-relational graph.

Contents: graph preprocessing, model and baselines, training, evaluation,
checkpoint-verified experiment scripts, unit tests, and the scalar result files
behind the paper tables. Raw association tables are not included.

## Layout

```
src/data        input parsing, ID normalisation, graph construction, splits and negative sampling
src/models      HiGNN and the four baselines
src/train.py    training loop and model selection
src/evaluate.py legacy sigmoid-score metrics and the standard logit-based ranking metrics
scripts/        experiment runners, replay verification, aggregation, input audit
tests/          CPU unit tests plus one tiny CUDA round trip
results/        recorded scalar results (see below)
provenance/     SHA256 manifest of the local AtMAD input files
```

## Environment

Tested with Python 3.10.9, PyTorch 2.5.1+cu121 and PyTorch Geometric 2.8.1 on
Windows and an NVIDIA RTX 3090. CPU is enough for the tests; full-graph training
needs a CUDA GPU. GPU scatter-add is not bitwise deterministic, which matters for
the replay tolerances below.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

## Data

AtMAD source paper: Lan Y, Sun R, Ouyang J, Ding W, Kim MJ, Wu J, Li Y, Shi T (2021)
AtMAD: Arabidopsis thaliana multi-omics association database. Nucleic Acids Research
49(D1):D1445–D1451. https://doi.org/10.1093/nar/gkaa1042

Download the ten tables from http://www.megabionet.org/atmad (download page:
http://119.3.41.228/atmad/download.php): `ciseqtl.xls`, `transeqtl.xls`,
`environmenteqtl.xlsx`, `gwaseqtl.csv`, `emqtl.csv`, `epathway.csv`,
`phenotypepathway.csv`, `gwas.csv`, `twas.csv`, `ewas.csv`.

Free download access is not a redistribution licence, so the tables are not shipped
here. `provenance/AtMAD_input_manifest.json` records the SHA256 of the exact local
copies used for the reported graph. The page label "June 2020" describes the server's
summary, not our snapshot, and the original acquisition date was not recorded. As of
October 2026 the current remote file lengths differ from the local ones, so a fresh
download will not reproduce these numbers byte for byte; rebuild the graph and rerun
if you need results from the live snapshot.

Point the code at your own copy and build the graph:

```bash
export HIGNN_RAW=/absolute/path/to/AtMAD_inputs      # default: ../rawdata
python scripts/audit_inputs.py --raw-dir "$HIGNN_RAW" --output input_manifest.json --check-remote
python -m src.data.loaders
python -m src.data.build_graph
```

Generated caches go to `data_cache/`, which is git-ignored. Only load `.pt`/`.pkl`
files you produced yourself.

## Models and protocol

The graph carries 21 directed relations; the target `assoc` layer is excluded from
message passing, leaving 20 auxiliary relations. HiGNN has three inspectable weight
levels: neighbour attention (α), relation mixing (β) and phenotype-conditioned task
pooling (γ). Weights are computable quantities, not validated explanations.

Baseline names refer to simplified controls, not to the published architectures:

| Name | What it actually is |
|---|---|
| `mlp` | no message passing, but uses graph-derived features and trainable node identity |
| `rgcn` | relation-specific **sum** aggregation |
| `gatv2` | edge-weight scaling control |
| `han` | relation-semantic attention control with unnormalised sum aggregation |

Training minimises binary cross-entropy on sampled negatives; the checkpoint is the
epoch with the best validation AUROC. This is association completion with sampled
unlabelled negatives, not individual-level phenotype prediction, and no correction for
unlabelled positives is applied.

## Historical runs (3 seeds)

`results/recorded_runs.jsonl` holds the 27 scalar records behind Tables 4–6: five
models and four HiGNN ablations at seeds 0–2. Those runs kept the selected state in
memory only, so they cannot be replayed from a checkpoint and are reported as recorded.

```bash
python -m src.train --model hignn --seed 0 --epochs 150   # appends to results/runs.jsonl
python scripts/run_legacy_grid.py                          # all 27 configurations
```

The legacy ranking path uses sigmoid scores and filters training positives only. Its
stored field `test_NDCG@10` is really mean `1/log2(rank+1)` over test-positive pairs,
with no cutoff and no ideal-list normalisation; the paper calls it DRS for that reason.

## Ten-seed verified runs

Separate from the historical log, and not merged with it. HiGNN, `han` and a uniform
task-pooling control (`hignn_uniformtask`, γ = 1/N_g) at seeds 0–9, 150 epochs each.
Each run directory keeps `best_model.pt`, `splits.pt`, `FP32logits.pt`, `metrics.json`
and a hash manifest, so every number can be recomputed from disk.

```bash
python scripts/run_enhanced.py --models hignn han hignn_uniformtask \
    --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 150 --device cuda --output-root results/enhanced
python scripts/verify_run.py results/enhanced/RUN_DIR --trusted-local-files --device cuda \
    --atol 1e-4 --rtol 1e-4 --metrics-atol 1e-6
python scripts/summarize_enhanced.py --root results/enhanced --output results/enhanced_summary.json --on-failed exclude
python scripts/compare_enhanced.py --root results/enhanced --output results/enhanced_comparison.json
python scripts/compare_enhanced.py --root results/enhanced --output results/enhanced_comparison_sensitivity.json --on-failed include
python scripts/diagnose_replay.py --root results/enhanced --output results/enhanced_replay_diagnostics.json --trusted-local-files
```

`verify_run.py` reloads the saved state and splits, recomputes logits and metrics, and
compares them with the persisted ones. Its defaults (1e-6/1e-5) are tighter than GPU
scatter-add reliably allows: the pilot failed them (max logit deviation 7.6e-6, metrics
within 4e-8), so the threshold above was fixed before the main batch. The pilot's failed
report is kept and the pilot is excluded. `diagnose_replay.py` is read-only and cannot
turn a failed run into a passed one.

`compare_enhanced.py` runs the two pre-specified paired comparisons (HiGNN − `han`,
HiGNN − uniform pooling) over seven metrics, reporting the mean paired difference with
a 95% t interval, Cohen's d_z, wins/ties/losses, an exact two-sided Wilcoxon p and
Holm correction across all 14 tests.

### What the ten-seed batch shows

Included are the aggregated JSON files only; the 30 run directories (checkpoints and
logits) are too large and are not redistributed here.

| Model (verified seeds) | AUROC | AUPRC |
|---|---|---|
| HiGNN (10) | 0.7840 ± 0.0098 | 0.2778 ± 0.0196 |
| `han` (7) | 0.6878 ± 0.0730 | 0.1949 ± 0.0577 |
| uniform pooling (10) | 0.7847 ± 0.0085 | 0.2848 ± 0.0283 |

- HiGNN exceeds the `han` control by 0.096 AUROC on average (95% CI 0.031–0.161), but
  nothing survives Holm correction. With seven pairs the smallest attainable adjusted
  p is 0.219, so this batch cannot establish significance either way.
- Seeds 5, 7 and 9 of the `han` control failed replay (max logit deviations 0.0078,
  0.047, 0.00068; that control's logits reach ~2.2e5). Diagnostics show at most 45 of
  5,796,552 logits outside tolerance and metrics within 8.6e-8, but they stay failed and
  are excluded from the primary analysis; the sensitivity run includes them and the
  conclusion is unchanged (smallest Holm p 0.055).
- Uniform pooling matches learned task attention on every metric, all intervals
  crossing zero. The γ weights are therefore not shown to add accuracy; at most the
  global gene context they carry matters.
- Retrieval stays weak: macro NDCG@10 below 0.05 and macro Recall@50 below 0.10.
- All seeds share one graph, so intervals describe split and initialisation variability
  only. Timing (about 54 s per HiGNN run, 5.0 GiB peak) is a single-device measurement,
  not an efficiency benchmark. 4/10 HiGNN and 8/10 uniform runs selected their final
  epoch, so the budget may not be saturated.

## Notes

- Rerunning a seed creates a new directory; there is no resume or dedup, and a rerun is
  not an extra independent replicate. Aggregate one verified run per model/seed pair.
- `han` seeds 0–2 under this protocol do not reproduce the historical values exactly
  (seed 0: AUROC 0.7675 → 0.6148). Splits and parameter states were not persisted
  historically, so the gap cannot be decomposed.
- Figures in the paper are produced by a separate local plotting script that is not part
  of this release.

## License

To be selected by the authors before publication. Until then no reuse rights are
granted. Please cite the paper if you build on this work.
