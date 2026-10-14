# -*- coding: utf-8 -*-
"""Isolated, auditable runs. torch.load is ONLY for trusted local graph files.

Never calls legacy run(), appends runs.jsonl, or writes legacy seed exports.
Repeated seeds/configurations are reruns, NOT additional independent replicates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.splits import G2P, split_data
from src.evaluate import (binary_metrics, binary_metrics_by_phenotype,
                          frequency_group_summary, ranking_metrics,
                          split_by_frequency, standard_ranking_metrics)
from src.models.hignn import HiGNN
from src.train import build_model, set_seed, train_one

MODELS = ("hignn", "han", "hignn_uniformtask")
SPLIT_PROTOCOL = {"seeded_fresh_split": True, "ratios": [0.8, 0.1, 0.1], "neg_ratio": 1}
METRIC_PROTOCOL = {
    "standard_ranking": "FP32 logits; remove train+validation; gene-index tie break; K=10,50",
    "binary": "pooled logits on original sampled negatives (multiplicity retained)",
    "macro_binary": "equal phenotype weights; original negative draws; both classes required",
    "head_tail_counts": "original negative draws; head if training positives >=20",
    "legacy": "sigmoid; training-only ranking filter; historical NDCG@10 is untruncated DRS",
    "negative_duplicates": "saved and evaluated unchanged; sampled multiplicity retained",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    # Exclusive creation: an existing artifact is never overwritten by the runner.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def code_hashes():
    paths = [ROOT / "src" / "__init__.py", ROOT / "src" / "train.py", ROOT / "src" / "evaluate.py"]
    for folder in ("data", "models"):
        paths.extend(sorted((ROOT / "src" / folder).glob("*.py")))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def runner_hashes():
    return {f"scripts/{name}": sha256_file(ROOT / "scripts" / name)
            for name in ("run_enhanced.py", "verify_run.py")}


def unique_pairs(pairs):
    return torch.unique(pairs.detach().cpu(), dim=1)


def prepare_graph(data):
    data = data.clone().cpu()
    for node in data.node_types:
        data[node].x = data[node].x.float()
    for edge in data.edge_types:
        if "edge_attr" in data[edge]:
            data[edge].edge_attr = data[edge].edge_attr.float()
    return data


def snapshot_splits(data, run_id, seed):
    store = data[G2P]
    return {
        "run_id": run_id, "seed": seed, "protocol": SPLIT_PROTOCOL,
        "edge_index": store.edge_index.detach().cpu().clone(),
        "edge_indices": {et: data[et].edge_index.detach().cpu().clone() for et in data.edge_types},
        **{key: store[key].detach().cpu().clone() for key in
           ("train_mask", "valid_mask", "test_mask", "neg_train", "neg_valid", "neg_test")},
    }


def restore_splits(data, saved):
    """Validate provenance against the hashed graph; never resample during verification."""
    if set(saved["edge_indices"]) != set(data.edge_types):
        raise ValueError("saved graph relation types differ")
    for edge, indices in saved["edge_indices"].items():
        if not torch.equal(data[edge].edge_index.cpu(), indices):
            raise ValueError(f"saved edge_index differs from graph: {edge}")
    if not torch.equal(data[G2P].edge_index.cpu(), saved["edge_index"]):
        raise ValueError("target edge_index differs from graph")
    store = data[G2P]
    for key in ("train_mask", "valid_mask", "test_mask", "neg_train", "neg_valid", "neg_test"):
        store[key] = saved[key].clone()
    validate_splits(data)
    return data


def validate_splits(data):
    store = data[G2P]
    edges = store.edge_index.cpu()
    nG, nP = int(data["gene"].num_nodes), int(data["phenotype"].num_nodes)

    def check_pairs(pairs):
        if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.size(0) != 2:
            raise ValueError("pairs must be LongTensor [2, N]")
        if (pairs < 0).any() or (pairs[0] >= nG).any() or (pairs[1] >= nP).any():
            raise ValueError("pair outside graph")

    check_pairs(edges)
    if unique_pairs(edges).size(1) != edges.size(1):
        raise ValueError("deduplicate target pairs before splitting; graph must have unique targets")
    indices = []
    for name in ("train", "valid", "test"):
        idx = store[f"{name}_mask"].cpu()
        if idx.dtype != torch.long or idx.ndim != 1 or not idx.numel():
            raise ValueError(f"{name} indices must be a nonempty LongTensor")
        indices.append(idx)
        neg = store[f"neg_{name}"].cpu()
        check_pairs(neg)
        if not neg.size(1):
            raise ValueError(f"{name} requires negatives")
        positive_ids = edges[0] * nP + edges[1]
        if torch.isin(neg[0] * nP + neg[1], positive_ids).any():
            raise ValueError(f"{name} negatives overlap observed positives")
    if not torch.equal(torch.cat(indices).sort().values, torch.arange(edges.size(1))):
        raise ValueError("split indices must be a disjoint, complete partition")


def compute_metrics(logits, sigmoid_scores, data, run_id):
    logits = logits.detach().cpu().float()
    sigmoid_scores = sigmoid_scores.detach().cpu().float()
    if not torch.isfinite(logits).all() or not torch.isfinite(sigmoid_scores).all():
        raise FloatingPointError("nonfinite evaluation scores")
    store = data[G2P]
    train, valid, test = [store.edge_index[:, store[f"{name}_mask"]].cpu()
                          for name in ("train", "valid", "test")]
    neg = store.neg_test.cpu()
    return {
        "run_id": run_id, "protocols": METRIC_PROTOCOL,
        "standard_metrics": {
            "ranking": standard_ranking_metrics(logits, train, valid, test),
            "binary_logits": binary_metrics(logits, test, neg),
            "macro_binary_logits": binary_metrics_by_phenotype(logits, test, neg),
            "head_tail_binary_logits": split_by_frequency(logits, data, train, test, neg),
            "head_tail_counts": frequency_group_summary(train, test, neg),
            "sampling_counts": {name: {"positive_pairs": int(store[f"{name}_mask"].numel()),
                                       "negative_draws": int(store[f"neg_{name}"].size(1)),
                                       "unique_negative_pairs": int(unique_pairs(store[f"neg_{name}"]).size(1))}
                                for name in ("train", "valid", "test")},
        },
        "legacy_sigmoid_metrics": {
            "binary": binary_metrics(sigmoid_scores, test, neg),
            "ranking": ranking_metrics(sigmoid_scores, train, test),
            "head_tail_binary": split_by_frequency(sigmoid_scores, data, train, test, neg),
        },
    }


def numeric_policy():
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "dtype": "float32", "autocast": False,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def environment(device):
    packages = {}
    for name in ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "numeric_policy": numeric_policy()}


def cached_explanation(model, run_id):
    """Only caches from the scoring forward; attention weights are not causal attribution."""
    if not isinstance(model, HiGNN):
        return None
    return {"run_id": run_id, "task_pooling": model.task_pooling,
            "interpretation": "forward attention/pooling weights; not faithful or causal explanation",
            "task_gamma": model.gamma.detach().cpu().clone() if model.gamma is not None else None,
            "rel_beta": [{node: (beta.detach().cpu().clone(), relations)
                          for node, (beta, relations) in layer.cache.get("rel_beta", {}).items()}
                         for layer in model.convs]}


def run_enhanced(name, seed=0, epochs=150, hidden=64, layers=2, dropout=0.2,
                 device=None, output_root=None, explain=False, graph_path=None,
                 lr=5e-3, wd=5e-4):
    """Always creates a unique directory, including reruns of identical seeds/configs.

    graph_path must be trusted local input (PyG graph deserialization uses pickle).
    FP32 full-matrix scoring is intentionally the same model/loss as historical runs.
    No automatic resume/skip: group by replicate_key, not directory count, in analyses.
    """
    if name not in MODELS:
        raise ValueError(f"enhanced models must be one of {MODELS}")
    if epochs < 1 or hidden < 1 or layers < 1:
        raise ValueError("epochs, hidden and layers must be positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only cpu/cuda are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    graph_path = Path(graph_path or ROOT / "data_cache" / "graph.pt").resolve()
    graph_hash = sha256_file(graph_path)
    sources, scripts = code_hashes(), runner_hashes()
    model_config = {"name": name, "hidden": hidden, "layers": layers, "dropout": dropout,
                    "task_pooling": {"hignn": "attention", "hignn_uniformtask": "uniform"}.get(name)}
    config = {"model_config": model_config, "seed": seed, "epochs": epochs, "lr": lr, "wd": wd,
              "split": SPLIT_PROTOCOL, "metrics": METRIC_PROTOCOL,
              "device": str(device), "numeric_policy": numeric_policy()}
    fingerprint = json_digest({"config": config, "code_sha256": sources,
                               "runner_sha256": scripts, "graph_sha256": graph_hash})
    replicate_key = json_digest({"model": name, "seed": seed, "graph_sha256": graph_hash})
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{name}_seed{seed}_{uuid.uuid4().hex}"
    output_root = Path(output_root or ROOT / "results" / "enhanced").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    # Explicit trusted local graph; never call load_graph()'s cached-split shortcut.
    data = prepare_graph(torch.load(graph_path, map_location="cpu", weights_only=False))
    if unique_pairs(data[G2P].edge_index).size(1) != data[G2P].edge_index.size(1):
        raise ValueError("graph targets contain duplicates; deduplicate before split_data")
    data = split_data(data, seed=seed)
    validate_splits(data)
    saved_splits = snapshot_splits(data, run_id, seed)
    torch.save(saved_splits, run_dir / "splits.pt")
    data = data.to(device)
    model = build_model(name, data, hidden, layers, dropout).float()
    with torch.autocast(device_type=device.type, enabled=False):
        model, selection = train_one(model, data, epochs=epochs, lr=lr, wd=wd,
                                     device=device, return_selection=True)
        model.eval()
        with torch.no_grad():
            sigmoid_scores = model.score_all()
            logits = model.logits.detach().cpu().float().clone()
            # Capture before any other forward. In particular, never call explain().
            explanation = cached_explanation(model, run_id) if explain else None
            sigmoid_scores = sigmoid_scores.detach().cpu().float()
    checkpoint = {"run_id": run_id, "model_config": model_config, "config": config,
                  "selection": selection,
                  "state_dict": {key: value.detach().cpu().clone()
                                 for key, value in model.state_dict().items()}}
    torch.save(checkpoint, run_dir / "best_model.pt")
    torch.save({"run_id": run_id, "logits": logits}, run_dir / "FP32logits.pt")
    metrics = compute_metrics(logits, sigmoid_scores, data, run_id)
    write_json(run_dir / "metrics.json", metrics)
    if explanation is not None:
        torch.save(explanation, run_dir / "explanation.pt")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if graph_hash != sha256_file(graph_path) or sources != code_hashes() or scripts != runner_hashes():
        raise RuntimeError("code or graph changed during run; artifacts are not finalized")
    artifacts = ["best_model.pt", "splits.pt", "FP32logits.pt", "metrics.json"]
    if explanation is not None:
        artifacts.append("explanation.pt")
    manifest = {
        "schema_version": 1, "status": "complete", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "config_code_graph_fingerprint": fingerprint, "replicate_key": replicate_key,
        "replicate_notice": "same model/graph/seed = rerun, not an additional independent replicate; no auto-skip",
        "graph": {"path": str(graph_path), "sha256": graph_hash},
        "code_sha256": sources, "runner_sha256": scripts,
        "artifacts_sha256": {name: sha256_file(run_dir / name) for name in artifacts},
        "selection": selection, "environment": environment(device),
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "gpu_memory_scope": "this process only; other processes/utilization are not measured",
        "trust": "torch.load only trusted local files; hashes detect corruption, not authenticity",
    }
    write_json(run_dir / "manifest.json", manifest)
    with (run_dir / "manifest.sha256").open("x", encoding="ascii") as stream:
        stream.write(sha256_file(run_dir / "manifest.json") + "\n")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "enhanced")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    models, seeds = list(dict.fromkeys(args.models)), list(dict.fromkeys(args.seeds))
    if len(models) != len(args.models) or len(seeds) != len(args.seeds):
        print("Duplicate models/seeds removed; they are not independent replicates.")
    print(json.dumps({"budget": {"models": models, "seeds": seeds, "runs": len(models) * len(seeds),
                                "epochs_per_run": args.epochs,
                                "total_training_epochs": len(models) * len(seeds) * args.epochs,
                                "device": args.device or "cuda if available else cpu",
                                "execution": "serial; FP32 full graph/matrix; no time estimate",
                                "output_root": str(args.output_root.resolve()),
                                "reruns": "unique folders; do not count repeated replicate_key as independent"}},
                     indent=2), flush=True)
    for name in models:
        for seed in seeds:
            directory = run_enhanced(name, seed, epochs=args.epochs, device=args.device,
                                     output_root=args.output_root, explain=args.explain)
            print(str(directory), flush=True)


if __name__ == "__main__":
    main()
