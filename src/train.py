# -*- coding: utf-8 -*-
"""训练单个模型（全图 transductive，二值交叉熵），返回测试指标与解释权重。"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.splits import load_graph, G2P
from src.models.hignn import HiGNN
from src.models.baselines import BASELINES
from src.evaluate import binary_metrics, ranking_metrics, split_by_frequency

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name, data, hidden=64, layers=2, dropout=0.2):
    if name == "hignn":
        return HiGNN(data, hidden=hidden, layers=layers, dropout=dropout)
    if name == "hignn_uniformtask":
        return HiGNN(data, hidden=hidden, layers=layers, dropout=dropout, task_pooling="uniform")
    if name == "hignn_nosign":
        return HiGNN(data, hidden=hidden, layers=layers, dropout=dropout, use_sign=False)
    if name == "hignn_noev":
        return HiGNN(data, hidden=hidden, layers=layers, dropout=dropout, use_evidence=False)
    if name == "hignn_norelattn":
        return HiGNN(data, hidden=hidden, layers=layers, dropout=dropout, use_rel_attn=False)
    if name == "hignn_notaskattn":
        return HiGNN(data, hidden=hidden, layers=layers, dropout=dropout, use_task_attn=False)
    if name in BASELINES:
        return BASELINES[name](data, hidden=hidden, layers=layers, dropout=dropout)
    raise ValueError(name)


def train_one(model, data, epochs=250, lr=5e-3, wd=5e-4, device="cpu", verbose=False,
              return_selection=False):
    if return_selection and epochs < 1:
        raise ValueError("audited training requires at least one epoch")
    model = model.to(device)
    if return_selection and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    store = data[G2P]
    tr = store.train_mask.to(device)
    pos_tr = store.edge_index[:, tr]
    neg_tr = store.neg_train.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = {"auc": -1, "state": None, "epoch": 0}
    checks, validation_history = [], []
    for ep in range(epochs):
        model.train()
        S = model.score_all()
        pos_loss = -torch.log(S[pos_tr[0], pos_tr[1]] + 1e-7).mean()
        neg_loss = -torch.log(1 - S[neg_tr[0], neg_tr[1]] + 1e-7).mean()
        loss = pos_loss + neg_loss
        if return_selection:
            if not all(torch.isfinite(t).all().item() for t in (S, model.logits, loss)):
                raise FloatingPointError(f"nonfinite training scores/loss at epoch {ep + 1}")
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if return_selection:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(torch.isfinite(grad_norm).item()) and all(
                torch.isfinite(g).all().item() for g in grads)
            nonzero = any(torch.count_nonzero(g).item() > 0 for g in grads)
            if not finite or not grads:
                raise FloatingPointError(f"nonfinite or absent gradients at epoch {ep + 1}")
        opt.step()
        if return_selection:
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise FloatingPointError(f"nonfinite parameters at epoch {ep + 1}")
            checks.append({"epoch": ep + 1, "loss": float(loss.detach()),
                           "grad_norm_before_clip": float(grad_norm),
                           "parameters_with_grad": len(grads), "nonzero_gradient": nonzero,
                           "scores_loss_finite": True, "gradients_finite": finite,
                           "parameters_finite": True})
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                Sv = model.score_all()
                if return_selection and not all(
                        torch.isfinite(t).all().item() for t in (Sv, model.logits)):
                    raise FloatingPointError(f"nonfinite validation scores at epoch {ep + 1}")
                m = binary_metrics(Sv, store.edge_index[:, store.valid_mask], store.neg_valid)
            if return_selection:
                if not np.isfinite(m["AUROC"]):
                    raise FloatingPointError("nonfinite validation AUROC")
                validation_history.append({"epoch": ep + 1, "sigmoid_AUROC": m["AUROC"]})
            if m["AUROC"] > best["auc"]:
                best = {"auc": m["AUROC"],
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                        "epoch": ep + 1}
            if verbose:
                print(f"  ep{ep+1:4d} loss={loss.item():.4f} val_AUROC={m['AUROC']:.4f}")
    if best["state"]:
        model.load_state_dict(best["state"])
    if return_selection:
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        return model, {"best_epoch": best["epoch"], "best_val_AUROC": best["auc"],
                       "training_seconds": time.perf_counter() - started,
                       "selection_protocol": "sigmoid validation AUROC; every 25 epochs and final; strict >",
                       "gradient_finite_checks": checks, "validation_history": validation_history}
    return model


def run(name, seed=0, epochs=250, hidden=64, layers=2, dropout=0.2, device=None,
        save_explain=False, log=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    data = load_graph(with_split=True, seed=seed)
    data = data.to(device)
    store = data[G2P]
    model = build_model(name, data, hidden, layers, dropout)
    if save_explain and isinstance(model, HiGNN):
        export_paths = {kind: os.path.join(RESULTS, f"{kind}_{name}_seed{seed}.pt")
                        for kind in ("explain", "score", "logits")}
        existing = [path for path in export_paths.values() if os.path.exists(path)]
        if existing:
            raise FileExistsError(f"refusing to overwrite historical exports: {existing}; use run_enhanced")
        os.makedirs(RESULTS, exist_ok=True)
    model = train_one(model, data, epochs=epochs, device=device)
    model.eval()
    with torch.no_grad():
        S = model.score_all()
        out = {"model": name, "seed": seed, "device": device, "epochs": epochs}
        out.update({f"test_{k}": v for k, v in binary_metrics(
            S, store.edge_index[:, store.test_mask], store.neg_test).items()})
        out.update({f"test_{k}": v for k, v in ranking_metrics(
            S, store.edge_index[:, store.train_mask], store.edge_index[:, store.test_mask]).items()})
        for grp, m in split_by_frequency(S, data, store.edge_index[:, store.train_mask],
                                         store.edge_index[:, store.test_mask], store.neg_test).items():
            out.update({f"test_{grp}_{k}": v for k, v in m.items()})
        if save_explain and isinstance(model, HiGNN):
            ex = model.explain()
            with open(export_paths["explain"], "xb") as stream:
                torch.save({"task_gamma": ex["task_gamma"].cpu(),
                            "rel_beta_L1": {t: (b[0].cpu(), b[1])
                                            for t, b in ex["layers"][1]["rel_beta"].items()}}, stream)
            # 预测得分矩阵 (nG, nP)：候选基因排序与案例作图的判别信号
            with open(export_paths["score"], "xb") as stream:
                torch.save(S.cpu().half(), stream)
            # 预 sigmoid 打分（排序判别、避免饱和并列）
            with open(export_paths["logits"], "xb") as stream:
                torch.save(model.logits.detach().cpu(), stream)
            out["sign_ev_params"] = model.parameter_report()
    os.makedirs(RESULTS, exist_ok=True)
    if log:
        with open(os.path.join(RESULTS, "runs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hignn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="不追加 runs.jsonl；仍会训练新模型，不是只读导出")
    a = ap.parse_args()
    r = run(a.model, a.seed, a.epochs, a.hidden, a.layers,
            save_explain=a.explain, log=not a.no_log)
    print(json.dumps(r, indent=2, ensure_ascii=False))
