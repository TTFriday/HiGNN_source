# -*- coding: utf-8 -*-
"""历史 sigmoid-score 评估接口，以及独立的标准 logit 排序/表型宏平均指标。"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def binary_metrics(S, pos, neg):
    """历史 pooled AUROC/AUPRC；pos/neg 为 LongTensor [2, N] (gene, phenotype)。

    使用调用方提供的分数（历史调用为 sigmoid 分数），不进行表型宏平均；
    保留 sklearn 对空输入/单一类别的原有行为。
    """
    p = S[pos[0].long(), pos[1].long()]
    n = S[neg[0].long(), neg[1].long()]
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    s = torch.cat([p, n]).detach().cpu().numpy()
    return {"AUROC": float(roc_auc_score(y, s)), "AUPRC": float(average_precision_score(y, s))}


def ranking_metrics(S, train_pos, test_pos, ks=(10, 50)):
    """历史 sigmoid ranking 接口，原有计算及返回键保持不变。

    S 是调用方提供的分数（历史调用为 sigmoid 分数），本函数不再做 sigmoid。
    仅将同表型训练阳性的分数设为 -inf；它们仍留在排序末端，未过滤验证阳性，
    且 np.argsort 没有显式 tie break。MRR/Hits@K 按测试阳性 pair 平均。
    旧键 NDCG@10 实为所有测试 pair 的 mean(1/log2(rank+1))（DRS），
    rank 从 1 开始，无 @10 截断或 ideal DCG 归一化，不是标准 NDCG@10。
    空输入仍保留原有 NaN 行为。
    """
    S_np = S.detach().cpu().numpy()
    tp_by_p, tr_by_p = {}, {}
    for g, p in test_pos.cpu().numpy().T:
        tp_by_p.setdefault(int(p), []).append(int(g))
    for g, p in train_pos.cpu().numpy().T:
        tr_by_p.setdefault(int(p), []).append(int(g))
    rr, ndcg, hits = [], [], {k: [] for k in ks}
    for p, gs in tp_by_p.items():
        s = S_np[:, p].copy()
        blocked = tr_by_p.get(p, [])
        if blocked:
            s[np.array(blocked)] = -np.inf
        order = np.argsort(-s)
        rank_of = np.empty(len(order), dtype=np.int64)
        rank_of[order] = np.arange(len(order))
        for g in gs:
            r = int(rank_of[g])
            rr.append(1.0 / (r + 1))
            ndcg.append(1.0 / np.log2(r + 2))
            for k in ks:
                hits[k].append(1.0 if r < k else 0.0)
    out = {"MRR": float(np.mean(rr)), "NDCG@10": float(np.mean(ndcg))}
    for k in ks:
        out[f"Hits@{k}"] = float(np.mean(hits[k]))
    return out


def split_by_frequency(S, data, train_pos, test_pos, neg_test, n_min=20):
    """历史 head/tail pooled AUROC/AUPRC，按训练阳性数 >= n_min 分 head。

    保留原行为：仅报告测试正、负 pair 各至少 5 个的组，不是表型宏平均。
    """
    nP = int(data["phenotype"].num_nodes)
    cnt = np.bincount(train_pos[1].cpu().numpy(), minlength=nP)
    res = {}
    for name, mask in [("head", cnt >= n_min), ("tail", cnt < n_min)]:
        sel = np.where(mask)[0]
        if sel.size == 0:
            continue
        sel_set = set(sel.tolist())
        pm = np.isin(test_pos[1].cpu().numpy(), sel)
        nm = np.isin(neg_test[1].cpu().numpy(), sel)
        if pm.sum() < 5 or nm.sum() < 5:
            continue
        res[name] = binary_metrics(S, test_pos[:, torch.tensor(pm.nonzero()[0])],
                                   neg_test[:, torch.tensor(nm.nonzero()[0])])
    return res


def _finite_scores(scores):
    """Return a finite, real FP32 NumPy [gene, phenotype] score matrix."""
    if isinstance(scores, torch.Tensor):
        if scores.is_complex():
            raise ValueError("scores must be real")
        scores = scores.detach().to(device="cpu", dtype=torch.float32).numpy()
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise ValueError("scores must be a real numeric [gene, phenotype] matrix")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = scores.astype(np.float32, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("scores must remain finite in FP32")
    return scores


def _pair_array(pairs, name, shape=None, unique=True):
    """Validate nonnegative integer pairs; allow repeated sampled negatives explicitly."""
    if isinstance(pairs, torch.Tensor):
        pairs = pairs.detach().cpu().numpy()
    pairs = np.asarray(pairs)
    if pairs.ndim == 1 and pairs.size == 0:
        pairs = np.empty((2, 0), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, N]")
    if not np.issubdtype(pairs.dtype, np.integer):
        if pairs.size == 0:
            return np.empty((2, 0), dtype=np.int64)
        raise ValueError(f"{name} must contain finite integer indices")
    if np.any(pairs < 0):
        raise ValueError(f"{name} indices must be nonnegative")
    if shape is not None and (np.any(pairs[0] >= shape[0]) or np.any(pairs[1] >= shape[1])):
        raise ValueError(f"{name} indices are outside the score matrix")
    if np.any(pairs > np.iinfo(np.int64).max):
        raise ValueError(f"{name} indices exceed int64 range")
    pairs = pairs.astype(np.int64, copy=False)
    if unique and len({(int(g), int(p)) for g, p in pairs.T}) != pairs.shape[1]:
        raise ValueError(f"{name} contains duplicate pairs")
    return pairs


def _genes_by_phenotype(pairs):
    grouped = {}
    for gene, phenotype in pairs.T:
        grouped.setdefault(int(phenotype), set()).add(int(gene))
    return grouped


def standard_ranking_metrics(logits, train_pos, valid_pos, test_pos, ks=(10, 50)):
    """标准排序：FP32 logits，分数降序，gene index 升序打破平局（lexsort）。

    输入为 Tensor 或 NumPy；logits 为 [gene, phenotype]，pair 为 [2, N]。
    候选集真正移除同表型 train + validation positives（不留在末端），
    不过滤其他 test positives。test 与 train/validation 的同一 pair 重叠抛
    ValueError。分数必须有限且 FP32 可表示；pair 必须唯一、整数且不越界。

    pair_MRR / pair_Hits@K 按测试 pair 平均；macro_Recall@K、macro_NDCG@K
    按含测试阳性的表型等权平均，binary relevance 的 IDCG 使用
    min(K, 该表型测试阳性数) 个理想命中。macro_MRR_first 为每个表型首个
    阳性 reciprocal rank 的宏平均。无测试阳性时各指标为 None、计数为 0。
    """
    scores = _finite_scores(logits)
    train = _genes_by_phenotype(_pair_array(train_pos, "train_pos", scores.shape))
    valid = _genes_by_phenotype(_pair_array(valid_pos, "valid_pos", scores.shape))
    test_pairs = _pair_array(test_pos, "test_pos", scores.shape)
    test = _genes_by_phenotype(test_pairs)
    ks = tuple(ks)
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0
           for k in ks):
        raise ValueError("ks must contain positive integers")
    ks = tuple(dict.fromkeys(int(k) for k in ks))
    for phenotype, genes in test.items():
        if genes & (train.get(phenotype, set()) | valid.get(phenotype, set())):
            raise ValueError("test_pos overlaps train_pos or valid_pos for the same phenotype")

    out = {"pair_MRR": None, "macro_MRR_first": None,
           "n_test_phenotypes": len(test), "n_test_pairs": int(test_pairs.shape[1])}
    for k in ks:
        out.update({f"pair_Hits@{k}": None, f"macro_Recall@{k}": None,
                    f"macro_NDCG@{k}": None})
    if not test:
        return out

    reciprocal_ranks, first_rr = [], []
    hits = {k: 0 for k in ks}
    recalls, ndcgs = {k: [] for k in ks}, {k: [] for k in ks}
    gene_ids = np.arange(scores.shape[0], dtype=np.int64)
    for phenotype, positives in test.items():
        blocked = train.get(phenotype, set()) | valid.get(phenotype, set())
        candidates = gene_ids[~np.isin(gene_ids, list(blocked))]
        order = candidates[np.lexsort((candidates, -scores[candidates, phenotype]))]
        # Only actual candidates receive ranks; blocked genes never enter the list.
        rank_of = np.full(scores.shape[0], -1, dtype=np.int64)
        rank_of[order] = np.arange(1, len(order) + 1)
        ranks = rank_of[np.array(sorted(positives), dtype=np.int64)]
        reciprocal_ranks.extend((1.0 / ranks).tolist())
        first_rr.append(1.0 / int(ranks.min()))
        for k in ks:
            in_top_k = ranks[ranks <= k]
            hits[k] += len(in_top_k)
            recalls[k].append(len(in_top_k) / len(positives))
            dcg = np.sum(1.0 / np.log2(in_top_k + 1))
            ideal_ranks = np.arange(1, min(k, len(positives)) + 1)
            idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
            ndcgs[k].append(float(dcg / idcg))

    out["pair_MRR"] = float(np.mean(reciprocal_ranks))
    out["macro_MRR_first"] = float(np.mean(first_rr))
    for k in ks:
        out[f"pair_Hits@{k}"] = hits[k] / out["n_test_pairs"]
        out[f"macro_Recall@{k}"] = float(np.mean(recalls[k]))
        out[f"macro_NDCG@{k}"] = float(np.mean(ndcgs[k]))
    return out


def binary_metrics_by_phenotype(S, pos, neg):
    """表型等权 macro_AUROC/macro_AUPRC，仅纳入同时有正、负样本的表型。

    S 为有限 FP32 可表示的 [gene, phenotype] logits 或分数；pos/neg 为
    [2, N] pair；正例必须唯一，负例保留有放回采样的重复项，互相重叠抛 ValueError。n_phenotypes 为纳入宏平均的
    表型数，n_pairs 为这些表型的正负 pair 总数；无可评估表型时指标为 None，
    两个计数为 0。AUPRC 使用 sklearn 的 average precision 定义。
    """
    scores = _finite_scores(S)
    pos_array = _pair_array(pos, "pos", scores.shape)
    neg_array = _pair_array(neg, "neg", scores.shape, unique=False)
    positive = _genes_by_phenotype(pos_array)
    negative = _genes_by_phenotype(neg_array)
    phenotypes = sorted(positive.keys() & negative.keys())
    aucs, aps, n_pairs = [], [], 0
    for phenotype in phenotypes:
        if positive[phenotype] & negative[phenotype]:
            raise ValueError("pos and neg overlap for the same phenotype")
        pos_genes = pos_array[0, pos_array[1] == phenotype].tolist()
        neg_genes = neg_array[0, neg_array[1] == phenotype].tolist()
        labels = np.concatenate([np.ones(len(pos_genes)), np.zeros(len(neg_genes))])
        values = scores[pos_genes + neg_genes, phenotype]
        aucs.append(float(roc_auc_score(labels, values)))
        aps.append(float(average_precision_score(labels, values)))
        n_pairs += len(labels)
    return {"macro_AUROC": float(np.mean(aucs)) if aucs else None,
            "macro_AUPRC": float(np.mean(aps)) if aps else None,
            "n_phenotypes": len(phenotypes), "n_pairs": n_pairs}


def frequency_group_summary(train_pos, test_pos, neg_test, n_min=20):
    """按同表型训练阳性数 >= n_min 分 head，否则 tail，不计算模型指标。

    输入为非负整数 [2, N] pair，训练/测试正例唯一，负例重复按样本数保留；test_pos/neg_test 重叠抛 ValueError。
    每组 n_phenotypes 只统计在 test_pos 或 neg_test 出现的表型（包含单类
    表型），不统计仅在训练集出现的表型。n_test_positives/n_test_negatives
    为该组测试 pair 数；prevalence = positives / (positives + negatives)，
    是组内 pair prevalence 而非表型宏平均。始终返回 head/tail，空组计数
    为 0、prevalence 为 None。n_min 为非负整数。
    """
    if (isinstance(n_min, (bool, np.bool_)) or not isinstance(n_min, (int, np.integer))
            or n_min < 0):
        raise ValueError("n_min must be a nonnegative integer")
    train = _genes_by_phenotype(_pair_array(train_pos, "train_pos"))
    positive = _genes_by_phenotype(_pair_array(test_pos, "test_pos"))
    neg_array = _pair_array(neg_test, "neg_test", unique=False)
    negative = _genes_by_phenotype(neg_array)
    result = {name: {"n_phenotypes": 0, "n_test_positives": 0,
                     "n_test_negatives": 0, "prevalence": None}
              for name in ("head", "tail")}
    for phenotype in sorted(positive.keys() | negative.keys()):
        pos_genes, neg_genes = positive.get(phenotype, set()), negative.get(phenotype, set())
        if pos_genes & neg_genes:
            raise ValueError("test_pos and neg_test overlap for the same phenotype")
        name = "head" if len(train.get(phenotype, set())) >= n_min else "tail"
        group = result[name]
        group["n_phenotypes"] += 1
        group["n_test_positives"] += len(pos_genes)
        group["n_test_negatives"] += int(np.count_nonzero(neg_array[1] == phenotype))
    for group in result.values():
        total = group["n_test_positives"] + group["n_test_negatives"]
        if total:
            group["prevalence"] = group["n_test_positives"] / total
    return result
