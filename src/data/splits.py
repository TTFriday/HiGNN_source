# -*- coding: utf-8 -*-
"""gene-phenotype 目标层的训练/验证/测试划分与负采样（防泄漏）。

用法:  python -m src.data.splits
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch_geometric.data import HeteroData

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(HERE, "data_cache")
G2P = ("gene", "assoc", "phenotype")


def split_data(data: HeteroData, seed: int = 0, ratios=(0.8, 0.1, 0.1), neg_ratio: int = 1):
    rng = np.random.default_rng(seed)
    ei = data[G2P].edge_index.numpy()          # [2, E]  gene, phenotype
    E = ei.shape[1]
    perm = rng.permutation(E)
    n_tr, n_va = int(E * ratios[0]), int(E * ratios[1])
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    nG, nP = data["gene"].num_nodes, data["phenotype"].num_nodes
    pos_mask = np.zeros((nG, nP), dtype=bool)
    pos_mask[ei[0], ei[1]] = True

    def sample_neg(idx_pos, k):
        # 按表型的正例数量比例抽负例，避开任何已观测正例
        ph = ei[1, idx_pos]
        w = np.bincount(ph, minlength=nP).astype(np.float64)
        w = w / w.sum()
        phn = rng.choice(nP, size=k, p=w)
        gn = rng.integers(0, nG, size=k)
        bad = pos_mask[gn, phn]
        tries = 0
        while bad.any() and tries < 10:
            fix = np.where(bad)[0]
            gn[fix] = rng.integers(0, nG, size=len(fix))
            bad = pos_mask[gn, phn]
            tries += 1
        keep = ~pos_mask[gn, phn]
        return np.stack([gn[keep], phn[keep]])

    ntr = len(tr) * neg_ratio
    data[G2P].train_mask = torch.tensor(tr, dtype=torch.long)
    data[G2P].valid_mask = torch.tensor(va, dtype=torch.long)
    data[G2P].test_mask = torch.tensor(te, dtype=torch.long)
    data[G2P].neg_train = torch.tensor(sample_neg(tr, ntr), dtype=torch.long)
    data[G2P].neg_valid = torch.tensor(sample_neg(va, len(va) * 10), dtype=torch.long)
    data[G2P].neg_test = torch.tensor(sample_neg(te, len(te) * 10), dtype=torch.long)
    return data


def load_graph(with_split=True, seed=0):
    data = torch.load(os.path.join(CACHE, "graph.pt"), weights_only=False)
    if with_split and not hasattr(data[G2P], "train_mask"):
        data = split_data(data, seed=seed)
    return data


if __name__ == "__main__":
    data = load_graph()
    s = data[G2P]
    print("train/valid/test 正例:", s.train_mask.numel(), s.valid_mask.numel(), s.test_mask.numel())
    print("负例:", s.neg_train.shape, s.neg_valid.shape, s.neg_test.shape)
    torch.save(data, os.path.join(CACHE, "graph_split.pt"))
    print("已保存 graph_split.pt")
