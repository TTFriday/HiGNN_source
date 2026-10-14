# -*- coding: utf-8 -*-
"""合成小图冒烟测试：验证 HiGNN 与基线可前向、可反向、可训练（不依赖真实数据）。

运行: python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import HeteroData

from src.models.hignn import HiGNN
from src.models.baselines import BASELINES


def make_fake(num_gene=200, num_pheno=8, num_snp=120, num_meth=90):
    d = HeteroData()
    d["gene"].num_nodes = num_gene
    d["phenotype"].num_nodes = num_pheno
    d["snp"].num_nodes = num_snp
    d["meth"].num_nodes = num_meth
    for t, f in [("gene", 6), ("phenotype", 3), ("snp", 8), ("meth", 9)]:
        d[t].x = torch.randn(d[t].num_nodes, f)

    def rel(s, r, dt, n, signed=True):
        ei = torch.randint(0, d[s].num_nodes, (n,)), torch.randint(0, d[dt].num_nodes, (n,))
        a = torch.stack([ei[0], ei[1]])
        sign = torch.randint(-1, 2, (n,)).float() if signed else torch.zeros(n)
        ev = torch.rand(n) * 5
        d[s, r, dt].edge_index = a
        d[s, r, dt].edge_attr = torch.stack([sign, torch.log1p(ev), (sign != 0).float()], dim=1)

    rel("gene", "assoc", "phenotype", 400)
    rel("snp", "gwas", "phenotype", 300, signed=False)
    rel("snp", "annot", "gene", 500, signed=False)
    rel("snp", "cis", "gene", 300)
    rel("meth", "emqtl", "gene", 400)
    rel("meth", "ewas", "phenotype", 300)
    # 反向
    for (s, r, dt) in [("snp", "gwas", "phenotype"), ("snp", "annot", "gene"),
                       ("snp", "cis", "gene"), ("meth", "emqtl", "gene"), ("meth", "ewas", "phenotype")]:
        st = d[s, r, dt]
        d[dt, f"rev_{r}", s].edge_index = st.edge_index.flip(0)
        d[dt, f"rev_{r}", s].edge_attr = st.edge_attr.clone()
    return d


def main():
    d = make_fake()
    models = {"hignn": HiGNN(d, hidden=32, layers=2)}
    for name, cls in BASELINES.items():
        models[name] = cls(d, hidden=32, layers=2)
    for name, m in models.items():
        S = m.score_all()
        assert S.shape == (d["gene"].num_nodes, d["phenotype"].num_nodes), name
        loss = S.mean()
        loss.backward()
        grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
        assert grad_ok, f"{name}: 无梯度"
        print(f"{name:10s} forward/backward OK  S range=[{S.min():.3f},{S.max():.3f}]")
    if isinstance(models["hignn"], HiGNN):
        ex = models["hignn"].explain()
        g = ex["task_gamma"]
        assert torch.allclose(g.sum(1), torch.ones(g.size(0)), atol=1e-4)
        print("hignn explain OK: gamma 归一化正确, 层数", len(ex["layers"]))
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
