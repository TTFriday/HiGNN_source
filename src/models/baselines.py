# -*- coding: utf-8 -*-
"""论文 A 的对比基线：MLP / R-GCN 风格 / GATv2 风格 / HAN 风格，统一接口。"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn
from torch.nn import Parameter, Linear

from .hignn import mp_edge_types


class MeanHetGNN(torch.nn.Module):
    """简化关系特定求和控制模型；不等同于原版 R-GCN/CompGCN，无注意力、无符号。"""

    def __init__(self, data, hidden=64, layers=2, dropout=0.2, use_edge_weight=False):
        super().__init__()
        self.data = data
        self.node_types = list(data.node_types)
        self.edge_types = mp_edge_types(data)
        self.use_edge_weight = use_edge_weight
        self.inp = torch.nn.ModuleDict({t: Linear(data[t].x.size(1), hidden) for t in self.node_types})
        self.base = torch.nn.ParameterDict({
            t: Parameter(torch.randn(int(data[t].num_nodes), hidden) * 0.1) for t in self.node_types})
        key = lambda et: f"{et[0]}__{et[1]}__{et[2]}"
        self.src_lin = torch.nn.ModuleList(
            [torch.nn.ModuleDict({key(et): Linear(hidden, hidden) for et in self.edge_types})
             for _ in range(layers)])
        self.self_lin = torch.nn.ModuleList(
            [torch.nn.ModuleDict({t: Linear(hidden, hidden) for t in self.node_types})
             for _ in range(layers)])
        self.out_g, self.out_p = Linear(hidden, hidden), Linear(hidden, hidden)
        self.hidden = hidden
        self.drop = torch.nn.Dropout(dropout)

    def encode(self):
        h = {t: torch.tanh(self.inp[t](self.data[t].x)) + self.base[t] for t in self.node_types}
        for li in range(len(self.src_lin)):
            out = {t: self.self_lin[li][t](h[t]) for t in self.node_types}
            for et in self.edge_types:
                s, r, d = et
                store = self.data[et]
                ei = store.edge_index
                m = self.src_lin[li][f"{s}__{r}__{d}"](h[s])[ei[0]]
                if self.use_edge_weight:
                    m = m * store.edge_attr[:, 1:2].clamp(max=5.0) / 5.0
                agg = torch.zeros_like(h[d]).index_add_(0, ei[1], self.drop(m))
                out[d] = out[d] + agg
            h = {t: Fn.elu(out[t]) for t in self.node_types}
        return h

    def score_all(self):
        h = self.encode()
        self.logits = (self.out_g(h["gene"]) @ self.out_p(h["phenotype"]).t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


class HANNet(MeanHetGNN):
    """HAN-style：求和聚合 + 关系级 softmax；不是原版 HAN（无符号/无边证据）。"""

    def __init__(self, data, hidden=64, layers=2, dropout=0.2):
        super().__init__(data, hidden, layers, dropout)
        self.rel_att = torch.nn.ParameterList([
            Parameter(torch.randn(len(self.edge_types), hidden) * 0.1) for _ in range(layers)])

    def encode(self):
        h = {t: torch.tanh(self.inp[t](self.data[t].x)) + self.base[t] for t in self.node_types}
        for li in range(len(self.src_lin)):
            out = {t: self.self_lin[li][t](h[t]) for t in self.node_types}
            incoming, rel_pos = {t: [] for t in self.node_types}, {t: [] for t in self.node_types}
            for i, et in enumerate(self.edge_types):
                s, r, d = et
                store = self.data[et]
                ei = store.edge_index
                m = self.drop(self.src_lin[li][f"{s}__{r}__{d}"](h[s])[ei[0]])
                agg = torch.zeros_like(h[d]).index_add_(0, ei[1], m)
                incoming[d].append(agg)
                rel_pos[d].append(i)
            for t, aggs in incoming.items():
                if not aggs:
                    continue
                A = torch.stack(aggs, dim=1)
                if A.size(1) > 1:
                    idx = torch.tensor(rel_pos[t], device=A.device)
                    score = (A * self.rel_att[li][idx].unsqueeze(0)).sum(-1)
                    beta = torch.softmax(score, dim=1)
                    out[t] = out[t] + (A * beta.unsqueeze(-1)).sum(1)
                else:
                    out[t] = out[t] + A[:, 0]
            h = {t: Fn.elu(out[t]) for t in self.node_types}
        return h


class MLPModel(torch.nn.Module):
    """无消息传递基线：图派生节点特征 + 可学习节点嵌入的双线性打分。"""

    def __init__(self, data, hidden=64, **kw):
        super().__init__()
        self.data = data
        self.g = Linear(data["gene"].x.size(1), hidden)
        self.p = Linear(data["phenotype"].x.size(1), hidden)
        self.gb = Parameter(torch.randn(int(data["gene"].num_nodes), hidden) * 0.1)
        self.pb = Parameter(torch.randn(int(data["phenotype"].num_nodes), hidden) * 0.1)
        self.hidden = hidden

    def score_all(self):
        hg = torch.tanh(self.g(self.data["gene"].x)) + self.gb
        hp = torch.tanh(self.p(self.data["phenotype"].x)) + self.pb
        self.logits = (hg @ hp.t()) / (self.hidden ** 0.5)
        return torch.sigmoid(self.logits)


BASELINES = {
    "mlp": MLPModel,
    "rgcn": MeanHetGNN,                # 简化关系特定求和聚合
    "gatv2": lambda data, **kw: MeanHetGNN(data, use_edge_weight=True, **kw),  # 边权感知聚合(无边注意力)
    "han": HANNet,
}
