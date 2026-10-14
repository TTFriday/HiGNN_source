# -*- coding: utf-8 -*-
"""HiGNN：面向有符号、证据加权多层异构图的关系感知层次注意力网络（论文 A 主模型）。

三级可检查权重（不自动构成忠实解释或因果归因）：
  L1 邻居级：每条边在该关系邻域中的归一化注意力 alpha；
  L2 关系级：每种入关系的语义混合权重 beta；
  L3 任务级：表型条件下全局基因池化权重 gamma。
边属性 edge_attr = [sign, log1p(evidence), coefficient_nonzero_indicator]。
第三属性不是可信概率，当前网络不独立使用。
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn
from torch.nn import Parameter, Linear, Dropout

G2P = ("gene", "assoc", "phenotype")
NEG_G2P = ("phenotype", "rev_assoc", "gene")


def mp_edge_types(data):
    """消息传递使用的关系：排除目标层 gene<->phenotype，防止监督信号泄漏。"""
    out = []
    for s, r, d in data.edge_types:
        if "assoc" in r:
            continue
        out.append((s, r, d))
    return out


class RelAttnLayer(torch.nn.Module):
    """一层：逐关系节点级注意力聚合 + 关系级语义注意力融合 + 自环。"""

    def __init__(self, edge_types, node_types, hidden, use_sign=True, use_evidence=True,
                 use_rel_attn=True, dropout=0.0):
        super().__init__()
        self.edge_types = edge_types
        self.node_types = node_types
        self.use_sign, self.use_evidence, self.use_rel_attn = use_sign, use_evidence, use_rel_attn
        key = lambda et: f"{et[0]}__{et[1]}__{et[2]}"
        self.src_lin = torch.nn.ModuleDict({key(et): Linear(hidden, hidden) for et in edge_types})
        self.att = Parameter(torch.randn(len(edge_types), hidden) * 0.1)      # 边注意力向量
        self.ev_w = Parameter(torch.zeros(len(edge_types)))                  # 证据强度偏置
        self.sg_w = Parameter(torch.zeros(len(edge_types)))                  # 符号偏置（进 logit）
        self.sg_m = Parameter(torch.zeros(len(edge_types)))                  # 符号调制（进 message）
        self.rel_att = Parameter(torch.randn(len(edge_types), hidden) * 0.1) # 关系级注意力
        self.self_lin = torch.nn.ModuleDict({t: Linear(hidden, hidden) for t in node_types})
        self.drop = Dropout(dropout)
        self.cache = {}   # 可解释输出: edge_attn / rel_beta

    def forward(self, h, data):
        # 自环项
        out = {t: self.self_lin[t](h[t]) for t in self.node_types}
        incoming = {t: [] for t in self.node_types}
        rel_pos = {t: [] for t in self.node_types}
        for i, et in enumerate(self.edge_types):
            s, r, d = et
            store = data[et]
            ei = store.edge_index
            ea = store.edge_attr
            n_d = h[d].size(0)
            q = self.src_lin[f"{s}__{r}__{d}"](h[s])[ei[0]]                    # (E, hidden)
            logit = (q * self.att[i]).sum(-1)
            if self.use_evidence:
                logit = logit + self.ev_w[i] * ea[:, 1]
            if self.use_sign:
                logit = logit + self.sg_w[i] * ea[:, 0] * ea[:, 1]
            alpha = _softmax_scatter(logit, ei[1], n_d)                 # 节点级注意力
            msg = q * alpha.unsqueeze(-1)
            if self.use_sign:
                msg = msg * (1.0 + torch.tanh(self.sg_m[i]) * ea[:, 0]).unsqueeze(-1)
            agg = torch.zeros_like(h[d]).index_add_(0, ei[1], self.drop(msg))
            incoming[d].append(agg)
            rel_pos[d].append(i)
            self.cache.setdefault("edge_attn", {})[f"{s}__{r}__{d}"] = alpha.detach()
        # 关系级注意力融合
        betas = {}
        for t, aggs in incoming.items():
            if not aggs:
                continue
            A = torch.stack(aggs, dim=1)                                        # (N, R, hidden)
            if self.use_rel_attn and A.size(1) > 1:
                idx = torch.tensor(rel_pos[t], device=A.device)
                score = (A * self.rel_att[idx].unsqueeze(0)).sum(-1)             # (N, R)
                beta = torch.softmax(score, dim=1)
                mix = (A * beta.unsqueeze(-1)).sum(1)
            else:
                beta = torch.full(A.shape[:2], 1.0 / A.size(1), device=A.device)
                mix = A.mean(1)
            out[t] = out[t] + mix
            betas[t] = (beta.detach(), [self.edge_types[i] for i in rel_pos[t]])
        self.cache["rel_beta"] = betas
        return out


def _softmax_scatter(logit, index, n):
    m = torch.zeros(n, device=logit.device).index_reduce_(0, index, logit, "amax", include_self=False)
    e = torch.exp(logit - m.index_select(0, index))
    z = torch.zeros(n, device=logit.device).index_add_(0, index, e)
    return e / (z.index_select(0, index) + 1e-12)


class HiGNN(torch.nn.Module):
    def __init__(self, data, hidden=64, layers=2, use_sign=True, use_evidence=True,
                 use_rel_attn=True, use_task_attn=True, dropout=0.2, task_pooling=None):
        super().__init__()
        self.data = data
        self.node_types = list(data.node_types)
        self.edge_types = mp_edge_types(data)
        # None preserves the historical use_task_attn switch, including its no-context ablation.
        self.task_pooling = task_pooling if task_pooling is not None else (
            "attention" if use_task_attn else "none")
        if self.task_pooling not in {"attention", "uniform", "none"}:
            raise ValueError("task_pooling must be attention, uniform or none")
        self.use_task_attn = self.task_pooling != "none"
        self.hidden = hidden
        self.inp = torch.nn.ModuleDict({t: Linear(data[t].x.size(1), hidden) for t in self.node_types})
        self.base = torch.nn.ParameterDict({
            t: Parameter(torch.randn(int(data[t].num_nodes), hidden) * 0.1) for t in self.node_types})
        self.convs = torch.nn.ModuleList([
            RelAttnLayer(self.edge_types, self.node_types, hidden, use_sign, use_evidence,
                         use_rel_attn, dropout) for _ in range(layers)])
        self.out_g = Linear(hidden, hidden)
        self.out_p = Linear(hidden, hidden)
        self.task_ctx = Linear(hidden, hidden)
        self.task_lin = Linear(hidden, hidden)

    def encode(self):
        h = {t: torch.tanh(self.inp[t](self.data[t].x)) + self.base[t] for t in self.node_types}
        for c in self.convs:
            hn = c(h, self.data)
            h = {t: Fn.elu(hn[t]) for t in self.node_types}
        return h

    # ---------- 打分头 ----------
    def phenotype_context(self, h):
        """Task pooling weights (nP, nG), not a claim of faithful explanation."""
        if self.task_pooling == "none":
            self.gamma = None
            return h["phenotype"]
        if self.task_pooling == "uniform":
            nP, nG = h["phenotype"].size(0), h["gene"].size(0)
            self.gamma = h["gene"].new_full((nP, nG), 1.0 / nG)
            # Preserve global gene context rather than removing the task head.
            return h["gene"].mean(dim=0, keepdim=True) + h["phenotype"]
        g = self.task_lin(h["gene"])                                    # (nG, d)
        ctx = self.task_ctx(h["phenotype"])                             # (nP, d)
        logit = ctx @ g.t() / (self.hidden ** 0.5)
        gamma = torch.softmax(logit, dim=1)
        self.gamma = gamma.detach()
        z = gamma @ h["gene"] + h["phenotype"]                          # 注意力池化 + 自身
        return z

    def score_all(self):
        h = self.encode()
        Hg = self.out_g(h["gene"])
        z = self.phenotype_context(h)
        Hp = self.out_p(z)
        self.logits = (Hg @ Hp.t()) / (self.hidden ** 0.5)     # 预 sigmoid 打分（排序用，避免饱和并列）
        return torch.sigmoid(self.logits)                      # (nG, nP)

    def explain(self):
        """返回三级内生注意力，供解释性与案例作图。"""
        with torch.no_grad():
            self.score_all()
        return {
            "task_gamma": getattr(self, "gamma", None),
            "layers": [{
                "edge_attn": c.cache.get("edge_attn", {}),
                "rel_beta": c.cache.get("rel_beta", {}),
            } for c in self.convs],
        }

    def parameter_report(self):
        rep = {}
        for li, c in enumerate(self.convs):
            rep[f"L{li}_ev_w"] = c.ev_w.detach().cpu().numpy().round(3).tolist()
            rep[f"L{li}_sg_w"] = c.sg_w.detach().cpu().numpy().round(3).tolist()
        return rep
