# -*- coding: utf-8 -*-
"""把 data_cache 中的证据表组装成统一的异质多层图 (PyG HeteroData)。

用法:  python -m src.data.build_graph
"""
from __future__ import annotations

import json
import os
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(HERE, "data_cache")

MAX_EDGES = {"ld": 100_000}          # 其余关系默认上限
DEFAULT_CAP = 200_000

# parquet 名 -> (src_type, relation, dst_type)
REL = {
    "gene_pheno":   ("gene", "assoc", "phenotype"),      # 预测目标层
    "snp_pheno":    ("snp", "gwas", "phenotype"),
    "snp_gene":     ("snp", "annot", "gene"),
    "snp_gene_cis": ("snp", "cis", "gene"),
    "snp_gene_trans": ("snp", "trans", "gene"),
    "snp_env":      ("snp", "env_qtl", "env"),
    "snp_snp":      ("snp", "ld", "snp"),
    "meth_pheno":   ("meth", "ewas", "phenotype"),
    "meth_gene":    ("meth", "emqtl", "gene"),
    "meth_pathway": ("meth", "epathway", "pathway"),
    "pheno_pathway": ("phenotype", "ppathway", "pathway"),
}
TYPES = ["gene", "phenotype", "snp", "meth", "pathway", "env"]


def load_edge_table(name):
    df = pd.read_pickle(os.path.join(CACHE, f"{name}.pkl"))
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce").fillna(1.0)
    corr = pd.to_numeric(df["corr"], errors="coerce") if "corr" in df else pd.Series(np.zeros(len(df)))
    df["corr"] = corr.fillna(0.0)
    # 同一对节点取证据最强的一条
    df = df.sort_values("strength", ascending=False).drop_duplicates(["src", "dst"], keep="first")
    cap = MAX_EDGES.get(name if name != "snp_snp" else "ld", DEFAULT_CAP)
    return df.head(cap)


def parse_meth_site(s):
    m = re.match(r"^([1-5])_(\d+)_(CG|CHG|CHH)", s.upper())
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None, None, "CG"


def build():
    tables = {k: load_edge_table(k) for k in REL}

    # ---- 节点集合：仅保留出现在保留边中的实体 ----
    nodes = {t: set() for t in TYPES}
    for name, (s, r, d) in REL.items():
        df = tables[name]
        nodes[s] |= set(df["src"])
        nodes[d] |= set(df["dst"])
    idx = {t: {v: i for i, v in enumerate(sorted(nodes[t]))} for t in TYPES}
    print({t: len(nodes[t]) for t in TYPES})

    data = HeteroData()
    for t in TYPES:
        data[t].num_nodes = len(idx[t])

    for name, (s, r, d) in REL.items():
        df = tables[name]
        si = df["src"].map(idx[s]).to_numpy()
        di = df["dst"].map(idx[d]).to_numpy()
        ei = torch.tensor(np.stack([si, di]), dtype=torch.long)
        ea = torch.tensor(np.stack([
            np.sign(df["corr"]).to_numpy(dtype="float32"),          # 符号
            np.log1p(df["strength"].to_numpy(dtype="float32")),      # 证据强度
            (df["corr"] != 0).to_numpy(dtype="float32"),             # 有无方向证据
        ], axis=1))
        data[s, r, d].edge_index = ei
        data[s, r, d].edge_attr = ea
        data[s, r, d].n_orig = len(df)
        # 反向边（目标层 gene->phenotype 不加反向，避免监督信息进入消息传递）
        if name != "gene_pheno":
            data[d, f"rev_{r}", s].edge_index = torch.tensor(np.stack([di, si]), dtype=torch.long)
            data[d, f"rev_{r}", s].edge_attr = ea.clone()

    # ---- 节点数值特征（结构统计量；类别/连续混合）----
    def src_count(src_type, rel, dst_type):
        e = data[src_type, rel, dst_type]
        return np.bincount(e.edge_index[0].numpy(), minlength=data[src_type].num_nodes).astype("float32")

    def dst_count(src_type, rel, dst_type):
        e = data[src_type, rel, dst_type]
        return np.bincount(e.edge_index[1].numpy(), minlength=data[dst_type].num_nodes).astype("float32")

    g_feats = []
    for t in TYPES:
        n = data[t].num_nodes
        F = []
        if t == "gene":
            F += [np.log1p(dst_count("snp", "annot", "gene")), np.log1p(dst_count("snp", "cis", "gene")),
                  np.log1p(dst_count("snp", "trans", "gene")), np.log1p(dst_count("meth", "emqtl", "gene"))]
            # 邻域证据平均强度（正负平衡度）
            st = np.zeros(n); pos = np.zeros(n); tot = np.zeros(n)
            for et in data.edge_types:
                if et[2] != "gene":
                    continue
                e = data[et]
                di = e.edge_index[1].numpy()
                wv = e.edge_attr[:, 1].numpy(); sg = e.edge_attr[:, 0].numpy()
                np.add.at(st, di, wv * sg); np.add.at(tot, di, wv); np.add.at(pos, di, np.maximum(sg, 0) * wv)
            F += [st / np.maximum(tot, 1), np.log1p(tot)]
        elif t == "phenotype":
            F += [np.log1p(dst_count("snp", "gwas", "phenotype")),
                  np.log1p(dst_count("meth", "ewas", "phenotype")),
                  np.log1p(src_count("phenotype", "ppathway", "pathway"))]
        elif t == "snp":
            arr = np.zeros((n, 8), dtype="float32")
            pos = np.zeros(n, dtype="float32"); maf = np.zeros(n, dtype="float32")
            chr_ = np.zeros(n, dtype="int64")
            id2s = {i: v for v, i in idx["snp"].items()}
            for i, v in id2s.items():
                m = re.match(r"^chr([1-5cm])_(\d+)$", v, re.I)
                if m:
                    c = m.group(1).upper()
                    chr_[i] = {"C": 0, "M": 1}.get(c, int(c)) if c in "CM" else int(c)
                    pos[i] = int(m.group(2))
            arr[np.arange(n), np.clip(chr_, 0, 4)] = 1.0
            arr[:, 5] = np.log1p(pos) / 20.0
            gdf = tables["snp_pheno"].copy()
            if "maf" in gdf:
                mm = gdf.groupby("src")["maf"].max()
                for k, v in mm.items():
                    if k in idx["snp"]:
                        maf[idx["snp"][k]] = float(v) if np.isfinite(v) else 0.0
            arr[:, 6] = np.log1p(maf * 1e4)
            arr[:, 7] = np.log1p(src_count("snp", "gwas", "phenotype") + src_count("snp", "annot", "gene"))
            F = [arr]
        elif t == "meth":
            arr = np.zeros((n, 9), dtype="float32")
            id2m = {i: v for v, i in idx["meth"].items()}
            for i, v in id2m.items():
                c, p, ct = parse_meth_site(v)
                if c:
                    arr[i, max(0, c - 1)] = 1.0
                    arr[i, 5] = np.log1p(p) / 20.0
                arr[i, 6 + {"CG": 0, "CHG": 1, "CHH": 2}.get(ct, 0)] = 1.0
            arr[:, 8] = np.log1p(src_count("meth", "emqtl", "gene") + src_count("meth", "ewas", "phenotype")
                                 + src_count("meth", "epathway", "pathway"))
            F = [arr]
        elif t == "pathway":
            F += [np.log1p(dst_count("meth", "epathway", "pathway")),
                  np.log1p(dst_count("phenotype", "ppathway", "pathway"))]
        elif t == "env":
            F += [np.log1p(dst_count("snp", "env_qtl", "env"))]
        X = np.stack([np.asarray(f, dtype="float32").reshape(n, -1) for f in F], axis=1)
        X = X.reshape(n, -1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        data[t].x = torch.tensor(X)

    data.node_index = {t: idx[t] for t in TYPES}
    torch.save(data, os.path.join(CACHE, "graph.pt"))
    stats = {f"{s}-{r}-{d}": int(data[s, r, d].edge_index.size(1)) for s, r, d in data.edge_types}
    stats.update({t: int(data[t].num_nodes) for t in TYPES})
    with open(os.path.join(CACHE, "graph_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, indent=2))
    return data


if __name__ == "__main__":
    build()
