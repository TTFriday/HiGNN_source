# -*- coding: utf-8 -*-
"""读取 rawdata 的 10 张关联证据表，标准化实体 ID，输出 parquet 缓存。

用法:  python -m src.data.loaders  (在项目根目录 hignn/ 下运行)
"""
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.environ.get("HIGNN_RAW", os.path.join(ROOT, "..", "rawdata"))
CACHE = os.path.join(ROOT, "data_cache")
os.makedirs(CACHE, exist_ok=True)

# EWAS 表约 400 万行，仅保留强证据边以控制图规模
EWAS_FDR = 1e-10
GENE_RX = re.compile(r"^AT([1-5CM])G(\d{5})$", re.I)


def norm_gene(s):
    """统一拟南芥基因 ID：去空白/转大写/补零，如 at1g1040 -> AT1G01040。非法返回 None。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().upper().replace("*", "")
    m = GENE_RX.match(s)
    if not m:
        return None
    return f"AT{m.group(1)}G{m.group(2).zfill(5)}"


def norm_snp(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().lower()
    m = re.match(r"^chr?([1-5cm])(\D*)(\d+)", s)
    if not m:
        return None
    return f"chr{m.group(1)}_{m.group(3)}"


def log10p(x):
    """把 p 值/FDR 转成 -log10 强度，并对 0 做下限截断。"""
    x = pd.to_numeric(x, errors="coerce").clip(lower=1e-300)
    return (-np.log10(x)).clip(upper=320.0).astype("float32")


def load_twas():
    df = R(os.path.join(RAW, "twas.csv"))
    out = pd.DataFrame({
        "src": df["Gene"].map(norm_gene),
        "dst": df["Phenotype"].astype(str).str.strip(),
        "corr": pd.to_numeric(df["Pearson"], errors="coerce").astype("float32"),
        "strength": log10p(df["FDR_Pearson"]),
        "study": df["Study"].astype(str),
    })
    return out.dropna(subset=["src", "dst", "corr"])


def load_gwas_snp_pheno():
    df = R(os.path.join(RAW, "gwas.csv"))
    df["snp"] = df["SNP"].map(norm_snp)
    df["pheno"] = df["Phenotype"].astype(str).str.strip()
    g = df.dropna(subset=["snp", "pheno"]).groupby(["snp", "pheno"], as_index=False).agg(
        score=("Score", lambda x: pd.to_numeric(x, errors="coerce").max()),
        maf=("MAF", lambda x: pd.to_numeric(x, errors="coerce").max()),
    )
    g["strength"] = (pd.to_numeric(g["score"], errors="coerce").clip(lower=0) / 10.0).astype("float32")
    return g.rename(columns={"snp": "src", "pheno": "dst"})


def load_gwas_snp_gene():
    df = R(os.path.join(RAW, "gwas.csv"))
    df["snp"] = df["SNP"].map(norm_snp)
    df["gene"] = df["Genename"].map(norm_gene)
    impact = {"HIGH": 3, "MODERATE": 2, "LOW": 1}
    df["impact"] = df["Impact"].astype(str).str.upper().map(impact).fillna(0)
    g = df.dropna(subset=["snp", "gene"]).groupby(["snp", "gene"], as_index=False).agg(
        impact=("impact", "max"))
    g["strength"] = g["impact"].astype("float32")
    return g.rename(columns={"snp": "src", "gene": "dst"})


def R(path, **kw):
    """统一 CSV 读取：latin-1 容错编码 + 跳过坏行（个别表含非 UTF-8 字节）。"""
    return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", low_memory=False, **kw)


def read_xls(path):
    """直接用 xlrd 读取老式 .xls，绕开 pandas 的版本检查。"""
    import xlrd
    b = xlrd.open_workbook(path)
    s = b.sheet_by_index(0)
    header = [str(x).strip() for x in s.row_values(0)]
    rows = [s.row_values(i) for i in range(1, s.nrows)]
    return pd.DataFrame(rows, columns=header)


def load_eqtl(which):
    df = read_xls(os.path.join(RAW, f"{which}eqtl.xls"))
    out = pd.DataFrame({
        "src": df["SNP"].map(norm_snp),
        "dst": df["Gene"].map(norm_gene),
        "corr": pd.to_numeric(df["Beta"], errors="coerce").astype("float32"),
        "strength": log10p(df["FDR"]),
    })
    return out.dropna(subset=["src", "dst", "corr"])


def load_env_eqtl():
    df = pd.read_excel(os.path.join(RAW, "environmenteqtl.xlsx"))
    out = pd.DataFrame({
        "src": df["eQTL"].map(norm_snp),
        "dst": df["Environment"].astype(str).str.strip(),
        "corr": pd.to_numeric(df["Beta"], errors="coerce").astype("float32"),
        "strength": log10p(df["FDR"]),
        "etype": df["type"].astype(str),
    })
    return out.dropna(subset=["src", "dst", "corr"])


def load_ld():
    df = R(os.path.join(RAW, "gwaseqtl.csv"))
    out = pd.DataFrame({
        "src": df["eQTL loci"].map(norm_snp),
        "dst": df["GWAS loci"].map(norm_snp),
        "corr": pd.to_numeric(df["LD"], errors="coerce").astype("float32"),
    })
    out["strength"] = out["corr"].abs() * 10.0
    out = out.dropna(subset=["src", "dst", "corr"])
    return out[out["src"] != out["dst"]]


def load_ewas():
    keep = []
    names = ["Phenotype", "Methylation", "Pearson", "FDR_Pearson", "FDR_Spearman"]
    kw = dict(usecols=[0, 2, 3, 5, 8], chunksize=500_000, encoding="latin-1", on_bad_lines="skip")
    for ch in pd.read_csv(os.path.join(RAW, "ewas.csv"), **kw):
        ch.columns = names
        fdr = pd.concat([pd.to_numeric(ch["FDR_Pearson"], errors="coerce"),
                         pd.to_numeric(ch["FDR_Spearman"], errors="coerce")], axis=1).min(axis=1)
        ch = ch[fdr < EWAS_FDR].copy()
        if len(ch) == 0:
            continue
        ch["fdr"] = fdr[fdr < EWAS_FDR]
        keep.append(ch)
    df = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=names + ["fdr"])
    if len(df) == 0:  # 阈值过严时自动放宽
        df = pd.read_csv(os.path.join(RAW, "ewas.csv"), usecols=[0, 2, 3, 5, 8],
                         encoding="latin-1", on_bad_lines="skip")
        df.columns = names
        df["fdr"] = pd.to_numeric(df["FDR_Pearson"], errors="coerce")
        df = df.nsmallest(300_000, "fdr")
    df = df.sort_values("fdr").groupby(["Methylation", "Phenotype"], as_index=False).first()
    return pd.DataFrame({
        "src": df["Methylation"].astype(str).str.strip(),
        "dst": df["Phenotype"].astype(str).str.strip(),
        "corr": pd.to_numeric(df["Pearson"], errors="coerce").astype("float32"),
        "strength": log10p(df["fdr"]),
    }).dropna()


def load_emqtl():
    df = R(os.path.join(RAW, "emqtl.csv"))
    return pd.DataFrame({
        "src": df["Methylation"].astype(str).str.strip(),
        "dst": df["Gene"].map(norm_gene),
        "corr": pd.to_numeric(df["Pearson"], errors="coerce").astype("float32"),
        "strength": log10p(df["FDR"]),
    }).dropna()


def load_epathway():
    df = R(os.path.join(RAW, "epathway.csv"))
    return pd.DataFrame({
        "src": df["Methylation"].astype(str).str.strip(),
        "dst": df["Pathway"].astype(str).str.strip(),
        "corr": pd.to_numeric(df["Pearson"], errors="coerce").astype("float32"),
        "strength": log10p(df["FDR_Pearson"]),
    }).dropna()


def load_pheno_pathway():
    df = R(os.path.join(RAW, "phenotypepathway.csv"))
    return pd.DataFrame({
        "src": df["Phenotype"].astype(str).str.strip(),
        "dst": df["Pathway"].astype(str).str.strip(),
        "corr": pd.to_numeric(df["Pearson"], errors="coerce").astype("float32"),
        "strength": log10p(df["FDR_Pearson"]),
    }).dropna()


JOBS = {
    "gene_pheno": load_twas,
    "snp_pheno": load_gwas_snp_pheno,
    "snp_gene": load_gwas_snp_gene,
    "snp_gene_cis": lambda: load_eqtl("cis"),
    "snp_gene_trans": lambda: load_eqtl("trans"),
    "snp_env": load_env_eqtl,
    "snp_snp": load_ld,
    "meth_pheno": load_ewas,
    "meth_gene": load_emqtl,
    "meth_pathway": load_epathway,
    "pheno_pathway": load_pheno_pathway,
}


def main():
    summary = {}
    for name, fn in JOBS.items():
        df = fn()
        df.to_pickle(os.path.join(CACHE, f"{name}.pkl"))
        summary[name] = {
            "edges": len(df),
            "src_nodes": int(df["src"].nunique()),
            "dst_nodes": int(df["dst"].nunique()),
            "signed": bool("corr" in df.columns),
        }
        print(f"{name:16s} edges={summary[name]['edges']:8d}  src={summary[name]['src_nodes']:7d}  dst={summary[name]['dst_nodes']:7d}")
    pd.DataFrame(summary).T.to_csv(os.path.join(CACHE, "load_summary.csv"))
    print("缓存写入完成:", CACHE)


if __name__ == "__main__":
    main()
