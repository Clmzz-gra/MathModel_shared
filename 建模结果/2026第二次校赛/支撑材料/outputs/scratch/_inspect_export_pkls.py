"""
目的：
    临时检查脚本：导出 MATLAB 交接数据前，核验各 pkl 的键结构与字段类型。

原理：
    打印关键路径的 type/shape/示例值，供 export_matlab_data.py 照抄读取方式。

性能：
    轻量-不适用（秒级只读检查）。

输入数据：
    - outputs/data/S1-results.pkl / S1-preprocessed.pkl / S2-results.pkl /
      S3-results.pkl / S3-preprocessed.pkl / c-data-cleaned.pkl

输出：
    - stdout 检查报告（不落盘）

对应论文章节：
    （交接工具脚本，不对应具体章节）
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def desc(x, depth=0):
    pad = "  " * depth
    if isinstance(x, dict):
        return f"dict<{len(x)}> keys={list(x.keys())[:12]}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}<{len(x)}>"
    if isinstance(x, np.ndarray):
        return f"ndarray{list(x.shape)} {x.dtype}"
    return f"{type(x).__name__} = {x!r:.80}" if not isinstance(x, str) else f"str({x[:60]!r})"


def walk(d, path="", max_depth=6, cur=0):
    if cur > max_depth:
        return
    if isinstance(d, dict):
        for k, v in list(d.items())[:14]:
            p = f"{path}.{k}" if path else str(k)
            print(f"{p} :: {desc(v)}")
            if isinstance(v, dict):
                walk(v, p, max_depth, cur + 1)
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], dict):
                print(f"  {p}[0] :: {desc(v[0])}")
                walk(v[0], f"{p}[0]", max_depth, cur + 1)


print("=" * 70)
print("S1-results.pkl")
s1 = load("S1-results.pkl")
print("top keys:", list(s1.keys()))
for d in ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]:
    if d in s1:
        print(f"[{d}] keys:", list(s1[d].keys()))
        for m in ["L2_CLR", "RF_raw"]:
            if m in s1[d]:
                print(f"  [{m}] keys:", {k: desc(v, 2) for k, v in s1[d][m].items()})
        if "baseline" in s1[d]:
            print(f"  [baseline] keys:", {k: desc(v, 2) for k, v in s1[d]["baseline"].items()})
print("[adenoma_sensitivity] keys:", list(s1.get("adenoma_sensitivity", {}).keys()))
ad = s1.get("adenoma_sensitivity", {})
for c in ["CRC_adenoma_as_healthy", "CRC_adenoma_as_diseased", "CRC_adenoma_excluded", "CRC_adenoma_separate"]:
    if c in ad:
        print(f"  [{c}] { {k: desc(v, 2) for k, v in ad[c].items()} }")

print("=" * 70)
print("S1-preprocessed.pkl")
s1p = load("S1-preprocessed.pkl")
print("top keys:", list(s1p.keys()))
print("feature_names type:", type(s1p["feature_names"]), "len:", len(s1p["feature_names"]))
for d in ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]:
    if "datasets" in s1p and d in s1p["datasets"]:
        dd = s1p["datasets"][d]
        print(f"[{d}] keys:", {k: desc(v, 1) for k, v in dd.items()})

print("=" * 70)
print("S2-results.pkl")
s2 = load("S2-results.pkl")
print("top keys:", list(s2.keys()))
print("meta keys:", list(s2.get("meta", {}).keys()))
print("meta.tau_grid:", s2["meta"].get("tau_grid"))
tc = s2["meta"].get("tau_counts")
print("meta.tau_counts:", {k: desc(v) for k, v in tc.items()} if isinstance(tc, dict) else tc)
print("cross_disease:", {k: desc(v) for k, v in s2.get("cross_disease", {}).items()})
pd_ = s2.get("per_disease", {})
for d in ["CRC", "IBD", "Obesity"]:
    if d in pd_:
        print(f"[per_disease.{d}] keys:", list(pd_[d].keys()))
        sf = pd_[d].get("stable_features")
        print(f"  stable_features: {desc(sf)}  first={sf[0] if sf else None}")
        co = pd_[d].get("cooccurrence", {})
        print(f"  cooccurrence keys:", list(co.keys()) if isinstance(co, dict) else type(co))
        sm = co.get("spearman_matrix")
        if isinstance(sm, dict):
            ks = list(sm.keys())
            print(f"  spearman_matrix: dict<{len(sm)}> first keys={ks[:4]}")
            print(f"  first val: {sm[ks[0]]}")

print("=" * 70)
print("S3-results.pkl")
s3 = load("S3-results.pkl")
print("top keys:", list(s3.keys()))
sc = s3.get("strategy_compare", {})
for st in list(sc.keys())[:8]:
    if isinstance(sc[st], dict):
        print(f"[strategy_compare.{st}] keys:", {k: desc(v, 1) for k, v in sc[st].items()})
    else:
        print(f"[strategy_compare.{st}]:", desc(sc[st]))
da = s3.get("decay_attribution", {})
for d in ["CRC", "IBD", "Obesity"]:
    if d in da:
        print(f"[decay_attribution.{d}]:", {k: desc(v) for k, v in da[d].items()})
ma = s3.get("migration_analysis", {})
print("[migration_analysis]:", {k: desc(v) for k, v in ma.items()})
td = s3.get("threshold_drift", {})
print("[threshold_drift]:", {k: desc(v) for k, v in td.items()})

print("=" * 70)
print("S3-preprocessed.pkl")
s3p = load("S3-preprocessed.pkl")
print("top keys:", list(s3p.keys()))
for k, v in s3p.items():
    if k == "lodo_combos":
        print(f"  lodo_combos keys:", list(v.keys()))
        c3 = v.get("C3", {})
        print(f"    C3 keys:", {kk: desc(vv, 3) for kk, vv in c3.items()})
    else:
        print(f"  {k}: {desc(v, 1)}")

print("=" * 70)
print("c-data-cleaned.pkl")
cd = load("c-data-cleaned.pkl")
print("type:", type(cd))
if hasattr(cd, "shape"):
    print("shape:", cd.shape)
    print("columns[:6]:", list(cd.columns[:6]))
    print("columns[-3:]:", list(cd.columns[-3:]))
    print("dataset_name value_counts:\n", cd["dataset_name"].value_counts())
    print("disease value_counts:\n", cd["disease"].value_counts())
