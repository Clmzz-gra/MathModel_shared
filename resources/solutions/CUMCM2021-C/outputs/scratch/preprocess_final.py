"""
目的：
    阶段 1.4 终版数据预处理（子问题 1 正式链）：从原始订货/供货矩阵计算 5 项供应商
    行为特征（不含品类），供 sub1-model.py 的 PCA 使用。

原理：
    1. 供货总量 = Σ_t s_t；供货周数 = Σ_t I[s_t > 0]（供货规模）。
    2. 供货满足率 = Σ_t I[s_t ≥ o_t ∧ o_t > 0] / Σ_t I[o_t > 0]（订货时供够的比例）。
    3. 供订CV差 = CV(supply_nz) − CV(order_nz)，CV 仅对非零供货/订货周计算
       （std/mean，ddof=0），供比订更稳则为负。
    4. 可靠性趋势 = 后半段满足率 − 前半段满足率 ∈ [−1,1]
       （前半=周1–120，后半=周121–240），>0 表示履约能力改善。
    5. 品类标签不纳入特征：它是供应商固定属性而非行为指标，one-hot 会在协方差矩阵
       产生二分信号劫持主成分权重（经 FA v1/v2 实证 A 类霸榜 46–50 家）。

输入数据：
    - order-raw.pkl (原始) — 供应商ID, 材料分类, W1..W240（订货量）
    - supply-raw.pkl (原始) — 供应商ID, W1..W240（供货量）

输出：
    - sub1-preprocessed-final.pkl (处理后) — 供应商ID, 品类, 5 特征
    - sub1-features-final.npy — 5 特征数值矩阵（402×5）

对应论文章节：
    论文「供应商重要性评价与筛选」（子问题 1，章节号论文写作时定稿）
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

df_order  = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))

week_cols = [c for c in df_order.columns if c.startswith("W")]
order_mat  = df_order[week_cols].values.astype(float)
supply_mat = df_supply[week_cols].values.astype(float)
categories = df_order["材料分类"].values
ids = df_order["供应商ID"].values
n = len(df_order)
half = 240 // 2

print("=" * 60)
print("阶段 1.4 终版 数据预处理")
print("=" * 60)

feat = {}
feat["供应商ID"]  = ids
feat["品类"]      = categories
feat["供货总量"]   = supply_mat.sum(axis=1)
feat["供货周数"]   = (supply_mat > 0).sum(axis=1)

# 供货满足率
order_active = order_mat > 0
supply_ge_order = (supply_mat >= order_mat) & order_active
feat["供货满足率"] = np.divide(
    supply_ge_order.sum(axis=1), order_active.sum(axis=1),
    where=order_active.sum(axis=1) > 0, out=np.zeros(n))

# 供订CV差
supply_cv = np.zeros(n); order_cv = np.zeros(n)
for i in range(n):
    s_nz = supply_mat[i][supply_mat[i] > 0]
    o_nz = order_mat[i][order_mat[i] > 0]
    if len(s_nz) > 1: supply_cv[i] = s_nz.std() / s_nz.mean()
    if len(o_nz) > 1: order_cv[i] = o_nz.std() / o_nz.mean()
feat["供订CV差"] = supply_cv - order_cv

# 可靠性趋势（后半满足率 - 前半满足率）
def fulfill_rate(sup_half, ord_half):
    active = ord_half > 0
    fulfill = (sup_half >= ord_half) & active
    return np.divide(fulfill.sum(axis=1), active.sum(axis=1),
                     where=active.sum(axis=1) > 0, out=np.zeros(n))
feat["可靠性趋势"] = (fulfill_rate(supply_mat[:, half:], order_mat[:, half:]) -
                    fulfill_rate(supply_mat[:, :half], order_mat[:, :half]))

df_feat = pd.DataFrame(feat)
num_cols = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势"]

print(f"特征: {num_cols}")
for col in num_cols:
    v = df_feat[col].values
    print(f"  {col}: [{v.min():.4f}, {v.max():.4f}] mean={v.mean():.4f}")

df_feat.to_pickle(os.path.join(DATA_DIR, "sub1-preprocessed-final.pkl"))
np.save(os.path.join(DATA_DIR, "sub1-features-final.npy"), df_feat[num_cols].values.astype(float))
print(f"\n已保存: sub1-preprocessed-final.pkl")
print("=" * 60)
