"""
阶段 1.1 A 类验证：子问题 1 供应商评价共享事实
实验设计：
  A1 — 简单基线：仅按供货总量排名（naive baseline）
  A2 — 指标独立性检验：候选指标间的 Spearman 相关矩阵
  A3 — 品类分布对比：不同排名方法下 Top 50 的品类构成
  A4 — 消耗率换算效果：换算前后排名的 Spearman 秩相关
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "figures"), exist_ok=True)

# ============================================================
# 加载数据
# ============================================================
df_order  = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))
week_cols = [c for c in df_order.columns if c.startswith("W")]
order_mat  = df_order[week_cols].values.astype(float)
supply_mat = df_supply[week_cols].values.astype(float)
categories = df_order["材料分类"].values
n = len(df_order)

# ============================================================
# A1: 简单基线 — 仅按供货总量排名
# ============================================================
print("=" * 60)
print("A1: 简单基线 — 仅按供货总量 Top 50")
print("=" * 60)
total_supply = supply_mat.sum(axis=1)
rank_simple = np.argsort(-total_supply)
top50_simple = rank_simple[:50]
top50_simple_ids = df_order.iloc[top50_simple]["供应商ID"].tolist()

# 品类分布
cats_simple = categories[top50_simple]
a_count = (cats_simple == "A").sum()
b_count = (cats_simple == "B").sum()
c_count = (cats_simple == "C").sum()
print(f"  Top 50 品类: A={a_count}, B={b_count}, C={c_count}")
print(f"  总供货占比: {total_supply[top50_simple].sum() / total_supply.sum() * 100:.1f}%")
print(f"  Top 5: {', '.join(top50_simple_ids[:5])}")

# ============================================================
# A2: 指标独立性检验
# ============================================================
print("\n" + "=" * 60)
print("A2: 候选指标 Spearman 相关矩阵")
print("=" * 60)

# 构造候选指标
indicators = pd.DataFrame({
    "供货总量":      total_supply,
    "供货周数":      (supply_mat > 0).sum(axis=1),
    "供货CV":        np.divide(supply_mat.std(axis=1), supply_mat.mean(axis=1),
                              where=supply_mat.mean(axis=1) > 0, out=np.zeros(n)),
    "供货满足率":    np.divide(
        ((supply_mat >= order_mat) & (order_mat > 0)).sum(axis=1),
        (order_mat > 0).sum(axis=1),
        where=(order_mat > 0).sum(axis=1) > 0, out=np.zeros(n)),
    "最大单周供货":  supply_mat.max(axis=1),
    "超供次数":      ((order_mat > 0) & (supply_mat > order_mat)).sum(axis=1),
    "欠供次数":      ((order_mat > 0) & (supply_mat < order_mat)).sum(axis=1),
    "净偏差总量":    (supply_mat - order_mat).sum(axis=1),
})

# 消耗率换算的供货能力
consumption_rate = {"A": 0.60, "B": 0.66, "C": 0.72}
cat_rates = np.array([consumption_rate[c] for c in categories])
# 可支撑产品量 = 供货量 / 消耗率
capacity_by_week = supply_mat / cat_rates[:, np.newaxis]
indicators["换算产能总量"] = capacity_by_week.sum(axis=1)
indicators["换算产能均值"] = capacity_by_week.mean(axis=1)

labels = list(indicators.columns)
corr_mat = np.zeros((len(labels), len(labels)))
for i in range(len(labels)):
    for j in range(len(labels)):
        corr_mat[i, j], _ = spearmanr(indicators.iloc[:, i], indicators.iloc[:, j])

print(f"{'':>12s}", end="")
for l in labels:
    print(f"{l[:6]:>8s}", end="")
print()
for i, l in enumerate(labels):
    print(f"{l:>12s}", end="")
    for j in range(len(labels)):
        v = corr_mat[i, j]
        if abs(v) > 0.8:
            flag = "⚠"
        else:
            flag = " "
        print(f"{v:>7.2f}{flag}", end="")
    print()

# 高相关对
print("\n  高相关 (|ρ| > 0.8) 指标对:")
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        if abs(corr_mat[i, j]) > 0.8:
            print(f"    {labels[i]} ↔ {labels[j]}: ρ = {corr_mat[i,j]:.3f}")

# ============================================================
# A3: 品类分布 — 不同排名方法对比
# ============================================================
print("\n" + "=" * 60)
print("A3: 不同方法 Top 50 品类分布对比")
print("=" * 60)

methods = {
    "仅供货总量": total_supply,
    "仅换算产能": indicators["换算产能总量"].values,
    "仅满足率":   indicators["供货满足率"].values,
    "仅供货周数": indicators["供货周数"].values,
}

for mname, scores in methods.items():
    top50 = np.argsort(-scores)[:50]
    cats = categories[top50]
    print(f"  {mname}: A={(cats=='A').sum()}, B={(cats=='B').sum()}, C={(cats=='C').sum()} | "
          f"总供占比={total_supply[top50].sum()/total_supply.sum()*100:.1f}%")

# ============================================================
# A4: 消耗率换算效果 — 排名变化
# ============================================================
print("\n" + "=" * 60)
print("A4: 消耗率换算前后排名 Spearman 秩相关")
print("=" * 60)

rank_raw = np.argsort(np.argsort(-total_supply))  # 0 = 最高
rank_cap = np.argsort(np.argsort(-indicators["换算产能总量"].values))
rho, p = spearmanr(rank_raw, rank_cap)
print(f"  供货总量 vs 换算产能总量: ρ = {rho:.4f} (p = {p:.2e})")

# 排名变动最大的供应商
rank_change = rank_raw - rank_cap
top_changers = np.argsort(-np.abs(rank_change))[:15]
print(f"\n  排名变动最大的 15 家:")
for idx in top_changers:
    sid = df_order.iloc[idx]["供应商ID"]
    cat = categories[idx]
    raw_r = rank_raw[idx] + 1
    cap_r = rank_cap[idx] + 1
    raw_val = total_supply[idx]
    cap_val = indicators["换算产能总量"].iloc[idx]
    print(f"    {sid} ({cat}): 供货排名 {raw_r} → 产能排名 {cap_r} (变动 {cap_r-raw_r:+d}), "
          f"供货={raw_val:.0f}, 产能当量={cap_val:.0f}")

print("\n" + "=" * 60)
print("A 类验证完成")
print("=" * 60)
