"""
验证：供应商时间模式变化是需求驱动还是能力驱动？
对比 订货量 和 供货量 的时间序列一致性
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

df_order  = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))
df_types  = pd.read_pickle(os.path.join(DATA_DIR, "supplier-types.pkl"))

week_cols = [c for c in df_order.columns if c.startswith("W")]
order_mat  = df_order[week_cols].values.astype(float)
supply_mat = df_supply[week_cols].values.astype(float)
ids = df_order["供应商ID"].values
categories = df_order["材料分类"].values
n = len(df_order)

# ============================================================
# 1. 突变型供应商：订货量是否也突变？
# ============================================================
print("=" * 60)
print("突变型供应商：订货 vs 供货同步性")
print("=" * 60)

mutation_idx = df_types[df_types["类型"] == "突变型"]["idx"].values.astype(int)

for idx in mutation_idx:
    sid = ids[idx]
    cat = categories[idx]
    sup = supply_mat[idx]
    ord_ = order_mat[idx]
    
    half = 240 // 2
    s1, s2 = sup[:half].mean(), sup[half:].mean()
    o1, o2 = ord_[:half].mean(), ord_[half:].mean()
    
    s_ratio = s2 / (s1 + 1e-10)
    o_ratio = o2 / (o1 + 1e-10)
    
    # 供货与订货的周级别 Spearman 相关
    active = (sup > 0) | (ord_ > 0)
    if active.sum() > 5:
        rho = np.corrcoef(sup[active], ord_[active])[0, 1]
    else:
        rho = np.nan
    
    # 满足率
    order_active = ord_ > 0
    if order_active.sum() > 0:
        fulfill = (sup[order_active] >= ord_[order_active]).mean()
    else:
        fulfill = np.nan
    
    print(f"\n  {sid} ({cat}):")
    print(f"    前半段 — 订货均值={o1:.1f}, 供货均值={s1:.1f}")
    print(f"    后半段 — 订货均值={o2:.1f}, 供货均值={s2:.1f}")
    print(f"    供货比 s₂/s₁={s_ratio:.2f}, 订货比 o₂/o₁={o_ratio:.2f}")
    print(f"    供货-订货周相关 ρ={rho:.3f}, 满足率={fulfill:.2%}")

# ============================================================
# 2. 全局：每家供应商的 订货-供货 相关性
# ============================================================
print("\n" + "=" * 60)
print("全局：供应商 订货-供货 周级别相关性分布")
print("=" * 60)

corrs = []
fulfills = []
for i in range(n):
    sup = supply_mat[i]
    ord_ = order_mat[i]
    active = (sup > 0) | (ord_ > 0)
    if active.sum() > 5:
        corrs.append(np.corrcoef(sup[active], ord_[active])[0, 1])
    else:
        corrs.append(np.nan)
    
    order_active = ord_ > 0
    if order_active.sum() > 0:
        fulfills.append((sup[order_active] >= ord_[order_active]).mean())
    else:
        fulfills.append(np.nan)

corrs = np.array(corrs)
fulfills = np.array(fulfills)

print(f"  有效样本: {(~np.isnan(corrs)).sum()}")
print(f"  ρ 均值: {np.nanmean(corrs):.3f}, 中位数: {np.nanmedian(corrs):.3f}")
print(f"  ρ > 0.8: {(corrs > 0.8).sum()} ({(corrs > 0.8).sum()/(~np.isnan(corrs)).sum()*100:.1f}%)")
print(f"  ρ > 0.5: {(corrs > 0.5).sum()} ({(corrs > 0.5).sum()/(~np.isnan(corrs)).sum()*100:.1f}%)")
print(f"  ρ < 0:   {(corrs < 0).sum()} ({(corrs < 0).sum()/(~np.isnan(corrs)).sum()*100:.1f}%)")

# ============================================================
# 3. 按时间模式类型的 订货-供货 相关性
# ============================================================
print("\n--- 各类型供应商的 订货-供货 ρ ---")
for t in ["平稳型", "周期型", "突变型", "泊松型", "无规律"]:
    sub = df_types[df_types["类型"] == t]
    idxs = sub["idx"].values.astype(int)
    c = corrs[idxs]
    f = fulfills[idxs]
    valid = ~np.isnan(c)
    print(f"  {t}: ρ均值={np.nanmean(c):.3f}, ρ中位={np.nanmedian(c):.3f}, "
          f"ρ>0.8={np.sum(c[valid]>0.8)}/{valid.sum()}, "
          f"满足率均值={np.nanmean(f):.2%}")

# ============================================================
# 4. 关键问题：供货波动是否由订货波动解释？
# ============================================================
print("\n" + "=" * 60)
print("核心问题：供应CV 能否被 订货CV 解释？")
print("=" * 60)

order_cv = np.divide(order_mat.std(axis=1), order_mat.mean(axis=1),
                     where=order_mat.mean(axis=1) > 0, out=np.zeros(n))
supply_cv = np.divide(supply_mat.std(axis=1), supply_mat.mean(axis=1),
                      where=supply_mat.mean(axis=1) > 0, out=np.zeros(n))

# 两者之差：供应CV超过订货CV的部分 = 供应商自身的不稳定性
cv_diff = supply_cv - order_cv
print(f"  订货CV均值: {order_cv[order_cv>0].mean():.3f}")
print(f"  供货CV均值: {supply_cv[supply_cv>0].mean():.3f}")
print(f"  CV差（供-订）均值: {cv_diff[(order_cv>0)&(supply_cv>0)].mean():.3f}")
print(f"  CV差 > 1 (供比订更波动): {(cv_diff > 1).sum()} 家")
print(f"  CV差 < 0 (供比订更稳): {(cv_diff < 0).sum()} 家")

# 哪些供应商 CV 差最大（供比订波动大很多 = 真正的不可靠）
print("\n  供比订波动大最多的 10 家:")
cv_diff_sorted = np.argsort(-cv_diff)
for rank, idx in enumerate(cv_diff_sorted[:10]):
    sid = ids[idx]
    cat = categories[idx]
    print(f"    {sid} ({cat}): 订货CV={order_cv[idx]:.2f}, 供货CV={supply_cv[idx]:.2f}, "
          f"差={cv_diff[idx]:.2f}, 满足率={fulfills[idx]:.2%}")

# ============================================================
# 5. 结论：重新审视「突变型」
# ============================================================
print("\n" + "=" * 60)
print("结论")
print("=" * 60)
# 对突变型，看供需是否同步
m_idx = df_types[df_types["类型"] == "突变型"]["idx"].values.astype(int)
synced = 0
for idx in m_idx:
    if not np.isnan(corrs[idx]) and corrs[idx] > 0.5:
        synced += 1
print(f"  突变型中订货-供货高相关(ρ>0.5): {synced}/{len(m_idx)}")
print(f"  → 其中 {synced}/{len(m_idx)} 家的供货波动主要是企业订货波动导致的")
print("=" * 60)
