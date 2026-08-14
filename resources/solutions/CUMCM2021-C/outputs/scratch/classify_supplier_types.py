"""
阶段 1.1 A 类验证补充：供应商时间模式分类
方法：K-S 检验 + ACF 周期检测 + 变异系数阈值
参考：MS-031, PR-007
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp, poisson

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ============================================================
# 加载
# ============================================================
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))
week_cols = [c for c in df_supply.columns if c.startswith("W")]
supply_mat = df_supply[week_cols].values.astype(float)
categories = df_supply["材料分类"].values
n_suppliers, n_weeks = supply_mat.shape
ids = df_supply["供应商ID"].values

print("=" * 60)
print("供应商时间模式分类 (n=402)")
print("=" * 60)

# ============================================================
# 1. 基础统计：供货间隔分布
# ============================================================
# 对每家供应商，统计非零周之间的间隔
intervals_stats = []
for i in range(n_suppliers):
    supply = supply_mat[i]
    nonzero_idx = np.where(supply > 0)[0]
    if len(nonzero_idx) < 3:
        intervals_stats.append({
            "idx": i, "mean_interval": 240, "std_interval": 0,
            "n_nonzero": len(nonzero_idx), "cv_supply": 0,
            "total_supply": supply.sum(),
            "is_too_sparse": True
        })
    else:
        intervals = np.diff(nonzero_idx)
        nz_vals = supply[nonzero_idx]
        intervals_stats.append({
            "idx": i,
            "mean_interval": intervals.mean(),
            "std_interval": intervals.std(),
            "n_nonzero": len(nonzero_idx),
            "cv_supply": nz_vals.std() / nz_vals.mean() if nz_vals.mean() > 0 else 0,
            "total_supply": supply.sum(),
            "is_too_sparse": len(nonzero_idx) < 10
        })

df_int = pd.DataFrame(intervals_stats)
df_int["供应商ID"] = ids
df_int["品类"] = categories

# ============================================================
# 2. K-S 检验：供货间隔是否服从泊松分布？
# ============================================================
print("\n--- K-S 检验：供货间隔 → Poisson 拟合 ---")
poisson_types = []
for i in range(n_suppliers):
    supply = supply_mat[i]
    nonzero_idx = np.where(supply > 0)[0]
    if len(nonzero_idx) < 10:
        poisson_types.append(False)
        continue
    intervals = np.diff(nonzero_idx)
    mean_int = intervals.mean()
    # 生成同均值的泊松分布样本做 KS 检验
    poisson_rv = np.random.poisson(mean_int, size=1000)
    # 把 intervals 和 poisson_rv 做双样本 KS
    stat, p = ks_2samp(intervals, poisson_rv)
    poisson_types.append(p > 0.05)  # α=0.05 不能拒绝 = 服从

df_int["泊松型"] = poisson_types
n_poisson = sum(poisson_types)
print(f"  泊松型（供货间隔随机）: {n_poisson} 家 ({n_poisson/n_suppliers*100:.1f}%)")

# ============================================================
# 3. ACF 周期检测
# ============================================================
print("\n--- ACF 周期检测 ---")
def detect_periodicity(series, max_lag=120, threshold=0.3):
    """检测最强的周期成分"""
    n = len(series)
    # 去均值
    x = series - series.mean()
    if np.std(x) < 1e-10:
        return 0, 0.0
    
    # 自相关
    acf = np.correlate(x, x, mode='full')[n-1:] / (n * np.var(x) + 1e-10)
    
    # 找峰值（排除 lag=0 和太近的 lag）
    best_lag, best_acf = 0, 0.0
    for lag in range(4, min(max_lag, n-1)):
        # 检测局部峰值
        if acf[lag] > acf[lag-1] and acf[lag] > acf[lag+1]:
            if acf[lag] > threshold and acf[lag] > best_acf:
                best_lag, best_acf = lag, acf[lag]
    return best_lag, best_acf

periodic_info = []
for i in range(n_suppliers):
    supply = supply_mat[i]
    lag, acf_val = detect_periodicity(supply)
    periodic_info.append({"lag": lag, "acf": acf_val, "is_periodic": lag > 0})

df_int["周期检测lag"] = [p["lag"] for p in periodic_info]
df_int["周期检测acf"] = [p["acf"] for p in periodic_info]
df_int["周期型"] = [p["is_periodic"] for p in periodic_info]
n_periodic = df_int["周期型"].sum()
print(f"  周期型（ACF > 0.3）: {n_periodic} 家 ({n_periodic/n_suppliers*100:.1f}%)")

# ============================================================
# 4. 变异系数阈值 → 平稳型
# ============================================================
# 对足够活跃的供应商，CV < 1.5 视为平稳
df_int["平稳型"] = (df_int["n_nonzero"] >= 10) & (df_int["cv_supply"] < 1.5) & (~df_int["周期型"])
n_stable = df_int["平稳型"].sum()
print(f"  平稳型（CV < 1.5, 非周期）: {n_stable} 家 ({n_stable/n_suppliers*100:.1f}%)")

# ============================================================
# 5. 突变型检测：前半 vs 后半供货量差异
# ============================================================
print("\n--- 突变型检测（前后半段均值差异） ---")
mutation_flags = []
for i in range(n_suppliers):
    supply = supply_mat[i]
    nonzero_idx = np.where(supply > 0)[0]
    if len(nonzero_idx) < 10:
        mutation_flags.append(False)
        continue
    half = n_weeks // 2
    first_half = supply[:half]
    second_half = supply[half:]
    
    m1 = first_half.mean()
    m2 = second_half.mean()
    
    if m1 + m2 < 1e-6:
        mutation_flags.append(False)
        continue
    
    ratio = m2 / (m1 + 1e-10) if m1 > 0 else float('inf')
    # 均值和 > 0 且前后差异超 3 倍或 < 1/3
    is_mutation = (ratio > 3.0 or ratio < 0.33) and (m1 + m2 > 5)
    mutation_flags.append(is_mutation)

df_int["突变型"] = mutation_flags
n_mutation = sum(mutation_flags)
print(f"  突变型（前后均值差 > 3×）: {n_mutation} 家 ({n_mutation/n_suppliers*100:.1f}%)")

# ============================================================
# 6. 综合分类（互斥优先级：突变 > 周期 > 泊松 > 平稳 > 稀疏）
# ============================================================
def classify(row):
    if row["突变型"]:
        return "突变型"
    if row["周期型"]:
        return "周期型"
    if row["泊松型"]:
        return "泊松型"
    if row["平稳型"]:
        return "平稳型"
    if row["n_nonzero"] < 5:
        return "极稀疏"
    return "无规律"

df_int["类型"] = df_int.apply(classify, axis=1)

print("\n" + "=" * 60)
print("分类结果汇总")
print("=" * 60)
type_counts = df_int["类型"].value_counts()
for t in ["泊松型", "周期型", "平稳型", "突变型", "无规律", "极稀疏"]:
    cnt = type_counts.get(t, 0)
    print(f"  {t}: {cnt} 家 ({cnt/n_suppliers*100:.1f}%)")

# ============================================================
# 7. 按品类 + 类型 交叉表
# ============================================================
print("\n--- 品类 × 类型交叉表 ---")
for cat in ["A", "B", "C"]:
    sub = df_int[df_int["品类"] == cat]
    print(f"\n  品类 {cat} ({len(sub)} 家):")
    for t in ["泊松型", "周期型", "平稳型", "突变型", "无规律", "极稀疏"]:
        cnt = (sub["类型"] == t).sum()
        if cnt > 0:
            print(f"    {t}: {cnt} ({cnt/len(sub)*100:.1f}%)")

# ============================================================
# 8. 各类型典型供应商
# ============================================================
print("\n--- 各类型 Top 5 示例 ---")
for t in ["泊松型", "周期型", "平稳型", "突变型"]:
    sub = df_int[df_int["类型"] == t].sort_values("total_supply", ascending=False).head(5)
    print(f"\n  {t}:")
    for _, row in sub.iterrows():
        print(f"    {row['供应商ID']} ({row['品类']}): 供货周={int(row['n_nonzero'])}, "
              f"CV={row['cv_supply']:.2f}, 间隔均值={row['mean_interval']:.1f}周, "
              f"ACF lag={int(row['周期检测lag'])}, 总供货={row['total_supply']:.0f}")

# ============================================================
# 9. 保存分类结果
# ============================================================
out_path = os.path.join(DATA_DIR, "supplier-types.pkl")
df_int.to_pickle(out_path)
print(f"\n已保存: {out_path}")
print("=" * 60)
print("时间模式分类完成")
print("=" * 60)
