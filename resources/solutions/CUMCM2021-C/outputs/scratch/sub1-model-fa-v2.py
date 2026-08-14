"""
阶段 2.1 v2: 因子分析 — 修正版
修改:
  1. 进步因子 → 可靠性趋势 = 后半满足率 - 前半满足率
  2. 品类 one-hot → 2列(去C), 避免完美共线性导致的方差膨胀
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ============================================================
# 1. 加载原始数据，重新构造特征
# ============================================================
df_order  = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))

week_cols = [c for c in df_order.columns if c.startswith("W")]
order_mat  = df_order[week_cols].values.astype(float)
supply_mat = df_supply[week_cols].values.astype(float)
categories = df_order["材料分类"].values
ids = df_order["供应商ID"].values
n, n_weeks = supply_mat.shape
half = n_weeks // 2

print("=" * 60)
print("阶段 2.1 v2: 因子分析 — 修正版")
print("=" * 60)

# ============================================================
# 2. 特征工程
# ============================================================

# D1: 供货规模
feat = {}
feat["供应商ID"]  = ids
feat["品类"]      = categories
feat["供货总量"]   = supply_mat.sum(axis=1)
feat["供货周数"]   = (supply_mat > 0).sum(axis=1)

# D2: 供货可靠性
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

# D3: 可靠性趋势 — 后半满足率 - 前半满足率
def calc_fulfill(ord_mat, sup_mat, start, end):
    o = ord_mat[:, start:end]
    s = sup_mat[:, start:end]
    o_act = o > 0
    fulfill = (s >= o) & o_act
    return np.divide(fulfill.sum(axis=1), o_act.sum(axis=1),
                     where=o_act.sum(axis=1) > 0, out=np.zeros(n))

first_fulfill = calc_fulfill(order_mat, supply_mat, 0, half)
second_fulfill = calc_fulfill(order_mat, supply_mat, half, n_weeks)
feat["可靠性趋势"] = second_fulfill - first_fulfill  # [-1, 1]

# D4: 品类 — 2列 one-hot (去C)
feat["品类A"] = (categories == "A").astype(int)
feat["品类B"] = (categories == "B").astype(int)
feat["品类C"] = (categories == "C").astype(int)  # 用于结果显示, 不参与FA

df_feat = pd.DataFrame(feat)
fa_cols = ["供货总量", "供货周数", "供货满足率", "供订CV差", "可靠性趋势", "品类A", "品类B"]
X_raw = df_feat[fa_cols].values.astype(float)
print(f"FA 输入特征: {fa_cols}")
print(f"可靠性趋势: min={feat['可靠性趋势'].min():.3f}, max={feat['可靠性趋势'].max():.3f}, "
      f"mean={feat['可靠性趋势'].mean():.3f}, >0 改善={(feat['可靠性趋势']>0).sum()}")

# ============================================================
# 3. 标准化
# ============================================================
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_raw)

# ============================================================
# 4. 相关矩阵 + 因子数
# ============================================================
R = np.corrcoef(X.T)
eigenvalues, eigenvectors = np.linalg.eigh(R)
idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
m = (eigenvalues >= 1).sum()
print(f"\n特征值: {np.round(eigenvalues, 3)}")
print(f"Kaiser 保留因子数: m = {m}")

# 载荷
L = eigenvectors[:, :m] * np.sqrt(eigenvalues[:m])

# ============================================================
# 5. Varimax 旋转
# ============================================================
def varimax(L, max_iter=200, tol=1e-6):
    p, m = L.shape
    L_rot = L.copy()
    for it in range(max_iter):
        old = L_rot.copy()
        h2 = (L_rot**2).sum(axis=1, keepdims=True)
        u = L_rot / np.sqrt(np.maximum(h2, 1e-10))
        for j in range(m):
            for k in range(j+1, m):
                uj, uk = u[:,j], u[:,k]
                A, B = uj**2 - uk**2, 2*uj*uk
                C, D = A.sum(), B.sum()
                num = D - 2*C*D/p
                den = C - (C**2 - D**2)/p
                phi = np.arctan2(num, den)/4.0
                cos_p, sin_p = np.cos(phi), np.sin(phi)
                L_rot[:,[j,k]] = old[:,[j,k]] @ np.array([[cos_p, -sin_p],[sin_p, cos_p]])
                u[:,[j,k]] = u[:,[j,k]] @ np.array([[cos_p, -sin_p],[sin_p, cos_p]])
                old = L_rot.copy()
        if np.abs(L_rot - old).max() < tol:
            print(f"  Varimax 收敛于迭代 {it+1}")
            break
    return L_rot

L_rot = varimax(L)

print(f"\n--- Varimax 旋转后载荷 ---")
print(f"{'':>10s}", end="")
for k in range(m):
    print(f"  {'F'+str(k+1):>8s}", end="")
print(f"  {'共同度':>8s}")
for j, name in enumerate(fa_cols):
    comm = (L_rot[j]**2).sum()
    print(f"  {name:>10s}", end="")
    for k in range(m):
        v = L_rot[j,k]
        mark = "*" if abs(v) > 0.5 else " "
        print(f"  {v:>7.3f}{mark}", end="")
    print(f"  {comm:>7.3f}")

print("\n因子命名:")
for k in range(m):
    top_idx = np.argsort(-np.abs(L_rot[:,k]))
    items = [f"{fa_cols[j]}({L_rot[j,k]:+.2f})" for j in top_idx if abs(L_rot[j,k])>0.3]
    print(f"  F{k+1}: {' | '.join(items[:4])}")

# ============================================================
# 6. 因子得分 + 综合评分
# ============================================================
R_inv = np.linalg.inv(R)
W_score = R_inv @ L_rot
F_scores = X @ W_score
var_total = eigenvalues[:m].sum()
weights = eigenvalues[:m] / var_total
print(f"\n因子权重: {np.round(weights, 3)}")

I_tmp = F_scores @ weights
I = (I_tmp - I_tmp.min()) / (I_tmp.max() - I_tmp.min())

# ============================================================
# 7. Top 50
# ============================================================
rank_order = np.argsort(-I)
top50_idx = rank_order[:50]

print(f"\n{'='*60}")
print("Top 50 供应商")
print(f"{'='*60}")
print(f"{'排名':>4s}  {'ID':>6s}  {'品类':>4s}  {'安全指数':>8s}  {'供货总量':>10s}  {'满足率':>8s}  {'可靠性趋势':>8s}")
for rank, idx in enumerate(top50_idx[:50]):
    print(f"{rank+1:>4d}  {ids[idx]:>6s}  {categories[idx]:>4s}  {I[idx]:>8.4f}  {feat['供货总量'][idx]:>10.0f}  {feat['供货满足率'][idx]:>8.3f}  {feat['可靠性趋势'][idx]:>8.3f}")

top_cats = categories[top50_idx]
print(f"\nTop 50 品类: A={(top_cats=='A').sum()}, B={(top_cats=='B').sum()}, C={(top_cats=='C').sum()}")

# 与基线对比
simple_top50 = set(np.argsort(-feat['供货总量'])[:50])
fa_top50 = set(top50_idx)
print(f"与仅供货总量基线重叠: {len(simple_top50 & fa_top50)}/50")

# 与大供应商重叠(前20)
top20_vol = set(np.argsort(-feat['供货总量'])[:20])
print(f"Top 50 含供货量前20中的: {len(top20_vol & fa_top50)}/20")
missing_top20 = top20_vol - fa_top50
if missing_top20:
    print(f"  缺失: {[ids[i] for i in missing_top20]}")

# ============================================================
# 8. 保存
# ============================================================
results = pd.DataFrame({
    "供应商ID": ids, "品类": categories,
    "安全指数_I": I,
    "排名": np.argsort(np.argsort(-I)) + 1,
})
for k in range(m):
    results[f"因子得分_F{k+1}"] = F_scores[:, k]
results = results.sort_values("排名")
results.to_csv(os.path.join(DATA_DIR, "sub1-results-fa-v2.csv"), index=False)
results.to_pickle(os.path.join(DATA_DIR, "sub1-results-fa-v2.pkl"))

print(f"\n已保存: sub1-results-fa-v2.csv")
print("=" * 60)
print("阶段 2.1 v2 完成")
print("=" * 60)
