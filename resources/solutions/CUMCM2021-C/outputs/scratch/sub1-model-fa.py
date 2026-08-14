"""
阶段 2.1: 因子分析 (FA) — 供应商重要性安全指数
流程: 标准化 → 相关矩阵 → 因子抽取 → Varimax 旋转 → 因子得分 → 安全指数
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ============================================================
# 1. 加载预处理数据
# ============================================================
df = pd.read_pickle(os.path.join(DATA_DIR, "sub1-preprocessed.pkl"))
num_cols = ["供货总量", "供货周数", "供货满足率", "供订CV差", "进步因子",
            "品类A", "品类B", "品类C"]
ids = df["供应商ID"].values
cats = df["品类"].values
X_raw = df[num_cols].values.astype(float)
n, p = X_raw.shape

print("=" * 60)
print("阶段 2.1: 因子分析 — 供应商重要性安全指数")
print("=" * 60)
print(f"样本: {n}, 特征: {p}")

# ============================================================
# 2. 标准化
# ============================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ============================================================
# 3. 相关矩阵 + 因子数确定 (Kaiser)
# ============================================================
R = np.corrcoef(X.T)
eigenvalues, eigenvectors = np.linalg.eigh(R)
# 降序排列
idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

m = (eigenvalues >= 1).sum()
print(f"\n特征值: {np.round(eigenvalues, 3)}")
print(f"Kaiser 保留因子数: m = {m}")

# ============================================================
# 4. 初始因子载荷
# ============================================================
L_unrotated = eigenvectors[:, :m] * np.sqrt(eigenvalues[:m])

print(f"\n--- 未旋转载荷矩阵 (p={p}, m={m}) ---")
for j, name in enumerate(num_cols):
    loads = L_unrotated[j]
    print(f"  {name:>10s}: " + " ".join(f"F{k+1}={v:+.3f}" for k, v in enumerate(loads)))

# ============================================================
# 5. Varimax 旋转
# ============================================================
def varimax(L, max_iter=200, tol=1e-6):
    """Varimax 正交旋转 (Kaiser, 1958)"""
    p, m = L.shape
    L_rot = L.copy()
    for iteration in range(max_iter):
        old_L = L_rot.copy()
        # 正规化载荷
        h2 = (L_rot ** 2).sum(axis=1, keepdims=True)
        u = L_rot / np.sqrt(np.maximum(h2, 1e-10))
        for j in range(m):
            for k in range(j + 1, m):
                uj, uk = u[:, j], u[:, k]
                A = uj**2 - uk**2
                B = 2 * uj * uk
                C = A.sum()
                D = B.sum()
                num = D - 2 * C * D / p
                den = C - (C**2 - D**2) / p
                phi = np.arctan2(num, den) / 4.0
                # 旋转矩阵
                cos_p, sin_p = np.cos(phi), np.sin(phi)
                L_rot[:, j] = old_L[:, j] * cos_p + old_L[:, k] * sin_p
                L_rot[:, k] = -old_L[:, j] * sin_p + old_L[:, k] * cos_p
                u[:, j] = u[:, j] * cos_p + u[:, k] * sin_p
                u[:, k] = -u[:, j] * cos_p + u[:, k] * sin_p
                old_L = L_rot.copy()
        if np.abs(L_rot - old_L).max() < tol:
            print(f"  Varimax 收敛于迭代 {iteration+1}")
            break
    return L_rot

L_rot = varimax(L_unrotated)

print(f"\n--- Varimax 旋转后载荷矩阵 ---")
print(f"{'':>10s}", end="")
for k in range(m):
    print(f"  {'F'+str(k+1):>8s}", end="")
print(f"  {'共同度':>8s}")
for j, name in enumerate(num_cols):
    communality = (L_rot[j]**2).sum()
    print(f"  {name:>10s}", end="")
    for k in range(m):
        v = L_rot[j, k]
        marker = "*" if abs(v) > 0.5 else " "
        print(f"  {v:>7.3f}{marker}", end="")
    print(f"  {communality:>7.3f}")

# 因子命名
print("\n因子命名:")
for k in range(m):
    top_idx = np.argsort(-np.abs(L_rot[:, k]))
    items = []
    for j in top_idx:
        if abs(L_rot[j, k]) > 0.3:
            items.append(f"{num_cols[j]}({L_rot[j,k]:+.2f})")
    print(f"  F{k+1}: {' | '.join(items[:4])}")

# ============================================================
# 6. 因子得分 (回归法: Bartlett)
# ============================================================
# F = X · R^{-1} · L_rot
R_inv = np.linalg.inv(R)
W_score = R_inv @ L_rot  # p × m
F_scores = X @ W_score   # n × m

# ============================================================
# 7. 综合评分 (方差贡献率加权)
# ============================================================
var_explained = eigenvalues[:m]
total_var = eigenvalues[:m].sum()
weights = var_explained / total_var
print(f"\n因子权重: {np.round(weights, 3)}")

I_tmp = F_scores @ weights  # n

# ============================================================
# 8. 归一化为安全指数
# ============================================================
I_min, I_max = I_tmp.min(), I_tmp.max()
I = (I_tmp - I_min) / (I_max - I_min)

# ============================================================
# 9. 排序 → Top 50
# ============================================================
rank_order = np.argsort(-I)
top50_idx = rank_order[:50]

print(f"\n{'='*60}")
print("Top 50 供应商")
print(f"{'='*60}")
print(f"{'排名':>4s}  {'ID':>6s}  {'品类':>4s}  {'安全指数':>8s}  {'供货总量':>10s}")
for rank, idx in enumerate(top50_idx[:50]):
    print(f"{rank+1:>4d}  {ids[idx]:>6s}  {cats[idx]:>4s}  {I[idx]:>8.4f}  {X_raw[idx,0]:>10.0f}")

# Top 50 品类分布
top_cats = cats[top50_idx]
a_cnt = (top_cats == "A").sum()
b_cnt = (top_cats == "B").sum()
c_cnt = (top_cats == "C").sum()
print(f"\nTop 50 品类分布: A={a_cnt}, B={b_cnt}, C={c_cnt}")

# ============================================================
# 10. 与简单基线对比
# ============================================================
total_supply = X_raw[:, 0]
simple_top50 = set(np.argsort(-total_supply)[:50])
fa_top50 = set(top50_idx)
overlap = len(simple_top50 & fa_top50)
print(f"\n与仅供货总量基线的 Top 50 重叠: {overlap}/50")

# ============================================================
# 11. 保存结果
# ============================================================
results = pd.DataFrame({
    "供应商ID": ids,
    "品类": cats,
    "安全指数_I": I,
    "排名": np.argsort(np.argsort(-I)) + 1,
})
for k in range(m):
    results[f"因子得分_F{k+1}"] = F_scores[:, k]
results = results.sort_values("排名")
results.to_csv(os.path.join(DATA_DIR, "sub1-results-fa.csv"), index=False)
results.to_pickle(os.path.join(DATA_DIR, "sub1-results-fa.pkl"))

print(f"\n已保存: sub1-results-fa.csv, sub1-results-fa.pkl")
print("=" * 60)
print("阶段 2.1 FA 完成")
print("=" * 60)
