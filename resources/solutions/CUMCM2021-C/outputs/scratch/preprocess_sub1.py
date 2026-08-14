"""
阶段 1.4 数据预处理 — 子问题 1
构造 PCA 输入特征矩阵，写入 outputs/data/sub1-preprocessed.pkl
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. 加载原始数据
# ============================================================
df_order  = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_supply = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))

week_cols = [c for c in df_order.columns if c.startswith("W")]
order_mat  = df_order[week_cols].values.astype(float)
supply_mat = df_supply[week_cols].values.astype(float)
categories = df_order["材料分类"].values
ids = df_order["供应商ID"].values
n = len(df_order)

print("=" * 60)
print("阶段 1.4 数据预处理 — 子问题 1")
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

# 供货CV 和 订货CV（仅非零周）
supply_cv = np.zeros(n)
order_cv = np.zeros(n)
for i in range(n):
    s_nz = supply_mat[i][supply_mat[i] > 0]
    o_nz = order_mat[i][order_mat[i] > 0]
    if len(s_nz) > 1:
        supply_cv[i] = s_nz.std() / s_nz.mean()
    if len(o_nz) > 1:
        order_cv[i] = o_nz.std() / o_nz.mean()

feat["供订CV差"] = supply_cv - order_cv  # 负=供更稳, 正=供更波动

# D3: 发展趋势 — 进步因子
half = 240 // 2
first_mean  = supply_mat[:, :half].mean(axis=1)
second_mean = supply_mat[:, half:].mean(axis=1)
progress = np.divide(second_mean, first_mean,
                     where=first_mean > 0, out=np.ones(n))
# 对极稀疏供应商（供货周数<20），设进步因子=1（无趋势）
sparse_mask = feat["供货周数"] < 20
progress[sparse_mask] = 1.0
# 截断极端值
progress = np.clip(progress, 0.1, 10.0)
feat["进步因子"] = progress

# D4: 品类价值 (one-hot)
feat["品类A"] = (categories == "A").astype(int)
feat["品类B"] = (categories == "B").astype(int)
feat["品类C"] = (categories == "C").astype(int)

# ============================================================
# 3. 组装 DataFrame
# ============================================================
df_feat = pd.DataFrame(feat)

# 数值特征列（用于 PCA 输入）
num_cols = ["供货总量", "供货周数", "供货满足率", "供订CV差", "进步因子",
            "品类A", "品类B", "品类C"]

print(f"\n特征矩阵: {df_feat.shape[0]} × {len(num_cols)}")
print(f"数值特征: {num_cols}")

# ============================================================
# 4. 预处理操作清单
# ============================================================
print(f"\n预处理操作清单:")
print(f"  1. 进步因子截断: clip(0.1, 10.0) — 极端值处理")
print(f"  2. 进步因子稀疏处理: 供货<20周 → 进步因子=1.0 — 无趋势默认")
print(f"  3. 供订CV差零除保护: 供货/订货均值=0 → CV差=0")
print(f"  4. 供货满足率零除保护: 无订货周 → 满足率=0")
print(f"  5. 不做标准化: PCA 阶段自行 StandardScaler，此处保留原始量纲便于审查")

# ============================================================
# 5. 数据质量摘要
# ============================================================
print(f"\n数据质量摘要:")
for col in num_cols:
    vals = df_feat[col].values
    print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, "
          f"mean={vals.mean():.4f}, NaN={np.isnan(vals).sum()}, inf={np.isinf(vals).sum()}")

# ============================================================
# 6. 保存
# ============================================================
out_path = os.path.join(DATA_DIR, "sub1-preprocessed.pkl")
df_feat.to_pickle(out_path)

# 也保存数值特征矩阵单独供 PCA 使用
X = df_feat[num_cols].values.astype(float)
np.save(os.path.join(DATA_DIR, "sub1-features.npy"), X)

print(f"\n已保存: {out_path}")
print(f"  columns: {list(df_feat.columns)}")
print(f"  shape: {df_feat.shape}")
print("=" * 60)
print("阶段 1.4 完成")
print("=" * 60)
