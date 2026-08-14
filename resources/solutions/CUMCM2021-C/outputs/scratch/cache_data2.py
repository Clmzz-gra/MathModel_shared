"""
阶段 0.2：数据缓存 + 深入探查
"""
import pandas as pd
import os, pickle

BASE = r"e:\MathModel\problems\CUMCM2021\C"
DATA = r"e:\MathModel\problems\CUMCM2021\C\outputs\data"

# ============================================================
# 1. 缓存原始数据
# ============================================================
f1 = os.path.join(BASE, "附件1 近5年402家供应商的相关数据.xlsx")
df_order = pd.read_excel(f1, sheet_name="企业的订货量（m³）")
df_supply = pd.read_excel(f1, sheet_name="供应商的供货量（m³）")
f2 = os.path.join(BASE, "附件2 近5年8家转运商的相关数据.xlsx")
df_loss = pd.read_excel(f2, sheet_name="运输损耗率（%）")

# 整理列名：统一 week 编号
week_cols = [f"W{i:03d}" for i in range(1, 241)]
df_order.columns = ["供应商ID", "材料分类"] + week_cols
df_supply.columns = ["供应商ID", "材料分类"] + week_cols
df_loss.columns = ["转运商ID"] + week_cols

# 存 pkl
df_order.to_pickle(os.path.join(DATA, "order-raw.pkl"))
df_supply.to_pickle(os.path.join(DATA, "supply-raw.pkl"))
df_loss.to_pickle(os.path.join(DATA, "loss-raw.pkl"))
print("缓存完成: order-raw.pkl, supply-raw.pkl, loss-raw.pkl")

# ============================================================
# 2. 订货 vs 供货 偏差分析
# ============================================================
print("\n" + "=" * 60)
print("订货 vs 供货 偏差分析")
print("=" * 60)

order_mat = df_order[week_cols].values
supply_mat = df_supply[week_cols].values
diff_mat = supply_mat - order_mat  # 正=多供，负=少供

# 只在订货>0或供货>0的条目上分析（双向非零）
active_mask = (order_mat > 0) | (supply_mat > 0)
diffs = diff_mat[active_mask]
print(f"  有订货或有供货的条目数: {active_mask.sum()} ({active_mask.sum()/active_mask.size*100:.1f}%)")
print(f"  偏差统计: min={diffs.min():.0f}, max={diffs.max():.0f}, mean={diffs.mean():.2f}, median={diffs[diffs!=0].mean() if len(diffs[diffs!=0])>0 else 0:.2f}")

# 供货 vs 订货 对比
both_active = (order_mat > 0) & (supply_mat > 0)
only_order = (order_mat > 0) & (supply_mat == 0)
only_supply = (order_mat == 0) & (supply_mat > 0)
print(f"  双向活跃（订+供均>0）: {both_active.sum()}")
print(f"  仅订货不供货: {only_order.sum()}")
print(f"  仅供货不订货: {only_supply.sum()}")

# 超供/欠供比例
over_supply = (both_active) & (diff_mat > 0)
under_supply = (both_active) & (diff_mat < 0)
exact_match = (both_active) & (diff_mat == 0)
total_both = both_active.sum()
print(f"\n  双向活跃中:")
print(f"    超供（供>订）: {over_supply.sum()} ({over_supply.sum()/total_both*100:.1f}%)")
print(f"    欠供（供<订）: {under_supply.sum()} ({under_supply.sum()/total_both*100:.1f}%)")
print(f"    精确匹配: {exact_match.sum()} ({exact_match.sum()/total_both*100:.1f}%)")

# ============================================================
# 3. 供应商活跃度分析
# ============================================================
print("\n" + "=" * 60)
print("供应商活跃度分析")
print("=" * 60)

# 每个供应商：有订货的周数、有供货的周数
df_info = pd.DataFrame({
    "供应商ID": df_order["供应商ID"],
    "材料分类": df_order["材料分类"],
    "订货周数": (order_mat > 0).sum(axis=1),
    "供货周数": (supply_mat > 0).sum(axis=1),
    "总订货量": order_mat.sum(axis=1),
    "总供货量": supply_mat.sum(axis=1),
    "平均订货量": order_mat.mean(axis=1),
    "平均供货量": supply_mat.mean(axis=1),
})

print(f"  供应商总订货量分布:\n{df_info['总订货量'].describe()}")
print(f"\n  供应商总供货量分布:\n{df_info['总供货量'].describe()}")
print(f"\n  供应商订货周数分布:\n{df_info['订货周数'].describe()}")
print(f"\n  供应商供货周数分布:\n{df_info['供货周数'].describe()}")

# 按品类分组
print("\n  按品类分组:")
for cat in ["A", "B", "C"]:
    sub = df_info[df_info["材料分类"] == cat]
    print(f"    {cat}类: {len(sub)}家, 总订货量={sub['总订货量'].sum():.0f}, 总供货量={sub['总供货量'].sum():.0f},")
    print(f"      平均每家周订货={sub['平均订货量'].mean():.2f}, 平均每家活跃周={sub['订货周数'].mean():.0f}")

# ============================================================
# 4. 损耗率时间维度分析
# ============================================================
print("\n" + "=" * 60)
print("损耗率分析")
print("=" * 60)

loss_mat = df_loss[week_cols].values
print(f"  8家转运商 × 240周 损耗率统计:")
for i in range(8):
    row = loss_mat[i]
    non_zero = row[row > 0]
    print(f"    T{i+1}: 非零周={len(non_zero)}, min={non_zero.min():.4f}, max={non_zero.max():.4f}, mean={non_zero.mean():.4f}, std={non_zero.std():.4f}")

# ============================================================
# 5. 模板结构分析
# ============================================================
print("\n" + "=" * 60)
print("模板结构分析")
print("=" * 60)

fa = os.path.join(BASE, "附件A 订购方案数据结果.xlsx")
dfa_detail = pd.read_excel(fa, header=None)
# 找到表头行
for i in range(min(20, dfa_detail.shape[0])):
    row = dfa_detail.iloc[i]
    non_null = row.dropna()
    if len(non_null) > 5:
        print(f"  附件A 第{i}行（可能表头）: {list(non_null[:10])}")

fb = os.path.join(BASE, "附件B 转运方案数据结果.xlsx")
dfb_detail = pd.read_excel(fb, header=None)
for i in range(min(20, dfb_detail.shape[0])):
    row = dfb_detail.iloc[i]
    non_null = row.dropna()
    if len(non_null) > 5:
        print(f"  附件B 第{i}行（可能表头）: {list(non_null[:10])}")

# 保存供应商信息汇总
df_info.to_pickle(os.path.join(DATA, "supplier-summary.pkl"))
print("\n缓存完成: supplier-summary.pkl")
