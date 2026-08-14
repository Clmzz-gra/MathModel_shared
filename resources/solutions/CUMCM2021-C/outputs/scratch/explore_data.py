"""
阶段 0.2 数据探查：CUMCM 2021 C 题附件
"""
import pandas as pd
import os

BASE = r"e:\MathModel\problems\CUMCM2021\C"

# ============================================================
# 附件1：供应商数据
# ============================================================
f1 = os.path.join(BASE, "附件1 近5年402家供应商的相关数据.xlsx")
print("=" * 60)
print("附件1 探查")
print("=" * 60)

# 查看 sheet 结构
xl1 = pd.ExcelFile(f1)
print(f"Sheet 数量: {len(xl1.sheet_names)}")
print(f"Sheet 名称: {xl1.sheet_names}")

for sname in xl1.sheet_names:
    df = pd.read_excel(f1, sheet_name=sname, nrows=5)
    print(f"\n--- Sheet: {sname} ---")
    print(f"  列数: {df.shape[1]}, 列名前5: {list(df.columns[:5])}")
    print(f"  列名后3: {list(df.columns[-3:])}")
    print(f"  前3行:")
    print(df.head(3).to_string())

# 完整读取第一个 sheet（订货量）
print("\n\n完整读取订货量 sheet...")
df_order = pd.read_excel(f1, sheet_name=xl1.sheet_names[0])
print(f"  订货量 shape: {df_order.shape}")
print(f"  列名: {list(df_order.columns[:5])} ... {list(df_order.columns[-3:])}")
print(f"  前3行前5列:\n{df_order.iloc[:3, :5]}")

# 识别数据类型
print(f"\n  第一列(供应商名) unique: {df_order.iloc[:, 0].nunique()}")
print(f"  第二列(品类) unique: {df_order.iloc[:, 1].unique()}")
print(f"  第二列 value_counts:\n{df_order.iloc[:, 1].value_counts()}")

# 数值列统计（跳过前两列描述列）
num_cols = df_order.columns[2:]
num_data = df_order[num_cols]
print(f"\n  数值区域（240周）:")
print(f"    总元素数: {num_data.size}")
print(f"    非零数: {(num_data != 0).sum().sum()}")
print(f"    零数: {(num_data == 0).sum().sum()}")
print(f"    NaN数: {num_data.isna().sum().sum()}")
print(f"    非零占比: {(num_data != 0).sum().sum() / num_data.size:.4f}")
flat = num_data.values.flatten()
nonzero = flat[flat != 0]
non_nan = flat[~pd.isna(flat)]
print(f"    非零值统计: min={nonzero.min():.2f}, max={nonzero.max():.2f}, mean={nonzero.mean():.2f}, median={pd.Series(nonzero).median():.2f}")

print(f"\n\n完整读取供货量 sheet...")
df_supply = pd.read_excel(f1, sheet_name=xl1.sheet_names[1])
print(f"  供货量 shape: {df_supply.shape}")
num_cols2 = df_supply.columns[2:]
num_data2 = df_supply[num_cols2]
flat2 = num_data2.values.flatten()
nonzero2 = flat2[flat2 != 0]
non_nan2 = flat2[~pd.isna(flat2)]
print(f"    非零数: {(num_data2 != 0).sum().sum()}, 零数: {(num_data2 == 0).sum().sum()}, NaN: {num_data2.isna().sum().sum()}")
print(f"    非零值统计: min={nonzero2.min():.2f}, max={nonzero2.max():.2f}, mean={nonzero2.mean():.2f}, median={pd.Series(non_nan2).median():.2f}")

# ============================================================
# 附件2：转运商数据
# ============================================================
f2 = os.path.join(BASE, "附件2 近5年8家转运商的相关数据.xlsx")
print("\n\n" + "=" * 60)
print("附件2 探查")
print("=" * 60)

xl2 = pd.ExcelFile(f2)
print(f"Sheet 数量: {len(xl2.sheet_names)}")
print(f"Sheet 名称: {xl2.sheet_names}")

for sname in xl2.sheet_names:
    df = pd.read_excel(f2, sheet_name=sname)
    print(f"\n--- Sheet: {sname} ---")
    print(f"  shape: {df.shape}")
    print(f"  列名前3: {list(df.columns[:3])}")
    print(f"  前2行:\n{df.head(2)}")
    
    # 数值统计
    num_cols = df.columns[1:]  # 跳过第一列名称
    num_data = df[num_cols]
    flat = num_data.values.flatten()
    non_nan = flat[~pd.isna(flat)]
    nonzero = non_nan[non_nan != 0]
    print(f"  总元素: {num_data.size}, NaN: {num_data.isna().sum().sum()}, 零: {(num_data == 0).sum().sum()}")
    if len(nonzero) > 0:
        print(f"  非零值: min={nonzero.min():.4f}, max={nonzero.max():.4f}, mean={nonzero.mean():.4f}")
    print(f"  转运商名称: {df.iloc[:, 0].unique()}")

# ============================================================
# 附件A/B 模板
# ============================================================
print("\n\n" + "=" * 60)
print("附件A 订购方案模板")
print("=" * 60)
fa = os.path.join(BASE, "附件A 订购方案数据结果.xlsx")
dfa = pd.read_excel(fa, header=None)
print(f"  shape: {dfa.shape}")
print(f"  前5行:\n{dfa.head(5)}")

print("\n\n附件B 转运方案模板")
fb = os.path.join(BASE, "附件B 转运方案数据结果.xlsx")
dfb = pd.read_excel(fb, header=None)
print(f"  shape: {dfb.shape}")
print(f"  前5行:\n{dfb.head(5)}")
