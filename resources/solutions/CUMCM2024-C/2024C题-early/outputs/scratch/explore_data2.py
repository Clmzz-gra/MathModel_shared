"""阶段 0.2：深入探索所有数据"""
import pandas as pd
from pathlib import Path

BASE = Path(r"e:\MathModel\problems\CUMCM2024\CUMCM2024Problems\C题")
OUT = Path(r"e:\MathModel\problems\CUMCM2024\C题\outputs\data")

# === 附件1: 耕地 ===
df_land = pd.read_excel(BASE / "附件1.xlsx", sheet_name="乡村的现有耕地", header=0)
print("=== 附件1: 耕地地块 ===")
print(f"Shape: {df_land.shape}")
print(f"地块类型分布:\n{df_land['地块类型'].value_counts()}")
print(f"\n总耕地面积: {df_land['地块面积/亩'].sum():.0f} 亩")
print(f"\n按类型汇总面积:")
print(df_land.groupby('地块类型')['地块面积/亩'].agg(['sum','count']))
print(f"\n前10个地块:")
print(df_land.head(10).to_string())

# === 附件1: 农作物 ===
df_crop = pd.read_excel(BASE / "附件1.xlsx", sheet_name="乡村种植的农作物", header=0)
print(f"\n\n=== 附件1: 农作物 ===")
print(f"Shape: {df_crop.shape}")
print(f"作物类型分布:\n{df_crop['作物类型'].value_counts()}")
print(f"\n全部作物:")
for _, row in df_crop.iterrows():
    land = str(row['种植耕地'])[:80] if pd.notna(row['种植耕地']) else "继承上一行"
    note = str(row['说明'])[:60] if pd.notna(row['说明']) else ""
    print(f"  {row['作物编号']:>2} {row['作物名称']:<6} {row['作物类型']:<12} | {land}{' | '+note if note else ''}")

# === 附件2: 2023种植情况 ===
df_2023 = pd.read_excel(BASE / "附件2.xlsx", sheet_name="2023年的农作物种植情况", header=0)
print(f"\n\n=== 附件2: 2023种植情况 ===")
print(f"Shape: {df_2023.shape}")
print(f"季次分布:\n{df_2023['种植季次'].value_counts()}")
print(f"总种植面积: {df_2023['种植面积/亩'].sum():.0f} 亩")
print(f"前10行:")
print(df_2023.head(10).to_string())

# === 附件2: 统计数据 ===
df_stat = pd.read_excel(BASE / "附件2.xlsx", sheet_name="2023年统计的相关数据", header=0)
print(f"\n\n=== 附件2: 统计数据 ===")
print(f"Shape: {df_stat.shape}")
print(f"Cols: {df_stat.columns.tolist()}")
print(f"地块类型: {df_stat['地块类型'].unique()}")
print(f"种植季次: {df_stat['种植季次'].unique()}")
print(f"\n全部数据:")
for _, row in df_stat.iterrows():
    print(f"  {int(row['序号']):>3} {int(row['作物编号']):>2} {row['作物名称']:<6} {row['地块类型']:<6} {row['种植季次']:<4} 亩产:{row['亩产量/斤']:>6} 成本:{row['种植成本/(元/亩)']:>6} 单价:{row['销售单价/(元/斤)']}")

# === 附件3: 结果模板 ===
for fname in ["result1_1.xlsx", "result1_2.xlsx", "result2.xlsx"]:
    fpath = BASE / "附件3" / fname
    if fpath.exists():
        df = pd.read_excel(fpath, header=None)
        print(f"\n\n=== 附件3: {fname} ===")
        print(f"Shape: {df.shape}")
        print(f"前8行:")
        for i in range(min(8, len(df))):
            vals = [str(v)[:60] if pd.notna(v) else "NaN" for v in df.iloc[i].values]
            print(f"  row{i}: {' | '.join(vals)}")
