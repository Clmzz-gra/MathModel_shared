"""阶段 0.2：数据转缓存 + 关联键检查 + 量纲初查"""
import pandas as pd
from pathlib import Path

BASE = Path(r"e:\MathModel\problems\CUMCM2024\CUMCM2024Problems\C题")
OUT = Path(r"e:\MathModel\problems\CUMCM2024\C题\outputs\data")
OUT.mkdir(parents=True, exist_ok=True)

# ==================== 加载并缓存 ====================

# 附件1
df_land = pd.read_excel(BASE / "附件1.xlsx", sheet_name="乡村的现有耕地")
df_crop_info = pd.read_excel(BASE / "附件1.xlsx", sheet_name="乡村种植的农作物")

# 附件2
df_plant2023 = pd.read_excel(BASE / "附件2.xlsx", sheet_name="2023年的农作物种植情况")
df_stat = pd.read_excel(BASE / "附件2.xlsx", sheet_name="2023年统计的相关数据")

# 保存
df_land.to_pickle(OUT / "land_plots.pkl")
df_crop_info.to_pickle(OUT / "crop_info.pkl")
df_plant2023.to_pickle(OUT / "planting_2023.pkl")
df_stat.to_pickle(OUT / "crop_stats.pkl")

print("数据已缓存到 outputs/data/")

# ==================== 关联键检查 ====================
print("\n=== 关联键检查 ===")

# 地块名一致性
land_from_plots = set(df_land['地块名称'].dropna())
land_from_planting = set(df_plant2023['种植地块'].dropna())
print(f"地块数: 耕地表={len(land_from_plots)}, 种植表={len(land_from_planting)}")
print(f"交集={len(land_from_plots & land_from_planting)}")
only_land = land_from_plots - land_from_planting
only_plant = land_from_planting - land_from_plots
if only_land: print(f"仅在耕地表: {only_land}")
if only_plant: print(f"仅在种植表: {only_plant}")

# 地块类型一致性
land_types = set(df_land['地块类型'].dropna().unique())
stat_types = set(df_stat['地块类型'].dropna().unique())
print(f"\n地块类型: 耕地表={land_types}")
print(f"地块类型: 统计表={stat_types}")

# 作物编号一致性
def safe_ids(series):
    return set(pd.to_numeric(series, errors='coerce').dropna().astype(int))

crop_ids_info = safe_ids(df_crop_info['作物编号'])
crop_ids_plant = safe_ids(df_plant2023['作物编号'])
crop_ids_stat = safe_ids(df_stat['作物编号'])
print(f"\n作物编号: 信息表={len(crop_ids_info)}, 种植表={len(crop_ids_plant)}, 统计表={len(crop_ids_stat)}")
print(f"信息表 vs 种植表: 仅信息={crop_ids_info-crop_ids_plant}, 仅种植={crop_ids_plant-crop_ids_info}")
print(f"信息表 vs 统计表: 仅信息={crop_ids_info-crop_ids_stat}, 仅统计={crop_ids_stat-crop_ids_info}")

# 季次一致性
seasons_plant = set(df_plant2023['种植季次'].dropna().unique())
seasons_stat = set(df_stat['种植季次'].dropna().unique())
print(f"\n季次: 种植表={seasons_plant}")
print(f"季次: 统计表={seasons_stat}")

# ==================== 量纲初查 ====================
print("\n=== 量纲初查 ===")
print(f"耕地面积范围: {df_land['地块面积/亩'].min():.0f} - {df_land['地块面积/亩'].max():.0f} 亩")
print(f"亩产量范围: {df_stat['亩产量/斤'].min():.0f} - {df_stat['亩产量/斤'].max():.0f} 斤/亩")
print(f"种植成本范围: {df_stat['种植成本/(元/亩)'].min():.0f} - {df_stat['种植成本/(元/亩)'].max():.0f} 元/亩")
print(f"种植面积范围: {df_plant2023['种植面积/亩'].min():.1f} - {df_plant2023['种植面积/亩'].max():.1f} 亩")
print(f"总种植面积(2023): {df_plant2023['种植面积/亩'].sum():.0f} 亩")
print(f"总耕地面积: {df_land['地块面积/亩'].sum():.0f} 亩")
print(f"差值: {df_plant2023['种植面积/亩'].sum() - df_land['地块面积/亩'].sum():.0f} (部分地块两季×面积)")

# ==================== 关键数值清单 ====================
print("\n=== 作物-地块可行性矩阵摘要 ===")
# 统计表覆盖了哪些 (地块类型, 季次, 作物编号) 组合
feasible = df_stat[['地块类型','种植季次','作物编号']].dropna()
print(f"可行组合数: {len(feasible)}")
print(f"按地块类型×季次分布:")
for (lt, s), grp in feasible.groupby(['地块类型','种植季次']):
    crop_list = sorted(grp['作物编号'].astype(int).unique())
    print(f"  {lt} × {s}: {len(crop_list)}种作物, 编号={crop_list[:5]}{'...' if len(crop_list)>5 else ''}")

# ==================== 2023年实际种植总结 ====================
print("\n=== 2023年种植概况 ===")
for season in ['单季','第一季','第二季']:
    sub = df_plant2023[df_plant2023['种植季次']==season]
    print(f"  {season}: {len(sub)}条记录, 总面积{sub['种植面积/亩'].sum():.0f}亩")
    top = sub.groupby('作物名称')['种植面积/亩'].sum().sort_values(ascending=False).head(5)
    print(f"    Top5: {dict(top)}")

# 豆类覆盖检查
legumes = {'黄豆','黑豆','红豆','绿豆','爬豆','豇豆','刀豆','芸豆'}
legume_plots = set(df_plant2023[df_plant2023['作物名称'].isin(legumes)]['种植地块'].unique())
all_plots = set(df_land['地块名称'].dropna())
print(f"\n  2023年种豆类的地块数: {len(legume_plots)}/{len(all_plots)}")

print("\n=== 数据盘点完成 ===")
