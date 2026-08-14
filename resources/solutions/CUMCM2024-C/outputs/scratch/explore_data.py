"""阶段 0.2 数据盘点：探索附件1/2/3的结构并缓存"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent.parent  # C题/
DATA_DIR = BASE / 'outputs' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── 附件1：耕地 + 作物 ───
print('=' * 60)
print('附件1: 耕地基本情况')
print('=' * 60)

# Sheet 1: 乡村的现有耕地
df_land = pd.read_excel(BASE / '附件1.xlsx', sheet_name='乡村的现有耕地')
print(f'\n【乡村的现有耕地】shape={df_land.shape}')
print(f'列名: {list(df_land.columns)}')
print(df_land.head(10).to_string())
print(f'\n耕地类型分布:')
print(df_land.iloc[:, 1].value_counts() if df_land.shape[1] > 1 else 'N/A')
print(f'\n面积统计:')
num_cols = df_land.select_dtypes(include='number')
if not num_cols.empty:
    print(num_cols.describe().to_string())

# Sheet 2: 乡村种植的农作物
df_crop = pd.read_excel(BASE / '附件1.xlsx', sheet_name='乡村种植的农作物')
print(f'\n【乡村种植的农作物】shape={df_crop.shape}')
print(f'列名: {list(df_crop.columns)}')
print(df_crop.head(20).to_string())

# ─── 附件2：2023数据 ───
print('\n' + '=' * 60)
print('附件2: 2023年种植与统计')
print('=' * 60)

df_plant23 = pd.read_excel(BASE / '附件2.xlsx', sheet_name='2023年的农作物种植情况')
print(f'\n【2023年的农作物种植情况】shape={df_plant23.shape}')
print(f'列名: {list(df_plant23.columns)}')
print(df_plant23.head(20).to_string())

df_stats23 = pd.read_excel(BASE / '附件2.xlsx', sheet_name='2023年统计的相关数据')
print(f'\n【2023年统计的相关数据】shape={df_stats23.shape}')
print(f'列名: {list(df_stats23.columns)}')
print(df_stats23.head(20).to_string())

# ─── 附件3：结果模板 ───
print('\n' + '=' * 60)
print('附件3: 结果模板')
print('=' * 60)
for tmpl_file in sorted((BASE / '附件3').glob('*.xlsx')):
    xls3 = pd.ExcelFile(tmpl_file)
    print(f'\n{tmpl_file.name}: sheets={xls3.sheet_names}')
    # 只读第一个sheet看结构
    df3 = pd.read_excel(tmpl_file, sheet_name=xls3.sheet_names[0])
    print(f'  shape={df3.shape}, columns={list(df3.columns)[:8]}...')
    print(f'  前3行:')
    print(df3.head(3).to_string())

# ─── 缓存 ───
print('\n' + '=' * 60)
print('缓存数据')
print('=' * 60)

df_land.to_pickle(DATA_DIR / 'land_plots.pkl')
print(f'land_plots.pkl ({df_land.shape})')

df_crop.to_pickle(DATA_DIR / 'crop_info.pkl')
print(f'crop_info.pkl ({df_crop.shape})')

df_plant23.to_pickle(DATA_DIR / 'planting_2023.pkl')
print(f'planting_2023.pkl ({df_plant23.shape})')

df_stats23.to_pickle(DATA_DIR / 'crop_stats.pkl')
print(f'crop_stats.pkl ({df_stats23.shape})')

print('\n全部完成')
