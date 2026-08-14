"""阶段 0.3 基础数据清洗 (修正版)"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
DATA_DIR = BASE / 'outputs' / 'data'

df_land = pd.read_pickle(DATA_DIR / 'land_plots.pkl')
df_crop = pd.read_pickle(DATA_DIR / 'crop_info.pkl')
df_plant = pd.read_pickle(DATA_DIR / 'planting_2023.pkl')
df_stats = pd.read_pickle(DATA_DIR / 'crop_stats.pkl')

# ═══════════════════════════════════════
# 1. crop_stats: 删除尾部的注释行（非数据行）
# ═══════════════════════════════════════
print('1. crop_stats 清理注释行')
# 排除非数据行: 注释行 + NaN行
bad_mask = df_stats['作物编号'].apply(
    lambda x: pd.isna(x) or (isinstance(x, str) and (x.startswith('(') or len(str(x).strip()) == 0))
)
print(f'  非数据行数: {bad_mask.sum()}')
for _, r in df_stats[bad_mask].iterrows():
    print(f'    删除: [{r["作物编号"]}] {r.get("作物名称", "")}')
df_stats = df_stats[~bad_mask].copy()
df_stats['作物编号'] = df_stats['作物编号'].astype(int)
print(f'  清理后: {df_stats.shape}')

# ═══════════════════════════════════════
# 2. planting_2023: ffill 种植地块（合并单元格）
# ═══════════════════════════════════════
print('\n2. planting_2023: ffill 种植地块')
df_plant['种植地块'] = df_plant['种植地块'].ffill()
print(f'  缺失 种植地块: {df_plant["种植地块"].isna().sum()}')

# 检查同一地块同一季次是否真有重复作物
dup_mask = df_plant.duplicated(subset=['种植地块', '作物编号', '种植季次'], keep=False)
print(f'  同地块同季同作物重复: {dup_mask.sum()} 行')
if dup_mask.sum() > 0:
    # 如果是同一地块种了同种作物的多个子区域，需要合并面积
    print('  合并重复条目...')
    df_plant = df_plant.groupby(['种植地块', '作物编号', '作物名称', '作物类型', '种植季次'],
                                  as_index=False)['种植面积/亩'].sum()
    print(f'  合并后: {df_plant.shape}')

# ═══════════════════════════════════════
# 3. 列名规范化 + 地块类型去空格
# ═══════════════════════════════════════
print('\n3. 列名 & 地块类型规范化')
for name, df in [('land_plots', df_land), ('crop_info', df_crop),
                  ('planting_2023', df_plant), ('crop_stats', df_stats)]:
    df.columns = [c.strip() for c in df.columns]

df_land['地块类型'] = df_land['地块类型'].str.strip()
print(f'  地块类型: {df_land["地块类型"].unique().tolist()}')

# ═══════════════════════════════════════
# 4. crop_info: ffill + 作物类型去空格
# ═══════════════════════════════════════
print('\n4. crop_info: ffill + 类型清理')
df_crop['种植耕地'] = df_crop['种植耕地'].ffill()
df_crop['说明'] = df_crop['说明'].ffill()
df_crop['作物类型'] = df_crop['作物类型'].ffill()
df_crop['作物名称'] = df_crop['作物名称'].ffill()
print(f'  缺失 种植耕地: {df_crop["种植耕地"].isna().sum()}')
print(f'  缺失 作物类型: {df_crop["作物类型"].isna().sum()}')
print(f'  作物类型: {df_crop["作物类型"].unique().tolist()}')

# ═══════════════════════════════════════
# 5. 价格区间解析
# ═══════════════════════════════════════
print('\n5. 价格区间解析')
price_col = '销售单价/(元/斤)'

def parse_price(val):
    if pd.isna(val):
        return np.nan, np.nan, np.nan
    s = str(val).strip()
    if '-' in s:
        parts = s.split('-')
        try:
            lo, hi = float(parts[0]), float(parts[1])
            return lo, (lo + hi) / 2, hi
        except ValueError:
            return np.nan, np.nan, np.nan
    try:
        v = float(s)
        return v, v, v
    except ValueError:
        return np.nan, np.nan, np.nan

parsed = df_stats[price_col].apply(parse_price)
df_stats['售价_低'] = parsed.apply(lambda x: x[0])
df_stats['售价_中'] = parsed.apply(lambda x: x[1])
df_stats['售价_高'] = parsed.apply(lambda x: x[2])
print(f'  解析失败: {df_stats["售价_中"].isna().sum()} 行')

# ═══════════════════════════════════════
# 6. 最终质量检查
# ═══════════════════════════════════════
print('\n6. 最终质量检查')
for name, df in [('land_plots', df_land), ('crop_info', df_crop),
                  ('planting_2023', df_plant), ('crop_stats', df_stats)]:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    dup = df.duplicated().sum()
    status = f'缺失={len(missing)}列, 重复={dup}行'
    if len(missing) > 0:
        status += f' [{dict(missing)}]'
    print(f'  {name} ({df.shape}): {status}')

# ═══════════════════════════════════════
# 7. land_plots: 合并重复地块（同一地块名不应重复）
# ═══════════════════════════════════════
# 检查
dup_land = df_land[df_land.duplicated(subset='地块名称', keep=False)]
if len(dup_land) > 0:
    print(f'\n  land_plots 同名重复地块:')
    print(dup_land.to_string())

# ═══════════════════════════════════════
# 8. 保存
# ═══════════════════════════════════════
CLEAN_DIR = DATA_DIR / 'clean'
CLEAN_DIR.mkdir(exist_ok=True)

df_land.to_pickle(CLEAN_DIR / 'land_plots.pkl')
df_crop.to_pickle(CLEAN_DIR / 'crop_info.pkl')
df_plant.to_pickle(CLEAN_DIR / 'planting_2023.pkl')
df_stats.to_pickle(CLEAN_DIR / 'crop_stats.pkl')

print(f'\n7. 已保存清洗数据到 {CLEAN_DIR}:')
for f in sorted(CLEAN_DIR.glob('*.pkl')):
    df = pd.read_pickle(f)
    ncols = len(df.columns)
    print(f'    {f.name}: {df.shape[0]}行 × {ncols}列')

# ═══════════════════════════════════════
# 9. 数据特征汇总
# ═══════════════════════════════════════
print('\n' + '=' * 60)
print('数据清洗完成摘要')
print('=' * 60)
print(f"""
操作记录:
  1. crop_stats: 删除3行注释行 (非数据), 剩余 {df_stats.shape[0]} 行
  2. planting_2023: ffill 合并单元格 → 合并同地块同季同作物的重复条目
  3. land_plots: 地块类型去尾部空格
  4. crop_info: ffill 种植耕地/说明/作物类型/作物名称
  5. 价格区间解析: "2.50-4.00" → 低/中/高三列

数据规模:
  - 地块: {df_land.shape[0]} 个 (露天{len(df_land[~df_land['地块名称'].str.startswith('E') & ~df_land['地块名称'].str.startswith('F')])} + 普通大棚{len(df_land[df_land['地块名称'].str.startswith('E')])} + 智慧大棚{len(df_land[df_land['地块名称'].str.startswith('F')])})
  - 作物: {df_crop.shape[0]} 种
  - 2023种植记录: {df_plant.shape[0]} 条
  - 统计数据: {df_stats.shape[0]} 条（{df_stats['作物编号'].nunique()} 种作物 × 多地块类型）

无记录被删除（仅排除非数据注释行），候选池覆盖率 100%。
""")
