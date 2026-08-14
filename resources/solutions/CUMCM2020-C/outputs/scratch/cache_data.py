"""
阶段 0.2 — 数据格式转缓存
将附件1/2/3加载为 .parquet，并输出数据盘点信息
"""
import pandas as pd
import os, sys

# 从 outputs/scratch/ → outputs/ → C题根目录
out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.dirname(out_dir)  # C题根目录
data_dir = os.path.join(out_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# ── 附件1 ──────────────────────────────────────
f1 = os.path.join(base, '附件1：123家有信贷记录企业的相关数据.xlsx')
print("=" * 60)
print("附件1: 123家有信贷记录企业")
print("=" * 60)

xls1 = pd.ExcelFile(f1)
sheets1 = {}
for s in xls1.sheet_names:
    df = pd.read_excel(xls1, s)
    sheets1[s] = df
    print(f"\n[{s}] shape={df.shape}, dtypes:\n{df.dtypes.to_string()}")
    print(f"  null counts: {df.isnull().sum().to_dict()}")
    if '金额' in s or '发票' in s:
        print(f"  日期范围: {df.iloc[:,2].min()} ~ {df.iloc[:,2].max()}")
        print(f"  金额范围: {df.iloc[:,4].min():.2f} ~ {df.iloc[:,4].max():.2f}")

# 企业信息
df1_info = sheets1[list(sheets1.keys())[0]]
print(f"\n  信誉评级分布:\n{df1_info.iloc[:,2].value_counts()}")
print(f"  违约分布:\n{df1_info.iloc[:,3].value_counts()}")

# ── 附件2 ──────────────────────────────────────
f2 = os.path.join(base, '附件2：302家无信贷记录企业的相关数据.xlsx')
print("\n" + "=" * 60)
print("附件2: 302家无信贷记录企业")
print("=" * 60)

xls2 = pd.ExcelFile(f2)
sheets2 = {}
for s in xls2.sheet_names:
    df = pd.read_excel(xls2, s)
    sheets2[s] = df
    print(f"\n[{s}] shape={df.shape}, dtypes:\n{df.dtypes.to_string()}")
    print(f"  null counts: {df.isnull().sum().to_dict()}")
    if '金额' in s or '发票' in s:
        print(f"  日期范围: {df.iloc[:,2].min()} ~ {df.iloc[:,2].max()}")
        print(f"  金额范围: {df.iloc[:,4].min():.2f} ~ {df.iloc[:,4].max():.2f}")

# ── 附件3 ──────────────────────────────────────
f3 = os.path.join(base, '附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx')
print("\n" + "=" * 60)
print("附件3: 利率-流失率关系")
print("=" * 60)

# 附件3第一行是子列标题(信誉评级A/B/C)，单独处理
df3_raw = pd.read_excel(f3)
# 提取实际列名
col_names = ['贷款年利率'] + [f'客户流失率_{x}' for x in df3_raw.iloc[0, 1:].values]
df3 = df3_raw.iloc[1:].copy()
df3.columns = col_names
df3 = df3.reset_index(drop=True)
for c in df3.columns:
    df3[c] = pd.to_numeric(df3[c], errors='coerce')
print(f"shape={df3.shape}, dtypes:\n{df3.dtypes.to_string()}")
print(f"null counts: {df3.isnull().sum().to_dict()}")
print(f"\n{df3.head(10).to_string()}")

# ── 关联键检查 ────────────────────────────────
print("\n" + "=" * 60)
print("关联键检查")
print("=" * 60)

# 附件1: 企业代号覆盖
info_eids_1 = set(sheets1[list(sheets1.keys())[0]].iloc[:,0])
for s in list(sheets1.keys())[1:]:
    eids = set(sheets1[s].iloc[:,0])
    print(f"附件1 [{s}] 企业代号数={len(eids)}, 与企业信息交集={len(info_eids_1 & eids)}, 仅在企业信息={len(info_eids_1 - eids)}")

# 附件2
info_eids_2 = set(sheets2[list(sheets2.keys())[0]].iloc[:,0])
for s in list(sheets2.keys())[1:]:
    eids = set(sheets2[s].iloc[:,0])
    print(f"附件2 [{s}] 企业代号数={len(eids)}, 与企业信息交集={len(info_eids_2 & eids)}, 仅在企业信息={len(info_eids_2 - eids)}")

# ── 写入 parquet ──────────────────────────────
print("\n" + "=" * 60)
print("写入 .parquet 缓存")
print("=" * 60)

for name, df in sheets1.items():
    path = os.path.join(data_dir, f'f1_{name}.parquet')
    df.to_parquet(path)
    print(f"  f1_{name}.parquet ({df.shape})")

for name, df in sheets2.items():
    path = os.path.join(data_dir, f'f2_{name}.parquet')
    df.to_parquet(path)
    print(f"  f2_{name}.parquet ({df.shape})")

path = os.path.join(data_dir, 'f3_rate_loss.parquet')
df3.to_parquet(path)
print(f"  f3_rate_loss.parquet ({df3.shape})")

print("\n完成。")
