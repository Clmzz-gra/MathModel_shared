"""
阶段 0.3 — 基础数据清洗
从 parquet 缓存加载，执行模型无关的清洗操作
"""
import pandas as pd
import numpy as np
import os

out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(out_dir, 'data')

# ══════════════════════════════════════════════
# 1. 加载缓存
# ══════════════════════════════════════════════
print("=" * 60)
print("加载 parquet 缓存")
print("=" * 60)

df1_info = pd.read_parquet(os.path.join(data_dir, 'f1_企业信息.parquet'))
df1_in   = pd.read_parquet(os.path.join(data_dir, 'f1_进项发票信息.parquet'))
df1_out  = pd.read_parquet(os.path.join(data_dir, 'f1_销项发票信息.parquet'))
df2_info = pd.read_parquet(os.path.join(data_dir, 'f2_企业信息.parquet'))
df2_in   = pd.read_parquet(os.path.join(data_dir, 'f2_进项发票信息.parquet'))
df2_out  = pd.read_parquet(os.path.join(data_dir, 'f2_销项发票信息.parquet'))
df3      = pd.read_parquet(os.path.join(data_dir, 'f3_rate_loss.parquet'))

print("全部加载完成。")

# ══════════════════════════════════════════════
# 2. 发票状态标准化
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("发票状态标准化")
print("=" * 60)

for label, df in [('f1_进项', df1_in), ('f1_销项', df1_out),
                   ('f2_进项', df2_in), ('f2_销项', df2_out)]:
    before = df['发票状态'].value_counts().to_dict()
    df['发票状态'] = df['发票状态'].str.strip()
    after = df['发票状态'].value_counts().to_dict()
    if before != after:
        print(f"  {label}: 前导空格已修正, before={before}, after={after}")
    else:
        print(f"  {label}: 无变化, 分布={after}")

# ══════════════════════════════════════════════
# 3. 发票分类与异常报告
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("发票分类统计")
print("=" * 60)

def classify_invoices(df, label):
    total = len(df)
    valid_mask = df['发票状态'] == '有效发票'
    void_mask  = df['发票状态'] == '作废发票'
    neg_mask   = df['金额'] < 0  # 负数发票（红冲/退货）
    
    n_valid = valid_mask.sum()
    n_void  = void_mask.sum()
    n_neg   = neg_mask.sum()
    n_neg_valid = (neg_mask & valid_mask).sum()  # 标记为有效但金额为负
    n_neg_void  = (neg_mask & void_mask).sum()   # 标记为作废且金额为负
    
    print(f"\n{label} ({total:,} 行):")
    print(f"  有效发票:        {n_valid:>8,} ({n_valid/total*100:5.1f}%)")
    print(f"  作废发票:        {n_void:>8,} ({n_void/total*100:5.1f}%)")
    print(f"  负数金额:        {n_neg:>8,} ({n_neg/total*100:5.1f}%)")
    print(f"    其中有效+负数:  {n_neg_valid:>8,}")
    print(f"    其中作废+负数:  {n_neg_void:>8,}")

classify_invoices(df1_in,  '附件1 进项')
classify_invoices(df1_out, '附件1 销项')
classify_invoices(df2_in,  '附件2 进项')
classify_invoices(df2_out, '附件2 销项')

# ══════════════════════════════════════════════
# 4. D 级企业统计
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("D 级企业（按题意不予放贷）")
print("=" * 60)

d_mask = df1_info['信誉评级'] == 'D'
d_eids = set(df1_info.loc[d_mask, '企业代号'])
d_default = (df1_info.loc[d_mask, '是否违约'] == '是').sum()
print(f"  D 级企业数: {len(d_eids)}")
print(f"  其中违约数: {d_default} ({d_default/len(d_eids)*100:.1f}%)")
print(f"  企业代号: {sorted(d_eids)[:10]}...")

# ══════════════════════════════════════════════
# 5. 金额异常值检测
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("金额异常值检测（有效发票、金额>0）")
print("=" * 60)

for label, df in [('f1_进项', df1_in), ('f1_销项', df1_out),
                   ('f2_进项', df2_in), ('f2_销项', df2_out)]:
    valid_pos = df[(df['发票状态'] == '有效发票') & (df['金额'] > 0)]
    s = valid_pos['金额']
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR
    extreme = (s > upper).sum()
    # 检查 >1000万的离群值（微型企业单笔超千万可疑）
    mega = (s > 10_000_000).sum()
    print(f"\n  {label}:")
    print(f"    有效正金额: {len(s):,} 行, median={s.median():.0f}, mean={s.mean():.0f}")
    print(f"    IQR异常上界: {upper:,.0f}, 超出: {extreme} 行 ({extreme/len(s)*100:.2f}%)")
    print(f"    单笔>1000万: {mega} 行")

# ══════════════════════════════════════════════
# 6. 写入清洗后缓存
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("写入清洗后缓存")
print("=" * 60)

# 6a. 发票表：仅保留有效发票 + 金额>0
#    （作废发票和负数发票在建模时不参与特征计算）
df1_in_clean = df1_in[(df1_in['发票状态'] == '有效发票') & (df1_in['金额'] > 0)].copy()
df1_out_clean = df1_out[(df1_out['发票状态'] == '有效发票') & (df1_out['金额'] > 0)].copy()
df2_in_clean = df2_in[(df2_in['发票状态'] == '有效发票') & (df2_in['金额'] > 0)].copy()
df2_out_clean = df2_out[(df2_out['发票状态'] == '有效发票') & (df2_out['金额'] > 0)].copy()

# 6b. 企业信息：标记D级但不在此阶段删除（建模时按需过滤）
df1_info_clean = df1_info.copy()
df1_info_clean['排除原因'] = ''
df1_info_clean.loc[df1_info_clean['信誉评级'] == 'D', '排除原因'] = '信誉评级D'

# 写入
df1_in_clean.to_parquet(os.path.join(data_dir, 'f1_进项_clean.parquet'))
df1_out_clean.to_parquet(os.path.join(data_dir, 'f1_销项_clean.parquet'))
df2_in_clean.to_parquet(os.path.join(data_dir, 'f2_进项_clean.parquet'))
df2_out_clean.to_parquet(os.path.join(data_dir, 'f2_销项_clean.parquet'))
df1_info_clean.to_parquet(os.path.join(data_dir, 'f1_企业信息_clean.parquet'))
df2_info.to_parquet(os.path.join(data_dir, 'f2_企业信息_clean.parquet'))

for name, df in [('f1_进项_clean', df1_in_clean), ('f1_销项_clean', df1_out_clean),
                  ('f2_进项_clean', df2_in_clean), ('f2_销项_clean', df2_out_clean),
                  ('f1_企业信息_clean', df1_info_clean), ('f2_企业信息_clean', df2_info)]:
    print(f"  {name}.parquet ({df.shape})")

# ══════════════════════════════════════════════
# 7. 候选池覆盖率
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("候选池覆盖率")
print("=" * 60)

# 排除D级后附件1有效企业数
n1_total = len(df1_info)
n1_valid = (df1_info['信誉评级'] != 'D').sum()
print(f"  附件1: {n1_total} 家 → 排除D级后 {n1_valid} 家 ({n1_valid/n1_total*100:.1f}%)")

# 附件2: 检查是否有企业无任何有效发票
eids_in  = set(df2_in_clean['企业代号'])
eids_out = set(df2_out_clean['企业代号'])
eids_info = set(df2_info['企业代号'])
no_invoice = eids_info - (eids_in | eids_out)
print(f"  附件2: {len(eids_info)} 家 → 无有效发票: {len(no_invoice)} 家")

print("\n清洗完成。")
