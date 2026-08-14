"""
量纲初查：各发票表的金额/税额/价税合计分布，以及发票状态分布
"""
import pandas as pd
import os

out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(out_dir, 'data')

for fname in ['f1_进项发票信息.parquet', 'f1_销项发票信息.parquet',
              'f2_进项发票信息.parquet', 'f2_销项发票信息.parquet']:
    df = pd.read_parquet(os.path.join(data_dir, fname))
    print(f"=== {fname} ===")
    print(f"  发票状态分布:\n{df['发票状态'].value_counts()}")
    num_cols = ['金额', '税额', '价税合计']
    for c in num_cols:
        s = df[c]
        print(f"  {c}: mean={s.mean():.2f}, std={s.std():.2f}, "
              f"min={s.min():.2f}, p25={s.quantile(0.25):.2f}, "
              f"p50={s.median():.2f}, p75={s.quantile(0.75):.2f}, max={s.max():.2f}")
    # 负数金额占比
    neg = (df['金额'] < 0).sum()
    print(f"  负数金额行数: {neg} ({neg/len(df)*100:.2f}%)")
    # 价税合计 vs 金额+税额 一致性
    diff = (df['价税合计'] - df['金额'] - df['税额']).abs()
    print(f"  价税合计≠金额+税额: {(diff > 0.01).sum()} 行")
    print()

# 日期分布
for fname in ['f1_进项发票信息.parquet', 'f1_销项发票信息.parquet']:
    df = pd.read_parquet(os.path.join(data_dir, fname))
    df['year'] = pd.DatetimeIndex(df['开票日期']).year
    print(f"=== {fname} 年度分布 ===")
    print(df['year'].value_counts().sort_index())
    print()
