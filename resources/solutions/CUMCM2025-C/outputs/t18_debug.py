import pandas as pd
import numpy as np

prep = pd.read_pickle(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-sub4-preprocessed.pkl')
clean = pd.read_pickle(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-female-clean.pkl')
prep = prep.drop_duplicates(subset='孕妇代码', keep='first').copy()
clean = clean.drop_duplicates(subset='孕妇代码', keep='first').copy()

extra_cols = ['X染色体浓度','13号染色体的GC含量','18号染色体的GC含量','21号染色体的GC含量',
    '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值',
    'GC含量','在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
    '孕妇BMI','年龄','体重','身高','孕周_数值','染色体的非整倍体','AB_异常']
extra_cols = [c for c in extra_cols if c in clean.columns]
clean_sub = clean[['孕妇代码'] + extra_cols].copy()
clean_sub = clean_sub.rename(columns={
    '13号染色体的Z值':'13号染色体的Z值_raw_clean',
    '18号染色体的Z值':'18号染色体的Z值_raw_clean',
    '21号染色体的Z值':'21号染色体的Z值_raw_clean',
    'X染色体的Z值':'X染色体的Z值_raw_clean',
})
df = prep.merge(clean_sub, on='孕妇代码', how='left', suffixes=('', '_y'))

print('Step1 - check col exists:')
col = '染色体的非整倍体_y'
print(f'  col exists: {col in df.columns}')
print(f'  dtype: {df[col].dtype}')
vals = df[col].values
print(f'  first 20 values:')
for i, v in enumerate(vals[:20]):
    print(f'    [{i}] type={type(v).__name__}, val={repr(v)}')

print(f'  count isinstance str: {sum(isinstance(v, str) for v in vals)}')
print(f'  count str.contains T18: {df[col].str.contains("T18", na=False).sum()}')
r1 = df[col].apply(lambda x: isinstance(x, str) and 'T18' in x).sum()
print(f'  apply lambda: {r1}')

# Show which rows have T18
mask = df[col].str.contains('T18', na=False)
print(f'  T18 rows: {mask.sum()}')
print(f'  T18 indices: {df.index[mask].tolist()}')
print(f'  T18 values: {df.loc[mask, col].tolist()}')
print(f'  T18 AB_异常_y: {df.loc[mask, "AB_异常_y"].tolist()}')

print()
print('AB_异常_y value_counts:')
print(df['AB_异常_y'].value_counts(dropna=False))
print()
print('AB_异常 value_counts:')
print(df['AB_异常'].value_counts(dropna=False))
