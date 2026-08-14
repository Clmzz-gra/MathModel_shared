import pandas as pd, os
cache=r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data'
raw=pd.read_pickle(os.path.join(cache,'2025C-raw-female.pkl'))
clean=pd.read_pickle(os.path.join(cache,'2025C-female-clean.pkl'))
sub4=pd.read_pickle(os.path.join(cache,'2025C-sub4-preprocessed.pkl'))

# 原始数据
print('=== 胎儿是否健康 ===')
print(raw['胎儿是否健康'].value_counts(dropna=False))

# 原始数据 Z>3 筛选
zcols = ['13号染色体的Z值','18号染色体的Z值','21号染色体的Z值']
z_alert = (raw[zcols].abs() > 3).any(axis=1)
unlabeled_raw = raw['胎儿是否健康'].isna() | (raw['胎儿是否健康'].astype(str).str.strip()=='') | (~raw['胎儿是否健康'].astype(str).isin(['0','1']))
print(f'原始 Z>3: {z_alert.sum()}, 未标注: {unlabeled_raw.sum()}, Z>3且未标注: {(z_alert & unlabeled_raw).sum()}')

print()
print('=== clean AB_异常 ===')
print(clean['AB_异常'].value_counts(dropna=False))

print()
print('=== sub4 AB_异常 ===')
print(sub4['AB_异常'].value_counts(dropna=False))

# sub4 中 Z>3 的情况
sub4_zcols_orig = ['13号染色体的Z值','18号染色体的Z值','21号染色体的Z值']
has_zcols = all(c in sub4.columns for c in sub4_zcols_orig)
print(f'sub4有原始Z值列: {has_zcols}')

# 用 corrected Z 值
zcols_corr = ['Z13_corrected','Z18_corrected','Z21_corrected']
z_alert_corr = (sub4[zcols_corr].abs() > 3).any(axis=1)
print(f'sub4中 Z_corrected>3: {z_alert_corr.sum()}')
print(f'  且 AB=0 (标为正常): {(z_alert_corr & (sub4["AB_异常"]==0)).sum()}')
print(f'  且 AB=1 (标为异常): {(z_alert_corr & (sub4["AB_异常"]==1)).sum()}')
print(f'  且 AB=NaN: {(z_alert_corr & sub4["AB_异常"].isna()).sum()}')
