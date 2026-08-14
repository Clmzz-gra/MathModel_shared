import pandas as pd, numpy as np
male = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
male['gw'] = male['孕周_数值']
male['pass'] = (male['Y染色体浓度'] >= 0.04).astype(int)

fp = male[male['pass']==1].groupby('孕妇代码')['gw'].min().reset_index()
fp.columns = ['code','first_pass']

person = male[['孕妇代码','孕妇BMI','bmi_group','IVF妊娠','IVF_编码','年龄','怀孕次数_num','生产次数']].drop_duplicates('孕妇代码')
fp = fp.merge(person, left_on='code', right_on='孕妇代码')

print('=== First pass time by fertility type ===')
for ft in fp['IVF妊娠'].unique():
    sub = fp[fp['IVF妊娠']==ft]
    print(f'{ft}: n={len(sub)}, mean={sub.first_pass.mean():.1f}w, median={sub.first_pass.median():.1f}w, SD={sub.first_pass.std():.1f}w, BMI={sub.孕妇BMI.mean():.1f}, age={sub.年龄.mean():.1f}')

print()
print('=== Assisted reproduction individuals ===')
assisted = fp[fp['IVF_编码']>0].sort_values('first_pass')
for _, r in assisted.iterrows():
    print(f'{r.code}: {r["IVF妊娠"]}, first_pass={r.first_pass:.1f}w, BMI={r.孕妇BMI:.1f}, age={r.年龄:.0f}, preg={r.怀孕次数_num:.0f}')

# Also check: total records for fertility types
print()
print('=== Total records by fertility type ===')
print(male['IVF妊娠'].value_counts())
