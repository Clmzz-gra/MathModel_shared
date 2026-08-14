import numpy as np, pandas as pd
male = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
male['gw'] = male['孕周_数值']
g = male[male['bmi_group']=='[32,36)']

print(f'[32,36): {len(g)} records, {g["孕妇代码"].nunique()} individuals')
print(f'GW range: {g["gw"].min():.1f} - {g["gw"].max():.1f}')
print()

print('GW distribution:')
for w in np.arange(10, 26):
    mask = (g['gw'] >= w-0.5) & (g['gw'] < w+0.5)
    n = mask.sum()
    k = (g.loc[mask, 'Y染色体浓度'] >= 0.04).sum()
    bar = '#' * n if n > 0 else ''
    print(f'  week {w:3.0f}: n={n:3d}  pass={k:3d}  {bar}')

print()
det_counts = g.groupby('孕妇代码').size()
print('Detection count per person:')
for cnt, num in det_counts.value_counts().sort_index().items():
    print(f'  {cnt} times: {num} people')

print()
g11 = g[(g['gw']>=10.5) & (g['gw']<11.5)]
print(f'11周: {len(g11)} records')
for _, row in g11.iterrows():
    y = row['Y染色体浓度']
    ps = 'PASS' if y >= 0.04 else 'FAIL'
    print(f'  {row["孕妇代码"]}: gw={row["gw"]:.1f}, Y={y:.4f}, {ps}')
