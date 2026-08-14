import pandas as pd

df = pd.read_pickle('E:/MathModel/problems/2025/C题/outputs/data/2025C-male-clean.pkl')
print('列名:', list(df.columns[:20]))

a = df[df['孕妇代码']=='A099']
print('\n=== A099 完整数据 ===')
print(a[['孕妇代码','孕周_数值','检测孕周','孕妇BMI','Y染色体浓度']].to_string())

print('\n=== 40+组所有人 ===')
over40 = df[df['孕妇BMI']>=40]
for pid, g in over40.groupby('孕妇代码'):
    rows = len(g)
    min_gw = g['孕周_数值'].min()
    max_gw = g['孕周_数值'].max()
    bmi = g['孕妇BMI'].iloc[0]
    y_vals = g['Y染色体浓度'].values
    print(f'{pid}: BMI={bmi:.1f}, {rows}条, 孕周{min_gw:.1f}-{max_gw:.1f}, Y={[round(y,4) for y in y_vals]}')
