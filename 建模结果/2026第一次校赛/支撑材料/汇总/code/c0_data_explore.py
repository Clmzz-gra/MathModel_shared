import pandas as pd
import os

base = r'e:\MathModel-school-competition\problems\2026第一次模拟赛赛题\C题 关于某竞赛网评结果的建模与分析\附件'

for f in ['A.xlsx','B.xlsx','C.xlsx','D.xlsx','E.xlsx']:
    df = pd.read_excel(os.path.join(base, f))
    print(f'=== {f} ===')
    print(f'总论文数: {len(df)}')
    print(f'有成绩(进集中评审): {df["成绩"].notna().sum()}')
    print(f'无成绩(网评淘汰): {df["成绩"].isna().sum()}')
    scores = []
    for i in range(4):
        col = '打分' if i == 0 else f'打分.{i}'
        scores.append(df[col])
    all_scores = pd.concat(scores)
    print(f'打分范围: {all_scores.min():.0f} - {all_scores.max():.0f}, 均值: {all_scores.mean():.2f}, 标准差: {all_scores.std():.2f}')
    awards = df[df['成绩'].notna()]['成绩'].value_counts()
    print(f'奖项分布: {dict(awards)}')
    print()
