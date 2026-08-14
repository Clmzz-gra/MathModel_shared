# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

DATA = "E:/MathModel/problems/2025/C题/2025C题测试/outputs/data"

dff = pd.read_pickle(f"{DATA}/2025C-female-clean.pkl")
df1 = dff.sort_values(["孕妇代码","孕周_数值"]).drop_duplicates("孕妇代码", keep="first")
cols = ['孕周_数值','孕妇BMI','年龄','X染色体浓度',
        'GC含量','在参考基因组上比对的比例','重复读段的比例','被过滤掉读段数的比例',
        '13号染色体的Z值','18号染色体的Z值','21号染色体的Z值','X染色体的Z值']
ivf = pd.get_dummies(df1['IVF妊娠'].fillna('自然受孕'), prefix='IVF').astype(float)
feat = pd.concat([df1[cols].reset_index(drop=True), ivf.reset_index(drop=True)], axis=1)
feat = feat.dropna().astype(float)
meta = df1.iloc[feat.index][['孕妇代码','AB_异常']].copy()
meta.to_pickle(f"{DATA}/umap_female_meta.pkl")
print(f"Female meta saved: {meta.shape}")
print(f"AB distribution: {meta['AB_异常'].value_counts().to_dict()}")
