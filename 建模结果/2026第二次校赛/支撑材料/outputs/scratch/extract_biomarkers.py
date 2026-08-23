"""
目的：
    计算 6 个已知标志物在各数据集的病/健存在率，并核对特征列名。

原理：
    存在率 = 该物种特征非零的样本占比。按 dataset 分组，病组/健组分别算。
    CRC: 病=cancer, 健=n；IBD: 病=ibd_uc+ibd_cd, 健=n；Obesity: 病=obesity, 健=leaness。

输入数据：
    - outputs/data/c-data-cleaned.pkl

输出：
    打印存在率

对应论文章节：数据理解节
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PKL = ROOT / 'outputs/data/c-data-cleaned.pkl'
with open(PKL, 'rb') as f:
    df = pickle.load(f)

feat_cols = [c for c in df.columns if c not in ('dataset_name', 'disease')]
X = df[feat_cols].astype(float)

# 目标标志物物种短名 -> 全特征列名匹配
targets = ['Fusobacterium_nucleatum','Peptostreptococcus_stomatis',
           'Peptostreptococcus_somerae','Bifidobacterium_bifidum',
           'Akkermansia_muciniphila','Bacteroides_fragilis']
print("特征列中匹配的目标物:")
found = {}
for t in targets:
    m = [c for c in feat_cols if t in c]
    print(f"  {t}: {m}")
    if m:
        found[t] = m[0]

def presence_rate(group_idx, col):
    v = X[col].iloc[group_idx]
    return (v > 0).mean()

for ds in df['dataset_name'].unique():
    sub = df[df['dataset_name'] == ds]
    idx = sub.index
    print(f"\n===== {ds} (n={len(sub)}) =====")
    # 组定义
    if 'Zeller' in ds:
        dis_idx = sub['disease'].isin(['cancer']).values
        heal_idx = sub['disease'].isin(['n']).values
        dis_name, heal_name = 'cancer', 'n'
    elif ds == 'metahit':
        dis_idx = sub['disease'].isin(['ibd_ulcerative_colitis','ibd_crohn_disease']).values
        heal_idx = sub['disease'].isin(['n']).values
        dis_name, heal_name = 'IBD(uc+cd)', 'n'
    else:
        dis_idx = sub['disease'].isin(['obesity']).values
        heal_idx = sub['disease'].isin(['leaness']).values
        dis_name, heal_name = 'obesity', 'leaness'
    print(f"  病组 {dis_name}: {dis_idx.sum()}, 健组 {heal_name}: {heal_idx.sum()}")
    subX = X.loc[sub.index]
    dis_df = subX[dis_idx]
    heal_df = subX[heal_idx]
    for t, col in found.items():
        if col in subX.columns:
            d_rate = (dis_df[col] > 0).mean()
            h_rate = (heal_df[col] > 0).mean()
            print(f"  {t}: 病组存在率={d_rate:.3f}, 健组存在率={h_rate:.3f}")
