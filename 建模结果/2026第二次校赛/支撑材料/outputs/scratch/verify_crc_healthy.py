"""
目的：核对 CRC 标志物健康组存在率口径（n 组 vs n+small_adenoma 组），对齐图内标注 2.7%/8.2%。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pickle
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / 'outputs/data/c-data-cleaned.pkl','rb') as f:
    df = pickle.load(f)
feat = [c for c in df.columns if c not in ('dataset_name','disease')]
X = df[feat].astype(float)
zeller = df[df['dataset_name']=='Zeller_fecal_colorectal_cancer']
for t in ['s__Fusobacterium_nucleatum','s__Peptostreptococcus_stomatis']:
    col = [c for c in feat if t in c][0]
    dis = zeller['disease']=='cancer'
    n_only = zeller['disease']=='n'
    n_plus = zeller['disease'].isin(['n','small_adenoma'])
    for name,mask in [('disease(cancer)',dis),('healthy(n)',n_only),('healthy(n+adenoma)',n_plus)]:
        v = X.loc[zeller[mask].index, col]
        print(f"{t} [{name}] n={mask.sum()} presence={(v>0).mean():.4f}")
