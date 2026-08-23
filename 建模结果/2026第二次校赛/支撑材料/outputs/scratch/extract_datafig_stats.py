"""
目的：
    只读提取 c-data-cleaned.pkl 的关键统计量 + 5 张 chart-*.pdf 的文本元素，供撰写数据特征图解读教学文档取数。

原理：
    数据特征图（S0 数据理解阶段）对应的统计量：三数据集样本构成、零值占比/稀疏度、非零丰度对数直方图、PCA/t-SNE 批次效应、已知标志物存在率。
    只做只读提取，不写新 pkl、不改脚本。

输入数据：
    - outputs/data/c-data-cleaned.pkl — 清洗后数据
    - outputs/figures/chart-*.pdf ×5 — PyMuPDF 提取文本

输出：
    打印关键统计量与各图文本元素

对应论文章节：
    数据理解节（阶段 0 数据特征图）
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PKL = ROOT / 'outputs/data/c-data-cleaned.pkl'
FIG = ROOT / 'outputs/figures'

print("=" * 70)
print("STEP 1: 解析 c-data-cleaned.pkl 结构")
print("=" * 70)
with open(PKL, 'rb') as f:
    data = pickle.load(f)

def describe(obj, name="root", depth=0, maxdepth=6):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict[{len(obj)}] keys={list(obj.keys())[:15]}")
        for k, v in obj.items():
            describe(v, k, depth + 1, maxdepth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__}[{len(obj)}]")
        if len(obj) > 0 and depth < maxdepth:
            describe(obj[0], "item[0]", depth + 1, maxdepth)
    else:
        s = str(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {s[:120]}")

describe(data, "data", 0, 5)
