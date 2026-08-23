"""
目的：
    S2 A 类验证公共工具：数据加载、标签映射、CLR 变换、零值占比、BH-FDR 等共享函数，
    供 verify-S2-v1~v6 六个验证脚本统一 import，避免重复实现。

原理：
    - 标签映射（problem-statement.md 口径）：每数据集二分类，患病=1/健康=0。
      CRC(Zeller): cancer=1, n+small_adenoma=0；IBD(metahit): ibd_ulcerative_colitis+ibd_crohn_disease=1, n=0；
      Obesity(Chatelier): obesity=1, leaness=0。
    - CLR 变换：成分数据（行和≈100）先乘法替换（零值→伪计数=最小正值的 1/2），
      归一化行和=1，再 log(x/g(x))，g=几何均值。CLR 消除定和约束引入的伪相关。
    - BH-FDR：Benjamini-Hochberg 多重比较校正，控制 FDR≤0.05。

性能：
    轻量-不适用（纯数据搬运与向量化变换，秒级，无并行需求）。

输入数据：
    - B-raw.pkl (原始) — dataset_name, disease, 1331 物种相对丰度特征

输出：
    - 无落盘（仅提供函数）

对应论文章节：
    §1.1 A 类验证（共享工具，非正式建模章节）
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体（探索图标题/标签用中文，避免 DejaVu 缺字显示为方框）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "B-raw.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

DATASETS = {
    "Zeller_fecal_colorectal_cancer": {
        "disease": ["cancer"],
        "healthy": ["n", "small_adenoma"],
        "short": "CRC",
    },
    "metahit": {
        "disease": ["ibd_ulcerative_colitis", "ibd_crohn_disease"],
        "healthy": ["n"],
        "short": "IBD",
    },
    "Chatelier_gut_obesity": {
        "disease": ["obesity"],
        "healthy": ["leaness"],
        "short": "Obesity",
    },
}


def load_df() -> pd.DataFrame:
    return pd.read_pickle(DATA_PKL)


def get_label(df: pd.DataFrame, dataset: str) -> np.ndarray:
    """返回该数据集的二分类标签（患病=1/健康=0），仅含该数据集样本。"""
    sub = df[df["dataset_name"] == dataset]
    cfg = DATASETS[dataset]
    return sub["disease"].isin(cfg["disease"]).astype(int).to_numpy()


def get_X(df: pd.DataFrame, dataset: str) -> np.ndarray:
    """返回该数据集的特征矩阵（float64，n_samples × 1331）。"""
    sub = df[df["dataset_name"] == dataset]
    feat_cols = [c for c in df.columns if c not in ("dataset_name", "disease")]
    return sub[feat_cols].astype(float).to_numpy()


def get_feature_names(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("dataset_name", "disease")]


def clr(X: np.ndarray) -> np.ndarray:
    """CLR 变换（乘法替换零值 + 归一化 + log 比几何均值）。"""
    X = np.asarray(X, dtype=float)
    pos = X[X > 0]
    minpos = pos.min() if pos.size else 1e-6
    pseudo = minpos / 2.0
    Xr = np.where(X == 0, pseudo, X)
    Xr = Xr / Xr.sum(axis=1, keepdims=True)
    g = np.exp(np.log(Xr).mean(axis=1, keepdims=True))
    return np.log(Xr / g)


def zero_fraction(X: np.ndarray) -> np.ndarray:
    """每特征零值占比（0~1）。"""
    return (X == 0).mean(axis=0)


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回布尔数组（True=显著）。"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    k = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    rejected = np.zeros(n, dtype=bool)
    rejected[order[:k]] = True
    return rejected
