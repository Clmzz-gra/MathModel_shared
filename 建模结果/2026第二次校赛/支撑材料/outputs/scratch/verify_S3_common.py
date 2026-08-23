"""
目的：
    S3 A 类验证共享工具模块：数据加载、二分类标签映射、CLR 变换、leave-one-disease-out
    三组合定义、分类学层级聚合（物种/属/门），供 verify-S3-a1~a5 脚本 import 复用。

原理：
    - 标签映射：患病=1（cancer / ibd_ulcerative_colitis / ibd_crohn_disease / obesity），
      健康=0（n / small_adenoma / leaness），与 inventory-B.txt 口径一致。
    - CLR（中心对数比变换）：对定和成分数据（每行丰度和≈100）解除定和约束。
      零值用乘法替换 δ=0.65×检出限（检出限=全局最小非零值 1e-5，故 δ=6.5e-6）。
      CLR_i = log(x_i) - mean_j(log(x_j))（逐样本），尺度不变故替换后无需重归一化。
      CLR 是逐样本变换，无跨样本参数，故不引入训练/测试泄漏。
    - 分类学聚合：特征名形如 k__..|p__..|c__..|o__..|f__..|g__X|s__Y，
      属级=按 g__ 段聚合（同属物种丰度求和），门级=按 p__ 段聚合。
    - 三组合（leave-one-disease-out）：C1 训练{metahit,Chatelier}测Zeller(CRC)；
      C2 训练{Zeller,Chatelier}测metahit(IBD)；C3 训练{Zeller,metahit}测Chatelier(Obesity)。

性能：
    轻量-不适用（484×1331 小数据，秒级；无并行需求）。

输入数据：
    - B-raw.pkl (处理后) — dataset_name, disease, 1331 个物种级相对丰度特征（float64）

输出：
    - 无落盘（只提供函数，供各 verify 脚本调用）

对应论文章节：
    §S3 跨疾病预测模型（A 类验证支撑，不入论文正文）
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "data" / "B-raw.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ

# 数据集 → 疾病名
DATASET_DISEASE = {
    "Zeller_fecal_colorectal_cancer": "CRC",
    "metahit": "IBD",
    "Chatelier_gut_obesity": "Obesity",
}

# 患病标签映射（1=患病，0=健康）
POSITIVE_LABELS = {"cancer", "ibd_ulcerative_colitis", "ibd_crohn_disease", "obesity"}

# leave-one-disease-out 三组合：训练数据集列表 → 测试数据集
COMBOS = {
    "C1": (["metahit", "Chatelier_gut_obesity"], "Zeller_fecal_colorectal_cancer"),
    "C2": (["Zeller_fecal_colorectal_cancer", "Chatelier_gut_obesity"], "metahit"),
    "C3": (["Zeller_fecal_colorectal_cancer", "metahit"], "Chatelier_gut_obesity"),
}


def load_data():
    """加载 B-raw.pkl，返回 (df, feature_cols)。"""
    df = pd.read_pickle(DATA)
    feature_cols = [c for c in df.columns if c not in ("dataset_name", "disease")]
    return df, feature_cols


def binary_label(disease_series):
    """disease 列 → 二分类标签（1=患病，0=健康）。"""
    return disease_series.map(lambda x: 1 if x in POSITIVE_LABELS else 0).astype(int)


def clr_transform(X):
    """CLR 变换（逐样本），X 为 DataFrame（行=样本，列=特征）。零值乘法替换 δ。"""
    X = X.replace(0.0, CLR_DELTA)
    logX = np.log(X)
    return logX.sub(logX.mean(axis=1), axis=0)


def taxonomy_aggregate(X, level):
    """按分类学层级聚合特征（求和）。level ∈ {'species','genus','phylum'}。

    species=原特征名；genus=按 g__ 段；phylum=按 p__ 段。
    返回聚合后的 DataFrame（列=聚合名）。
    """
    if level == "species":
        return X.copy()
    seg = "g__" if level == "genus" else "p__"
    # 从特征名提取聚合键：取 seg 段（到下一个 | 或 s__ 前）
    def key(col):
        parts = col.split("|")
        for p in parts:
            if p.startswith(seg):
                return p
        return col  # 兜底：无该段则用原名
    # pandas 3.0 移除 groupby(axis=1)，改用转置后按行分组再转回
    agg = X.T.groupby(key).sum().T
    return agg


def get_combo_data(df, feature_cols, combo_name):
    """按组合切分训练/测试，返回 (X_train, y_train, X_test, y_test, test_pos_frac)。"""
    train_ds, test_ds = COMBOS[combo_name]
    train_mask = df["dataset_name"].isin(train_ds)
    test_mask = df["dataset_name"] == test_ds
    X_train = df.loc[train_mask, feature_cols].astype(float)
    y_train = binary_label(df.loc[train_mask, "disease"])
    X_test = df.loc[test_mask, feature_cols].astype(float)
    y_test = binary_label(df.loc[test_mask, "disease"])
    test_pos_frac = y_test.mean()
    return X_train, y_train, X_test, y_test, test_pos_frac
