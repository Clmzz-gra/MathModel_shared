"""
目的：
    S1 A 类验证共享工具模块：数据加载、标签映射、CLR 变换、分层 CV 评估。

原理：
    三数据集二分类标签映射（患病=1 / 健康=0）；CLR 变换 = log(x) - mean(log(x))（逐行几何均值中心化），
    零值先做乘法替换（AL-007：δ=0.65×检出限，检出限≈非零最小值 1e-05 → δ=6.5e-06）；
    评估用分层 K 折 CV（K=5），报告 AUC（阈值无关）+ ACC + F1/Recall（按少数类为正类）。

性能：
    轻量-不适用（小数据 484×1331，单次 CV 秒级；RF 用 n_jobs=-1 并行）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度特征

输出：
    - 无（纯函数模块，供 verify-S1-a*.py 调用）

对应论文章节：
    §1.1 A 类验证（探索，不入论文）
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # worktree 根
DATA = ROOT / "outputs" / "data" / "B-raw.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

# 标签映射：患病=1 / 健康=0；minority = 少数类标签（F1/Recall 正类）
DATASETS = {
    "Zeller_fecal_colorectal_cancer": {
        "positive": ["cancer"],
        "negative": ["n", "small_adenoma"],
        "minority": 1,
    },
    "metahit": {
        "positive": ["ibd_ulcerative_colitis", "ibd_crohn_disease"],
        "negative": ["n"],
        "minority": 1,
    },
    "Chatelier_gut_obesity": {
        "positive": ["obesity"],
        "negative": ["leaness"],
        "minority": 0,
    },
}

DELTA = 0.65 * 1e-05  # 乘法替换 δ = 0.65 × 检出限(1e-05)


def load_data():
    return pd.read_pickle(DATA)


def get_dataset(df, name):
    cfg = DATASETS[name]
    sub = df[df["dataset_name"] == name].copy()
    X = sub.drop(columns=["dataset_name", "disease"]).values.astype(np.float64)
    y = sub["disease"].isin(cfg["positive"]).astype(int).values
    return X, y, cfg["minority"]


def clr_transform(X, delta=DELTA):
    X = X.copy()
    X[X == 0] = delta
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


def cv_evaluate(X, y, make_model, k=5, minority=1, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs, accs, f1s, recs = [], [], [], []
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yp = m.predict(X[te])
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, pos_label=minority))
        recs.append(recall_score(y[te], yp, pos_label=minority))
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "acc": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "recall": float(np.mean(recs)),
    }


def make_logistic(C=1.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000, random_state=42),
    )


def make_rf(n_estimators=500):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)


def ensure_fig_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR
