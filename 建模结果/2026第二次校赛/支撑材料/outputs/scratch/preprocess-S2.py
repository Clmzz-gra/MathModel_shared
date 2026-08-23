"""
目的：
    S2 阶段 1.4 数据预处理：从共享清洗数据 c-data-cleaned.pkl 产出 S2-preprocessed.pkl，
    为 2.1 特征选择（Lasso+bootstrap 稳定性选择）与生物标志物解读提供统一口径的输入。

原理：
    - 标签映射（handoff-S2-code-agent.md §2 / problem-statement.md 口径，二分类患病=1/健康=0）：
        CRC(Zeller_fecal_colorectal_cancer): cancer=1, n+small_adenoma=0（small_adenoma 跟随 S1 主口径，
        未选定前按题面口径归健康，见 R4）；
        IBD(metahit): ibd_ulcerative_colitis+ibd_crohn_disease=1, n=0；
        Obesity(Chatelier_gut_obesity): obesity=1, leaness=0。
    - 近全零过滤（approach-S2-confirmed.md §1.2 / handoff §1.1）：零值占比 >95% 的特征剔除，
        三病并集统一口径（在全部 484 样本上计算零值占比，一次过滤），1331→264（与 A 类验证 V2 一致）。
    - CLR 中心对数比变换（approach §3.4 / handoff §1.2）：乘法替换零值→伪计数 δ=6.5e-06（与 S1 一致，
        proxy-replacement-checklist-S2.md P6），重归一化行和=1，再 clr(x)_j = ln x_j - (1/D)Σ_k ln x_k
        = ln(x_j / g(x))，g 为几何均值。CLR 消除成分数据定和约束引入的伪相关（F6）。
    - 分层 CV 折划分：StratifiedKFold(n_splits=5, shuffle=True, random_state=42)，每病独立划分，
        折索引落盘供 2.1 折内 Lasso（防泄漏）复用。
    - 特征名元数据：1331 物种特征名形如 k__..|p__..|c__..|o__..|f__..|g__..|s__..，
        按 '|' 拆分为 7 级分类学层级（界/门/纲/目/科/属/种），供属级聚合与标志物解读。

性能：
    轻量-不适用（纯数据搬运 + 向量化 CLR 变换 + 5 折划分，秒级，无并行需求；
    bootstrap/Lasso 等重计算留给 2.1，本任务不涉及）。

输入数据：
    - c-data-cleaned.pkl (处理后/清洗后) — dataset_name, disease, 1331 物种相对丰度特征列
      （相对丰度 0-100 量级，行和≈100；484 样本 × 1333 列）

输出：
    - outputs/data/S2-preprocessed.pkl — 三病标签 + 过滤后 264 特征原始丰度 + CLR 变换后丰度
      + 特征名分类学元数据 + 分层 CV 折索引 + meta（过滤阈值/δ/CV 参数/字段语义）
    - outputs/data/preprocess-report-S2.txt — 预处理关键数字报告（shape/过滤后特征数/标签分布/CLR 验证）

对应论文章节：
    §1.4 数据预处理（S2 特征选择与生物标志物）
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_IN = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
DATA_OUT = ROOT / "outputs" / "data" / "S2-preprocessed.pkl"
REPORT_OUT = ROOT / "outputs" / "data" / "preprocess-report-S2.txt"

# 参数（proxy-replacement-checklist-S2.md 当前值，无占位符）
FILTER_THRESHOLD = 0.95          # P5：零值占比 >95% 剔除
CLR_DELTA = 6.5e-06              # P6：CLR 伪计数 δ（与 S1 一致）
CV_N_SPLITS = 5                  # 分层 CV 折数
CV_SEED = 42                     # 分层 CV 随机种子
TAXONOMY_RANKS = ["k", "p", "c", "o", "f", "g", "s"]  # 界/门/纲/目/科/属/种

# 标签映射（handoff §2）
DATASETS = {
    "CRC": {
        "dataset_name": "Zeller_fecal_colorectal_cancer",
        "disease": ["cancer"],
        "healthy": ["n", "small_adenoma"],
    },
    "IBD": {
        "dataset_name": "metahit",
        "disease": ["ibd_ulcerative_colitis", "ibd_crohn_disease"],
        "healthy": ["n"],
    },
    "Obesity": {
        "dataset_name": "Chatelier_gut_obesity",
        "disease": ["obesity"],
        "healthy": ["leaness"],
    },
}


def clr(X: np.ndarray, delta: float = CLR_DELTA) -> np.ndarray:
    """CLR 变换：乘法替换零值→δ，重归一化行和=1，再 log(x/g(x))，g=几何均值。"""
    X = np.asarray(X, dtype=float)
    Xr = np.where(X == 0, delta, X)
    Xr = Xr / Xr.sum(axis=1, keepdims=True)
    g = np.exp(np.log(Xr).mean(axis=1, keepdims=True))
    return np.log(Xr / g)


def parse_taxonomy(feature: str) -> dict:
    """把 'k__..|p__..|...|s__..' 拆成 7 级分类学层级 dict。"""
    parts = feature.split("|")
    out = {rank: "" for rank in TAXONOMY_RANKS}
    for part in parts:
        for rank in TAXONOMY_RANKS:
            prefix = f"{rank}__"
            if part.startswith(prefix):
                out[rank] = part[len(prefix):]
                break
    return out


def main() -> None:
    df = pd.read_pickle(DATA_IN)
    feat_cols = [c for c in df.columns if c not in ("dataset_name", "disease")]
    X_all = df[feat_cols].astype(float).to_numpy()

    # 1. 近全零过滤（三病并集统一口径：全 484 样本上计算零值占比）
    zero_frac = (X_all == 0).mean(axis=0)
    keep_mask = zero_frac <= FILTER_THRESHOLD
    kept_cols = [c for c, m in zip(feat_cols, keep_mask) if m]
    n_before = len(feat_cols)
    n_after = int(keep_mask.sum())

    # 2. 特征名分类学元数据（264 特征）
    taxo = {rank: [] for rank in TAXONOMY_RANKS}
    for c in kept_cols:
        parsed = parse_taxonomy(c)
        for rank in TAXONOMY_RANKS:
            taxo[rank].append(parsed[rank])

    # 3. 每病：标签 + 过滤后原始丰度 + CLR + 分层 CV 折
    per_disease = {}
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("S2 阶段 1.4 预处理报告")
    report_lines.append(f"生成时间: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"输入: {DATA_IN.name}  shape={df.shape}")
    report_lines.append("=" * 70)
    report_lines.append(f"[过滤] 零值占比>{FILTER_THRESHOLD} 剔除特征数: {n_before - n_after}")
    report_lines.append(f"[过滤] 过滤后特征数: {n_after} / {n_before}")

    for short, cfg in DATASETS.items():
        sub = df[df["dataset_name"] == cfg["dataset_name"]]
        y = sub["disease"].isin(cfg["disease"]).astype(int).to_numpy()
        X_raw = sub[kept_cols].astype(float).to_numpy()
        X_clr = clr(X_raw)

        # 分层 CV 折（每病独立）
        skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_SEED)
        folds = [(tr.tolist(), te.tolist()) for tr, te in skf.split(X_raw, y)]

        per_disease[short] = {
            "dataset_name": cfg["dataset_name"],
            "X_raw": X_raw,          # 过滤后原始相对丰度（n × 264）
            "X_clr": X_clr,          # CLR 变换后丰度（n × 264）
            "y": y,                  # 标签（患病=1/健康=0）
            "cv_folds": folds,       # 5 折 (train_idx, test_idx)
            "n_samples": int(len(y)),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
        }

        # CLR 验证：每行 CLR 后均值应≈0（几何均值中心化性质）
        clr_row_mean = X_clr.mean(axis=1)
        report_lines.append(
            f"[{short}] n={len(y)} 患病={int(y.sum())} 健康={int((1-y).sum())} "
            f"| CLR 行均值 mean={clr_row_mean.mean():.3e} std={clr_row_mean.std():.3e}"
        )

    # 4. 落盘 pkl
    result = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "source": DATA_IN.name,
            "note": "S2 1.4 预处理产物：过滤+CLR+分层CV折，供 2.1 特征选择复用",
            "filter_threshold": FILTER_THRESHOLD,
            "clr_delta": CLR_DELTA,
            "n_features_before": n_before,
            "n_features_after": n_after,
            "cv": {"n_splits": CV_N_SPLITS, "shuffle": True, "seed": CV_SEED},
            "field_semantics": {
                "y": "二分类标签：患病=1/健康=0（CRC: cancer=1, n+small_adenoma=0；"
                     "IBD: ibd_ulcerative_colitis+ibd_crohn_disease=1, n=0；Obesity: obesity=1, leaness=0）",
                "X_raw": "过滤后原始相对丰度（0-100 量级，行和≈100），未 CLR",
                "X_clr": "CLR 变换后丰度（乘法替换 δ=6.5e-06 + 几何均值中心化），行均值≈0",
                "cv_folds": "StratifiedKFold(5, shuffle, seed=42) 折索引 (train_idx, test_idx)，"
                            "供 2.1 折内 Lasso 防泄漏复用",
                "feature_taxonomy": "264 特征 7 级分类学层级（k/p/c/o/f/g/s），供属级聚合与标志物解读",
            },
        },
        "feature_names": kept_cols,
        "feature_taxonomy": taxo,
        "per_disease": per_disease,
    }

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT, "wb") as f:
        pickle.dump(result, f, protocol=4)

    # 5. 报告落盘 + stdout
    report_text = "\n".join(report_lines) + "\n"
    REPORT_OUT.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"[输出] {DATA_OUT}")
    print(f"[报告] {REPORT_OUT}")


if __name__ == "__main__":
    main()
