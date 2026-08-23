"""
目的：
    S1 阶段 1.4 数据预处理：从共享清洗数据 c-data-cleaned.pkl 产出 S1-preprocessed.pkl，
    供 2.1 正式模型（L2+CLR 主模型 / RF 对照 / 基线）与 small_adenoma 四口径敏感性分析直接加载。

原理：
    本脚本只做「标签构造 + 特征过滤 + CLR 变换 + 折划分」，不做任何模型训练（那是 2.1）。
    1) 标签映射（三数据集统一患病=1 / 健康=0，见 handoff-S1-code-agent.md §1.1）：
         Zeller CRC: 患病=cancer(48)；健康=n(47)+small_adenoma(26)（口径①默认主口径）
         metahit IBD: 患病=ibd_ulcerative_colitis(21)+ibd_crohn_disease(4)；健康=n(85)
         Chatelier Obesity: 患病=obesity(164)；健康=leaness(89)（少数类=健康，方向特殊）
    2) small_adenoma 四口径（2026-08-21 人类裁定，全做择优，仅影响 Zeller）：
         ① 归健康（默认主口径）：n+small_adenoma=0，cancer=1，n=121
         ② 归病变：cancer+small_adenoma=1，n=0，n=121
         ③ 剔除：剔除 26 例 small_adenoma，cancer=1/n=0，n=95
         ④ 单开一类：small_adenoma 不参与二分类（cancer=1/n=0，n=95），26 例单独作第三类
    3) 近全零过滤（与 S2 口径统一）：零值占比 >95% 的特征剔除，1331→264；三病并集统一口径
        （对全部 484 样本逐特征算零值占比，同一 264 特征集用于三数据集）。
    4) CLR 变换（仅主模型 L2 需要）：零值乘法替换 x←max(x,δ)，δ=0.65×1e-05=6.5e-06（AL-007），
        再逐行几何均值中心化 clr(x_ij)=ln(x_ij)-mean_k(ln(x_ik))，解除定和约束伪相关。
    5) 分层 CV 折划分：StratifiedKFold(n_splits=5, shuffle=True, random_state=42)，
        折索引预生成存 pkl（主口径三数据集 + 四口径各一套），2.1 直接复用，保证可复现。

性能：
    轻量-不适用（484×1331 小数据，过滤/CLR/折划分均为秒级向量化操作，无并行需求）。

输入数据：
    - c-data-cleaned.pkl（处理后，共享清洗数据）— dataset_name(数据集名, str), disease(疾病标签, str),
      1331 个物种级相对丰度特征列（float32，列名=7 级分类学层级 k__域|...|s__种）

输出：
    - S1-preprocessed.pkl — 预处理产物（三数据集 X_raw/X_clr/y/折划分 + 四口径标签/样本集 + 过滤特征集）
    - preprocess-report-S1.txt — 预处理报告（shape/标签分布/过滤特征数/四口径样本数/折划分）

对应论文章节：
    §1.4 数据预处理（特征过滤 + CLR + 标签构造 + 折划分，非正式建模章节）
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 项目根目录 = 本脚本上三级（outputs/scratch -> outputs -> 根）
ROOT = Path(__file__).resolve().parent.parent.parent
IN_PKL = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S1-preprocessed.pkl"
OUT_REPORT = ROOT / "outputs" / "data" / "preprocess-report-S1.txt"

META_COLS = ["dataset_name", "disease"]

# 标签映射：患病=1 / 健康=0；minority = 少数类标签（F1/Recall 正类）
DATASETS = {
    "Zeller_fecal_colorectal_cancer": {
        "positive": ["cancer"],
        "negative": ["n", "small_adenoma"],  # 口径①默认主口径：small_adenoma 归健康
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
        "minority": 0,  # 少数类=健康（方向特殊）
    },
}

DELTA = 0.65 * 1e-05  # 乘法替换 δ = 0.65 × 检出限(1e-05) = 6.5e-06
ZERO_RATIO_THRESHOLD = 0.95  # 近全零过滤阈值：零值占比 >95% 剔除
N_FOLDS = 5
SEED = 42


def clr_transform(X: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """CLR 变换：零值乘法替换 δ → 逐行几何均值中心化。

    clr(x_ij) = ln(max(x_ij, δ)) - (1/p) * sum_k ln(max(x_ik, δ))
    """
    X = X.copy()
    X[X == 0] = delta
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


def make_folds(y: np.ndarray, n_splits: int = N_FOLDS, seed: int = SEED):
    """分层 K 折折划分，返回 [{"train": [...], "test": [...]} x n_splits]（索引为 Python int）。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        folds.append({
            "train": [int(i) for i in tr],
            "test": [int(i) for i in te],
        })
    return folds


def main() -> None:
    # 1. 读共享清洗数据
    df = pd.read_pickle(IN_PKL)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    n_feat = len(feat_cols)
    assert n_feat == 1331, f"特征列数异常: {n_feat}（预期 1331）"
    X_all = df[feat_cols].values.astype(np.float64)  # 484 × 1331

    # 2. 近全零过滤（三病并集统一口径）：零值占比 >95% 剔除
    zero_ratio = (X_all == 0).mean(axis=0)  # 每特征零值占比（484 样本）
    keep_mask = zero_ratio <= ZERO_RATIO_THRESHOLD
    kept_indices = np.where(keep_mask)[0]
    n_kept = int(keep_mask.sum())
    n_removed = n_feat - n_kept
    assert n_kept == 264, f"过滤后特征数异常: {n_kept}（预期 264）"
    kept_feat_cols = [feat_cols[i] for i in kept_indices]
    X_all_f = X_all[:, keep_mask]  # 484 × 264

    # 3. 逐数据集构造标签 + 特征 + CLR + 折划分
    datasets = {}
    for name, cfg in DATASETS.items():
        sub_mask = (df["dataset_name"] == name).values
        X_raw = X_all_f[sub_mask]  # n × 264（过滤后原始丰度）
        y = df.loc[sub_mask, "disease"].isin(cfg["positive"]).astype(int).values
        X_clr = clr_transform(X_raw)
        folds = make_folds(y)
        datasets[name] = {
            "X_raw": X_raw.astype(np.float32),
            "X_clr": X_clr.astype(np.float32),
            "y": y.astype(np.int8),
            "minority": int(cfg["minority"]),
            "folds": folds,
            "n_samples": int(len(y)),
        }

    # 4. small_adenoma 四口径（仅 Zeller，样本集/标签预生成）
    zeller_name = "Zeller_fecal_colorectal_cancer"
    z_mask = (df["dataset_name"] == zeller_name).values
    z_disease = df.loc[z_mask, "disease"].values
    z_idx = np.where(z_mask)[0]  # 在 484 中的位置（仅用于报告，不落盘）
    is_cancer = (z_disease == "cancer")
    is_n = (z_disease == "n")
    is_adenoma = (z_disease == "small_adenoma")
    n_zeller = int(z_mask.sum())
    assert n_zeller == 121, f"Zeller 样本数异常: {n_zeller}（预期 121）"

    # 口径① 归健康（默认主口径）：n+small_adenoma=0，cancer=1
    y_c1 = is_cancer.astype(int)
    # 口径② 归病变：cancer+small_adenoma=1，n=0
    y_c2 = (is_cancer | is_adenoma).astype(int)
    # 口径③ 剔除：剔除 small_adenoma，cancer=1/n=0
    keep_c3 = ~is_adenoma
    y_c3 = is_cancer[keep_c3].astype(int)
    # 口径④ 单开一类：small_adenoma 不参与二分类（cancer=1/n=0），26 例单独第三类
    y_c4 = is_cancer[keep_c3].astype(int)
    adenoma_idx = np.where(is_adenoma)[0]  # 在 Zeller 121 内的位置

    adenoma_calibers = {
        "CRC_adenoma_as_healthy": {
            "y": y_c1.astype(np.int8),
            "n_samples": n_zeller,
            "minority": 1,
            "folds": make_folds(y_c1),
            "note": "口径① 归健康（默认主口径）：n+small_adenoma=0，cancer=1",
        },
        "CRC_adenoma_as_diseased": {
            "y": y_c2.astype(np.int8),
            "n_samples": n_zeller,
            "minority": 1,
            "folds": make_folds(y_c2),
            "note": "口径② 归病变：cancer+small_adenoma=1，n=0",
        },
        "CRC_adenoma_excluded": {
            "y": y_c3.astype(np.int8),
            "n_samples": int(keep_c3.sum()),
            "minority": 1,
            "folds": make_folds(y_c3),
            "keep_mask": keep_c3,  # 在 Zeller 121 内的布尔掩码（True=保留）
            "note": "口径③ 剔除：剔除 26 例 small_adenoma，cancer=1/n=0",
        },
        "CRC_adenoma_separate": {
            "y": y_c4.astype(np.int8),
            "n_samples": int(keep_c3.sum()),
            "minority": 1,
            "folds": make_folds(y_c4),
            "keep_mask": keep_c3,
            "adenoma_indices": [int(i) for i in adenoma_idx],
            "note": "口径④ 单开一类：small_adenoma 不参与二分类，26 例单独第三类",
        },
    }

    # 5. 组装 pkl
    out = {
        "meta": {
            "generated": "S1 1.4 预处理",
            "source": "c-data-cleaned.pkl",
            "note": "标签映射患病=1/健康=0；近全零过滤 1331→264；CLR δ=6.5e-06；分层5折CV seed=42",
            "field_semantics": {
                "datasets.<name>.y": "患病=1/健康=0（口径①默认主口径，small_adenoma 归健康）",
                "datasets.<name>.X_raw": "近全零过滤后 264 维原始相对丰度（RF 对照输入）",
                "datasets.<name>.X_clr": "264 维 CLR 变换后特征（L2 主模型输入）",
                "datasets.<name>.folds": "StratifiedKFold(5,shuffle,seed=42) 折索引（train/test 为样本内位置）",
                "adenoma_calibers.*.y": "small_adenoma 四口径标签（仅 Zeller），keep_mask 为 Zeller 121 内布尔掩码",
                "adenoma_calibers.CRC_adenoma_separate.adenoma_indices": "small_adenoma 26 例在 Zeller 121 内的位置（第三类）",
            },
        },
        "feature_names": kept_feat_cols,
        "filter": {
            "n_features_before": n_feat,
            "n_features_after": n_kept,
            "n_removed": n_removed,
            "zero_ratio_threshold": ZERO_RATIO_THRESHOLD,
            "kept_indices": [int(i) for i in kept_indices],
        },
        "clr": {
            "delta": DELTA,
            "function": "clr_transform(x) = ln(max(x,δ)) - mean_k(ln(max(x_k,δ)))",
        },
        "datasets": datasets,
        "adenoma_calibers": adenoma_calibers,
    }

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        import pickle
        pickle.dump(out, f, protocol=4)

    # 6. 组装报告
    lines = []
    lines.append("=" * 70)
    lines.append("S1 数据预处理报告（阶段 1.4）")
    lines.append("=" * 70)
    lines.append("")
    lines.append("[1] 输入数据")
    lines.append(f"    c-data-cleaned.pkl: {df.shape}（2 元数据 + {n_feat} float32 特征）")
    lines.append("")
    lines.append("[2] 近全零过滤（零值占比 >95% 剔除，三病并集统一口径）")
    lines.append(f"    过滤前特征数: {n_feat}")
    lines.append(f"    过滤后特征数: {n_kept}")
    lines.append(f"    剔除特征数: {n_removed}")
    lines.append("")
    lines.append("[3] 三数据集标签分布（患病=1 / 健康=0，口径①默认主口径）")
    for name, cfg in DATASETS.items():
        d = datasets[name]
        n1 = int((d["y"] == 1).sum())
        n0 = int((d["y"] == 0).sum())
        lines.append(f"    {name}: n={d['n_samples']}，患病={n1}，健康={n0}，少数类={d['minority']}")
    lines.append("")
    lines.append("[4] small_adenoma 四口径样本数（仅 Zeller）")
    for cal, c in adenoma_calibers.items():
        n1 = int((c["y"] == 1).sum())
        n0 = int((c["y"] == 0).sum())
        lines.append(f"    {cal}: n={c['n_samples']}，患病={n1}，健康={n0} — {c['note']}")
    lines.append("")
    lines.append("[5] CLR 变换")
    lines.append(f"    δ = {DELTA}（乘法替换，AL-007）")
    lines.append("    clr(x_ij) = ln(max(x_ij,δ)) - mean_k(ln(max(x_ik,δ)))（逐行几何均值中心化）")
    lines.append("")
    lines.append("[6] 分层 CV 折划分")
    lines.append(f"    StratifiedKFold(n_splits={N_FOLDS}, shuffle=True, random_state={SEED})")
    for name in DATASETS:
        d = datasets[name]
        sizes = [len(f["test"]) for f in d["folds"]]
        lines.append(f"    {name}: 5 折 test 大小 = {sizes}")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"[OK] 预处理完成，产物已落盘: {OUT_PKL}")
    print(f"[OK] 预处理报告已落盘: {OUT_REPORT}")


if __name__ == "__main__":
    main()
