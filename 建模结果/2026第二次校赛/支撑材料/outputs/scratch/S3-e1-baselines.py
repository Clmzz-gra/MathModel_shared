"""
目的：
    S3 补充实验 E1：免训练/低容量基线（NCM 最近类均值原型 + PLS-DA 偏最小二乘判别）跨疾病
    LODO 评估。检验「跨疾病 AUC 天花板（~0.56）是否与模型容量无关」——若免训练（NCM，仅类均值
    两个参数）与低容量（PLS-DA，线性潜变量）方法也 ≈0.56，则坐实数据固有限制（疾病特异信号 +
    标签语义漂移），而非 Logistic/RF/DANN 等模型容量不足所致。

原理：
    - 数据口径（与 S3 正式实现同口径）：c-data-cleaned.pkl（484×1331）→ 近全零过滤（零值占比
      >95% 剔除，1331→264，三病并集）→ CLR（零值乘法替换 δ=0.65×检出限=6.5e-6，逐样本 log 后
      减行均值）。本脚本直接复用 S3-preprocessed.pkl 的 X_filtered（484×264 原始丰度，与 S3
      正式实现同口径），再施加 CLR，避免重解析原始 Excel。
    - LODO 协议：每次留一种疾病作测试集，其余两种疾病作训练集，共 3 组合（C1 测 CRC / C2 测
      IBD / C3 测 Obesity）。测试疾病在训练阶段完全不可见（标签与特征均不参与训练）。
    - NCM（最近类均值原型，免训练）：每类原型 = 训练集该类样本的 CLR 空间均值向量 μ_pos（患病）、
      μ_neg（健康），仅 2×264 个「参数」（类均值），无任何迭代训练。测试样本按距离归最近原型：
      欧氏距离 score = ||x−μ_neg|| − ||x−μ_pos||（越大越接近患病原型）；余弦距离 score =
      cos(x,μ_pos) − cos(x,μ_neg)（越大越相似于患病原型）。score 为连续量，直接算 AUC（阈值无关）。
      「阈值 0.5」等价于 softmax 概率 P(患病|x)=exp(−d_pos)/(exp(−d_pos)+exp(−d_neg)) 的 0.5
      决策边界，即 d_pos=d_neg（最近原型），与 score=0 同界；AUC 不依赖该阈值。
    - PLS-DA（低容量）：sklearn PLSRegression 对二分类标签 y∈{0,1} 回归，预测连续分数作判别
      分数。n_components 用内层 5 折分层 CV（仅训练集，候选 2~10）按 AUC 选最优，再全训练集
      重拟合预测测试集；另报固定 n_components=5 作敏感性对照。StandardScaler 仅训练集 fit
      （与 S3 正式实现同口径，防泄漏），PLSRegression(scale=False) 避免双重标准化。
    - 判读：若 NCM/PLS-DA 的 3 组合 AUC 均值与策略 A（0.5603）同量级（近随机 0.5），且显著低于
      回退 R3 加权（0.6068）/R4 DANN（0.5947），则说明「近随机天花板」与模型容量无关，是数据
      固有限制（疾病特异信号 + 标签语义漂移），而非模型欠拟合。

性能：
    任务级并行（ProcessPoolExecutor，max_workers=3）：3 个 LODO 组合彼此独立（无数据依赖），
    每 worker 内完成 NCM（欧氏/余弦）+ PLS-DA（内层 CV 选 n_components + 固定 5）全部计算。
    数据 484×264 小样本，NCM 为纯向量化均值/距离运算（毫秒级），PLS-DA 内层 CV 为 9 候选 ×
    5 折 = 45 次 PLS 拟合/组合（每次秒级），整体预计 <1 分钟。随机性隔离：内层 CV 用
    StratifiedKFold(random_state=42)，每组合独立，与串行基准结果一致（无跨任务共享可变状态）。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32) —
      X_filtered(484×264 过滤后物种级原始丰度), y(484 二分类标签), lodo_combos(C1/C2/C3 样本索引)

输出：
    - outputs/data/S3-e1-baselines.pkl — NCM(欧氏/余弦) + PLS-DA(内层CV/固定5) 各 3 组合 AUC
      + 均值 + 内层 CV 选出的 n_components（含 meta：方法/日期/口径）
    - 控制台对比表：NCM/PLS-DA 各 3 组合 AUC + 均值，与策略 A/R3/R4 基线对比

对应论文章节：
    §S3 跨疾病预测模型（补充实验 E1：免训练/低容量基线，归因检验，不入论文正文）
"""
from __future__ import annotations

import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S3-e1-baselines.pkl"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

SEED = 42
COMBOS = ["C1", "C2", "C3"]
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}
PLS_N_COMPONENTS_CANDIDATES = list(range(2, 11))  # 2~10
PLS_FIXED_N = 5

# 现有基线（S3-results.pkl 实际值，供对比，非本脚本产出）
BASELINE = {
    "A_direct": 0.5603121010295473,
    "R3_weighted": 0.6068496899732825,
    "R4_dann": 0.5947079507148955,
}


# ---------------------------------------------------------------------------
# 数据变换（与 S3-model.py 口径一致）
# ---------------------------------------------------------------------------
def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值（几何均值中心化）。"""
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# NCM（最近类均值原型，免训练）
# ---------------------------------------------------------------------------
def ncm_score(X_clr: np.ndarray, y: np.ndarray, train_idx, test_idx, metric: str) -> np.ndarray:
    """NCM 判别分数：score = 距健康原型 − 距患病原型（欧氏）或 相似患病 − 相似健康（余弦）。

    越大越倾向患病。仅用训练集类均值（2 个原型），无迭代训练。
    """
    Xtr = X_clr[train_idx]
    ytr = y[train_idx]
    Xte = X_clr[test_idx]

    mu_pos = Xtr[ytr == 1].mean(axis=0)
    mu_neg = Xtr[ytr == 0].mean(axis=0)

    if metric == "euclidean":
        d_pos = np.linalg.norm(Xte - mu_pos, axis=1)
        d_neg = np.linalg.norm(Xte - mu_neg, axis=1)
        return d_neg - d_pos  # 越大越接近患病原型
    if metric == "cosine":
        def cos_sim(a, b):
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-12
            return np.dot(a, b) / denom
        s_pos = cos_sim(Xte, mu_pos)
        s_neg = cos_sim(Xte, mu_neg)
        return s_pos - s_neg  # 越大越相似于患病原型
    raise ValueError(f"未知 metric: {metric}")


# ---------------------------------------------------------------------------
# PLS-DA（低容量）
# ---------------------------------------------------------------------------
def select_n_components(Xtr_s: np.ndarray, ytr: np.ndarray) -> tuple[int, float]:
    """内层 5 折分层 CV（仅训练集）按 AUC 选 n_components（候选 2~10）。返回 (best_n, best_cv_auc)。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_n, best_auc = None, -1.0
    for n in PLS_N_COMPONENTS_CANDIDATES:
        aucs = []
        for tr, te in skf.split(Xtr_s, ytr):
            pls = PLSRegression(n_components=n, scale=False)
            pls.fit(Xtr_s[tr], ytr[tr].reshape(-1, 1))
            score = pls.predict(Xtr_s[te]).ravel()
            aucs.append(roc_auc_score(ytr[te], score))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_n = n
    return best_n, best_auc


def plsda_score(Xtr_s, ytr, Xte_s, n_components):
    """PLS-DA：PLSRegression 对二分类标签回归，预测连续分数作判别分数。"""
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xtr_s, ytr.reshape(-1, 1))
    return pls.predict(Xte_s).ravel()


# ---------------------------------------------------------------------------
# 并行 worker（模块级，供 ProcessPoolExecutor pickle）
# ---------------------------------------------------------------------------
def _e1_worker(args):
    """worker：对单个组合完成 NCM(欧氏/余弦) + PLS-DA(内层CV/固定5) 全部计算。

    返回 (combo, result_dict)。result 含各方法 AUC + PLS-DA 选出的 n_components。
    """
    combo, X_clr, y, train_idx, test_idx = args
    yte = y[test_idx]
    result = {
        "combo": combo,
        "test_disease": COMBO_TO_DISEASE[combo],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_pos_frac": float(yte.mean()),
    }

    # NCM（免训练，CLR 空间直接算，无 StandardScaler）
    for metric in ["euclidean", "cosine"]:
        score = ncm_score(X_clr, y, train_idx, test_idx, metric)
        result[f"ncm_{metric}_auc"] = float(roc_auc_score(yte, score))

    # PLS-DA（低容量，StandardScaler 仅训练集 fit，与 S3 正式实现同口径）
    Xtr = X_clr[train_idx]
    ytr = y[train_idx]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(X_clr[test_idx])

    best_n, best_cv_auc = select_n_components(Xtr_s, ytr)
    score = plsda_score(Xtr_s, ytr, Xte_s, best_n)
    result["plsda_n_components"] = best_n
    result["plsda_cv_auc"] = best_cv_auc
    result["plsda_auc"] = float(roc_auc_score(yte, score))

    score5 = plsda_score(Xtr_s, ytr, Xte_s, PLS_FIXED_N)
    result["plsda_fixed5_auc"] = float(roc_auc_score(yte, score5))

    return combo, result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("S3 补充实验 E1：免训练/低容量基线（NCM + PLS-DA）跨疾病 LODO")
    print("=" * 78)

    # 1. 加载预处理缓存（复用 S3-preprocessed.pkl，与 S3 正式实现同口径）
    with open(DATA_PKL, "rb") as f:
        pre = pickle.load(f)
    X_filtered = pre["X_filtered"]  # 484×264 DataFrame（原始丰度）
    y = np.asarray(pre["y"], dtype=int)
    lodo_combos = pre["lodo_combos"]

    # 2. CLR 变换（逐样本，无泄漏）
    X_clr = clr_transform(X_filtered.to_numpy())  # 484×264
    print(f"特征维度：{X_clr.shape[1]}（过滤后物种级，CLR 后）")

    # 3. 并行执行 3 组合（NCM + PLS-DA）
    tasks = [
        (c, X_clr, y, lodo_combos[c]["train_idx"], lodo_combos[c]["test_idx"])
        for c in COMBOS
    ]
    results = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        for combo, m in ex.map(_e1_worker, tasks):
            results[combo] = m

    # 4. 汇总均值
    methods = ["ncm_euclidean_auc", "ncm_cosine_auc", "plsda_auc", "plsda_fixed5_auc"]
    means = {m: float(np.mean([results[c][m] for c in COMBOS])) for m in methods}

    # 5. 落盘 pkl
    meta = {
        "sub": "S3",
        "stage": "补充实验 E1（免训练/低容量基线）",
        "methods": "NCM(欧氏/余弦) + PLS-DA(内层CV选n_components 2~10 / 固定5)",
        "clr_delta": CLR_DELTA,
        "seed": SEED,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "outputs/data/S3-preprocessed.pkl (源自 c-data-cleaned.pkl float32)",
        "note": "NCM 免训练（仅类均值 2 原型，CLR 空间直接算，无 StandardScaler）；"
                "PLS-DA 低容量（StandardScaler 仅训练集 fit + PLSRegression(scale=False)，"
                "n_components 内层 5 折分层 CV 按 AUC 选 2~10，另报固定 5 对照）。"
                "LODO 测试疾病训练阶段完全不可见。",
        "field_semantics": {
            "ncm_euclidean_auc": "NCM 欧氏距离判别分数（距健康原型−距患病原型）的测试集 AUC",
            "ncm_cosine_auc": "NCM 余弦相似判别分数（相似患病−相似健康）的测试集 AUC",
            "plsda_auc": "PLS-DA（内层 CV 选 n_components）测试集 AUC",
            "plsda_cv_auc": "PLS-DA 内层 5 折 CV 的 AUC（仅训练集，选 n_components 用，非测试指标）",
            "plsda_fixed5_auc": "PLS-DA（固定 n_components=5）测试集 AUC（敏感性对照）",
            "plsda_n_components": "内层 CV 选出的最优 n_components（2~10）",
        },
    }
    payload = {
        "meta": meta,
        "results": results,
        "mean_auc": means,
        "baseline_reference": BASELINE,
    }
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n结果已落盘: {OUT_PKL}")

    # 6. 控制台对比表
    print("\n" + "-" * 78)
    print("NCM / PLS-DA 3 组合 LODO AUC")
    print("-" * 78)
    header = f"{'组合':<6}{'测试疾病':<10}" + "".join(f"{m:<18}" for m in methods)
    print(header)
    for c in COMBOS:
        row = f"{c:<6}{COMBO_TO_DISEASE[c]:<10}"
        for m in methods:
            row += f"{results[c][m]:<18.4f}"
        print(row)
    mean_row = f"{'均值':<6}{'':<10}"
    for m in methods:
        mean_row += f"{means[m]:<18.4f}"
    print(mean_row)
    print("-" * 78)
    print("PLS-DA 内层 CV 选出的 n_components：",
          {c: results[c]["plsda_n_components"] for c in COMBOS})

    # 7. 与现有基线对比
    print("\n" + "-" * 78)
    print("与现有基线对比（3 组合 AUC 均值）")
    print("-" * 78)
    print(f"  NCM 欧氏        = {means['ncm_euclidean_auc']:.4f}")
    print(f"  NCM 余弦        = {means['ncm_cosine_auc']:.4f}")
    print(f"  PLS-DA (内层CV) = {means['plsda_auc']:.4f}")
    print(f"  PLS-DA (固定5)  = {means['plsda_fixed5_auc']:.4f}")
    print(f"  --- 现有基线 ---")
    print(f"  A 直接迁移      = {BASELINE['A_direct']:.4f}")
    print(f"  R3 加权         = {BASELINE['R3_weighted']:.4f}")
    print(f"  R4 DANN         = {BASELINE['R4_dann']:.4f}")
    print("-" * 78)


if __name__ == "__main__":
    main()
