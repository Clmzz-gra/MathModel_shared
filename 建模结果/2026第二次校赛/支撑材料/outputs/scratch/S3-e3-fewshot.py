"""
目的：
    S3 补充实验 E3：LODO few-shot 敏感性分析——逐步给测试疾病少量带标签样本加入训练
    （k=0/5/20/50，分层抽样患病/健康，种子 0/1/2 各跑一次取均值），看跨疾病 AUC 是否随
    k 跳升，据此判断「LODO 零样本协议是否过严」（零样本是失败主因 vs 信号稀缺）。

原理：
    - LODO 协议（approach-S3-confirmed.md §4.1）：每次留一种疾病作测试集，其余两种疾病作
      训练集，共 3 组合（C1 测 CRC / C2 测 IBD / C3 测 Obesity）。零样本（k=0）时测试疾病
      在训练阶段完全不可见。
    - few-shot 干预：对每个组合，从测试疾病样本中**分层抽样** k 个（按患病/健康比例，即
      k_pos=round(k·n_pos/n_test)），把这 k 个带标签样本并入训练集，剩余 n_test−k 个测试
      疾病样本作评估集。k=0 即现状（零样本协议），k>0 模拟「部署时少量标注」。
    - 预处理（与 S3-model.py 口径一致，防泄漏）：近全零过滤（1331→264，已在
      S3-preprocessed.pkl 完成）→ CLR（零值乘法替换 δ=0.65×检出限=6.5e-6，逐样本 log 后
      减行均值，无跨样本参数）→ StandardScaler（均值/方差仅训练集估计）。
    - 模型：LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=2000,
      random_state=42)，与 S3 正式实现策略 A 完全同口径。
    - 主指标 AUC（阈值无关）；辅指标（ACC/灵敏度/特异度/F1）用训练集 Youden J 最优阈值
      τ*（max 灵敏度+特异度−1）迁移到评估集，禁止评估集重定阈值（防泄漏）。
    - 判读逻辑：若 AUC 随 k 大幅跳升（如 k=50 时均值显著 >0.60 或相对 k=0 提升 ≥0.10）→
      零样本协议过严是失败主因（实际部署需少量标注）；若 AUC 随 k 仅有限提升（仍近随机）→
      信号稀缺，LODO 零样本结论稳健。

性能：
    任务级并行（ProcessPoolExecutor，max_workers=min(8, cpu_count)）：组合×档×种子
    （3×4×3=36 任务）彼此独立（无数据依赖），并行执行；每 worker 内 Logistic 单核（lbfgs
    无嵌套并行）。数据 484×264 小样本，单次 Logistic 拟合毫秒级，整体预计 <5 秒（轻量，
    并行仅为遵循 C8 任务独立性，非性能瓶颈）。随机性隔离：每 worker 用 default_rng(seed)
    独立抽样（seed∈{0,1,2}），Logistic random_state=42 固定（lbfgs 确定性），k=0 无抽样
    随机性（3 种子结果一致）。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32) —
      X_filtered(484×264 过滤后物种级丰度，1331→264 近全零过滤已完成), y(484 二分类标签，
      1=患病/0=健康), dataset_name(484 数据集名), lodo_combos(C1/C2/C3 样本索引)

输出：
    - outputs/data/S3-e3-fewshot.pkl — few-shot 敏感性曲线数据（3 组合 × 4 档 × 3 种子 AUC
      + 辅指标 + 汇总均值/标准差）
    - outputs/figures/_explore/S3-e3-fewshot-curve.pdf — AUC vs k 敏感性曲线（3 组合 + 均值）
    - solution/model-notes/experiment-e3-fewshot-S3.md — 实验报告（由本脚本结果撰写）

对应论文章节：
    §S3 跨疾病预测模型（补充实验 E3，敏感性分析，探索图不入论文正文）
"""
from __future__ import annotations

import pickle
import warnings

# sklearn 1.9 弃用 penalty 参数（改用 l1_ratio），但规格明确要求 penalty='l2'（S1/S2/S3 口径），
# 保留 penalty='l2' 并抑制该 FutureWarning（非可操作项）
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S3-e3-fewshot.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

SEED = 42  # Logistic 固定随机种子（lbfgs 确定性，与 S3-model.py 一致）
SAMPLING_SEEDS = [0, 1, 2]  # 分层抽样随机种子
K_VALUES = [0, 5, 20, 50]  # few-shot 档位（0=现状零样本）
COMBOS = ["C1", "C2", "C3"]
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}


# ---------------------------------------------------------------------------
# 数据变换（与 S3-model.py 口径一致）
# ---------------------------------------------------------------------------
def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值（几何均值中心化）。

    无跨样本参数，不引入训练/测试泄漏。接受 ndarray，返回 ndarray。
    """
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """训练集 Youden J 最优阈值（max 灵敏度+特异度-1）。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """计算 AUC + 阈值迁移下的 ACC/灵敏度/特异度/F1。"""
    auc = float(roc_auc_score(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return dict(auc=auc, acc=acc, sensitivity=sens, specificity=spec, f1=f1)


def stratified_sample_k(test_idx: np.ndarray, y: np.ndarray, k: int, rng) -> np.ndarray:
    """从测试疾病样本中分层抽样 k 个（按患病/健康比例），返回抽样索引（0-based 行号）。

    比例分层：k_pos = round(k · n_pos / n_test)，k_neg = k − k_pos；边界裁剪保证
    k_pos ≤ n_pos、k_neg ≤ n_neg。k=0 返回空数组；k ≥ n_test 返回全部（本实验 k≤50，
    n_test≥110，不触发）。
    """
    test_y = y[test_idx]
    pos_idx = test_idx[test_y == 1]
    neg_idx = test_idx[test_y == 0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_test = n_pos + n_neg
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n_test:
        return test_idx.copy()
    k_pos = int(round(k * n_pos / n_test))
    k_pos = max(0, min(k_pos, n_pos))
    k_neg = k - k_pos
    if k_neg > n_neg:
        k_neg = n_neg
        k_pos = k - k_neg
    sampled_pos = rng.choice(pos_idx, size=k_pos, replace=False)
    sampled_neg = rng.choice(neg_idx, size=k_neg, replace=False)
    return np.concatenate([sampled_pos, sampled_neg])


# ---------------------------------------------------------------------------
# 并行 worker（模块级，供 ProcessPoolExecutor pickle）
# ---------------------------------------------------------------------------
def _fewshot_worker(args):
    """worker：对单个 (combo, k, seed) 做 few-shot 训练评估。

    训练 = 2 疾病全量 + k 个测试疾病样本（分层抽样带标签）；评估 = 剩余 n_test−k 个
    测试疾病样本。返回 (combo, k, seed, metrics_dict)。
    """
    combo, k, seed, X_clr, y, lodo_combos = args
    train_idx = lodo_combos[combo]["train_idx"]
    test_idx = lodo_combos[combo]["test_idx"]

    rng = np.random.default_rng(seed)
    sampled = stratified_sample_k(test_idx, y, k, rng)
    train = np.concatenate([train_idx, sampled])
    test = np.setdiff1d(test_idx, sampled)  # 剩余测试疾病样本

    Xtr = X_clr[train]
    Xte = X_clr[test]
    ytr = y[train]
    yte = y[test]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=SEED,
    )
    clf.fit(Xtr_s, ytr)

    train_score = clf.predict_proba(Xtr_s)[:, 1]
    thr = youden_threshold(ytr, train_score)
    test_score = clf.predict_proba(Xte_s)[:, 1]
    y_pred = (test_score >= thr).astype(int)

    m = compute_metrics(yte, y_pred, test_score)
    m["youden_threshold"] = thr
    m["n_train"] = int(len(ytr))
    m["n_test"] = int(len(yte))
    m["n_sampled_pos"] = int((y[sampled] == 1).sum())
    m["n_sampled_neg"] = int((y[sampled] == 0).sum())
    m["test_pos_frac"] = float(yte.mean())
    return combo, k, seed, m


# ---------------------------------------------------------------------------
# 探索图
# ---------------------------------------------------------------------------
def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = {"C1": "#4C72B0", "C2": "#DD8452", "C3": "#55A868"}
    labels = {"C1": "C1 测 CRC", "C2": "C2 测 IBD", "C3": "C3 测 Obesity"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for combo in COMBOS:
        ks = K_VALUES
        means = [summary[combo][k]["mean_auc"] for k in ks]
        stds = [summary[combo][k]["std_auc"] for k in ks]
        ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, lw=1.5,
                    color=colors[combo], label=labels[combo])
    # 3 组合均值线
    overall = [float(np.mean([summary[c][k]["mean_auc"] for c in COMBOS])) for k in K_VALUES]
    ax.plot(K_VALUES, overall, marker="s", lw=2, color="#C44E52", ls="--",
            label="3 组合均值")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="随机 0.5")
    ax.axhline(0.6, color="red", ls=":", lw=1, label="可用线 0.6")
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("few-shot 样本数 k（测试疾病带标签样本并入训练）")
    ax.set_ylabel("AUC（剩余测试疾病样本）")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("LODO few-shot 敏感性：AUC vs k（3 种子均值 ± 标准差）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S3-e3-fewshot-curve.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("S3 补充实验 E3：LODO few-shot 敏感性分析")
    print("=" * 72)

    # 1. 加载预处理缓存（X_filtered 已 1331→264 过滤，源自 c-data-cleaned.pkl）
    with open(DATA_PKL, "rb") as f:
        pre = pickle.load(f)
    X_filtered = pre["X_filtered"]
    y = np.asarray(pre["y"], dtype=int)
    lodo_combos = pre["lodo_combos"]

    # 2. CLR 变换（逐样本，无泄漏）
    X_clr = clr_transform(X_filtered.to_numpy())
    print(f"特征维度：{X_clr.shape[1]}（过滤后物种级）")

    # 3. 构造任务（组合 × 档 × 种子，全独立）
    tasks = [
        (combo, k, seed, X_clr, y, lodo_combos)
        for combo in COMBOS
        for k in K_VALUES
        for seed in SAMPLING_SEEDS
    ]
    print(f"任务数：{len(tasks)}（{len(COMBOS)} 组合 × {len(K_VALUES)} 档 × "
          f"{len(SAMPLING_SEEDS)} 种子）")

    # 4. 并行执行
    results = {}
    with ProcessPoolExecutor(max_workers=min(8, __import__("os").cpu_count())) as ex:
        for combo, k, seed, m in ex.map(_fewshot_worker, tasks):
            results[(combo, k, seed)] = m

    # 5. 汇总（combo × k → 均值/标准差/种子列表）
    summary = {}
    for combo in COMBOS:
        summary[combo] = {}
        for k in K_VALUES:
            aucs = [results[(combo, k, s)]["auc"] for s in SAMPLING_SEEDS]
            summary[combo][k] = {
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
                "seed_aucs": aucs,
            }

    # 6. 组装 pkl
    meta = {
        "sub": "S3",
        "stage": "E3-fewshot",
        "model": "LogisticRegression(L2, C=1.0, class_weight=balanced, max_iter=2000) + CLR + StandardScaler",
        "seed": SEED,
        "sampling_seeds": SAMPLING_SEEDS,
        "k_values": K_VALUES,
        "clr_delta": CLR_DELTA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "outputs/data/S3-preprocessed.pkl (源自 c-data-cleaned.pkl float32)",
        "note": "few-shot 敏感性：k 个测试疾病样本（分层抽样患病/健康）并入训练，"
                "剩余测试疾病样本评估 AUC；Youden 阈值仅训练集估计；k=0 即零样本协议基线",
        "field_semantics": {
            "results.<combo,k,seed>.auc": "剩余测试疾病样本 AUC（阈值无关主指标）",
            "results.<combo,k,seed>.sensitivity": "训练集 Youden J 阈值迁移到评估集的灵敏度",
            "results.<combo,k,seed>.n_sampled_pos/neg": "并入训练的 k 个样本中患病/健康数（分层抽样）",
            "summary.<combo>.<k>.mean_auc": "3 种子 AUC 均值",
            "summary.<combo>.<k>.std_auc": "3 种子 AUC 标准差（抽样随机性）",
        },
    }
    payload = {
        "meta": meta,
        "results": {f"{c}|{k}|{s}": v for (c, k, s), v in results.items()},
        "summary": summary,
    }

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n结果已落盘: {OUT_PKL}")

    # 7. 探索图
    make_figure(summary)
    print("探索图已保存到 outputs/figures/_explore/S3-e3-fewshot-curve.pdf")

    # 8. 关键数字摘要（stdout）
    print("\n" + "=" * 72)
    print("AUC vs k 汇总（3 种子均值）")
    print("=" * 72)
    header = "组合".ljust(6) + "".join(f"k={k}".rjust(10) for k in K_VALUES)
    print(header)
    for combo in COMBOS:
        row = combo.ljust(6) + "".join(
            f"{summary[combo][k]['mean_auc']:.4f}".rjust(10) for k in K_VALUES
        )
        print(row)
    overall = [float(np.mean([summary[c][k]["mean_auc"] for c in COMBOS])) for k in K_VALUES]
    print("均值".ljust(6) + "".join(f"{v:.4f}".rjust(10) for v in overall))
    # 提升量（相对 k=0）
    print("\n相对 k=0 提升量（3 组合均值）：")
    for k in K_VALUES[1:]:
        print(f"  k={k}: {overall[K_VALUES.index(k)] - overall[0]:+.4f}")


if __name__ == "__main__":
    main()
