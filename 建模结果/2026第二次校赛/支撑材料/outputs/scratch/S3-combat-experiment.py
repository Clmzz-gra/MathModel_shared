"""
目的：
    S3 扩展实验：ComBat 批次校正 + LODO 复现对比，检验「跨疾病失败 = 批次混杂」假设。
    对 CLR 后的 264 维特征矩阵按批次（dataset_name，3 批次）做 ComBat 校正（保留疾病信号），
    对比校正前 vs 校正后的策略 A（L2+CLR）3 组合 LODO AUC 与共享物种方向一致性，
    判定跨疾病失败主因是「批次伪影」还是「疾病特异信号」。

原理：
    - 数据准备：c-data-cleaned.pkl（484×1331）→ 近全零过滤（零值占比>95% 剔除，1331→264，
      三病并集）→ CLR（δ=6.5e-6 乘法替换 + 逐样本几何均值中心化）。本脚本直接复用
      S3-preprocessed.pkl 的 X_filtered（264 特征原始丰度，与 S3 正式实现同口径），
      避免重解析原始 Excel。
    - ComBat（pycombat 0.20，标准参数化 empirical Bayes）：Y_ijg = α_g + Xβ_g + γ_ig + δ_ig·ε_ijg。
      批次 = dataset_name（3 批次：Zeller/metahit/Chatelier，与疾病类型完全共线）；
      设计矩阵 X = 二分类疾病标签 y（「患病 vs 健康」，需保留的生物信号），
      故 ComBat 移除批次（数据集）效应、保留疾病（患病 vs 健康）信号。
      校正施加于 CLR 后的 264 维矩阵（全 484 样本一次校正，3 批次共同估计批次参数）。
    - LODO 复现（与 S3 正式实现同口径）：LogisticRegression(penalty='l2', C=1.0,
      class_weight='balanced', max_iter=2000)，StandardScaler 仅训练集 fit，Youden 阈值仅训练集
      估计；3 组合（C1 测 CRC / C2 测 IBD / C3 测 Obesity），测试疾病训练阶段完全不可见。
    - 共享方向检验：对 252 个共享物种，逐组合计算训练疾病与测试疾病的「患病 vs 健康」丰度方向
      （sign(mean_diseased - mean_healthy)），统计方向一致/翻转占比，配符号检验（二项检验 null=0.5）。
      校正前后各算一次，看 51.2% 是否变化——这是「批次混杂 vs 疾病特异」的判别关键。
    - 判定：若校正后 A 策略均值显著提升（>0.02）→ 之前学的是批次伪影；若不提升 → 疾病特异信号
      为真，归因更干净。

性能：
    任务级并行（ProcessPoolExecutor，max_workers=min(6, cpu)）：校正前/后各 3 组合 LODO 共 6 个
    任务彼此独立（无数据依赖），并行执行。数据 484×264 小样本，Logistic 毫秒级，整体预计 <1 分钟
    （ComBat 为 264 特征 × 3 批次的矩阵运算，秒级）。随机性隔离：Logistic random_state=42，
    每组合独立拟合，与串行基准结果一致（无跨任务共享可变状态）。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32) —
      X_filtered(484×264 过滤后物种级原始丰度), y(484 二分类标签), dataset_name(3 批次),
      feature_names(264), shared_features(252), lodo_combos(C1/C2/C3 样本索引)

输出：
    - outputs/data/S3-combat-corrected.pkl — ComBat 校正后特征矩阵（含 meta：方法/日期/口径）
    - 控制台对比表：策略 A 校正前 vs 校正后 3 组合 AUC + 均值；共享方向前后对比（一致数/比例/p）

对应论文章节：
    §S3 跨疾病预测模型（扩展实验：批次校正归因检验，探索性诊断，不入论文正文）
"""
from __future__ import annotations

import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pycombat
from scipy.stats import binomtest
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
OUT_PKL = ROOT / "outputs" / "data" / "S3-combat-corrected.pkl"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

SEED = 42
COMBOS = ["C1", "C2", "C3"]
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}


# ---------------------------------------------------------------------------
# 数据变换（与 S3-model.py 口径一致）
# ---------------------------------------------------------------------------
def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值（几何均值中心化）。"""
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


# ---------------------------------------------------------------------------
# 并行 worker（模块级，供 ProcessPoolExecutor pickle）
# ---------------------------------------------------------------------------
def _fit_eval_worker(args):
    """worker：对单个 (combo, X, y, train_idx, test_idx) 拟合评估（策略 A 口径）。

    X 为已 CLR（或 CLR+ComBat）的特征矩阵；StandardScaler 仅训练集 fit，Youden 阈值仅训练集估计。
    返回 (combo, result_dict)。
    """
    combo, X, y, train_idx, test_idx = args
    Xtr = X[train_idx]
    Xte = X[test_idx]
    ytr = y[train_idx]
    yte = y[test_idx]

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
    m["test_pos_frac"] = float(yte.mean())
    return combo, m


def run_lodo(X: np.ndarray, y: np.ndarray, lodo_combos: dict, max_workers: int = 6) -> dict:
    """策略 A 直接迁移：3 组合 LODO 并行执行，返回 {combo: result}。"""
    tasks = [
        (c, X, y, lodo_combos[c]["train_idx"], lodo_combos[c]["test_idx"])
        for c in COMBOS
    ]
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for combo, m in ex.map(_fit_eval_worker, tasks):
            results[combo] = m
    return results


# ---------------------------------------------------------------------------
# 共享方向一致性（与 S3-model.py migration_analysis 口径一致）
# ---------------------------------------------------------------------------
def migration_analysis(X, y, shared_features, feature_names, lodo_combos):
    """共享物种在训练/测试疾病的「患病 vs 健康」丰度方向一致性，配符号检验。"""
    shared_idx = [feature_names.index(s) for s in shared_features if s in feature_names]

    consistent_total = 0
    flipped_total = 0
    for combo in COMBOS:
        train_idx = lodo_combos[combo]["train_idx"]
        test_idx = lodo_combos[combo]["test_idx"]
        ytr = y[train_idx]
        yte = y[test_idx]
        for si in shared_idx:
            tr_pos = X[train_idx][ytr == 1, si].mean()
            tr_neg = X[train_idx][ytr == 0, si].mean()
            train_dir = np.sign(tr_pos - tr_neg)
            te_pos = X[test_idx][yte == 1, si].mean()
            te_neg = X[test_idx][yte == 0, si].mean()
            test_dir = np.sign(te_pos - te_neg)
            if train_dir == 0 or test_dir == 0:
                continue
            if train_dir == test_dir:
                consistent_total += 1
            else:
                flipped_total += 1

    n_valid = consistent_total + flipped_total
    p_value = float(binomtest(consistent_total, n_valid, 0.5).pvalue) if n_valid > 0 else float("nan")
    consistent_frac = consistent_total / n_valid if n_valid > 0 else float("nan")
    return dict(
        direction_consistent_count=consistent_total,
        direction_flipped_count=flipped_total,
        n_valid=n_valid,
        consistent_fraction=consistent_frac,
        sign_test_pvalue=p_value,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("S3 扩展实验：ComBat 批次校正 + LODO 复现对比（批次混杂 vs 疾病特异）")
    print("=" * 78)

    # 1. 加载预处理缓存（复用 S3-preprocessed.pkl，与 S3 正式实现同口径）
    with open(DATA_PKL, "rb") as f:
        pre = pickle.load(f)
    X_filtered = pre["X_filtered"]  # 484×264 DataFrame（原始丰度）
    y = np.asarray(pre["y"], dtype=int)
    dataset_name = np.asarray(pre["dataset_name"])
    feature_names = list(pre["feature_names"])
    lodo_combos = pre["lodo_combos"]
    shared_features = list(pre["shared_features"])

    # 2. CLR 变换（逐样本，无泄漏）
    X_clr = clr_transform(X_filtered.to_numpy())  # 484×264
    print(f"特征维度：{X_clr.shape[1]}（过滤后物种级）；共享物种 {len(shared_features)} 个")
    print(f"批次（dataset_name）：{np.unique(dataset_name).tolist()}")

    # 3. ComBat 批次校正（批次=dataset_name，保留疾病信号 y）
    print("\n[ComBat] pycombat 0.20 参数化 empirical Bayes，batch=dataset_name，X=疾病标签 y")
    batch = dataset_name
    X_design = y.reshape(-1, 1)  # 保留「患病 vs 健康」生物信号
    combat = pycombat.Combat(mode="p", conv=0.0001)
    X_combat = combat.fit_transform(X_clr, batch, X=X_design)
    print(f"  校正后矩阵形状：{X_combat.shape}（CLR + ComBat）")

    # 4. 落盘校正后特征矩阵（含 meta）
    meta = {
        "sub": "S3",
        "stage": "扩展实验（批次校正归因检验）",
        "method": "ComBat（pycombat 0.20，参数化 empirical Bayes，mode='p'）",
        "batch": "dataset_name（3 批次：Zeller/metahit/Chatelier，与疾病类型完全共线）",
        "design_matrix": "X = 二分类疾病标签 y（保留「患病 vs 健康」信号，移除批次效应）",
        "input": "CLR 后 264 维特征矩阵（δ=6.5e-6 乘法替换 + 几何均值中心化）",
        "clr_delta": CLR_DELTA,
        "seed": SEED,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "outputs/data/S3-preprocessed.pkl (源自 c-data-cleaned.pkl float32)",
        "note": "诊断性预处理（非可部署模型）：ComBat 用疾病标签 y 作保留协变量，"
                "且全 484 样本一次校正（3 批次共同估计批次参数），故含测试疾病标签/特征，"
                "仅用于归因检验，不用于 LODO 部署口径。",
        "field_semantics": {
            "X_combat": "ComBat 校正后 CLR 特征矩阵（484×264），批次效应已移除、疾病信号保留",
            "batch": "dataset_name 字符串数组（3 批次）",
        },
    }
    payload = {
        "meta": meta,
        "X_combat": X_combat,
        "X_clr": X_clr,
        "y": y,
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "shared_features": shared_features,
        "lodo_combos": lodo_combos,
    }
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  校正后特征矩阵已落盘: {OUT_PKL}")

    # 5. LODO 复现：校正前（复现 S3 基线）vs 校正后（ComBat）
    print("\n[LODO 复现] 策略 A（L2+CLR+StandardScaler），校正前 vs 校正后")
    before = run_lodo(X_clr, y, lodo_combos)
    after = run_lodo(X_combat, y, lodo_combos)

    before_mean = float(np.mean([before[c]["auc"] for c in COMBOS]))
    after_mean = float(np.mean([after[c]["auc"] for c in COMBOS]))
    delta_mean = after_mean - before_mean

    print("\n" + "-" * 78)
    print("策略 A 3 组合 LODO AUC 对比（校正前 vs 校正后）")
    print("-" * 78)
    print(f"{'组合':<6}{'测试疾病':<10}{'校正前 AUC':<14}{'校正后 AUC':<14}{'Δ':<10}")
    for c in COMBOS:
        b = before[c]["auc"]
        a = after[c]["auc"]
        print(f"{c:<6}{COMBO_TO_DISEASE[c]:<10}{b:<14.4f}{a:<14.4f}{a - b:<+10.4f}")
    print(f"{'均值':<6}{'':<10}{before_mean:<14.4f}{after_mean:<14.4f}{delta_mean:<+10.4f}")
    print("-" * 78)

    # 6. 共享方向一致性（校正前 vs 校正后）
    print("\n[共享方向检验] 共享物种「患病 vs 健康」方向一致性（校正前 vs 校正后）")
    mig_before = migration_analysis(X_clr, y, shared_features, feature_names, lodo_combos)
    mig_after = migration_analysis(X_combat, y, shared_features, feature_names, lodo_combos)

    print("-" * 78)
    print(f"{'':<12}{'一致数':<10}{'翻转数':<10}{'总数':<10}{'一致占比':<12}{'符号检验 p':<12}")
    for label, mig in [("校正前", mig_before), ("校正后", mig_after)]:
        print(f"{label:<12}{mig['direction_consistent_count']:<10}"
              f"{mig['direction_flipped_count']:<10}{mig['n_valid']:<10}"
              f"{mig['consistent_fraction']:<12.3f}{mig['sign_test_pvalue']:<12.4f}")
    print("-" * 78)

    # 7. 结论判定
    print("\n[结论判定]")
    if delta_mean > 0.02:
        verdict = (
            f"校正后 A 策略均值提升 {delta_mean:+.4f}（>0.02）→ 之前学的是批次伪影，"
            f"跨疾病失败主因是「批次混杂」。"
        )
    elif delta_mean < -0.02:
        verdict = (
            f"校正后 A 策略均值下降 {delta_mean:+.4f}（<-0.02）→ 批次校正反而损害性能，"
            f"说明原信号含疾病特异成分，批次效应非主因。"
        )
    else:
        verdict = (
            f"校正后 A 策略均值变化 {delta_mean:+.4f}（|Δ|≤0.02，不显著）→ 批次校正未改变"
            f"跨疾病性能，疾病特异信号为真，归因更干净（批次混杂假设被证伪）。"
        )
    print(f"  {verdict}")

    # 共享方向变化解读
    frac_delta = mig_after["consistent_fraction"] - mig_before["consistent_fraction"]
    print(f"  共享方向一致占比：校正前 {mig_before['consistent_fraction']:.3f} → "
          f"校正后 {mig_after['consistent_fraction']:.3f}（Δ={frac_delta:+.3f}）")
    if abs(frac_delta) < 0.05:
        print("  共享方向一致性基本不变 → 方向翻转（疾病特异信号）非批次伪影，"
              "进一步支持「疾病特异信号为真」。")
    else:
        print("  共享方向一致性明显变化 → 批次校正改变了方向结构，需结合 AUC 变化综合判定。")

    print("\n" + "=" * 78)
    print("关键数字摘要")
    print("=" * 78)
    print(f"  校正前 A 策略均值 AUC = {before_mean:.4f}")
    print(f"  校正后 A 策略均值 AUC = {after_mean:.4f}（Δ={delta_mean:+.4f}）")
    print(f"  共享方向一致占比：校正前 {mig_before['consistent_fraction']:.3f} "
          f"(p={mig_before['sign_test_pvalue']:.4f}) → "
          f"校正后 {mig_after['consistent_fraction']:.3f} "
          f"(p={mig_after['sign_test_pvalue']:.4f})")


if __name__ == "__main__":
    main()
