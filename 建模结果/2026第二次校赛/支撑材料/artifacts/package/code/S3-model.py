"""
目的：
    S3 跨疾病预测模型（leave-one-disease-out 泛化评估）2.1 正式实现：构建并对比四策略
    （A 直接迁移 / B 共享标志物 / C 分类学聚合 / D Platt 部署校正），选出最优交付模型；
    若四策略 3 组合 AUC 均值全部 <0.60 触发紧急回退（R1 树模型 → R2 样本合并 → R3 密度比
    重加权 → R4 对抗式域适应），达可用线（均值 ≥0.65 或相对 A 提升 ≥0.10）即交付；
    并产出衰减归因（三分法）、深度迁移分析、C3 阈值漂移量化，落盘 S3-results.pkl。

原理：
    - LODO 协议：每次留一种疾病作测试集，其余两种疾病作训练集，共 3 组合（C1 测 CRC /
      C2 测 IBD / C3 测 Obesity）。测试疾病在训练阶段完全不可见（标签与特征均不参与训练）。
    - 预处理（与 S1 口径一致，防泄漏）：近全零过滤（零值占比>95% 剔除，1331→264，三病并集）
      → CLR 变换（零值乘法替换 δ=0.65×检出限=6.5e-6，逐样本 log 后减行均值，无跨样本参数）
      → StandardScaler（均值/方差仅训练集估计，防泄漏）。
    - 模型：LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=2000)。
    - 阈值迁移：训练集 Youden J 最优阈值 τ*（max 灵敏度+特异度-1）只在训练集估计，原样搬到
      测试集；禁止测试集重定阈值（评估泄漏）。
    - 策略 B：过滤后 264 特征内三病按特征名交集（252 个，仅用特征存在性，绝不用测试标签）。
    - 策略 C：物种级按 g__（属）/ p__（门）层级聚合（同层丰度求和），CLR 后重训。
    - 策略 D：在 A/B/C 中 AUC 最优者上叠加 Platt 缩放 P=1/(1+exp(A·f+B))，A,B 仅训练集估计
      （Logistic 拟合，把训练分数 f 作唯一特征、标签作目标）；单调变换不改 AUC，只修复可部署
      指标。注：sklearn Logistic 拟合得 P=1/(1+exp(-(w·f+b)))，故 A=-w、B=-b；单调性校验
      w>0（等价 A<0，即分数越高概率越高），与 math-S3.tex 的「A>0」符号约定相反，见口径修正。
    - 回退 R1：RandomForest(n_estimators=500, random_state=42, class_weight='balanced')
      （XGBoost 未安装，仅 RF）；R2：样本合并 Logistic（≡策略 A 口径，显式一环）；R3：密度比
      重加权（域分类器估计 w=P(test|x)/P(train|x)，权重裁剪上界 10，转导式：用测试特征不用
      测试标签）；R4：对抗式域适应（DANN，梯度反转层，最后手段）。
    - 衰减归因（三分法）：批次效应（silhouette 0.070 近 0，不主导）/ 疾病特异信号（衰减量
      Δ=跨疾病 AUC−域内 AUC）/ 标签语义漂移（C3 训练阈值迁移灵敏度崩溃）。
    - 深度迁移分析：共享物种在训练疾病与测试疾病的「患病 vs 健康」丰度方向一致性，配符号检验。
    - C3 阈值漂移：训练/测试患病概率基线差 Δ_baseline + Youden 阈值在测试分数分布的位置。

性能：
    任务级并行（ProcessPoolExecutor，max_workers=min(8, cpu)）：策略 A/B/C（属/门）与回退
    R1/R2/R3 的「策略 × 组合」任务彼此独立（无数据依赖），并行执行；每 worker 内 RF 设 n_jobs=1
    避免嵌套并行。数据 484×264 小样本，Logistic 毫秒级、RF 秒级，整体预计 <2 分钟（R4 DANN
    为最后手段，仅 R1-R3 全败时按需运行，PyTorch CPU 约 1-3 分钟）。随机性隔离：Logistic/RF
    均 random_state=42，每组合独立拟合，与串行基准结果一致（无跨任务共享可变状态）。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32，非 B-raw.pkl) —
      X_filtered(484×264 过滤后物种级丰度), y(484 二分类标签), dataset_name, disease,
      feature_names(264), feature_taxonomy, lodo_combos(C1/C2/C3 样本索引),
      shared_features(252), genus_features(106), phylum_features(11)

输出：
    - outputs/data/S3-results.pkl — 四策略对比 + 回退记录 + 衰减归因 + 迁移分析 + 阈值漂移
    - outputs/figures/_explore/S3-strategy-compare-auc.pdf — 四策略 × 3 组合 AUC 分组柱状图
    - outputs/figures/_explore/S3-decay-attribution.pdf — 域内 vs 跨疾病 AUC + 衰减量
    - outputs/figures/_explore/S3-threshold-drift.pdf — C3 训练/测试分数分布 + Youden 阈值

对应论文章节：
    §S3 跨疾病预测模型（2.1 正式实现，探索图不入论文正文）
"""
from __future__ import annotations

import json
import pickle
import warnings

# sklearn 1.9 弃用 penalty 参数（改用 l1_ratio），但规格明确要求 penalty='l2'（S1/S2 口径），
# 保留 penalty='l2' 并抑制该 FutureWarning（非可操作项）
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S3-results.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

SEED = 42
COMBOS = ["C1", "C2", "C3"]
# 组合 → 测试疾病（C1 测 CRC / C2 测 IBD / C3 测 Obesity）
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}
# 域内 AUC 参考（A3 在 1331 全集上测得，handoff §2.8；正式实现另在 264 过滤集重算）
DOMAIN_AUC_REF_A3 = {"CRC": 0.814, "IBD": 0.885, "Obesity": 0.644}


# ---------------------------------------------------------------------------
# 数据变换（与 preprocess-S3.py 口径一致）
# ---------------------------------------------------------------------------
def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值（几何均值中心化）。

    无跨样本参数，不引入训练/测试泄漏。接受 ndarray，返回 ndarray。
    """
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def taxonomy_aggregate(X: pd.DataFrame, level: str) -> pd.DataFrame:
    """按分类学层级聚合特征（同层丰度求和）。level ∈ {'species','genus','phylum'}。

    genus=按 g__ 段；phylum=按 p__ 段。返回聚合后 DataFrame。
    """
    if level == "species":
        return X.copy()
    seg = "g__" if level == "genus" else "p__"

    def key(col):
        parts = col.split("|")
        for p in parts:
            if p.startswith(seg):
                return p
        return col  # 兜底：无该段则用原名

    # pandas 3.0 移除 groupby(axis=1)，转置后按行分组再转回
    return X.T.groupby(key).sum().T


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
def _make_model(model_type: str):
    if model_type == "logistic":
        return LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=SEED,
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=500, random_state=SEED, class_weight="balanced", n_jobs=1,
        )
    raise ValueError(f"未知 model_type: {model_type}")


def _fit_eval_worker(args):
    """worker：对单个 (X_clr, y, train_idx, test_idx, model_type, sample_weight) 拟合评估。

    返回 (combo, result_dict)。result 含 auc/acc/sensitivity/specificity/f1/youden_threshold
    及 train_score/test_score（供 Platt 校准与阈值漂移分析）。
    """
    combo, X_clr, y, train_idx, test_idx, model_type, sample_weight = args
    Xtr = X_clr[train_idx]
    Xte = X_clr[test_idx]
    ytr = y[train_idx]
    yte = y[test_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = _make_model(model_type)
    if sample_weight is not None:
        clf.fit(Xtr_s, ytr, sample_weight=sample_weight[train_idx])
    else:
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
    # 供后续分析（Platt 校准 / 阈值漂移）
    m["_train_score"] = train_score
    m["_test_score"] = test_score
    m["_ytr"] = ytr
    m["_yte"] = yte
    return combo, m


def _run_parallel(tasks, max_workers=8):
    """并行执行任务列表，返回 {combo: result}。"""
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for combo, m in ex.map(_fit_eval_worker, tasks):
            results[combo] = m
    return results


# ---------------------------------------------------------------------------
# 策略 A/B/C（Logistic 族）
# ---------------------------------------------------------------------------
def _logistic_tasks(X_clr, y, lodo_combos, model_type="logistic", sample_weight=None):
    return [
        (c, X_clr, y, lodo_combos[c]["train_idx"], lodo_combos[c]["test_idx"],
         model_type, sample_weight)
        for c in COMBOS
    ]


def run_strategy_A(X_clr, y, lodo_combos):
    """策略 A：直接迁移（物种级 264 特征 Logistic L2 + CLR）。"""
    tasks = _logistic_tasks(X_clr, y, lodo_combos)
    return _run_parallel(tasks)


def run_strategy_B(X_clr_shared, y, lodo_combos, n_shared):
    """策略 B：共享标志物（252 特征交集 Logistic L2 + CLR）。"""
    tasks = _logistic_tasks(X_clr_shared, y, lodo_combos)
    res = _run_parallel(tasks)
    res["shared_feature_count"] = n_shared
    return res


def run_strategy_C(X_clr_genus, X_clr_phylum, y, lodo_combos):
    """策略 C：分类学聚合（属级 106 / 门级 11 + CLR）。"""
    genus = _run_parallel(_logistic_tasks(X_clr_genus, y, lodo_combos))
    phylum = _run_parallel(_logistic_tasks(X_clr_phylum, y, lodo_combos))
    return {"genus": genus, "phylum": phylum}


# ---------------------------------------------------------------------------
# 策略 D：Platt 校准（在 A/B/C 最优者上叠加）
# ---------------------------------------------------------------------------
def platt_calibrate(train_score, ytr, test_score):
    """Platt 缩放：训练集拟合 A,B（Logistic 回归，分数作唯一特征），校准测试分数。

    返回 (cal_test_score, A, B, w)。sklearn Logistic 得 P=1/(1+exp(-(w·f+b)))，
    故按 math-S3.tex 形式 P=1/(1+exp(A·f+B)) 有 A=-w、B=-b。单调性校验 w>0。
    """
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, random_state=SEED)
    clf.fit(train_score.reshape(-1, 1), ytr)
    w = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    A = -w  # math-S3.tex 形式
    B = -b
    cal_test = clf.predict_proba(test_score.reshape(-1, 1))[:, 1]
    return cal_test, A, B, w


def run_strategy_D(base_result, base_name):
    """策略 D：在 base_result（A/B/C 最优者）上叠加 Platt 校准，重算可部署指标。

    AUC 不变（单调变换），只修复 ACC/灵敏度/特异度/F1。返回 {C1,C2,C3, base_strategy, mean_auc}。
    """
    out = {"base_strategy": base_name}
    for combo in COMBOS:
        m = base_result[combo]
        cal_test, A, B, w = platt_calibrate(m["_train_score"], m["_ytr"], m["_test_score"])
        # 单调性校验：w>0（分数越高概率越高）；w<=0 触发警告（排序反转异常）
        if w <= 0:
            print(f"  [警告] 策略 D {combo} Platt 系数 w={w:.4f}<=0，排序反转异常，检查训练过程")
        thr = youden_threshold(m["_ytr"], m["_train_score"])  # 训练集 Youden 阈值（原始分数）
        # 口径 1（规格）：把训练 Youden 分数阈值 τ* 映射到校准概率阈值 τ_prob，再判测试
        #   τ_prob = sigmoid(A·τ*+B)；因 Platt 单调，决策边界与原始 τ* 完全一致 → 可部署指标不变
        tau_prob = 1.0 / (1.0 + np.exp(A * thr + B))
        y_pred_tau = (cal_test >= tau_prob).astype(int)
        m_cal = compute_metrics(m["_yte"], y_pred_tau, cal_test)
        # 口径 2（对照）：校准后直接用自然概率阈值 0.5（非 Youden 迁移），观察是否修复可部署指标
        y_pred_05 = (cal_test >= 0.5).astype(int)
        m_05 = compute_metrics(m["_yte"], y_pred_05, cal_test)
        out[combo] = {
            "auc": m["auc"],  # AUC 不变
            "cal_acc": m_cal["acc"],
            "cal_sensitivity": m_cal["sensitivity"],
            "cal_specificity": m_cal["specificity"],
            "cal_f1": m_cal["f1"],
            "A": A,
            "B": B,
            "platt_w": w,
            "tau_prob": float(tau_prob),
            # 对照口径（0.5 概率阈值，非 Youden 迁移）
            "thr05_acc": m_05["acc"],
            "thr05_sensitivity": m_05["sensitivity"],
            "thr05_specificity": m_05["specificity"],
            "thr05_f1": m_05["f1"],
        }
    out["mean_auc"] = float(np.mean([base_result[c]["auc"] for c in COMBOS]))
    return out


# ---------------------------------------------------------------------------
# 回退 R1-R3
# ---------------------------------------------------------------------------
def run_R1_rf(X_clr, y, lodo_combos):
    """R1：树模型（RandomForest 500 树，物种级过滤后特征）。"""
    tasks = _logistic_tasks(X_clr, y, lodo_combos, model_type="rf")
    return _run_parallel(tasks)


def run_R2_pooled(X_clr, y, lodo_combos):
    """R2：样本合并通用模型（Logistic L2，≡策略 A 口径，显式一环）。"""
    tasks = _logistic_tasks(X_clr, y, lodo_combos, model_type="logistic")
    return _run_parallel(tasks)


def _density_ratio_weights(X_clr, train_idx, test_idx, clip=10.0):
    """密度比估计（域分类器法）：w(x)=P(test|x)/P(train|x)，转导式（用测试特征不用测试标签）。

    训练一个 Logistic 区分 train/test 样本，w = exp(logit) × (n_train/n_test)，裁剪上界 clip。
    """
    n_train = len(train_idx)
    n_test = len(test_idx)
    idx_all = np.concatenate([train_idx, test_idx])
    X_all = X_clr[idx_all]
    domain = np.concatenate([np.zeros(n_train), np.ones(n_test)])  # 0=train, 1=test
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)
    clf.fit(X_all, domain)
    logit = clf.decision_function(X_all)
    # w = P(test|x)/P(train|x) = exp(logit)（logit = log P(test|x)/P(train|x)）
    w = np.exp(logit) * (n_train / n_test)
    w = np.clip(w, 0.0, clip)
    # 权重按样本索引对齐（train 部分）
    w_train = w[:n_train]
    return w_train


def run_R3_weighted(X_clr, y, lodo_combos):
    """R3：密度比重加权（importance weighting），给训练样本加权匹配测试分布。"""
    results = {}
    for combo in COMBOS:
        train_idx = lodo_combos[combo]["train_idx"]
        test_idx = lodo_combos[combo]["test_idx"]
        w_train = _density_ratio_weights(X_clr, train_idx, test_idx)
        # 构造全样本权重（仅训练部分非零，测试部分不用）
        sample_weight = np.zeros(len(y))
        sample_weight[train_idx] = w_train
        # 复用 worker（sample_weight 传入，仅训练样本加权）
        args = (combo, X_clr, y, train_idx, test_idx, "logistic", sample_weight)
        _, m = _fit_eval_worker(args)
        results[combo] = m
    return results


# ---------------------------------------------------------------------------
# 回退 R4：对抗式域适应（DANN，最后手段）
# ---------------------------------------------------------------------------
def run_R4_dann(X_clr, y, lodo_combos, epochs=200, lr=1e-3, hidden=64):
    """R4：对抗式域适应（DANN 思想，梯度反转层）。PyTorch CPU，最后手段。

    特征提取器（线性→ReLU）+ 标签分类器 + 域判别器（梯度反转），学域不变特征。
    严格防泄漏：域判别器只用 train/test 特征（不用测试标签），标签分类器只用训练标签。
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    class DANN(nn.Module):
        def __init__(self, d_in, hidden):
            super().__init__()
            self.feature = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
            self.label_clf = nn.Linear(hidden, 1)
            self.domain_clf = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x, alpha):
            f = self.feature(x)
            y_logit = self.label_clf(f)
            # 梯度反转：域判别器反向传播时乘 -alpha
            rev_f = _GradReverse.apply(f, alpha)
            d_logit = self.domain_clf(rev_f)
            return y_logit, d_logit

    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.alpha, None

    results = {}
    for combo in COMBOS:
        train_idx = lodo_combos[combo]["train_idx"]
        test_idx = lodo_combos[combo]["test_idx"]
        Xtr = X_clr[train_idx]
        Xte = X_clr[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        d_in = Xtr_s.shape[1]
        model = DANN(d_in, hidden)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()

        Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        Xte_t = torch.tensor(Xte_s, dtype=torch.float32)
        n_train = Xtr_t.shape[0]
        n_test = Xte_t.shape[0]

        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            # 域标签：train=0, test=1
            y_logit_tr, d_logit_tr = model(Xtr_t, alpha=1.0)
            y_logit_te, d_logit_te = model(Xte_t, alpha=1.0)
            loss_label = bce(y_logit_tr, ytr_t)
            loss_domain = bce(
                torch.cat([d_logit_tr, d_logit_te]),
                torch.cat([torch.zeros(n_train, 1), torch.ones(n_test, 1)]),
            )
            loss = loss_label + loss_domain
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_score = torch.sigmoid(model(Xtr_t, alpha=0.0)[0]).numpy().ravel()
            test_score = torch.sigmoid(model(Xte_t, alpha=0.0)[0]).numpy().ravel()

        thr = youden_threshold(ytr, train_score)
        y_pred = (test_score >= thr).astype(int)
        m = compute_metrics(yte, y_pred, test_score)
        m["youden_threshold"] = thr
        m["n_train"] = n_train
        m["n_test"] = n_test
        m["test_pos_frac"] = float(yte.mean())
        m["_train_score"] = train_score
        m["_test_score"] = test_score
        m["_ytr"] = ytr
        m["_yte"] = yte
        results[combo] = m
    return results


# ---------------------------------------------------------------------------
# 分析：衰减归因 / 深度迁移 / 阈值漂移
# ---------------------------------------------------------------------------
def domain_auc_264(X_clr, y, dataset_name, n_splits=5):
    """在过滤后 264 特征上重算各疾病域内 AUC（5 折分层 CV，与 A3 同协议）。"""
    ds_map = {"Zeller_fecal_colorectal_cancer": "CRC", "metahit": "IBD",
              "Chatelier_gut_obesity": "Obesity"}
    out = {}
    for ds, disease in ds_map.items():
        mask = dataset_name == ds
        X = X_clr[mask]
        yy = y[mask]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        aucs = []
        for tr, te in skf.split(X, yy):
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                                     class_weight="balanced", random_state=SEED)
            clf.fit(scaler.transform(X[tr]), yy[tr])
            score = clf.predict_proba(scaler.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(yy[te], score))
        out[disease] = float(np.mean(aucs))
    return out


def decay_attribution(domain_auc, cross_auc, strategy_A):
    """三分法衰减归因：对每疾病给 domain_auc/cross_auc/decay/dominant_cause。"""
    out = {}
    for disease in ["CRC", "IBD", "Obesity"]:
        da = domain_auc[disease]
        ca = cross_auc[disease]
        decay = ca - da
        # 主导归因判定：
        #   - 标签语义漂移：该疾病作测试时训练阈值迁移灵敏度 <0.10（决策边界失准）
        #   - 疾病特异信号：衰减量大（|decay| 大）
        #   - 批次效应：silhouette 0.070 近 0，不主导（全局排除）
        combo = {v: k for k, v in COMBO_TO_DISEASE.items()}[disease]
        sens = strategy_A[combo]["sensitivity"]
        if sens < 0.10:
            cause = "标签语义漂移"
        elif abs(decay) >= 0.20:
            cause = "疾病特异信号"
        else:
            cause = "疾病特异信号（弱）"
        out[disease] = dict(domain_auc=da, cross_auc=ca, decay=decay, dominant_cause=cause)
    return out


def migration_analysis(X_clr, y, dataset_name, shared_features, feature_names, lodo_combos):
    """深度迁移分析：共享物种在训练/测试疾病的「患病 vs 健康」丰度方向一致性。

    对每个组合（测试疾病 d），对每个共享物种 s：
      train_dir = sign(mean_diseased - mean_healthy) 在训练疾病样本上
      test_dir  = sign(mean_diseased - mean_healthy) 在测试疾病样本上
    方向一致（可迁移）vs 方向翻转（疾病特异），配符号检验（二项检验，null=0.5）。
    """
    ds_map = {"Zeller_fecal_colorectal_cancer": "CRC", "metahit": "IBD",
              "Chatelier_gut_obesity": "Obesity"}
    # 共享特征在 X_clr 中的列索引
    shared_idx = [feature_names.index(s) for s in shared_features if s in feature_names]

    consistent_total = 0
    flipped_total = 0
    species_dir = {}  # species -> {combo: 'consistent'|'flipped'|'zero'}

    for combo in COMBOS:
        train_idx = lodo_combos[combo]["train_idx"]
        test_idx = lodo_combos[combo]["test_idx"]
        ytr = y[train_idx]
        yte = y[test_idx]
        for si in shared_idx:
            s = feature_names[si]
            # 训练疾病方向
            tr_pos = X_clr[train_idx][ytr == 1, si].mean()
            tr_neg = X_clr[train_idx][ytr == 0, si].mean()
            train_dir = np.sign(tr_pos - tr_neg)
            # 测试疾病方向
            te_pos = X_clr[test_idx][yte == 1, si].mean()
            te_neg = X_clr[test_idx][yte == 0, si].mean()
            test_dir = np.sign(te_pos - te_neg)
            if train_dir == 0 or test_dir == 0:
                status = "zero"
            elif train_dir == test_dir:
                status = "consistent"
                consistent_total += 1
            else:
                status = "flipped"
                flipped_total += 1
            species_dir.setdefault(s, {})[combo] = status

    n_valid = consistent_total + flipped_total
    # 符号检验（二项检验，null：方向一致概率 0.5）
    from scipy.stats import binomtest
    p_value = float(binomtest(consistent_total, n_valid, 0.5).pvalue) if n_valid > 0 else float("nan")
    consistent_frac = consistent_total / n_valid if n_valid > 0 else float("nan")

    return dict(
        direction_consistent_count=consistent_total,
        direction_flipped_count=flipped_total,
        n_valid=n_valid,
        consistent_fraction=consistent_frac,
        sign_test_pvalue=p_value,
        shared_species_list=shared_features,
        species_direction=species_dir,
    )


def threshold_drift(strategy_A, lodo_combos, y):
    """C3 阈值漂移量化：训练/测试患病概率基线差 + Youden 阈值在测试分数分布的位置。"""
    combo = "C3"
    m = strategy_A[combo]
    train_idx = lodo_combos[combo]["train_idx"]
    test_idx = lodo_combos[combo]["test_idx"]
    ytr = y[train_idx]
    yte = y[test_idx]
    train_baseline = float(ytr.mean())
    test_baseline = float(yte.mean())
    delta_baseline = test_baseline - train_baseline
    thr = m["youden_threshold"]
    test_score = m["_test_score"]
    # Youden 阈值在测试分数分布中的位置（低于阈值的测试样本占比 = 判健康占比）
    boundary_position = float((test_score < thr).mean())
    # 诊断
    diagnosis = (
        f"训练患病基线 {train_baseline:.3f} vs 测试 {test_baseline:.3f}，"
        f"基线差 Δ={delta_baseline:+.3f}；Youden 阈值 τ*={thr:.4f} 落在测试分数分布的 "
        f"{boundary_position:.1%} 分位（{boundary_position:.1%} 测试样本被判健康），"
        f"灵敏度 {m['sensitivity']:.3f} 崩溃 → 标签语义漂移（决策边界跨疾病失准）。"
    )
    return dict(
        train_baseline=train_baseline,
        test_baseline=test_baseline,
        delta_baseline=delta_baseline,
        youden_threshold=thr,
        boundary_position=boundary_position,
        sensitivity=m["sensitivity"],
        diagnosis=diagnosis,
    )


# ---------------------------------------------------------------------------
# 探索图
# ---------------------------------------------------------------------------
def make_figures(strategy_compare, fallback, decay, drift, A_raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # CJK 字体（Windows 中文字体，防探索图中文乱码）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 图 1：四策略 × 3 组合 AUC 分组柱状图
    strategies = ["A_direct", "B_shared", "C_genus", "C_phylum"]
    labels = ["A 直接迁移", "B 共享标志物", "C 属级聚合", "C 门级聚合"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(COMBOS))
    width = 0.18
    for k, (skey, lab, col) in enumerate(zip(strategies, labels, colors)):
        vals = [strategy_compare[skey][c]["auc"] for c in COMBOS]
        ax.bar(x + (k - 1.5) * width, vals, width, label=lab, color=col)
        for xi, v in zip(x + (k - 1.5) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="随机 0.5")
    ax.axhline(0.6, color="red", ls=":", lw=1, label="回退触发线 0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(["C1\nCRC", "C2\nIBD", "C3\nObesity"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("四策略跨疾病 AUC（leave-one-disease-out）")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S3-strategy-compare-auc.pdf")
    plt.close(fig)

    # 图 2：域内 vs 跨疾病 AUC + 衰减量
    diseases = ["CRC", "IBD", "Obesity"]
    dom = [decay[d]["domain_auc"] for d in diseases]
    cross = [decay[d]["cross_auc"] for d in diseases]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(diseases))
    ax.bar(x, dom, 0.35, label="域内 AUC（264 特征 5 折 CV）", color="#4C72B0")
    ax.bar(x + 0.35, cross, 0.35, label="跨疾病 AUC（策略 A）", color="#DD8452")
    for xi, v in zip(x, dom):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.35, cross):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(diseases)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("各疾病域内 vs 跨疾病 AUC（衰减归因）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S3-decay-attribution.pdf")
    plt.close(fig)

    # 图 3：C3 阈值漂移（训练/测试分数分布 + Youden 阈值）
    m = A_raw["C3"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(m["_train_score"], bins=40, alpha=0.6, label="训练分数", color="#4C72B0", density=True)
    ax.hist(m["_test_score"], bins=40, alpha=0.6, label="测试分数", color="#DD8452", density=True)
    ax.axvline(m["youden_threshold"], color="red", ls="--", lw=1.5,
               label=f"Youden 阈值 τ*={m['youden_threshold']:.3f}")
    ax.set_xlabel("模型分数（logit）")
    ax.set_ylabel("密度")
    ax.set_title("C3 阈值漂移：训练/测试分数分布 + Youden 阈值")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S3-threshold-drift.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("S3 2.1 正式模型实现：跨疾病预测（leave-one-disease-out）")
    print("=" * 72)

    # 1. 加载预处理缓存（源自 c-data-cleaned.pkl，非 B-raw.pkl）
    with open(DATA_PKL, "rb") as f:
        pre = pickle.load(f)
    X_filtered = pre["X_filtered"]  # 484×264 DataFrame
    y = np.asarray(pre["y"], dtype=int)
    dataset_name = pre["dataset_name"]
    feature_names = list(pre["feature_names"])
    lodo_combos = pre["lodo_combos"]
    shared_features = list(pre["shared_features"])
    genus_features = list(pre["genus_features"])
    phylum_features = list(pre["phylum_features"])

    # 2. 预计算各策略 CLR 特征矩阵（逐样本，无泄漏）
    X_clr = clr_transform(X_filtered.to_numpy())  # 484×264
    X_clr_shared = clr_transform(X_filtered[shared_features].to_numpy())  # 484×252
    X_genus = taxonomy_aggregate(X_filtered, "genus")
    X_phylum = taxonomy_aggregate(X_filtered, "phylum")
    X_clr_genus = clr_transform(X_genus.to_numpy())  # 484×106
    X_clr_phylum = clr_transform(X_phylum.to_numpy())  # 484×11
    print(f"特征维度：物种 {X_clr.shape[1]} / 共享 {X_clr_shared.shape[1]} / "
          f"属 {X_clr_genus.shape[1]} / 门 {X_clr_phylum.shape[1]}")

    # 3. 策略 A/B/C
    print("\n[策略 A] 直接迁移（物种级 264）")
    A = run_strategy_A(X_clr, y, lodo_combos)
    print("[策略 B] 共享标志物（252）")
    B = run_strategy_B(X_clr_shared, y, lodo_combos, len(shared_features))
    print("[策略 C] 分类学聚合（属/门）")
    C = run_strategy_C(X_clr_genus, X_clr_phylum, y, lodo_combos)

    # 4. 汇总策略对比（去掉内部 _ 字段）
    def _clean(res):
        return {c: {k: v for k, v in res[c].items() if not k.startswith("_")}
                for c in COMBOS}

    strategy_compare = {
        "A_direct": _clean(A),
        "B_shared": _clean(B),
        "C_genus": _clean(C["genus"]),
        "C_phylum": _clean(C["phylum"]),
    }
    # 补回 _clean 丢弃的元字段（shared_feature_count / 层级维度）
    strategy_compare["B_shared"]["shared_feature_count"] = len(shared_features)
    strategy_compare["C_genus"]["level"] = "genus"
    strategy_compare["C_genus"]["n_features"] = X_clr_genus.shape[1]
    strategy_compare["C_phylum"]["level"] = "phylum"
    strategy_compare["C_phylum"]["n_features"] = X_clr_phylum.shape[1]
    mean_aucs = {
        "A_direct": float(np.mean([A[c]["auc"] for c in COMBOS])),
        "B_shared": float(np.mean([B[c]["auc"] for c in COMBOS])),
        "C_genus": float(np.mean([C["genus"][c]["auc"] for c in COMBOS])),
        "C_phylum": float(np.mean([C["phylum"][c]["auc"] for c in COMBOS])),
    }
    for k, v in mean_aucs.items():
        strategy_compare[k]["mean_auc"] = v
        print(f"  {k}: 3 组合 AUC 均值 = {v:.4f}")

    # 5. 策略 D：在 A/B/C 中 AUC 最优者上叠加 Platt 校准
    best_base = max(mean_aucs, key=mean_aucs.get)
    best_base_result = {"A_direct": A, "B_shared": B, "C_genus": C["genus"],
                        "C_phylum": C["phylum"]}[best_base]
    print(f"\n[策略 D] Platt 校准（base={best_base}）")
    D = run_strategy_D(best_base_result, best_base)
    strategy_compare["D_calibrated"] = D

    # 6. 回退触发判定
    four_means = [mean_aucs["A_direct"], mean_aucs["B_shared"],
                  mean_aucs["C_genus"], mean_aucs["C_phylum"]]
    triggered = all(m < 0.60 for m in four_means)
    print(f"\n回退触发判定：四策略均值 {[f'{m:.4f}' for m in four_means]}，"
          f"全部 <0.60 → {triggered}")

    fallback = {"triggered": bool(triggered)}
    if triggered:
        print("\n[回退 R1] 树模型（RandomForest 500 树）")
        R1 = run_R1_rf(X_clr, y, lodo_combos)
        print("[回退 R2] 样本合并（Logistic，≡策略 A 口径）")
        R2 = run_R2_pooled(X_clr, y, lodo_combos)
        print("[回退 R3] 密度比重加权（importance weighting）")
        R3 = run_R3_weighted(X_clr, y, lodo_combos)

        def _mean(res):
            return float(np.mean([res[c]["auc"] for c in COMBOS]))

        r1_mean, r2_mean, r3_mean = _mean(R1), _mean(R2), _mean(R3)
        fallback["R1_tree"] = {"mean_auc": r1_mean, **_clean(R1)}
        fallback["R2_pooled"] = {"mean_auc": r2_mean, **_clean(R2)}
        fallback["R3_weighted"] = {"mean_auc": r3_mean, **_clean(R3)}
        print(f"  R1 RF 均值={r1_mean:.4f}  R2 合并均值={r2_mean:.4f}  "
              f"R3 加权均值={r3_mean:.4f}")

        # 可用线判定：均值 ≥0.65 或相对 A 提升 ≥0.10
        a_mean = mean_aucs["A_direct"]
        candidates = {"R1_tree": r1_mean, "R2_pooled": r2_mean, "R3_weighted": r3_mean}
        usable = None
        for name, m in candidates.items():
            if m >= 0.65 or (m - a_mean) >= 0.10:
                usable = name
                break

        if usable is not None:
            fallback["usable"] = True
            fallback["delivered_strategy"] = usable
            print(f"  回退达可用线：{usable}（均值 {candidates[usable]:.4f}）")
        else:
            # R4 对抗式域适应（最后手段）
            print("[回退 R4] 对抗式域适应（DANN，最后手段）")
            try:
                R4 = run_R4_dann(X_clr, y, lodo_combos)
                r4_mean = _mean(R4)
                fallback["R4_dann"] = {"mean_auc": r4_mean, **_clean(R4)}
                print(f"  R4 DANN 均值={r4_mean:.4f}")
                if r4_mean >= 0.65 or (r4_mean - a_mean) >= 0.10:
                    fallback["usable"] = True
                    fallback["delivered_strategy"] = "R4_dann"
                else:
                    fallback["usable"] = False
                    fallback["delivered_strategy"] = None
            except Exception as e:  # noqa: BLE001
                print(f"  [警告] R4 DANN 失败：{e}，跳过（R1-R3 已有证据链）")
                fallback["R4_dann"] = {"error": str(e)}
                fallback["usable"] = False
                fallback["delivered_strategy"] = None

        if not fallback.get("usable", False):
            # 穷尽证据链
            best_all = max(candidates, key=candidates.get)
            fallback["exhausted_evidence"] = {
                "best_strategy": best_all,
                "best_mean_auc": candidates[best_all],
                "usable_line": 0.65,
                "conclusion": (
                    f"在现有数据与协议下，跨疾病预测模型最优可达 AUC {candidates[best_all]:.4f}"
                    f"（{best_all}），低于可用线 0.65；证据链见 decay_attribution / "
                    f"migration_analysis / threshold_drift。"
                ),
            }
            print(f"  穷尽出口：最优 {best_all} AUC={candidates[best_all]:.4f}，未达可用线")
    else:
        fallback["usable"] = True
        fallback["delivered_strategy"] = "D_calibrated" if "D_calibrated" in strategy_compare else best_base

    # 7. 域内 AUC（264 特征重算）+ 衰减归因
    print("\n[域内 AUC] 264 特征 5 折 CV 重算")
    domain_auc = domain_auc_264(X_clr, y, dataset_name)
    for d, v in domain_auc.items():
        print(f"  {d}: 域内 AUC = {v:.4f}（A3 参考 {DOMAIN_AUC_REF_A3[d]:.3f}）")

    cross_auc = {COMBO_TO_DISEASE[c]: A[c]["auc"] for c in COMBOS}
    decay = decay_attribution(domain_auc, cross_auc, A)
    print("\n[衰减归因]")
    for d in ["CRC", "IBD", "Obesity"]:
        r = decay[d]
        print(f"  {d}: 域内 {r['domain_auc']:.4f} → 跨疾病 {r['cross_auc']:.4f} "
              f"衰减 {r['decay']:+.4f}，主导归因={r['dominant_cause']}")

    # 8. 深度迁移分析
    print("\n[深度迁移分析] 共享物种方向一致性")
    mig = migration_analysis(X_clr, y, dataset_name, shared_features, feature_names, lodo_combos)
    print(f"  方向一致 {mig['direction_consistent_count']} / 方向翻转 "
          f"{mig['direction_flipped_count']}（一致占比 {mig['consistent_fraction']:.3f}，"
          f"符号检验 p={mig['sign_test_pvalue']:.4f}）")

    # 9. C3 阈值漂移
    print("\n[C3 阈值漂移]")
    drift = threshold_drift(A, lodo_combos, y)
    print(f"  {drift['diagnosis']}")

    # 10. 组装 S3-results.pkl
    meta = {
        "sub": "S3",
        "stage": "2.1",
        "model": "LogisticRegression(L2, C=1.0, class_weight=balanced, max_iter=2000) + CLR",
        "seed": SEED,
        "clr_delta": CLR_DELTA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "outputs/data/S3-preprocessed.pkl (源自 c-data-cleaned.pkl float32)",
        "note": "四策略 A/B/C/D + 紧急回退 R1-R4；阈值迁移禁测试集重定；"
                "Platt 校准 A=-w,B=-b（sklearn 形式），单调性校验 w>0（等价 math-S3.tex A<0，"
                "符号约定相反，见口径修正）",
        "budget_limited": False,
        "field_semantics": {
            "strategy_compare.<S>.<C>.auc": "该策略该组合测试集 AUC（阈值无关主指标）",
            "strategy_compare.<S>.<C>.sensitivity": "训练集 Youden J 阈值迁移到测试集的灵敏度（非测试集重定阈值）",
            "strategy_compare.D_calibrated.<C>.auc": "校准后 AUC 与 base 策略相同（Platt 单调变换不改排序）",
            "strategy_compare.D_calibrated.<C>.A/B": "Platt 参数，P=1/(1+exp(A·f+B))，A=-w（sklearn 系数取负）",
            "fallback.triggered": "四策略（A/B/C属/C门）3 组合 AUC 均值全部 <0.60 才为 True",
            "fallback.usable": "回退候选达可用线（均值≥0.65 或相对 A 提升≥0.10）",
            "decay_attribution.<D>.decay": "跨疾病 AUC − 域内 AUC（264 特征重算域内，非 A3 的 1331 口径）",
            "threshold_drift.boundary_position": "Youden 阈值在测试分数分布中的分位（低于阈值=判健康占比）",
            "migration_analysis.direction_consistent_count": "共享物种在训练/测试疾病患病方向一致的计数（跨 3 组合累计）",
        },
    }

    # 交付模型选择：回退达可用线 → delivered_strategy；回退穷尽 → 最优可达（exhausted best）；
    # 未触发回退 → D_calibrated（最优 base 的部署形态）
    if fallback.get("delivered_strategy"):
        best_strategy = fallback["delivered_strategy"]
    elif fallback.get("exhausted_evidence"):
        best_strategy = fallback["exhausted_evidence"]["best_strategy"]
    else:
        best_strategy = "D_calibrated"
    strategy_compare["best_strategy"] = best_base  # A/B/C/D 中 AUC 最优 base

    payload = {
        "meta": meta,
        "strategy_compare": strategy_compare,
        "fallback": fallback,
        "domain_auc": domain_auc,
        "domain_auc_reference_A3": DOMAIN_AUC_REF_A3,
        "decay_attribution": decay,
        "migration_analysis": {k: v for k, v in mig.items() if k != "species_direction"},
        "migration_analysis_species_detail": mig["species_direction"],
        "threshold_drift": drift,
        "best_strategy": best_strategy,
    }

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n结果已落盘: {OUT_PKL}")

    # 11. 探索图
    make_figures(strategy_compare, fallback, decay, drift, A)
    print("探索图已保存到 outputs/figures/_explore/")

    # 12. 关键数字摘要（stdout）
    print("\n" + "=" * 72)
    print("关键数字摘要")
    print("=" * 72)
    for k in ["A_direct", "B_shared", "C_genus", "C_phylum"]:
        print(f"  {k}: C1={strategy_compare[k]['C1']['auc']:.4f} "
              f"C2={strategy_compare[k]['C2']['auc']:.4f} "
              f"C3={strategy_compare[k]['C3']['auc']:.4f} "
              f"均值={strategy_compare[k]['mean_auc']:.4f}")
    print(f"  策略 D base={D['base_strategy']} 均值={D['mean_auc']:.4f}")
    print(f"  回退触发={triggered}  交付策略={fallback.get('delivered_strategy')}")
    if triggered:
        for k in ["R1_tree", "R2_pooled", "R3_weighted"]:
            if k in fallback:
                print(f"  {k}: 均值={fallback[k]['mean_auc']:.4f}")
        if "R4_dann" in fallback and "mean_auc" in fallback["R4_dann"]:
            print(f"  R4_dann: 均值={fallback['R4_dann']['mean_auc']:.4f}")


if __name__ == "__main__":
    main()
