"""
目的：
    S3 补充实验 E6：深度学习直接分类（MLP）跑 LODO——补齐归因证据链最后一档。
    R4 DANN（0.5947）是「域适应深度网络」，纯深度分类器从未直接测过；本实验用 MLP 直接
    分类跑同一 LODO 协议，回答「深度学习本身是否也无效」。

性能：
    瓶颈分析（C8 决策树）：Q0 长耗时？数据 484×264 小样本，单次 MLP 拟合（adam ≤500 迭代 +
    early stopping）秒级，完整运行分钟级以内。Q1 GPU？PyTorch/RAPIDS 有成熟方案，但任务规格
    指定 sklearn MLPClassifier 且单任务秒级，GPU 收益低于维护成本，按 C3 例外条款回退 CPU。
    Q2 任务独立？组合×配置互不依赖 → Q3b 任务级并行：ProcessPoolExecutor，完整运行
    3 组合 × 3 配置 = 9 个 MLP 任务 + 3 个基线任务，max_workers=min(8, cpu_count)；预计完整
    运行 <2 分钟、--smoke（2 任务）<40 秒。随机性隔离：每 worker 内 MLP random_state=42 固定
    （adam 随机性仅来自权重初始化与 batch 顺序，固定种子即确定），Logistic lbfgs 确定性，
    worker 间无随机耦合，结果与串行执行一致。

原理：
    - LODO 协议（approach-S3-confirmed.md §4.1）：C1 测 CRC / C2 测 IBD / C3 测 Obesity，
      训练集 = 其余 2 疾病，与策略 A / E0-E3 / E5 完全同口径。
    - 数据链路（与 S3-e3-fewshot.py 一致，防泄漏）：X_filtered(484×264 过滤后丰度) → CLR
      （零值乘法替换 δ=0.65×检出限=6.5e-6 + 逐样本 log 减行均值，无跨样本参数）→
      StandardScaler 仅训练集 fit。
    - 模型：sklearn MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
      alpha=1e-3, max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.15,
      n_iter_no_change=10)。MLP 无 class_weight 参数——类别不平衡用 sample_weight 传入 fit
      （sklearn>=1.7 支持；本环境 1.9.0 已验证支持），权重口径与 class_weight='balanced'
      同款：w_c = n/(n_classes·n_c)，保证与合并训练 Logistic 基线可比。
    - 敏感性对照（同协议重跑）：α=1e-2（更强正则，防过拟合对照）；hidden=(128,64)（更大
      容量对照）。
    - 训练集内 early-stopping 验证 AUC（train_val_auc）：MLP 开 early_stopping 后从训练集内
      部再切 15% 作分层验证集。本脚本**精确截获**该内部验证集——给 sklearn 模块级
      train_test_split 包一层间谍函数，捕获内部划分返回的 (X_val, y_val)
      （train_val_source=internal_early_stopping_val），在其上算 AUC——展示「训练疾病域内强、
      跨疾病塌缩」模式；并用「最终模型（early stopping 回滚最优权重后）在该验证集上的准确率
      == best_validation_score_」交叉核验（val_split_verified 字段，False 时 train_val_auc
      仅供参考并在报告标注）。注意不能按文档直觉用同参同种子重放该划分：实测 sklearn 1.9
      中权重初始化（_initialize 的 uniform 抽样）先于划分消耗同一随机流，且各配置网络规模
      不同消耗量不同，外部无法按序复现（冒烟实测：重放划分 acc=0.9091 vs 内部 0.6560）；
      另外不能用 max(validation_scores_) 作参照——sklearn 仅在 score>best+tol 时更新最优
      权重，最终模型可能停在略低于历史峰值的迭代上，参照必须用 best_validation_score_；
      且该分数是**加权**准确率（_score 传 sample_weight=sample_weight_val，权重=balanced
      口径在验证子集上的取值，见 _multilayer_perceptron.py L802），核验必须用同权重准确率
      （冒烟实测：按文档直觉重放的划分上无权重 acc=0.9091，而内部加权最优验证 acc=0.6560
      ——两者连样本成员都不同，证实重放不可行）。
      截获失败时回退为自行同参划分（train_val_source=reproduced_split_unverified，
      val_split_verified 恒 False）。
    - 主指标 AUC（阈值无关）；基线核对：脚本内重算合并训练 Logistic（策略 A 完全同口径：
      L2/C=1.0/class_weight=balanced/max_iter=2000/lbfgs/random_state=42），与官方
      S3-results.pkl::strategy_compare.A_direct.<C>.auc 核对 |Δ|<1e-6（确定性复现）。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32) —
      X_filtered(484×264 过滤后物种级丰度，1331→264 近全零过滤已完成), y(484 二分类标签，
      1=患病/0=健康), lodo_combos(C1/C2/C3 的 train_idx/test_idx/train_datasets/test_disease)
    - S3-results.pkl (结果缓存) — strategy_compare.A_direct.<C>.auc（官方基线，仅核对用）
    中文指标↔代码变量名映射：跨疾病测试 AUC = per_combo[c].configs[cfg].test_auc；
    训练集内 early-stopping 验证 AUC = per_combo[c].configs[cfg].train_val_auc；
    合并训练基线 = per_combo[c].baseline_pooled_A.in_script_auc。

输出：
    - outputs/data/S3-e6-mlp.pkl — meta（口径/日期/脚本路径/字段语义）+
      per_combo（每组合 3 配置 test_auc/train_val_auc + 合并基线核对）+ summary
      （mean_auc_by_config + official_baseline_mean_auc + baseline_check + R4 DANN 对照）

对应论文章节：
    §S3 跨疾病预测模型（补充实验 E6，归因分析第 6 环，探索性，不入论文正文）
"""
from __future__ import annotations

import inspect
import os
import pickle
import warnings

# sklearn 1.9 弃用 penalty 参数（改用 l1_ratio），但规格明确要求 penalty='l2'（S1/S2/S3 口径），
# 保留 penalty='l2' 并抑制该 FutureWarning（非可操作项）
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
REF_PKL = ROOT / "outputs" / "data" / "S3-results.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S3-e6-mlp.pkl"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

SEED = 42  # MLP/Logistic 固定随机种子 + early-stopping 验证划分种子（与全管线一致）
COMBOS = ["C1", "C2", "C3"]
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}

# MLP 配置表：主配置 + α=1e-2 强正则对照 + 宽网络容量对照（其余超参全部一致）
CONFIGS = {
    "mlp_main": {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
    "mlp_alpha1e-2": {"hidden_layer_sizes": (64, 32), "alpha": 1e-2},
    "mlp_wide": {"hidden_layer_sizes": (128, 64), "alpha": 1e-3},
}
SMOKE_CONFIGS = ["mlp_main"]  # 冒烟模式只跑主配置

MAX_ITER = 500
VALIDATION_FRACTION = 0.15
N_ITER_NO_CHANGE = 10

# R4 DANN 对照（文献复现值，证据链第 4 环；非本管线 LODO 重跑，仅作档位对照）
DANN_REF_AUC = 0.5947
DANN_REF_NOTE = (
    "R4 DANN 对抗式域适应深度网络（文献复现值）；E6 为纯深度直接分类器（无域适应机制），"
    "两者构成「深度学习跨疾病」的两档对照：若 E6 ≈ DANN ≈ 0.56~0.59 则深度模型同样受限于信号稀缺"
)

# 运行时能力探测：MLPClassifier.fit 的 sample_weight 支持（sklearn>=1.7 引入）
FIT_SUPPORTS_SAMPLE_WEIGHT = (
    "sample_weight" in inspect.signature(MLPClassifier.fit).parameters
)


# ---------------------------------------------------------------------------
# 公共口径（与 S3-model.py / S3-e3-fewshot.py / S3-e5-source-ensemble.py 一致）
# ---------------------------------------------------------------------------
def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值（几何均值中心化）。

    无跨样本参数，不引入训练/测试泄漏。接受 ndarray，返回 ndarray。
    """
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def make_logistic() -> LogisticRegression:
    """策略 A 同参数 Logistic（penalty/C/class_weight/max_iter/random_state 全一致）。"""
    return LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=SEED,
    )


def build_mlp(hidden_layer_sizes: tuple, alpha: float) -> MLPClassifier:
    """任务规格指定的 MLP 直接分类器（除 hidden/alpha 外所有超参各配置一致）。"""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        max_iter=MAX_ITER,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
    )


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """class_weight='balanced' 同款样本权重：w_c = n/(n_classes·n_c)，逐样本展开。

    与基线 Logistic 的 class_weight='balanced' 加权口径完全一致（E2 加权似然同源），
    保证 MLP 与基线的类别不平衡处理可比。
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)
    w = np.zeros(n, dtype=float)
    for c in classes:
        n_c = int((y == c).sum())
        if n_c > 0:
            w[y == c] = n / (n_classes * n_c)
    return w


# ---------------------------------------------------------------------------
# 并行 worker（模块级，供 ProcessPoolExecutor pickle）
# ---------------------------------------------------------------------------
def _mlp_worker(args):
    """worker：单个 (combo, config) 的 MLP 直接分类训练评估。

    步骤：① 取 LODO 训练/测试划分；② StandardScaler 仅训练集 fit；③ balanced 口径
    sample_weight 传入 fit；④ 截获内部 early-stopping 验证集算 train_val_auc 并交叉核验；
    ⑤ 留出疾病打分算 test_auc。返回 (combo, cfg_name, info_dict)。
    """
    combo, cfg_name, hidden, alpha, X_clr, y, lodo_combos = args
    train_idx = np.asarray(lodo_combos[combo]["train_idx"])
    test_idx = np.asarray(lodo_combos[combo]["test_idx"])
    Xtr = X_clr[train_idx]
    Xte = X_clr[test_idx]
    ytr = y[train_idx]
    yte = y[test_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if not FIT_SUPPORTS_SAMPLE_WEIGHT:
        # 显式失败而非静默降级：降级方案（加权重采样）会改变 early-stopping 划分语义，
        # 必须先登记口径偏差，不得混入正式产物
        raise RuntimeError(
            "当前 sklearn 的 MLPClassifier.fit 不支持 sample_weight"
            "（需 sklearn>=1.7）；请登记口径偏差后改用加权重采样方案"
        )
    sw = balanced_sample_weights(ytr)
    clf = build_mlp(hidden, alpha)

    # 精确截获 MLP 内部 early-stopping 验证集：给 sklearn 模块级 train_test_split
    # （_multilayer_perceptron.py 顶部 from-import 的名字）包一层间谍函数，捕获内部划分
    # 返回的 (X_val, y_val)。不能按文档直觉重放划分——权重初始化先于划分消耗同一随机流
    # （见文件头「原理」）。参照值用 best_validation_score_（early stopping 回滚的最优权重
    # 对应的验证准确率），不能用 max(validation_scores_)（见「原理」）。
    cap: dict = {}
    val_source = "internal_early_stopping_val"
    try:
        import sklearn.neural_network._multilayer_perceptron as _mlp_mod

        _orig_tts = getattr(_mlp_mod, "train_test_split", None)
        if _orig_tts is None:
            raise ImportError("module attr 'train_test_split' not found")

        def _tts_spy(*a, **kw):
            out = _orig_tts(*a, **kw)
            # 内部有两种调用布局（见 _multilayer_perceptron.py L651/L668）：
            #   无 sample_weight：(X_tr, X_val, y_tr, y_val)                 → 4 元组
            #   有 sample_weight：(X_tr, X_val, y_tr, y_val, sw_tr, sw_val)  → 6 元组
            # 本脚本走 6 元组分支；按长度自适应防版本/布局漂移
            if len(out) == 6:
                cap["X_val"], cap["y_val"], cap["sw_val"] = out[1], out[3], out[5]
            else:
                cap["X_val"], cap["y_val"], cap["sw_val"] = out[2], out[3], None
            return out

        _mlp_mod.train_test_split = _tts_spy
        try:
            clf.fit(Xtr_s, ytr, sample_weight=sw)
        finally:
            _mlp_mod.train_test_split = _orig_tts  # 无条件还原，防污染同进程后续任务
        # 内部切分发生在 y 重塑为 (n,1) 之后（_fit 先 reshape 再进 _fit_stochastic），
        # 截获的 y_val/sw_val 是 2 维——必须拉平回 1-D，否则与 1-D 预测广播成 (n,n) 矩阵
        if cap.get("X_val") is not None and cap.get("y_val") is not None:
            X_va = np.asarray(cap["X_val"])
            y_va = np.asarray(cap["y_val"]).ravel()
        else:
            X_va = y_va = None
    except Exception:
        X_va = y_va = None

    if X_va is None or y_va is None:
        # 回退：同参同种子自行划分。与内部划分不保证一致（随机流已被权重初始化消耗），
        # 仅作量级参考；val_split_verified 恒 False 并在报告中标注
        val_source = "reproduced_split_unverified"
        _, X_va, _, y_va = train_test_split(
            Xtr_s, ytr, test_size=VALIDATION_FRACTION,
            random_state=SEED, stratify=ytr,
        )

    val_auc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
    y_pred_va = clf.predict(X_va)
    sw_va_raw = cap.get("sw_val")
    val_acc_rep = float(np.mean(y_pred_va == y_va))  # 无权重准确率（诊断用）
    if sw_va_raw is not None:
        # 内部验证分数是加权准确率（_score 传 sample_weight=sample_weight_val，见「原理」），
        # 核验必须同权重比较
        val_acc_rep_w = float(np.average(
            y_pred_va == y_va, weights=np.asarray(sw_va_raw).ravel()))
    else:
        val_acc_rep_w = val_acc_rep
    v_scores = getattr(clf, "validation_scores_", None)
    val_scores_max = float(np.max(v_scores)) if v_scores else float("nan")
    ref_best = getattr(clf, "best_validation_score_", None)
    val_best_acc = float(ref_best) if ref_best is not None else float("nan")
    split_verified = bool(
        val_source == "internal_early_stopping_val"
        and np.isfinite(val_best_acc)
        and abs(val_acc_rep_w - val_best_acc) <= 1e-10
    )

    info = {
        "test_auc": float(roc_auc_score(yte, clf.predict_proba(Xte_s)[:, 1])),
        "train_val_auc": val_auc,
        "train_val_source": val_source,
        "val_split_verified": split_verified,
        "val_best_accuracy_weighted": val_best_acc,
        "val_accuracy_reproduced_weighted": val_acc_rep_w,
        "val_accuracy_reproduced_unweighted": val_acc_rep,
        "val_scores_max_weighted": val_scores_max,
        "n_iter": int(clf.n_iter_),
        "hidden_layer_sizes": tuple(int(h) for h in hidden),
        "alpha": float(alpha),
        "imbalance_method": "sample_weight(w_c=n/(n_classes*n_c), balanced)",
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }
    return combo, cfg_name, info


def _baseline_worker(args):
    """worker：策略 A 同口径合并训练模型（两源样本合并、scaler 仅合并训练集估计）。

    返回 (combo, test_auc, n_train)。用于同日基线重算 + 官方 0.5603 核对。
    """
    combo, X_clr, y, lodo_combos = args
    train_idx = np.asarray(lodo_combos[combo]["train_idx"])
    test_idx = np.asarray(lodo_combos[combo]["test_idx"])
    Xtr = X_clr[train_idx]
    ytr = y[train_idx]

    scaler = StandardScaler().fit(Xtr)
    clf = make_logistic().fit(scaler.transform(Xtr), ytr)
    f_te = clf.decision_function(scaler.transform(X_clr[test_idx]))
    return combo, float(roc_auc_score(y[test_idx], f_te)), int(len(ytr))


# ---------------------------------------------------------------------------
# stdout 摘要（人读用，正式数值以 pkl 为准）
# ---------------------------------------------------------------------------
def print_report(per_combo: dict, summary: dict, combos_run: list,
                 configs_run: list, smoke: bool):
    """stdout 摘要表。注意：Windows GBK 控制台无法编码 ✓/✗ 等符号，一律用 ASCII 标记。"""
    tag = "[SMOKE] " if smoke else ""
    print("\n" + "=" * 100)
    scope = "+".join(combos_run)
    print(f"{tag}E6 MLP 直接分类 LODO（{'仅 ' + scope if smoke else '全部 3 组合'}）")
    print("=" * 100)

    cfg_cols = "  ".join(f"{name}(test/val)".rjust(22) for name in configs_run)
    hdr = f"{'组合':<4} {'测试':<8} | {cfg_cols} | {'基线A重算':>9} {'官方pkl':>8}"
    print(hdr)
    print("-" * 100)
    for c in combos_run:
        e = per_combo[c]
        cells = "  ".join(
            f"{e['configs'][name]['test_auc']:.4f}/"
            f"{e['configs'][name]['train_val_auc']:.4f}".rjust(22)
            for name in configs_run
        )
        b = e["baseline_pooled_A"]
        print(f"{c:<4} {e['test_disease']:<8} | {cells} | "
              f"{b['in_script_auc']:>9.4f} {b['official_pkl_auc']:>8.4f}")

    # 划分核验（任何 False 都要在报告中标注）
    bad = [
        f"{c}/{name}"
        for c in combos_run for name in configs_run
        if not per_combo[c]["configs"][name]["val_split_verified"]
    ]
    if bad:
        for c in combos_run:
            for name in configs_run:
                cfg = per_combo[c]["configs"][name]
                if not cfg["val_split_verified"]:
                    print(f"[警告] {c}/{name} (source={cfg['train_val_source']}): "
                          f"复现加权acc={cfg['val_accuracy_reproduced_weighted']:.4f}"
                          f" vs 最优验证加权acc={cfg['val_best_accuracy_weighted']:.4f}"
                          f" —— train_val_auc 仅供参考")
    else:
        print("[OK] 全部配置的内部验证集截获与核验通过（加权 acc 差 <= 1e-10）")

    print("-" * 100)
    m = summary["mean_auc_by_config"]
    parts = "  ".join(f"{k}={v:.4f}" for k, v in m.items())
    scope_note = f"({scope} 均值" + ("，非完整结论" if smoke else "") + ")"
    print(f"均值{scope_note}: {parts}")

    bc = summary["baseline_check"]
    flag = "[一致]" if bc["match_within_1e-6"] else "[不一致，需排查]"
    print(f"\n基线核对: 脚本内重算均值={bc['in_script_mean']:.6f}  "
          f"官方pkl(同组合均值)={bc['official_mean_same_combos']:.6f}  "
          f"|D|={bc['abs_diff']:.2e}  {flag}")
    print(f"(官方 3 组合全量均值 reference={bc['official_baseline_all_mean']:.4f}，"
          f"完整运行时同组合均值应与其一致)")

    dr = summary["dann_reference"]
    if dr.get("mlp_main_mean_auc") is not None:
        print(f"DANN 对照: E6 mlp_main 均值={dr['mlp_main_mean_auc']:.4f}  "
              f"R4 DANN={dr['reference_auc']:.4f}  "
              f"gap(E6-DANN)={dr['gap_mlp_main_minus_dann']:+.4f}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S3-E6 深度学习直接分类（MLP）LODO 实验")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：只跑 C1 组合 MLP 主配置 + 基线（验证代码路径），不落盘 pkl")
    args = ap.parse_args()
    smoke = args.smoke
    combos_run = ["C1"] if smoke else COMBOS
    configs_run = SMOKE_CONFIGS if smoke else list(CONFIGS.keys())

    print("=" * 72)
    print(f"S3 补充实验 E6：深度学习直接分类（MLP）"
          f"{'（SMOKE：仅 C1 x mlp_main，不落盘）' if smoke else ''}")
    print("=" * 72)
    print(f"sklearn={sklearn.__version__}  "
          f"MLPClassifier.fit 支持 sample_weight={FIT_SUPPORTS_SAMPLE_WEIGHT}")

    # 1. 加载预处理缓存（X_filtered 已 1331→264 过滤，源自 c-data-cleaned.pkl）
    with open(DATA_PKL, "rb") as fh:
        pre = pickle.load(fh)
    X_filtered = pre["X_filtered"]
    y = np.asarray(pre["y"], dtype=int)
    lodo_combos = pre["lodo_combos"]

    # 2. CLR 变换（逐样本，全数据做过，无泄漏——与策略 A/E3/E5 口径一致）
    X_clr = clr_transform(X_filtered.to_numpy())
    print(f"特征维度：{X_clr.shape[1]}（过滤后物种级）")

    # 3. 官方基线（S3-results.pkl，仅核对用，不重算替代）
    with open(REF_PKL, "rb") as fh:
        ref = pickle.load(fh)
    ad = ref["strategy_compare"]["A_direct"]

    # 4. 构造并行任务：组合 × 配置（拟合+验证复现+打分一体） + 组合 × 合并基线
    mlp_tasks = [
        (c, name, CONFIGS[name]["hidden_layer_sizes"], CONFIGS[name]["alpha"],
         X_clr, y, lodo_combos)
        for c in combos_run
        for name in configs_run
    ]
    base_tasks = [(c, X_clr, y, lodo_combos) for c in combos_run]
    max_workers = min(8, os.cpu_count() or 4)
    print(f"并行任务：{len(mlp_tasks)} 个 MLP 任务 + {len(base_tasks)} 个基线任务"
          f"（max_workers={max_workers}）")

    mlp_results: dict = {}
    base_results: dict = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for combo, name, info in ex.map(_mlp_worker, mlp_tasks):
            mlp_results[(combo, name)] = info
        for combo, auc, n_tr in ex.map(_baseline_worker, base_tasks):
            base_results[combo] = (auc, n_tr)

    # 官方基线值预置进 base_results（组装阶段统一取用，与 E5 同构）
    for c in combos_run:
        base_results[c + "_official"] = ad[c]["auc"]

    # 5. 组装 per_combo（主进程，毫秒级）
    per_combo = {}
    for c in combos_run:
        base_auc, n_train = base_results[c]
        off_auc = float(base_results[c + "_official"])
        per_combo[c] = {
            "test_disease": lodo_combos[c]["test_disease"],
            "n_train": int(n_train),
            "n_test": mlp_results[(c, configs_run[0])]["n_test"],
            "configs": {name: mlp_results[(c, name)] for name in configs_run},
            "baseline_pooled_A": {
                "in_script_auc": float(base_auc),
                "official_pkl_auc": off_auc,
                "abs_diff": abs(float(base_auc) - off_auc),
                "match_within_1e-6": bool(abs(float(base_auc) - off_auc) < 1e-6),
            },
        }

    # 6. 汇总均值 + 基线核对 + DANN 对照
    def _mean(fn) -> float:
        return float(np.mean([fn(per_combo[c]) for c in combos_run]))

    in_script_mean = _mean(lambda e: e["baseline_pooled_A"]["in_script_auc"])
    official_mean_all = float(ad["mean_auc"])  # 官方 3 组合全量均值（引用值 0.5603）
    # 核对口径：与实际运行的组合集合对比（冒烟只跑 C1 时对 C1 官方值，避免跨集合误报）
    official_mean_run = float(np.mean([ad[c]["auc"] for c in combos_run]))

    mlp_main_mean = (
        _mean(lambda e: e["configs"]["mlp_main"]["test_auc"])
        if "mlp_main" in configs_run else None
    )
    summary = {
        "combos_included": list(combos_run),
        "mean_auc_by_config": {
            **{name: _mean(lambda e, nm=name: e["configs"][nm]["test_auc"])
               for name in configs_run},
            "baseline_pooled_A": in_script_mean,
        },
        "official_baseline_mean_auc": official_mean_all,
        "baseline_check": {
            "in_script_mean": in_script_mean,
            "official_mean_same_combos": official_mean_run,
            "official_baseline_all_mean": official_mean_all,
            "abs_diff": abs(in_script_mean - official_mean_run),
            "match_within_1e-6": bool(abs(in_script_mean - official_mean_run) < 1e-6),
        },
        "dann_reference": {
            "reference_name": "R4_DANN",
            "reference_auc": DANN_REF_AUC,
            "note": DANN_REF_NOTE,
            "mlp_main_mean_auc": mlp_main_mean,
            "gap_mlp_main_minus_dann": (
                mlp_main_mean - DANN_REF_AUC if mlp_main_mean is not None else None
            ),
        },
    }

    # 7. stdout 摘要
    print_report(per_combo, summary, combos_run, configs_run, smoke)

    # 8. 落盘（冒烟模式跳过，防部分数据混入正式产物）
    if smoke:
        print("\n[SMOKE] 冒烟模式不落盘 pkl；完整运行后产物为 outputs/data/S3-e6-mlp.pkl")
        return

    meta = {
        "sub": "S3",
        "stage": "E6-mlp-direct",
        "script_path": str(Path(__file__).resolve()),
        "model": "MLPClassifier(relu, adam, max_iter=500, random_state=42, "
                 "early_stopping=True, validation_fraction=0.15, n_iter_no_change=10; "
                 "hidden/alpha 见 configs) + CLR(delta=6.5e-6) + StandardScaler（仅训练集 fit）",
        "configs": {name: dict(v) for name, v in CONFIGS.items()},
        "imbalance_handling": "MLP 无 class_weight：sample_weight=w_c=n/(n_classes*n_c) 传入 "
                              "fit（sklearn>=1.7 支持，环境 sklearn=" + sklearn.__version__ +
                              "已验证），与基线 class_weight='balanced' 加权口径一致",
        "seed": SEED,
        "max_iter": MAX_ITER,
        "validation_fraction": VALIDATION_FRACTION,
        "n_iter_no_change": N_ITER_NO_CHANGE,
        "clr_delta": CLR_DELTA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_data": "outputs/data/S3-preprocessed.pkl (源自 c-data-cleaned.pkl float32)",
        "baseline_reference": "outputs/data/S3-results.pkl :: strategy_compare.A_direct"
                              "（官方均值 0.5603，仅核对不替代）",
        "note": "深度学习直接分类（MLP）LODO：证据链最后一档——R4 DANN 是域适应深度网络，"
                "本实验测纯深度分类器（无域适应机制）是否同样无效；train_val_auc 取自精确截获的"
                " early-stopping 内部验证划分并经准确率交叉核验，展示域内强/跨疾病塌缩对照；"
                "合并基线同日重算并核对官方值",
        "field_semantics": {
            "per_combo.<C>.configs.<cfg>.test_auc":
                "留出疾病上的 AUC（主指标，阈值无关）",
            "per_combo.<C>.configs.<cfg>.train_val_auc":
                "训练集内部 early-stopping 验证划分（15% 分层）上的 AUC"
                "（域内指标，用于展示「域内强、跨疾病塌缩」，不可与 test_auc 混比）",
            "per_combo.<C>.configs.<cfg>.train_val_source":
                "train_val_auc 所用验证集来源：internal_early_stopping_val=精确截获的 MLP 内部"
                "验证集（首选）；reproduced_split_unverified=截获失败的回退划分（仅供参考）",
            "per_combo.<C>.configs.<cfg>.val_split_verified":
                "内部验证集核验（最终模型在该验证集上的加权准确率==best_validation_score_，"
                "权重=balanced sample_weight 在验证子集上的取值，与 sklearn 内部评分同口径）；"
                "False 时 train_val_auc 仅供参考",
            "per_combo.<C>.configs.<cfg>.val_best_accuracy_weighted":
                "sklearn best_validation_score_（加权准确率，early stopping 的监控量）",
            "per_combo.<C>.configs.<cfg>.imbalance_method":
                "类别不平衡处理方式（balanced 口径样本权重，与基线可比）",
            "per_combo.<C>.baseline_pooled_A.abs_diff":
                "|脚本内重算−官方 pkl|，应 <1e-6（确定性复现核对）",
            "summary.mean_auc_by_config.*":
                "运行组合集合的 AUC 均值（完整运行=3 组合，与官方 0.5603 同口径对比）",
            "summary.baseline_check.abs_diff":
                "|重算均值−官方同组合均值|，应 <1e-6",
            "summary.dann_reference.*":
                "R4 DANN 0.5947 为文献复现值（对抗式域适应深度网络），与本实验同为 LODO 协议"
                "但实现独立，仅作「深度学习两档」（域适应 vs 纯分类）对照，不作同日复现核对",
        },
    }
    payload = {"meta": meta, "per_combo": per_combo, "summary": summary}
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n结果已落盘: {OUT_PKL}")


if __name__ == "__main__":
    main()
