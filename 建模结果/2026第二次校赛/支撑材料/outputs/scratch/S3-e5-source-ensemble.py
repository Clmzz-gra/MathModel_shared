"""
目的：
    S3 补充实验 E5：源分离学习 + 输出级融合——LODO 两训练疾病分别独立训练源模型，再用
    4 种输出级融合方式结合打分测留出疾病，验证「模型组合方式」维度能否突破 0.56~0.60 区间。

原理：
    - LODO 协议（approach-S3-confirmed.md §4.1）：C1 测 CRC / C2 测 IBD / C3 测 Obesity，
      训练集=其余 2 疾病。与策略 A（合并样本训一个模型）不同，本实验按源疾病切分为 S_a/S_b，
      源模型 M_a/M_b 各自只用源子集拟合（StandardScaler 仅该源子集估计，「分别学习」）；
      其余口径与策略 A 完全一致（CLR 全数据逐样本做过、Logistic 同参数）。
    - 输出级融合（测试打分全用训练侧信息，无泄漏）：F1 logit 平均 score=(f_a+f_b)/2
      （f=decision_function 对率）；F2 概率平均 P=(P_a+P_b)/2；F3 域内 CV-AUC 加权——每源在
      源子集上 5 折分层 CV（seed=42，Pipeline 内 scaler 逐折重估）得池化 OOF AUC，
      权重 w_d=AUC_d/(AUC_a+AUC_b)，score=w_a·f_a+w_b·f_b（有害源 AUC<0.5 自动拉低）；
      F4 单源对照：M_a/M_b 单独打分（报告用）。
    - 基线核对：脚本内重算合并训练模型（=策略 A 完全同口径）作同日对比，并与 S3-results.pkl
      的 strategy_compare.A_direct.<C>.auc（官方均值 0.5603）核对，|差| 应 <1e-6。

性能：
    任务级并行（ProcessPoolExecutor，max_workers=min(8, cpu_count)）：3 组合 × {源a、源b、
    合并基线} = 9 个独立拟合评分任务并行（融合算术在主进程对已返回的测试分数做毫秒级
    合成，无数据依赖）；数据 484×264 小样本，单任务含 5 折 CV 仍为秒级以下，整体预计
    <15 秒（轻量，并行仅为遵循 C8 任务独立性纪律，非性能瓶颈）。随机性隔离：CV 用
    StratifiedKFold(shuffle=True, random_state=42)，Logistic random_state=42 固定（lbfgs
    确定性求解），worker 间无随机耦合，结果与串行执行完全一致。

输入数据：
    - S3-preprocessed.pkl (处理后，源自 c-data-cleaned.pkl float32) —
      X_filtered(484×264 过滤后物种级丰度，1331→264 近全零过滤已完成), y(484 二分类标签，
      1=患病/0=健康), dataset_name(484 数据集名), lodo_combos(C1/C2/C3 的 train_idx/
      test_idx/train_datasets/test_disease)
    - S3-results.pkl (结果缓存) — strategy_compare.A_direct.<C>.auc（官方基线，仅核对用）
    中文指标↔代码变量名映射：跨疾病融合 AUC = per_combo[c].fusion_auc.*；单源迁移 AUC =
    per_combo[c].sources[D].single_test_auc；域内 OOF AUC（仅权重用）=
    per_combo[c].sources[D].oof_cv_auc；合并基线 = per_combo[c].baseline_pooled_A.in_script_auc。

输出：
    - outputs/data/S3-e5-source-ensemble.pkl — meta（口径/日期/脚本路径/字段语义）+
      per_combo（每组合单源/融合/基线 AUC 与 F3 权重）+ summary（3 组合均值表 + 基线核对）

对应论文章节：
    §S3 跨疾病预测模型（补充实验 E5，归因分析第 5 环，探索性，不入论文正文）
"""
from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
REF_PKL = ROOT / "outputs" / "data" / "S3-results.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S3-e5-source-ensemble.pkl"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

SEED = 42  # Logistic 固定随机种子 + CV 打乱种子（lbfgs 确定性，与 S3-model.py 一致）
CV_FOLDS = 5  # F3 权重的域内分层 CV 折数
COMBOS = ["C1", "C2", "C3"]
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}
DATASET_DISEASE = {
    "Zeller_fecal_colorectal_cancer": "CRC",
    "metahit": "IBD",
    "Chatelier_gut_obesity": "Obesity",
}


# ---------------------------------------------------------------------------
# 公共口径（与 S3-model.py / S3-e3-fewshot.py 一致）
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


# ---------------------------------------------------------------------------
# 并行 worker（模块级，供 ProcessPoolExecutor pickle）
# ---------------------------------------------------------------------------
def _source_worker(args):
    """worker：单个源模型的「分别学习」pipeline + 域内 OOF CV-AUC + 留出疾病打分。

    步骤：① 从该组合训练集中按 dataset_name 切出本源子集 S_d；② StandardScaler 仅在
    S_d 上 fit + Logistic 拟合；③ 在 S_d 上做 5 折分层 CV（seed=42，Pipeline 内 scaler
    逐折重估防折内泄漏）得池化 OOF AUC（F3 权重用）；④ 对留出疾病打分（logit + 概率）。
    返回 (combo, ds_name, info_dict)。logit = decision_function（对率）。
    """
    combo, ds_name, X_clr, y, dataset_name, lodo_combos = args
    train_idx = np.asarray(lodo_combos[combo]["train_idx"])
    test_idx = np.asarray(lodo_combos[combo]["test_idx"])

    src_idx = train_idx[dataset_name[train_idx] == ds_name]
    X_src = X_clr[src_idx]
    y_src = y[src_idx]

    # 源独立 pipeline：「分别学习」的核心——scaler 不看另一个源的样本
    scaler = StandardScaler().fit(X_src)
    X_src_s = scaler.transform(X_src)
    clf = make_logistic().fit(X_src_s, y_src)

    # 域内池化 OOF AUC（仅作 F3 权重估计）：Pipeline 保证每折 scaler 重估，无折内泄漏
    counts = np.bincount(y_src.astype(int))
    n_splits = min(CV_FOLDS, int(counts.min())) if len(counts) == 2 else 0
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_logistic())])
        oof = cross_val_predict(pipe, X_src, y_src, cv=skf, method="decision_function")
        oof_auc = float(roc_auc_score(y_src, oof))
    else:
        oof_auc = float("nan")  # 类别过少退化情形（本数据不触发）

    X_te_s = scaler.transform(X_clr[test_idx])
    return combo, ds_name, {
        "oof_cv_auc": oof_auc,
        "oof_cv_folds": int(n_splits),
        "single_test_logit": clf.decision_function(X_te_s),
        "single_test_proba": clf.predict_proba(X_te_s)[:, 1],
        "n_samples": int(len(src_idx)),
        "n_pos": int((y_src == 1).sum()),
        "n_neg": int((y_src == 0).sum()),
    }


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
# 融合与汇总
# ---------------------------------------------------------------------------
def fuse_combo(combo: str, src_results: dict, base_results: dict, lodo_combos: dict, y: np.ndarray) -> dict:
    """对单个组合做 F1/F2/F3 融合打分 + F4 单源对照 + 基线核对，返回结构化结果。"""
    info_c = lodo_combos[combo]
    ds_a, ds_b = info_c["train_datasets"]
    dis_a, dis_b = DATASET_DISEASE[ds_a], DATASET_DISEASE[ds_b]
    ra, rb = src_results[(combo, ds_a)], src_results[(combo, ds_b)]
    yte = y[np.asarray(info_c["test_idx"])]

    f_a, f_b = ra["single_test_logit"], rb["single_test_logit"]
    p_a, p_b = ra["single_test_proba"], rb["single_test_proba"]

    # F4 单源对照
    auc_a = float(roc_auc_score(yte, f_a))
    auc_b = float(roc_auc_score(yte, f_b))

    # F1 logit 平均 / F2 概率平均
    f1_auc = float(roc_auc_score(yte, (f_a + f_b) / 2.0))
    f2_auc = float(roc_auc_score(yte, (p_a + p_b) / 2.0))

    # F3 域内 CV-AUC 加权（权重归一化；nan/退化回退等权）
    w_a_raw = float(ra["oof_cv_auc"]) if np.isfinite(ra["oof_cv_auc"]) else 0.5
    w_b_raw = float(rb["oof_cv_auc"]) if np.isfinite(rb["oof_cv_auc"]) else 0.5
    denom = w_a_raw + w_b_raw
    w_a = w_a_raw / denom if denom > 0 else 0.5
    w_b = 1.0 - w_a
    f3_auc = float(roc_auc_score(yte, w_a * f_a + w_b * f_b))

    base_auc, n_train = base_results[combo]
    off_auc = float(base_results[combo + "_official"])  # 由 main 预置的官方值

    return {
        "test_disease": info_c["test_disease"],
        "n_train": int(n_train),
        "n_test": int(len(yte)),
        "sources": {
            dis_a: {
                "dataset": ds_a, "n_samples": ra["n_samples"],
                "n_pos": ra["n_pos"], "n_neg": ra["n_neg"],
                "oof_cv_auc": ra["oof_cv_auc"], "oof_cv_folds": ra["oof_cv_folds"],
                "single_test_auc": auc_a,
            },
            dis_b: {
                "dataset": ds_b, "n_samples": rb["n_samples"],
                "n_pos": rb["n_pos"], "n_neg": rb["n_neg"],
                "oof_cv_auc": rb["oof_cv_auc"], "oof_cv_folds": rb["oof_cv_folds"],
                "single_test_auc": auc_b,
            },
        },
        "single_source_auc": {dis_a: auc_a, dis_b: auc_b},  # F4 对照
        "F3_weights": {dis_a: w_a, dis_b: w_b},
        "fusion_auc": {
            "F1_logit_mean": f1_auc,
            "F2_prob_mean": f2_auc,
            "F3_cvauc_weighted": f3_auc,
        },
        "baseline_pooled_A": {
            "in_script_auc": float(base_auc),
            "official_pkl_auc": off_auc,
            "abs_diff": abs(float(base_auc) - off_auc),
            "match_within_1e-6": bool(abs(float(base_auc) - off_auc) < 1e-6),
        },
    }


def print_report(per_combo: dict, summary: dict, combos_run: list, smoke: bool):
    """stdout 摘要表（人读用，正式数值以 pkl 为准）。"""
    tag = "[SMOKE] " if smoke else ""
    print("\n" + "=" * 88)
    print(f"{tag}E5 源分离学习 + 输出级融合（{'仅 ' + '+'.join(combos_run) if smoke else '全部 3 组合'}）")
    print("=" * 88)
    hdr = (f"{'组合':<4} {'测试':<8} | {'单源a':>6} {'单源b':>6} | "
           f"{'F1':>6} {'F2':>6} {'F3':>6} | {'基线A重算':>9} {'官方pkl':>8}")
    print(hdr)
    print("-" * 88)
    for c in combos_run:
        e = per_combo[c]
        dis = list(e["single_source_auc"].keys())
        b = e["baseline_pooled_A"]
        print(f"{c:<4} {e['test_disease']:<8} | "
              f"{e['single_source_auc'][dis[0]]:>6.4f} {e['single_source_auc'][dis[1]]:>6.4f} | "
              f"{e['fusion_auc']['F1_logit_mean']:>6.4f} {e['fusion_auc']['F2_prob_mean']:>6.4f} "
              f"{e['fusion_auc']['F3_cvauc_weighted']:>6.4f} | "
              f"{b['in_script_auc']:>9.4f} {b['official_pkl_auc']:>8.4f}")
        # F3 权重明细（判读「有害源是否被降权」时用）
        wts = ", ".join(f"w({d})={e['F3_weights'][d]:.3f}(域内OOF="
                        f"{e['sources'][d]['oof_cv_auc']:.4f})" for d in dis)
        print(f"     F3 权重: {wts}")

    print("-" * 88)
    m = summary["mean_auc_by_method"]
    scope = f"({'+'.join(combos_run)} 均值" + ("，非完整结论" if smoke else "") + ")"
    print(f"均值{scope}:")
    print(f"  基线A(重算)={m['baseline_pooled_A']:.4f}  F1={m['F1_logit_mean']:.4f}  "
          f"F2={m['F2_prob_mean']:.4f}  F3={m['F3_cvauc_weighted']:.4f}  "
          f"F4单源均={m['F4_single_source_avg']:.4f}")

    bc = summary["baseline_check"]
    # 注意：Windows GBK 控制台无法编码 ✓/✗ 等符号，此处用 ASCII 标记
    flag = "[一致]" if bc["match_within_1e-6"] else "[不一致，需排查]"
    print(f"\n基线核对: 脚本内重算均值={bc['in_script_mean']:.6f}  "
          f"官方pkl(同组合均值)={bc['official_mean_same_combos']:.6f}  "
          f"|Δ|={bc['abs_diff']:.2e}  {flag}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S3-E5 源分离学习 + 输出级融合实验")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：只跑 C1 组合（验证全部代码路径），不落盘 pkl")
    args = ap.parse_args()
    smoke = args.smoke
    combos_run = ["C1"] if smoke else COMBOS

    print("=" * 72)
    print(f"S3 补充实验 E5：源分离学习 + 输出级融合{'（SMOKE：仅 C1，不落盘）' if smoke else ''}")
    print("=" * 72)

    # 1. 加载预处理缓存（X_filtered 已 1331→264 过滤，源自 c-data-cleaned.pkl）
    with open(DATA_PKL, "rb") as fh:
        pre = pickle.load(fh)
    X_filtered = pre["X_filtered"]
    y = np.asarray(pre["y"], dtype=int)
    dataset_name = np.asarray(pre["dataset_name"])
    lodo_combos = pre["lodo_combos"]

    # 2. CLR 变换（逐样本，全数据做过，无泄漏——与策略 A/E3 口径一致）
    X_clr = clr_transform(X_filtered.to_numpy())
    print(f"特征维度：{X_clr.shape[1]}（过滤后物种级）")

    # 3. 官方基线（S3-results.pkl，仅核对用，不重算替代）
    with open(REF_PKL, "rb") as fh:
        ref = pickle.load(fh)
    ad = ref["strategy_compare"]["A_direct"]

    # 4. 构造并行任务：3 组合 × {源a, 源b}（拟合+CV+打分一体） + 3 组合 × 合并基线
    src_tasks = [
        (c, ds, X_clr, y, dataset_name, lodo_combos)
        for c in combos_run
        for ds in lodo_combos[c]["train_datasets"]
    ]
    base_tasks = [(c, X_clr, y, lodo_combos) for c in combos_run]
    print(f"并行任务：{len(src_tasks)} 个源模型任务 + {len(base_tasks)} 个基线任务"
          f"（max_workers={min(8, os.cpu_count() or 4)}）")

    src_results: dict = {}
    base_results: dict = {}
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for combo, ds_name, info in ex.map(_source_worker, src_tasks):
            src_results[(combo, ds_name)] = info
        for combo, auc, n_tr in ex.map(_baseline_worker, base_tasks):
            base_results[combo] = (auc, n_tr)

    # 官方基线值预置进 base_results（fuse_combo 内统一取用）
    for c in combos_run:
        base_results[c + "_official"] = ad[c]["auc"]

    # 5. 融合打分 + 结构化结果（主进程，毫秒级算术）
    per_combo = {c: fuse_combo(c, src_results, base_results, lodo_combos, y) for c in combos_run}

    # 6. 汇总均值 + 基线核对
    def _mean(fn) -> float:
        return float(np.mean([fn(per_combo[c]) for c in combos_run]))

    in_script_mean = _mean(lambda e: e["baseline_pooled_A"]["in_script_auc"])
    official_mean_all = float(ad["mean_auc"])  # 官方 3 组合全量均值（引用值 0.5603）
    # 核对口径：与实际运行的组合集合对比（冒烟只跑 C1 时对 C1 官方值，避免跨集合误报）
    official_mean_run = float(np.mean([ad[c]["auc"] for c in combos_run]))
    summary = {
        "combos_included": list(combos_run),
        "mean_auc_by_method": {
            "baseline_pooled_A": in_script_mean,
            "F1_logit_mean": _mean(lambda e: e["fusion_auc"]["F1_logit_mean"]),
            "F2_prob_mean": _mean(lambda e: e["fusion_auc"]["F2_prob_mean"]),
            "F3_cvauc_weighted": _mean(lambda e: e["fusion_auc"]["F3_cvauc_weighted"]),
            "F4_single_source_avg": _mean(
                lambda e: float(np.mean(list(e["single_source_auc"].values())))),
        },
        "official_baseline_mean_auc": official_mean_all,
        "baseline_check": {
            "in_script_mean": in_script_mean,
            "official_mean_same_combos": official_mean_run,
            "abs_diff": abs(in_script_mean - official_mean_run),
            "match_within_1e-6": bool(abs(in_script_mean - official_mean_run) < 1e-6),
        },
    }

    # 7. stdout 摘要
    print_report(per_combo, summary, combos_run, smoke)

    # 8. 落盘（冒烟模式跳过，防部分数据混入正式产物）
    if smoke:
        print("\n[SMOKE] 冒烟模式不落盘 pkl；完整运行后产物为 outputs/data/S3-e5-source-ensemble.pkl")
        return

    meta = {
        "sub": "S3",
        "stage": "E5-source-ensemble",
        "script_path": str(Path(__file__).resolve()),
        "model": "LogisticRegression(L2, C=1.0, class_weight=balanced, max_iter=2000, "
                 "random_state=42) + CLR(delta=6.5e-6) + 源分离 StandardScaler（各源子集独立估计）",
        "fusion_methods": {
            "F1_logit_mean": "(f_a+f_b)/2，f=decision_function（对率）",
            "F2_prob_mean": "(P_a+P_b)/2",
            "F3_cvauc_weighted": "score=w_a·f_a+w_b·f_b，w_d=源子集池化OOF AUC 归一化"
                                 "（5 折分层 CV，shuffle+seed=42，Pipeline 逐折重估 scaler）",
            "F4_single_source": "M_a/M_b 单独打分（对照组）",
        },
        "seed": SEED,
        "cv_seed": SEED,
        "cv_folds": CV_FOLDS,
        "clr_delta": CLR_DELTA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_data": "outputs/data/S3-preprocessed.pkl (源自 c-data-cleaned.pkl float32)",
        "baseline_reference": "outputs/data/S3-results.pkl :: strategy_compare.A_direct"
                              "（官方均值 0.5603，仅核对不替代）",
        "note": "源分离学习+输出级融合：两训练疾病分别独立训练（scaler 仅各自源子集估计，"
                "体现「分别学习」），4 种输出级融合测留出疾病；合并基线在脚本内同日重算并核对官方值",
        "field_semantics": {
            "per_combo.<C>.sources.<D>.oof_cv_auc": "源子集 D 上分层 CV 池化 OOF AUC"
                "（仅作 F3 权重估计的域内指标，非跨疾病迁移指标）",
            "per_combo.<C>.sources.<D>.single_test_auc": "源模型 D 单独在留出疾病上的 AUC（F4 对照）",
            "per_combo.<C>.fusion_auc.*": "融合打分在留出疾病上的 AUC（主指标，阈值无关）",
            "per_combo.<C>.F3_weights.<D>": "F3 中源 D 的归一化权重（有害源 AUC<0.5 自动拉低）",
            "per_combo.<C>.baseline_pooled_A.abs_diff": "|脚本内重算−官方 pkl|，应 <1e-6（确定性复现核对）",
            "summary.mean_auc_by_method.*": "3 组合 AUC 均值（与官方 0.5603 同口径对比）",
        },
    }
    payload = {"meta": meta, "per_combo": per_combo, "summary": summary}
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n结果已落盘: {OUT_PKL}")


if __name__ == "__main__":
    main()
