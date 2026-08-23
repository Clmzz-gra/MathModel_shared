"""
目的：
    S1/S3 补充实验 E2：贝叶斯回归（ARD/稀疏先验 + 变分推断）。①S1 域内三数据集跑贝叶斯
    Logistic（ARD 稀疏先验 + 贝叶斯 L2 高斯先验对照），5 折 CV 同口径，输出 AUC + 后验 95% CI，
    与 L2（0.7907/0.8871/0.6496）对比，检验 L2 是否接近上限；②S3 LODO 跑贝叶斯 L2 与贝叶斯
    ARD 各 3 组合，与策略 A（0.5603）对比，坐实天花板与模型族无关。

原理：
    - 贝叶斯 Logistic 回归：y_i ~ Bernoulli(σ(w^T x_i + b))，σ=sigmoid，logit_p=w^T x + b。
    - 贝叶斯 L2（高斯先验，≈ sklearn L2 的贝叶斯化）：全局尺度 τ ~ HalfCauchy(β=1)，
      w_j ~ N(0, τ)，b ~ N(0, 1)。单一 τ 对所有特征统一收缩，等价于 L2 惩罚的贝叶斯形式。
    - 贝叶斯 ARD（稀疏先验，自动相关确定 Automatic Relevance Determination）：每特征独立尺度
      τ_j ~ HalfCauchy(β=1)，w_j ~ N(0, τ_j)，b ~ N(0, 1)。τ_j 由数据自动收缩，无关特征
      τ_j→0（稀疏化），是「稀疏先验」的代表（Horseshoe 的轻量替代，避免 Horseshoe 的
      非凸后验几何给 ADVI 带来的收敛困难）。
    - 类别不平衡：加权似然（class_weight='balanced' 口径），w_c = n/(n_classes·n_c)，
      经 pm.Potential 对逐样本 logp 加权，与 sklearn class_weight='balanced' 一致。
    - 推断：ADVI（mean-field 变分推断，pm.fit(n=20000)），后验 q(w)=N(μ, diag(σ²))；
      点估计用后验均值 μ（后验采样均值），后验 95% CI 用后验采样 S=200 的 AUC 分位数。
    - 评估：S1 复用 S1-preprocessed.pkl 的 5 折分层索引（seed=42，与 S1-model.py 同口径）；
      S3 复用 lodo_combos（C1 测 CRC / C2 测 IBD / C3 测 Obesity），CLR + StandardScaler
      （仅训练集估计，防泄漏，与 S3-model.py 同口径）。
    - 后验 95% CI：对每折采样 S 个后验 (w,b)，算测试 AUC，跨折+跨采样池化取 2.5%/97.5% 分位
      （后验预测分布上的 AUC 区间，含参数不确定性与折间不确定性）。

性能：
    任务级并行（ProcessPoolExecutor，max_workers=8）：S1 3 数据集 × 2 先验 × 5 折 = 30 任务，
    S3 3 组合 × 2 先验 = 6 任务，共 36 个独立 ADVI 拟合任务（无数据依赖），并行执行；
    每 worker 设 OMP_NUM_THREADS=1 防 BLAS 线程过订阅（进程级并行替代线程级，32 核分 8 进程）。
    数据 484×264 小样本，单次 ADVI 拟合 ~15-30s，8 进程并行预计总墙钟 ~2-4 分钟。
    随机性隔离：ADVI 初始化 random_seed=42，每任务独立拟合，与串行基准一致（无跨任务共享可变状态）。
    NUTS 小链仅 metahit 1 折做 sanity check（可选，--skip-nuts 跳过）。

输入数据：
    - S1-preprocessed.pkl（处理后，1.4 预处理产物）— datasets.<name>.{X_clr(264维CLR),
      y(患病=1/健康=0), folds(5折索引)}, feature_names(264)
    - S3-preprocessed.pkl（处理后）— X_filtered(484×264 过滤后物种级丰度), y(484 二分类标签),
      dataset_name, feature_names(264), lodo_combos(C1/C2/C3 的 train_idx/test_idx)

输出：
    - outputs/data/E2-bayes-results.pkl — S1 三数据集 ARD/贝叶斯L2 的 AUC+CI，S3 贝叶斯L2/ARD
      3 组合 AUC+均值，NUTS sanity check
    - solution/model-notes/experiment-e2-bayes.md — 实验报告（两表 + 判读）

对应论文章节：
    §S1/S3 补充实验 E2（贝叶斯回归稳健性检验，附录）
"""
import os

# 进程级并行：每 worker 单 BLAS 线程，防线程过订阅（须在 numpy/pytensor 导入前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import pymc as pm

ROOT = Path(__file__).resolve().parent.parent.parent
S1_PKL = ROOT / "outputs" / "data" / "S1-preprocessed.pkl"
S3_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "E2-bayes-results.pkl"

SEED = 42
N_ITER = 20000   # ADVI 迭代数
N_DRAWS = 200    # 后验采样数（CI 用）

# 检出限 = 全局最小非零丰度（与 S3-model.py 一致）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

# S1 三数据集短名（与 S1-model.py SHORT 一致）
SHORT = {
    "Zeller_fecal_colorectal_cancer": "Zeller CRC",
    "metahit": "metahit IBD",
    "Chatelier_gut_obesity": "Chatelier Obesity",
}
# S1 L2 基线（S1-results.pkl 实测，供 Δ 对比）
S1_L2_AUC = {
    "Zeller_fecal_colorectal_cancer": 0.7907,
    "metahit": 0.8871,
    "Chatelier_gut_obesity": 0.6496,
}
# S3 策略 A 基线（S3-results.pkl 实测）
S3_A_MEAN = 0.5603
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}


def clr_transform(X):
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值（几何均值中心化）。"""
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def balanced_weights(y):
    """class_weight='balanced' 口径的逐样本权重：w_c = n/(n_classes·n_c)。"""
    y = np.asarray(y)
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    w = np.ones(n, dtype=np.float64)
    for c, cnt in zip(classes, counts):
        w[y == c] = n / (n_classes * cnt)
    return w


def build_and_fit(Xtr, ytr, prior_type, sample_weights, n_iter, seed):
    """构建贝叶斯 Logistic 模型并 ADVI 拟合，返回 (approx, trace)。"""
    p = Xtr.shape[1]
    with pm.Model() as model:
        if prior_type == "ard":
            tau = pm.HalfCauchy("tau", beta=1.0, shape=p)  # 每特征独立尺度（稀疏）
            w = pm.Normal("w", mu=0.0, sigma=tau, shape=p)
        elif prior_type == "l2":
            tau = pm.HalfCauchy("tau", beta=1.0)  # 全局尺度（高斯先验）
            w = pm.Normal("w", mu=0.0, sigma=tau, shape=p)
        else:
            raise ValueError(f"未知 prior_type: {prior_type}")
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        logits = pm.math.dot(Xtr, w) + b
        # 加权似然（class_weight='balanced' 口径）
        ll = pm.logp(pm.Bernoulli.dist(logit_p=logits), ytr)
        pm.Potential("weighted_ll", sample_weights * ll)
        approx = pm.fit(n=n_iter, method=pm.ADVI(), random_seed=seed, progressbar=False)
        trace = approx.sample(draws=N_DRAWS, random_seed=seed)
    return approx, trace


def fit_eval_task(args):
    """单个 ADVI 拟合评估任务（模块级，供 ProcessPoolExecutor pickle）。

    args = (Xtr, ytr, Xte, yte, prior_type, standardize, seed, n_iter)
    返回 dict(point_auc, point_scores, yte, posterior_aucs, n_train, n_test)。
    """
    Xtr, ytr, Xte, yte, prior_type, standardize, seed, n_iter = args
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)
    if standardize:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
    sw = balanced_weights(ytr)
    approx, trace = build_and_fit(Xtr, ytr, prior_type, sw, n_iter, seed)

    w_samples = trace.posterior["w"].values  # (chain=1, draw=N_DRAWS, p)
    b_samples = trace.posterior["b"].values  # (chain=1, draw=N_DRAWS,)
    w_mean = w_samples.mean(axis=(0, 1))     # (p,)
    b_mean = b_samples.mean(axis=(0, 1))     # scalar

    point_scores = Xte @ w_mean + b_mean
    point_auc = float(roc_auc_score(yte, point_scores))

    aucs = np.empty(w_samples.shape[1])
    for s in range(w_samples.shape[1]):
        scores = Xte @ w_samples[0, s, :] + b_samples[0, s]
        aucs[s] = roc_auc_score(yte, scores)

    return {
        "point_auc": point_auc,
        "point_scores": point_scores,
        "yte": yte,
        "posterior_aucs": aucs,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }


def aggregate_folds(fold_results):
    """聚合某 (数据集/组合, 先验) 的 5 折（或 1 组合）结果 → 点 AUC + CI + 折间 std。"""
    oof_scores = np.concatenate([r["point_scores"] for r in fold_results])
    oof_y = np.concatenate([r["yte"] for r in fold_results])
    auc = float(roc_auc_score(oof_y, oof_scores))
    fold_aucs = [r["point_auc"] for r in fold_results]
    auc_std = float(np.std(fold_aucs))
    pooled = np.concatenate([r["posterior_aucs"] for r in fold_results])
    ci_low, ci_high = np.percentile(pooled, [2.5, 97.5])
    return {
        "AUC": auc,
        "AUC_std": auc_std,
        "CI_low": float(ci_low),
        "CI_high": float(ci_high),
        "fold_aucs": [float(a) for a in fold_aucs],
    }


def run_nuts_sanity(Xtr, ytr, Xte, yte, prior_type="ard", draws=500, tune=500):
    """NUTS 小链 sanity check（仅 metahit 1 折），验证 ADVI 点估计量级。"""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)
    p = Xtr.shape[1]
    sw = balanced_weights(ytr)
    with pm.Model() as model:
        if prior_type == "ard":
            tau = pm.HalfCauchy("tau", beta=1.0, shape=p)
            w = pm.Normal("w", mu=0.0, sigma=tau, shape=p)
        else:
            tau = pm.HalfCauchy("tau", beta=1.0)
            w = pm.Normal("w", mu=0.0, sigma=tau, shape=p)
        b = pm.Normal("b", mu=0.0, sigma=1.0)
        logits = pm.math.dot(Xtr, w) + b
        ll = pm.logp(pm.Bernoulli.dist(logit_p=logits), ytr)
        pm.Potential("weighted_ll", sw * ll)
        trace = pm.sample(draws=draws, tune=tune, chains=1, random_seed=SEED,
                          progressbar=False, compute_convergence_checks=False)
    w_mean = trace.posterior["w"].mean(dim=("chain", "draw")).values
    b_mean = trace.posterior["b"].mean(dim=("chain", "draw")).values
    scores = Xte @ w_mean + b_mean
    return float(roc_auc_score(yte, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：小迭代+子集任务")
    ap.add_argument("--skip-nuts", action="store_true", help="跳过 NUTS sanity check")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    n_iter = 2000 if args.smoke else N_ITER
    n_draws = 50 if args.smoke else N_DRAWS

    # ---- 加载预处理缓存 ----
    with open(S1_PKL, "rb") as f:
        pre1 = pickle.load(f)
    datasets = list(pre1["datasets"].keys())

    with open(S3_PKL, "rb") as f:
        pre3 = pickle.load(f)
    X_filtered = pre3["X_filtered"]
    y3 = np.asarray(pre3["y"], dtype=int)
    lodo_combos = pre3["lodo_combos"]
    X_clr3 = clr_transform(X_filtered.to_numpy())  # 484×264

    # ---- 构造任务 ----
    tasks = []
    task_meta = []  # (kind, name, prior, fold/combo)

    for name in datasets:
        d = pre1["datasets"][name]
        X_clr = d["X_clr"].astype(np.float64)
        y = d["y"].astype(int)
        folds = d["folds"]
        for prior in ["ard", "l2"]:
            for fi, f in enumerate(folds):
                tr = np.asarray(f["train"], dtype=int)
                te = np.asarray(f["test"], dtype=int)
                tasks.append((X_clr[tr], y[tr], X_clr[te], y[te], prior, False, SEED, n_iter))
                task_meta.append(("S1", name, prior, fi))

    for combo in ["C1", "C2", "C3"]:
        tr = np.asarray(lodo_combos[combo]["train_idx"], dtype=int)
        te = np.asarray(lodo_combos[combo]["test_idx"], dtype=int)
        for prior in ["l2", "ard"]:
            tasks.append((X_clr3[tr], y3[tr], X_clr3[te], y3[te], prior, True, SEED, n_iter))
            task_meta.append(("S3", combo, prior, 0))

    if args.smoke:
        # 冒烟：仅 metahit(ard+l2) fold0 + S3 C1(l2+ard)
        keep = []
        for i, (kind, name, prior, fi) in enumerate(task_meta):
            if kind == "S1" and name == "metahit" and fi == 0:
                keep.append(i)
            if kind == "S3" and name == "C1":
                keep.append(i)
        tasks = [tasks[i] for i in keep]
        task_meta = [task_meta[i] for i in keep]
        print(f"[smoke] 任务数={len(tasks)} n_iter={n_iter} n_draws={n_draws}")

    # ---- 并行执行 ----
    print(f"[E2] 共 {len(tasks)} 个 ADVI 任务，max_workers={args.workers}")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(fit_eval_task, tasks):
            results.append(r)

    # ---- 聚合 ----
    # 按 (kind, name, prior) 分组
    groups = {}
    for meta, r in zip(task_meta, results):
        key = (meta[0], meta[1], meta[2])
        groups.setdefault(key, []).append(r)

    s1_out = {}
    s3_out = {}
    for (kind, name, prior), rs in groups.items():
        agg = aggregate_folds(rs)
        if kind == "S1":
            s1_out.setdefault(name, {})[prior] = agg
        else:
            s3_out.setdefault(name, {})[prior] = agg

    # ---- 打印摘要 ----
    print("\n===== S1 域内（贝叶斯 ARD / 贝叶斯 L2 vs sklearn L2）=====")
    for name in s1_out:
        for prior in ["l2", "ard"]:
            if prior not in s1_out[name]:
                continue
            a = s1_out[name][prior]
            delta = a["AUC"] - S1_L2_AUC[name]
            print(f"  {SHORT[name]:<18} {prior:>3}: AUC={a['AUC']:.4f} "
                  f"CI=[{a['CI_low']:.4f},{a['CI_high']:.4f}] "
                  f"ΔvsL2={delta:+.4f} (L2={S1_L2_AUC[name]:.4f})")

    print("\n===== S3 LODO（贝叶斯 L2 / 贝叶斯 ARD vs 策略 A 0.5603）=====")
    for combo in s3_out:
        for prior in ["l2", "ard"]:
            if prior not in s3_out[combo]:
                continue
            a = s3_out[combo][prior]
            print(f"  {combo}({COMBO_TO_DISEASE[combo]:<7}) {prior:>3}: AUC={a['AUC']:.4f} "
                  f"CI=[{a['CI_low']:.4f},{a['CI_high']:.4f}]")
    for prior in ["l2", "ard"]:
        if all(c in s3_out and prior in s3_out[c] for c in ["C1", "C2", "C3"]):
            mean = float(np.mean([s3_out[c][prior]["AUC"] for c in ["C1", "C2", "C3"]]))
            print(f"  {prior:>3} 3 组合均值 = {mean:.4f} (ΔvsA={mean - S3_A_MEAN:+.4f})")

    # ---- NUTS sanity check（可选）----
    nuts = None
    if not args.smoke and not args.skip_nuts:
        print("\n[NUTS sanity check] metahit fold0 (ard)")
        try:
            mh = pre1["datasets"]["metahit"]
            X_clr = mh["X_clr"].astype(np.float64)
            y = mh["y"].astype(int)
            f0 = mh["folds"][0]
            tr = np.asarray(f0["train"], dtype=int)
            te = np.asarray(f0["test"], dtype=int)
            nuts_auc = run_nuts_sanity(X_clr[tr], y[tr], X_clr[te], y[te], "ard")
            nuts = {"dataset": "metahit", "fold": 0, "prior": "ard",
                    "AUC": nuts_auc, "advi_fold_auc": s1_out["metahit"]["ard"]["fold_aucs"][0]}
            print(f"  NUTS AUC={nuts_auc:.4f} vs ADVI fold0 AUC={nuts['advi_fold_auc']:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"  [警告] NUTS sanity check 失败：{e}")
            nuts = {"error": str(e)}

    # ---- 落盘 ----
    meta = {
        "sub": "E2",
        "stage": "补充实验",
        "model": "贝叶斯 Logistic（ARD 稀疏先验 / 贝叶斯 L2 高斯先验）+ ADVI 变分推断",
        "prior": "ARD: w_j~N(0,τ_j), τ_j~HalfCauchy(1)；L2: w_j~N(0,τ), τ~HalfCauchy(1)；b~N(0,1)",
        "inference": f"ADVI mean-field, n_iter={n_iter}, posterior draws={n_draws}",
        "class_weight": "balanced（加权似然，w_c=n/(n_classes·n_c)）",
        "seed": SEED,
        "clr_delta": CLR_DELTA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "S1-preprocessed.pkl + S3-preprocessed.pkl（源自 c-data-cleaned.pkl）",
        "note": "S1 复用 5 折分层索引(seed=42)；S3 复用 lodo_combos，CLR+StandardScaler(仅训练集估计)。"
                "点 AUC=后验均值 OOF；95% CI=后验采样 AUC 跨折池化 2.5/97.5 分位。",
        "field_semantics": {
            "s1.<ds>.<prior>.AUC": "后验均值 OOF AUC（5 折，阈值无关主指标）",
            "s1.<ds>.<prior>.CI_low/CI_high": "后验 95% CI（后验采样 AUC 跨折池化分位）",
            "s1.<ds>.<prior>.AUC_std": "5 折点 AUC 标准差（折间不确定性）",
            "s3.<combo>.<prior>.AUC": "该 LODO 组合测试集 AUC（后验均值）",
            "s3.<combo>.<prior>.CI_low/CI_high": "后验 95% CI",
            "nuts.AUC": "NUTS 小链 sanity check AUC（metahit fold0，验证 ADVI 量级）",
        },
    }
    payload = {
        "meta": meta,
        "s1": s1_out,
        "s3": s3_out,
        "s1_l2_baseline": S1_L2_AUC,
        "s3_A_baseline_mean": S3_A_MEAN,
        "nuts_sanity": nuts,
    }
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n[OK] E2-bayes-results.pkl 已落盘: {OUT_PKL}")


if __name__ == "__main__":
    main()
