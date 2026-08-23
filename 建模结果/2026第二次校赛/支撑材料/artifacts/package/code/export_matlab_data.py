"""
目的：
    将 20 张论文正式图的「最终可画数据」从 pkl 导出为 MATLAB 可直接 readtable 的 CSV
    （UTF-8 BOM，列名全英文/ASCII），写入 solution/handoff/图像美化交接/data/，
    供队友用 MATLAB 重绘，保证数值与现有图逐位一致。

原理：
    所有统计计算在本脚本内完成，CSV 只含可直接 plot/bar/imagesc/histogram 的最终数据：
    - ROC / 阈值-指标曲线：与 plot_s1.py 同参数（sklearn roc_curve；排序唯一 oof_prob + 端点
      逐阈值扫描 ACC/F1/Recall/Specificity，Youden J = TPR - FPR，minority 决定少数类口径）；
    - KDE：与 plot_s3_redraw.py 同参数——C3 组合训练/测试预测概率按 S3 正式口径重算
      （CLR δ=6.5e-6 → StandardScaler(train fit) → LogisticRegression(l2, C=1,
      class_weight=balanced, max_iter=2000, random_state=42) → predict_proba[:,1]），
      gaussian_kde 在 xs=linspace(0,1,600) 求密度；
    - 直方图类（零值占比/丰度 log10）：导出原始值长列，MATLAB 用 BinEdges 复现 numpy 等宽分箱
      （零值占比: edges=0:0.05:1；丰度 log10: edges=linspace(min,max,51)，即 np.histogram(bins=50)）；
    - 批次效应 / 画像：与 chart-data-features-45.py / profile-B.py 同参数
      （近全零过滤 1331→264 → CLR → StandardScaler → PCA/t-SNE seed=42；
       StandardScaler(1331) → numpy SVD 方差解释率；K-Means++(n_init=10, seed=42) +
       silhouette 选 K → t-SNE(perplexity=30, seed=42, init=pca)）；
    - CSV 数值统一 %.6f（AUC/ROC/KDE 点列 6 位小数，满足"关键值 ≥6 位有效数字"），
      缺失单元格写 NaN；列名一律英文，避免 MATLAB 中文编码坑。

性能：
    轻量-不适用（一次性小数据导出：2 次 Logistic 拟合 / 2 次 t-SNE / 1 次 K-Means 扫描
    / 1 次 484×1331 SVD，预计 <1 分钟，秒级-分钟级边界取轻量档，无并行必要——C8 决策树 0 轻量分支）。

输入数据：
    - S1-results.pkl (结果) — <ds>.L2_CLR.{AUC,AUC_std,oof_prob,coefficients} /
      <ds>.RF_raw.{AUC,AUC_std,oof_prob} / <ds>.baseline.single_feature_best_AUC /
      adenoma_sensitivity.*.{L2_AUC,RF_AUC,n_samples}（ds ∈ Zeller_fecal_colorectal_cancer /
      metahit / Chatelier_gut_obesity；中文指标↔变量：AUC↔auc、oof_prob↔OOF 预测概率、
      coefficients↔L2 系数、single_feature_best_AUC↔单特征基线 AUC）
    - S1-preprocessed.pkl (预处理) — datasets.<ds>.{y,minority} / feature_names(264)
    - S2-results.pkl (结果) — per_disease.<D>.stable_features.{feature,frequency,cv_frequency,rank} /
      meta.{tau_grid,tau_counts} / per_disease.<D>.cooccurrence.spearman_matrix /
      cross_disease.jaccard_matrix（D ∈ CRC/IBD/Obesity）
    - S3-results.pkl (结果) — strategy_compare.<S>.<C>.auc / mean_auc /
      decay_attribution.<D>.{domain_auc,cross_auc,decay,dominant_cause} /
      migration_analysis.{direction_consistent_count,direction_flipped_count,n_valid,
      consistent_fraction,sign_test_pvalue} / threshold_drift.{train_baseline,test_baseline,
      delta_baseline,youden_threshold,boundary_position,sensitivity}
    - S3-preprocessed.pkl (处理后) — X_filtered(484×264), y(484), lodo_combos.C3.{train_idx,test_idx}
    - c-data-cleaned.pkl (共享清洗后) — dataset_name, disease, 1331 物种丰度特征列

输出：
    - solution/handoff/图像美化交接/data/fig-*.csv（20 张图主 CSV + 10 个标量/辅助 CSV，UTF-8 BOM）

对应论文章节：
    论文全部图表（S1/S2/S3 结果图 + 数据特征图 + 数据画像图）
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "data"
OUT = ROOT / "solution" / "handoff" / "图像美化交接" / "data"

DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 6.5e-6，与 S3-model.py / 出图脚本一致
SEED = 42

DS = ["Zeller_fecal_colorectal_cancer", "metahit", "Chatelier_gut_obesity"]
DS_SHORT = ["Zeller", "metahit", "Chatelier"]
S3_DISEASES = ["CRC", "IBD", "Obesity"]


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def species_name(feature):
    """从完整分类学名提取物种名（s__ 之后），无 s__ 取末段。与出图脚本一致。"""
    if "|s__" in feature:
        return feature.split("|s__")[-1]
    return feature.split("|")[-1]


def clr_transform(X: np.ndarray) -> np.ndarray:
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减行均值。与 S3-model.py 一致。"""
    arr = np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    return logX - logX.mean(axis=1, keepdims=True)


def log_csv(name, df):
    """写 CSV（UTF-8 BOM，NaN 显式），打印行数/列名/数值 min-max-mean 供抽查。"""
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.6f")
    print(f"== {name}: rows={len(df)} cols={list(df.columns)}")
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                print(f"    {c}: min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g} (n={len(s)})")
    return OUT / name


# ============================================================
# S1：疾病预测（5 张）
# ============================================================
def threshold_metrics(y, prob, minority):
    """逐阈值扫描（排序唯一 oof_prob + 端点），与 plot_s1.py 完全一致。"""
    thresh = np.unique(np.concatenate([[0.0], np.asarray(prob), [1.0]]))
    accs, f1s, recs, specs, tprs, fprs = [], [], [], [], [], []
    for t in thresh:
        pred = (prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        accs.append((tp + tn) / len(y))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
        if minority == 1:
            rec = tpr
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        else:
            rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
        recs.append(rec)
        f1s.append(f1)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return (thresh, np.asarray(accs), np.asarray(f1s), np.asarray(recs),
            np.asarray(specs), np.asarray(tprs), np.asarray(fprs))


def export_s1(s1, s1p):
    feature_names = s1p["feature_names"]

    # 图 1：ROC 点列（6 条曲线）+ AUC 标量
    rows, auc_rows = [], []
    for d, ds in zip(DS, DS_SHORT):
        y = np.asarray(s1p["datasets"][d]["y"])
        for model, key in [("L2", "L2_CLR"), ("RF", "RF_raw")]:
            prob = np.asarray(s1[d][key]["oof_prob"])
            fpr, tpr, _ = roc_curve(y, prob)
            for a, b in zip(fpr, tpr):
                rows.append((ds, model, a, b))
            auc_rows.append((ds, model, s1[d][key]["AUC"], s1[d][key]["AUC_std"]))
    log_csv("fig-S1-roc-curve.csv",
            pd.DataFrame(rows, columns=["dataset", "model", "fpr", "tpr"]))
    log_csv("fig-S1-roc-curve-auc.csv",
            pd.DataFrame(auc_rows, columns=["dataset", "model", "auc", "auc_std"]))

    # 图 2：L2/RF/单特征基线 AUC 对比（误差棒 = 5 折标准差）
    rows = []
    for d, ds in zip(DS, DS_SHORT):
        rows.append((ds, "L2", s1[d]["L2_CLR"]["AUC"], s1[d]["L2_CLR"]["AUC_std"]))
        rows.append((ds, "RF", s1[d]["RF_raw"]["AUC"], s1[d]["RF_raw"]["AUC_std"]))
        rows.append((ds, "baseline", s1[d]["baseline"]["single_feature_best_AUC"], 0.0))
    log_csv("fig-S1-performance-compare.csv",
            pd.DataFrame(rows, columns=["dataset", "model", "auc", "auc_std"]))

    # 图 3：small_adenoma 四口径敏感性
    ad = s1["adenoma_sensitivity"]
    calibers = ["CRC_adenoma_as_healthy", "CRC_adenoma_as_diseased",
                "CRC_adenoma_excluded", "CRC_adenoma_separate"]
    rows = [(c, ad[c]["L2_AUC"], ad[c]["RF_AUC"], ad[c]["n_samples"]) for c in calibers]
    log_csv("fig-S1-adenoma-sensitivity.csv",
            pd.DataFrame(rows, columns=["scenario", "L2_auc", "RF_auc", "n"]))

    # 图 4：L2 系数 Top10（按 |系数| 降序，rank 1 = 图中最顶行）
    rows = []
    for d, ds in zip(DS, DS_SHORT):
        coef = np.asarray(s1[d]["L2_CLR"]["coefficients"])
        idx = np.argsort(np.abs(coef))[::-1][:10]
        for r, i in enumerate(idx, 1):
            rows.append((ds, r, species_name(feature_names[i]), feature_names[i], float(coef[i])))
    log_csv("fig-S1-feature-importance.csv",
            pd.DataFrame(rows, columns=["dataset", "rank", "feature", "full_feature", "coefficient"]))

    # 图 5：阈值-指标曲线（长格式）+ Youden 最优标量
    rows, you_rows = [], []
    for d, ds in zip(DS, DS_SHORT):
        y = np.asarray(s1p["datasets"][d]["y"])
        minority = int(s1p["datasets"][d]["minority"])
        prob = np.asarray(s1[d]["L2_CLR"]["oof_prob"])
        thresh, accs, f1s, recs, specs, tprs, fprs = threshold_metrics(y, prob, minority)
        youden = tprs - fprs
        j_best = float(youden.max())
        t_best = float(thresh[int(np.argmax(youden))])
        for i, t in enumerate(thresh):
            rows.append((ds, t, accs[i], f1s[i], recs[i], specs[i], youden[i]))
        you_rows.append((ds, t_best, j_best))
    log_csv("fig-S1-threshold-analysis.csv",
            pd.DataFrame(rows, columns=["dataset", "threshold", "acc", "f1", "recall",
                                        "specificity", "youden_j"]))
    log_csv("fig-S1-threshold-analysis-youden.csv",
            pd.DataFrame(you_rows, columns=["dataset", "youden_threshold", "youden_j"]))


# ============================================================
# S2：特征选择（4 张）
# ============================================================
def export_s2(s2):
    # 图 1：稳定标志物频率（长格式，rank 与出图一致）
    rows = []
    for d in S3_DISEASES:
        for sf in s2["per_disease"][d]["stable_features"]:
            rows.append((d, species_name(sf["feature"]), sf["feature"],
                         sf["frequency"], sf["cv_frequency"], sf.get("rank")))
    log_csv("fig-S2-stable-frequency.csv",
            pd.DataFrame(rows, columns=["disease", "feature", "full_feature",
                                        "frequency", "cv_frequency", "rank"]))

    # 图 2：τ 敏感性
    tg = s2["meta"]["tau_grid"]
    tc = s2["meta"]["tau_counts"]
    rows = [(t, tc["CRC"][i], tc["IBD"][i], tc["Obesity"][i]) for i, t in enumerate(tg)]
    log_csv("fig-S2-tau-sensitivity.csv",
            pd.DataFrame(rows, columns=["tau", "CRC_count", "IBD_count", "Obesity_count"]))

    # 图 3：共现 Spearman 热图（长格式，N/A 单元格 = NaN；4×4 全网格）
    rows = []
    for d in ["CRC", "IBD"]:
        sm = s2["per_disease"][d]["cooccurrence"]["spearman_matrix"]
        feats = []
        for (a, b) in sm.keys():
            for f in (a, b):
                if f not in feats:
                    feats.append(f)
        n = len(feats)
        M = np.full((n, n), np.nan)
        np.fill_diagonal(M, 1.0)
        for (a, b), v in sm.items():
            i, j = feats.index(a), feats.index(b)
            if v is not None:
                M[i, j] = v
                M[j, i] = v
        for i in range(n):
            for j in range(n):
                rows.append((d, species_name(feats[i]), species_name(feats[j]), M[i, j]))
    log_csv("fig-S2-cooccurrence-heatmap.csv",
            pd.DataFrame(rows, columns=["disease", "feature", "feature2", "spearman"]))

    # 图 4：三病 Jaccard 重叠矩阵（3×3，对角 1）
    jm = s2["cross_disease"]["jaccard_matrix"]
    J = np.eye(3)
    J[0, 1] = J[1, 0] = jm["CRC_IBD"]
    J[0, 2] = J[2, 0] = jm["CRC_Obesity"]
    J[1, 2] = J[2, 1] = jm["IBD_Obesity"]
    labels = S3_DISEASES
    rows = [(labels[i], labels[j], J[i, j]) for i in range(3) for j in range(3)]
    log_csv("fig-S2-cross-disease.csv",
            pd.DataFrame(rows, columns=["disease", "disease2", "jaccard"]))


# ============================================================
# S3：跨疾病（4 张）
# ============================================================
def export_s3(s3):
    # 图 1：五策略 × 3 组合 AUC + 均值
    sc = s3["strategy_compare"]
    strats = ["A_direct", "B_shared", "C_genus", "C_phylum", "D_calibrated"]
    rows = []
    for st in strats:
        d = sc[st]
        rows.append((st, d["C1"]["auc"], d["C2"]["auc"], d["C3"]["auc"], d["mean_auc"]))
    log_csv("fig-S3-strategy-compare.csv",
            pd.DataFrame(rows, columns=["strategy", "C1_auc", "C2_auc", "C3_auc", "mean_auc"]))

    # 图 2：域内 vs 跨病衰减归因（dominant_cause 英文化）
    cause_map = {"疾病特异信号": "disease_specific", "标签语义漂移": "label_semantic"}
    da = s3["decay_attribution"]
    rows = []
    for d in S3_DISEASES:
        v = da[d]
        cause = cause_map.get(v["dominant_cause"], v["dominant_cause"])
        rows.append((d, v["domain_auc"], v["cross_auc"], v["decay"], cause))
    log_csv("fig-S3-decay-attribution.csv",
            pd.DataFrame(rows, columns=["disease", "domain_auc", "cross_auc",
                                        "decay", "dominant_cause"]))

    # 图 3：迁移方向（蝴蝶图）+ 标量
    ma = s3["migration_analysis"]
    n_cons = int(ma["direction_consistent_count"])
    n_flip = int(ma["direction_flipped_count"])
    frac = float(ma["consistent_fraction"])
    pval = float(ma["sign_test_pvalue"])
    log_csv("fig-S3-migration-direction.csv",
            pd.DataFrame([("consistent", n_cons), ("flipped", n_flip)],
                         columns=["direction", "count"]))
    log_csv("fig-S3-migration-direction-scalars.csv",
            pd.DataFrame([{"consistent_fraction": frac, "flip_fraction": 1.0 - frac,
                           "n_valid": int(ma["n_valid"]), "sign_test_pvalue": pval}]))


def export_s3_threshold_drift(s3, s3p):
    # 图 4：C3 阈值漂移 KDE（训练/测试密度序列）+ 标量
    td = s3["threshold_drift"]
    train_b = float(td["train_baseline"])
    test_b = float(td["test_baseline"])
    delta = float(td["delta_baseline"])
    tau = float(td["youden_threshold"])
    bpos = float(td["boundary_position"])
    sens = float(td["sensitivity"])

    X_filtered = s3p["X_filtered"]
    y = np.asarray(s3p["y"], dtype=int)
    lodo_combos = s3p["lodo_combos"]
    X_clr = clr_transform(X_filtered.to_numpy())
    train_idx = lodo_combos["C3"]["train_idx"]
    test_idx = lodo_combos["C3"]["test_idx"]
    Xtr, Xte = X_clr[train_idx], X_clr[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                             class_weight="balanced", random_state=SEED)
    clf.fit(scaler.transform(Xtr), ytr)
    train_score = clf.predict_proba(scaler.transform(Xtr))[:, 1]
    test_score = clf.predict_proba(scaler.transform(Xte))[:, 1]

    # 核销（与 plot_s3_redraw.py 相同口径）：重算 Youden 阈值 vs pkl
    fpr, tpr, thresholds = roc_curve(ytr, train_score)
    j = tpr - fpr
    thr_recalc = float(thresholds[int(np.argmax(j))])
    train_pos_mean = float(train_score[ytr == 1].mean())
    test_pos_mean = float(test_score[yte == 1].mean())
    print(f"  [核销] youden_threshold 重算 {thr_recalc:.6f} vs pkl {tau:.6f}；"
          f"train_pos_mean={train_pos_mean:.4f} test_pos_mean={test_pos_mean:.4f}")

    xs = np.linspace(0, 1, 600)
    d_train = gaussian_kde(train_score)(xs)
    d_test = gaussian_kde(test_score)(xs)
    rows = [(x, a, b) for x, a, b in zip(xs, d_train, d_test)]
    log_csv("fig-S3-threshold-drift.csv",
            pd.DataFrame(rows, columns=["probability", "train_density", "test_density"]))
    log_csv("fig-S3-threshold-drift-scalars.csv",
            pd.DataFrame([{"train_baseline": train_b, "test_baseline": test_b,
                           "delta_baseline": delta, "youden_threshold": tau,
                           "boundary_position": bpos, "sensitivity": sens,
                           "train_pos_mean": train_pos_mean, "test_pos_mean": test_pos_mean}]))


# ============================================================
# 数据特征图（共享数据层，5 张）
# ============================================================
def export_charts(df):
    meta = ["dataset_name", "disease"]
    feat_cols = [c for c in df.columns if c not in meta]
    mat = df[feat_cols].to_numpy(dtype=float)
    df = df.copy()

    # 图：样本构成（患病/健康/腺瘤；腺瘤仅 Zeller 有，其余 NaN）
    state_map = {"cancer": "患病", "small_adenoma": "腺瘤", "ibd_crohn_disease": "患病",
                 "ibd_ulcerative_colitis": "患病", "obesity": "患病", "n": "健康", "leaness": "健康"}
    df["_state"] = df["disease"].map(state_map)
    state_ct = df.groupby(["dataset_name", "_state"]).size()
    ds_order = [("Zeller_fecal_colorectal_cancer", "CRC"),
                ("metahit", "IBD"),
                ("Chatelier_gut_obesity", "Obesity")]
    rows, srows = [], []
    for raw, disp in ds_order:
        sub = state_ct.get(raw, pd.Series(dtype=int))
        case = int(sub.get("患病", 0))
        healthy = int(sub.get("健康", 0))
        adenoma = int(sub.get("腺瘤", 0))
        total = int(sub.sum())
        ad_val = adenoma if raw == "Zeller_fecal_colorectal_cancer" else float("nan")
        rows.append((disp, case, healthy, ad_val))
        srows.append((disp, total, case / total))
    log_csv("fig-chart-sample-composition.csv",
            pd.DataFrame(rows, columns=["dataset", "case_count", "healthy_count", "adenoma_count"]))
    log_csv("fig-chart-sample-composition-scalars.csv",
            pd.DataFrame(srows, columns=["dataset", "total", "prevalence"]))

    # 图：特征零值占比（每特征一个值；MATLAB histogram BinEdges=0:0.05:1 复现 np.histogram）
    zero_per_feat = np.mean(mat == 0, axis=0)
    overall_zero_ratio = np.mean(mat == 0)
    n_gt95 = int(np.sum(zero_per_feat > 0.95))
    n_keep = int(np.sum(zero_per_feat <= 0.95))
    log_csv("fig-chart-zero-sparsity.csv",
            pd.DataFrame(zero_per_feat, columns=["feature_zero_fraction"]))
    log_csv("fig-chart-zero-sparsity-scalars.csv",
            pd.DataFrame([{"total_features": len(feat_cols), "removed_features": n_gt95,
                           "kept_features": n_keep, "threshold": 0.95,
                           "global_zero_fraction": overall_zero_ratio}]))

    # 图：非零丰度 log10 分布（长列；MATLAB BinEdges=linspace(min,max,51) 复现 50 等宽分箱）
    nz = mat[mat != 0]
    log_nz = np.log10(nz)
    log_csv("fig-chart-abundance-distribution.csv",
            pd.DataFrame(log_nz, columns=["log10_abundance"]))
    log_csv("fig-chart-abundance-distribution-scalars.csv",
            pd.DataFrame([{"min": nz.min(), "median": np.median(nz), "max": nz.max(),
                           "n_nonzero": nz.size, "log10_min": log_nz.min(),
                           "log10_median": float(np.median(log_nz)), "log10_max": log_nz.max()}]))

    # 图：批次效应 PCA/t-SNE（近全零过滤 1331→264 → CLR → StandardScaler）
    X_raw = mat
    zero_ratio = (X_raw == 0.0).mean(axis=0)
    keep = zero_ratio <= 0.95
    Xf = X_raw[:, keep]
    X_clr = clr_transform(Xf)
    X_std = StandardScaler().fit_transform(X_clr)
    ds_short = df["dataset_name"].map(
        {"Zeller_fecal_colorectal_cancer": "CRC", "metahit": "IBD",
         "Chatelier_gut_obesity": "Obesity"}).to_numpy()
    pca = PCA(n_components=2, random_state=SEED).fit(X_std)
    pc = pca.transform(X_std)
    exp = pca.explained_variance_ratio_
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca")
    ts = tsne.fit_transform(X_std)
    rows = [(int(i), ds_short[i], pc[i, 0], pc[i, 1], ts[i, 0], ts[i, 1]) for i in range(len(df))]
    log_csv("fig-chart-batch-effect.csv",
            pd.DataFrame(rows, columns=["sample_id", "dataset", "pc1", "pc2", "tsne1", "tsne2"]))
    log_csv("fig-chart-batch-effect-scalars.csv",
            pd.DataFrame([{"pc1_variance": exp[0], "pc2_variance": exp[1],
                           "total_variance": exp[0] + exp[1]}]))

    # 图：已知标志物存在率（%）
    def disease_healthy_masks(df_, dataset):
        m = df_["dataset_name"] == dataset
        if dataset == "Zeller_fecal_colorectal_cancer":
            dis = df_["disease"] == "cancer"
            h = df_["disease"].isin(["n", "small_adenoma"])
        elif dataset == "metahit":
            dis = df_["disease"].isin(["ibd_ulcerative_colitis", "ibd_crohn_disease"])
            h = df_["disease"] == "n"
        else:  # Chatelier_gut_obesity
            dis = df_["disease"] == "obesity"
            h = df_["disease"] == "leaness"
        return m & dis, m & h

    known = [
        ("Fusobacterium_nucleatum", "nucleatum", "Zeller_fecal_colorectal_cancer", "CRC"),
        ("Peptostreptococcus_stomatis", "stomatis", "Zeller_fecal_colorectal_cancer", "CRC"),
        ("Porphyromonas_somerae", "somerae", "Zeller_fecal_colorectal_cancer", "CRC"),
        ("Bifidobacterium_bifidum", "bifidum", "metahit", "IBD"),
        ("Akkermansia_muciniphila", "muciniphila", "metahit", "IBD"),
        ("Bacteroides_fragilis", "fragilis", "Chatelier_gut_obesity", "Obesity"),
    ]
    rows = []
    for label, key, ds, disp in known:
        cols = [c for c in feat_cols if key.lower() in c.lower()]
        if not cols:
            rows.append((label, disp, float("nan"), float("nan")))
            continue
        dis_mask, hea_mask = disease_healthy_masks(df, ds)
        xd = df.loc[dis_mask, cols].to_numpy()
        xh = df.loc[hea_mask, cols].to_numpy()
        rows.append((label, disp,
                     float((xd > 0).any(axis=1).mean()) * 100.0,
                     float((xh > 0).any(axis=1).mean()) * 100.0))
    log_csv("fig-chart-known-biomarker-presence.csv",
            pd.DataFrame(rows, columns=["biomarker", "disease",
                                        "presence_case_pct", "presence_control_pct"]))


# ============================================================
# 数据画像（2 张）：PCA 碎石图 + 聚类 t-SNE（与 profile-B.py 同参数）
# ============================================================
def export_profile(df):
    X = df.iloc[:, 2:].values.astype(np.float64)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)  # 零值保留（与 profile-B.py 口径一致）

    # --- PCA 碎石图（numpy SVD）---
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    eigvals = S ** 2
    var_ratio = eigvals / eigvals.sum()
    cum_ratio = np.cumsum(var_ratio)
    scores = U * S
    k_pca = int(np.searchsorted(cum_ratio, 0.60) + 1)
    rows = [(i + 1, var_ratio[i]) for i in range(30)]
    log_csv("fig-pca-scree.csv",
            pd.DataFrame(rows, columns=["component", "variance_explained"]))
    log_csv("fig-pca-scree-scalars.csv",
            pd.DataFrame([{"kaiser": 1.0, "n_features": X.shape[1], "n_samples": X.shape[0]}]))

    # --- K-Means++ + silhouette 选 K（与 profile-B.py 逐行一致）---
    def kmeans_pp(Xm, k, n_init=10, seed=42):
        rng = np.random.default_rng(seed)
        n = Xm.shape[0]
        best = None
        for _ in range(n_init):
            centers = np.empty((k, Xm.shape[1]))
            c0 = rng.integers(n)
            centers[0] = Xm[c0]
            for j in range(1, k):
                d2 = ((Xm[:, None, :] - centers[None, :j, :]) ** 2).sum(-1).min(1)
                probs = d2 / d2.sum()
                centers[j] = Xm[rng.choice(n, p=probs)]
            for _ in range(300):
                d2 = ((Xm[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                labels = d2.argmin(1)
                new_centers = np.array([Xm[labels == j].mean(0) if (labels == j).any()
                                        else centers[j] for j in range(k)])
                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers
            d2 = ((Xm[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = d2.argmin(1)
            inertia = d2[np.arange(n), labels].sum()
            if best is None or inertia < best[1]:
                best = (labels, inertia, centers)
        return best

    def silhouette(Xm, labels):
        n = Xm.shape[0]
        D = ((Xm[:, None, :] - Xm[None, :, :]) ** 2).sum(-1) ** 0.5
        s = np.zeros(n)
        for i in range(n):
            same = labels == labels[i]
            same[i] = False
            a = D[i, same].mean() if same.any() else 0.0
            b = np.inf
            for c in np.unique(labels):
                if c == labels[i]:
                    continue
                b = min(b, D[i, labels == c].mean())
            s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        return s.mean()

    X_pca = scores[:, :k_pca]
    k_range = [2, 3, 4]
    kmeans_results, sils = {}, []
    for k in k_range:
        labels, inertia, centers = kmeans_pp(X_pca, k)
        sil = silhouette(X_pca, labels)
        kmeans_results[k] = (labels, inertia, centers)
        sils.append(sil)
    best_k = k_range[int(np.argmax(sils))]
    best_labels, _, _ = kmeans_results[best_k]
    print(f"  [画像] best_k={best_k} sils={[round(x, 4) for x in sils]}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    X_tsne = tsne.fit_transform(X_std)
    rows = [(int(i), X_tsne[i, 0], X_tsne[i, 1], int(best_labels[i])) for i in range(len(df))]
    log_csv("fig-cluster-tsne.csv",
            pd.DataFrame(rows, columns=["sample_id", "tsne1", "tsne2", "cluster"]))
    sizes = [(c, int((best_labels == c).sum())) for c in range(best_k)]
    log_csv("fig-cluster-tsne-scalars.csv",
            pd.DataFrame([{"best_k": best_k, "cluster": c, "size": sz} for c, sz in sizes]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(">>> 导出 S1 ...")
    s1 = load("S1-results.pkl")
    s1p = load("S1-preprocessed.pkl")
    export_s1(s1, s1p)

    print(">>> 导出 S2 ...")
    s2 = load("S2-results.pkl")
    export_s2(s2)

    print(">>> 导出 S3 ...")
    s3 = load("S3-results.pkl")
    s3p = load("S3-preprocessed.pkl")
    export_s3(s3)
    export_s3_threshold_drift(s3, s3p)

    print(">>> 导出数据特征图 ...")
    df = load("c-data-cleaned.pkl")
    export_charts(df)

    print(">>> 导出数据画像图 ...")
    export_profile(df)

    files = sorted(p.name for p in OUT.glob("fig-*.csv"))
    print(f"\n=== 导出完成，共 {len(files)} 个 CSV ===")
    for f in files:
        print("  " + f)


if __name__ == "__main__":
    main()
