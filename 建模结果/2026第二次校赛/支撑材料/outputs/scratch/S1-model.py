"""
目的：
    S1 阶段 2.1 正式模型实现：三数据集（Zeller CRC / metahit IBD / Chatelier Obesity）各自
    「患病 vs 健康」二分类，主模型 Logistic(L2)+CLR、对照 RF(原始丰度)、基线（单特征+Dummy）、
    LOOCV 兜底、small_adenoma 四口径敏感性、B2/B3/B4 验证，产出 S1-results.pkl + 4 张探索图。

原理：
    1) 数据加载：直接读 S1-preprocessed.pkl（1.4 预处理产物，含 X_raw/X_clr/y/folds/minority
       与 adenoma_calibers 四口径），禁止重解析原始 xlsx；特征已近全零过滤 1331→264（零值占比>95% 剔除，
       三病并集统一口径），CLR 已做（δ=6.5e-06 乘法替换 + 逐行几何均值中心化）。
    2) 主模型 L2：LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs',
       max_iter=1000)，输入 X_clr（264 维 CLR 空间）；C=1.0 即 λ=1.0（默认起点，P6 代理值，
       未调参——AUC 已达标，无需内层 CV 调参，决策留痕于此）。
       class_weight='balanced'：w_c = n/(n_classes·n_c)，放大少数类损失（针对 metahit Recall=0.400）。
    3) 对照 RF：RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1,
       random_state=42, n_jobs=-1)，输入 X_raw（原始丰度，含零值，免 CLR）；特征重要性用
       permutation importance（scoring=roc_auc，n_repeats=5，n_jobs=-1）。
    4) 评估协议：分层 5 折 CV（StratifiedKFold(5, shuffle, seed=42)，折索引复用预处理产物），
       主指标 AUC（阈值无关）+ ACC + F1/Recall(少数类)；OOF 概率聚合出混淆矩阵与 ROC。
       LOOCV 兜底：全量 AUC - LOOCV AUC > 0.1 判过拟合。
    5) 基线：单特征最佳阈值（逐特征 roc_auc_score 取 max(auc,1-auc) 处理方向，取全特征最大，
       样本内乐观上界）+ DummyClassifier(strategy='most_frequent')（ACC=多数类占比，AUC=0.5）。
    6) small_adenoma 四口径（仅 Zeller，人类裁定全做择优）：①归健康（默认主口径）②归病变
       ③剔除 26 例 ④单开一类（26 例单独第三类，报告丰度画像）；各跑 L2+RF 的 5 折 CV AUC。
    7) B2 Soft Voting（条件触发）：L2 与 RF 两方法 AUC 均 ≥0.75 才做概率平均集成，对比单最佳。
    8) B3 metahit class_weight：class_weight='balanced' vs None 的 Recall(少数类) head-to-head 对比。
    9) B4 14 离群样本剔除：复现 0.4 画像的 PCA(64PC)+K-Means++(k=2,seed=42) 定位簇1的 14 个
       Zeller 样本，剔除后重训 Zeller 对比 AUC。

性能：
    轻量-不适用（整体 <2 分钟，一次性小数据 484×264）。RF 训练与 permutation importance 用
    n_jobs=-1（sklearn 内部线程池并行，32 核）；L2 lbfgs 单核但单次拟合毫秒级（264 维 × ≤253 样本）；
    LOOCV 为 3 数据集 × ~250 次 lbfgs 拟合，秒级~十秒级。无 seed/实例级独立任务，不触发 C8 单核红线，
    无需 ProcessPoolExecutor。

输入数据：
    - S1-preprocessed.pkl（处理后，1.4 预处理产物）— datasets.<name>.{X_raw(264维原始丰度),
      X_clr(264维CLR), y(患病=1/健康=0), minority(少数类标签), folds(5折索引)},
      adenoma_calibers.{四口径}.{y, folds, keep_mask, adenoma_indices}, feature_names(264), filter, clr
    - c-data-cleaned.pkl（处理后，共享清洗数据）— 仅用于 B4 复现 0.4 聚类定位 14 离群样本

输出：
    - S1-results.pkl — 三数据集 L2/RF/基线/LOOCV/soft_voting + adenoma_sensitivity + B3/B4 结果
    - outputs/figures/_explore/S1-roc-curve-explore.pdf — 三数据集 ROC 曲线 + AUC
    - outputs/figures/_explore/S1-confusion-matrix-explore.pdf — 三数据集混淆矩阵热力图
    - outputs/figures/_explore/S1-feature-importance-explore.pdf — L2 系数 Top10 + RF importance Top10
    - outputs/figures/_explore/S1-threshold-analysis-explore.pdf — 阈值-指标曲线

对应论文章节：
    §2.1 正式模型实现（疾病预测模型）
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_curve, precision_score,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# ---- 路径定位（相对脚本位置，禁止硬编码盘符）----
ROOT = Path(__file__).resolve().parent.parent.parent
PRE_PKL = ROOT / "outputs" / "data" / "S1-preprocessed.pkl"
CLEAN_PKL = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S1-results.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

# 中文字体
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SHORT = {
    "Zeller_fecal_colorectal_cancer": "Zeller CRC",
    "metahit": "metahit IBD",
    "Chatelier_gut_obesity": "Chatelier Obesity",
}

# ---- 模型工厂（严格按 handoff §1.3/§1.4 规格）----
def make_l2(C=1.0, class_weight="balanced"):
    return LogisticRegression(
        penalty="l2", C=C, class_weight=class_weight,
        solver="lbfgs", max_iter=1000, random_state=42,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        random_state=42, n_jobs=-1,
    )


# ---- 评估：分层 5 折 CV（复用预处理折索引）+ OOF 聚合 ----
def cv_evaluate(X, y, make_model, folds, minority):
    """返回 {auc, acc, f1, recall, confusion_matrix, cv_folds, oof_prob, oof_y}。

    AUC/ACC/F1/Recall 取 5 折均值（与 A 类验证 utils.cv_evaluate 口径一致）；
    confusion_matrix 由 OOF 预测（阈值 0.5）聚合；oof_prob/oof_y 供 ROC/soft_voting 复用。
    """
    aucs, accs, f1s, recs = [], [], [], []
    oof_prob = np.zeros(len(y))
    oof_y = np.zeros(len(y), dtype=int)
    fold_details = []
    for f in folds:
        tr = np.asarray(f["train"], dtype=int)
        te = np.asarray(f["test"], dtype=int)
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yp = m.predict(X[te])
        oof_prob[te] = p
        oof_y[te] = yp
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, pos_label=minority))
        recs.append(recall_score(y[te], yp, pos_label=minority))
        fold_details.append({
            "AUC": float(aucs[-1]), "ACC": float(accs[-1]),
            "F1": float(f1s[-1]), "Recall": float(recs[-1]),
        })
    oof_pred = (oof_prob >= 0.5).astype(int)
    cm = confusion_matrix(y, oof_pred, labels=[0, 1]).tolist()  # [[TN,FP],[FN,TP]]
    return {
        "AUC": float(np.mean(aucs)),
        "AUC_std": float(np.std(aucs)),
        "ACC": float(np.mean(accs)),
        "F1_minority": float(np.mean(f1s)),
        "Recall_minority": float(np.mean(recs)),
        "confusion_matrix": cm,
        "cv_folds": fold_details,
        "oof_prob": oof_prob,
        "oof_y": oof_y,
    }


def loocv_auc(X, y, make_model):
    """LOOCV 兜底：留一法 AUC（主模型 L2）。"""
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        probs[te[0]] = m.predict_proba(X[te])[:, 1][0]
    return float(roc_auc_score(y, probs))


def full_auc(X, y, make_model):
    """全量训练 AUC（样本内乐观上界，供过拟合判定）。"""
    m = make_model()
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, p))


def single_feature_best_auc(X, y):
    """单特征最佳阈值 AUC（样本内乐观上界，性能地板）。"""
    best = 0.0
    for j in range(X.shape[1]):
        a = roc_auc_score(y, X[:, j])
        a = max(a, 1 - a)  # 处理方向
        if a > best:
            best = a
    return float(best)


def dummy_baseline(y):
    """Dummy 多数类：ACC=多数类占比，AUC=0.5。"""
    n = len(y)
    majority = max(int(y.sum()), n - int(y.sum()))
    return {"dummy_ACC": float(majority / n), "dummy_AUC": 0.5}


def fit_full_coef(X, y, make_model):
    """全量拟合，返回 (coef_, intercept_)。"""
    m = make_model()
    m.fit(X, y)
    return m.coef_[0], float(m.intercept_[0])


def rf_permutation_importance(X, y):
    """RF 全量拟合 + permutation importance（scoring=roc_auc，n_repeats=5，n_jobs=-1）。"""
    m = make_rf()
    m.fit(X, y)
    r = permutation_importance(
        m, X, y, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1,
    )
    return r.importances_mean


# ---- B4：复现 0.4 画像聚类定位 14 离群样本 ----
def kmeans_pp(X, k, n_init=10, seed=42):
    """K-Means++ 初始化 + Lloyd 迭代（复现 profile-B.py 的 kmeans_pp 实现，k=2 固定，非 K 扫描）。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best = None
    for _ in range(n_init):
        centers = np.empty((k, X.shape[1]))
        c0 = rng.integers(n)
        centers[0] = X[c0]
        for j in range(1, k):
            d2 = ((X[:, None, :] - centers[None, :j, :]) ** 2).sum(-1).min(1)
            probs = d2 / d2.sum()
            centers[j] = X[rng.choice(n, p=probs)]
        for _ in range(300):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            labels = d2.argmin(1)
            new_centers = np.array([X[labels == j].mean(0) if (labels == j).any()
                                    else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d2.argmin(1)
        inertia = d2[np.arange(n), labels].sum()
        if best is None or inertia < best[1]:
            best = (labels, inertia, centers)
    return best


def identify_zeller_outliers():
    """复现 0.4 画像：PCA(64PC)+K-Means++(k=2) 定位簇1的 14 个 Zeller 样本。

    返回 Zeller 121 样本内的布尔掩码（True=离群，共 14 个）。
    """
    df = pd.read_pickle(CLEAN_PKL)
    meta = df[["dataset_name", "disease"]].copy()
    X = df.iloc[:, 2:].values.astype(np.float64)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    eigvals = S ** 2
    var_ratio = eigvals / eigvals.sum()
    cum_ratio = np.cumsum(var_ratio)
    k_pca = int(np.searchsorted(cum_ratio, 0.60) + 1)
    scores = U * S
    X_pca = scores[:, :k_pca]
    labels, _, _ = kmeans_pp(X_pca, 2)
    cluster_sizes = np.array([(labels == c).sum() for c in range(2)])
    # 最小簇 = 0.4 画像的「簇1」（14 样本），argmin 定位比硬编码标签编号更健壮
    small_cluster = int(np.argmin(cluster_sizes))
    outlier_global = labels == small_cluster  # 484 内布尔
    zeller_global = (meta["dataset_name"] == "Zeller_fecal_colorectal_cancer").values
    zeller_global_idx = np.where(zeller_global)[0]
    outlier_in_zeller = outlier_global[zeller_global_idx]  # 121 内布尔
    return outlier_in_zeller, int(k_pca)


# ---- 探索图 ----
def plot_roc(results, datasets):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, datasets):
        d = results[name]
        for key, color, label in [("L2_CLR", "#4C72B0", "L2(CLR)"),
                                  ("RF_raw", "#55A868", "RF(raw)")]:
            y = d["_y"]
            p = d[key]["oof_prob"]
            fpr, tpr, _ = roc_curve(y, p)
            auc = d[key]["AUC"]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(SHORT[name]); ax.legend(fontsize=8)
    fig.suptitle("S1 ROC curves (OOF, 5-fold CV)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S1-roc-curve-explore.pdf")
    plt.close(fig)


def plot_confusion(results, datasets):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, datasets):
        cm = np.array(results[name]["L2_CLR"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{SHORT[name]} L2(CLR)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("S1 confusion matrices (L2 CLR, OOF)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S1-confusion-matrix-explore.pdf")
    plt.close(fig)


def plot_feature_importance(results, datasets, feat_names):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for r, name in enumerate(datasets):
        d = results[name]
        coef = np.array(d["L2_CLR"]["coefficients"])
        imp = np.array(d["RF_raw"]["feature_importances"])
        # L2 系数 Top10（按 |系数|）
        idx_l2 = np.argsort(-np.abs(coef))[:10]
        ax = axes[r, 0]
        ax.barh(range(10), coef[idx_l2][::-1], color="#4C72B0")
        ax.set_yticks(range(10))
        ax.set_yticklabels([feat_names[i].split("|")[-1].replace("s__", "") for i in idx_l2][::-1], fontsize=7)
        ax.set_title(f"{SHORT[name]} L2 coef Top10")
        # RF importance Top10
        idx_rf = np.argsort(-imp)[:10]
        ax = axes[r, 1]
        ax.barh(range(10), imp[idx_rf][::-1], color="#55A868")
        ax.set_yticks(range(10))
        ax.set_yticklabels([feat_names[i].split("|")[-1].replace("s__", "") for i in idx_rf][::-1], fontsize=7)
        ax.set_title(f"{SHORT[name]} RF perm-imp Top10")
    fig.suptitle("S1 feature importance (L2 coef vs RF permutation)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S1-feature-importance-explore.pdf")
    plt.close(fig)


def plot_threshold(results, datasets):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, datasets):
        d = results[name]
        y = d["_y"]
        p = d["L2_CLR"]["oof_prob"]
        minority = d["_minority"]
        ths = np.linspace(0.01, 0.99, 99)
        prec, rec, f1, acc = [], [], [], []
        for t in ths:
            yp = (p >= t).astype(int)
            prec.append(precision_score(y, yp, pos_label=minority, zero_division=0))
            rec.append(recall_score(y, yp, pos_label=minority))
            f1.append(f1_score(y, yp, pos_label=minority, zero_division=0))
            acc.append(accuracy_score(y, yp))
        ax.plot(ths, prec, label="Precision(min)")
        ax.plot(ths, rec, label="Recall(min)")
        ax.plot(ths, f1, label="F1(min)")
        ax.plot(ths, acc, label="ACC")
        ax.set_xlabel("threshold"); ax.set_ylabel("score")
        ax.set_title(SHORT[name]); ax.legend(fontsize=7)
    fig.suptitle("S1 threshold analysis (L2 CLR, OOF)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "S1-threshold-analysis-explore.pdf")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRE_PKL, "rb") as f:
        pre = pickle.load(f)
    feat_names = pre["feature_names"]
    datasets = list(pre["datasets"].keys())

    results = {"meta": {
        "generated": "S1 2.1 正式模型",
        "source": "S1-preprocessed.pkl (← c-data-cleaned.pkl)",
        "note": "主模型 L2(CLR)+class_weight=balanced；对照 RF(原始丰度)；分层5折CV seed=42；"
                "近全零过滤 1331→264；CLR δ=6.5e-06；AUC 主指标 + F1/Recall(少数类)",
        "field_semantics": {
            "<ds>.L2_CLR.AUC": "5 折 CV AUC 均值（阈值无关，主指标）",
            "<ds>.L2_CLR.confusion_matrix": "OOF 预测(阈值0.5)聚合，[[TN,FP],[FN,TP]]，正类=1(患病)",
            "<ds>.L2_CLR.coefficients": "264 维 CLR 空间系数（全量拟合），非原始丰度效应",
            "<ds>.RF_raw.feature_importances": "permutation importance(roc_auc,n_repeats=5) 264 维",
            "<ds>.baseline.single_feature_best_AUC": "单特征最佳阈值 AUC（样本内乐观上界，性能地板）",
            "<ds>.LOOCV.AUC": "留一法 AUC（主模型 L2，诚实估计）",
            "<ds>.soft_voting": "仅 L2 与 RF AUC 均≥0.75 时输出（B2 条件触发）",
            "adenoma_sensitivity.selected_main_caliber": "默认 healthy(口径①)，最终主口径待建模/人类择优",
            "B3_class_weight": "metahit class_weight=balanced vs None 的 Recall(少数类) 对比",
            "B4_outlier_removal": "剔除 14 离群样本(簇1)重训 Zeller 对比 AUC",
        },
    }}

    # ---- 主模型 + 对照 + 基线 + LOOCV（三数据集）----
    for name in datasets:
        d = pre["datasets"][name]
        X_raw = d["X_raw"].astype(np.float64)
        X_clr = d["X_clr"].astype(np.float64)
        y = d["y"].astype(int)
        minority = int(d["minority"])
        folds = d["folds"]

        l2 = cv_evaluate(X_clr, y, make_l2, folds, minority)
        rf = cv_evaluate(X_raw, y, make_rf, folds, minority)

        # 全量拟合系数 / 重要性
        coef, intercept = fit_full_coef(X_clr, y, make_l2)
        rf_imp = rf_permutation_importance(X_raw, y)

        # 基线
        sfa = single_feature_best_auc(X_raw, y)
        dum = dummy_baseline(y)

        # LOOCV 兜底（主模型 L2）
        loo_auc = loocv_auc(X_clr, y, make_l2)
        full_a = full_auc(X_clr, y, make_l2)

        entry = {
            "L2_CLR": {
                "AUC": l2["AUC"], "AUC_std": l2["AUC_std"],
                "ACC": l2["ACC"], "F1_minority": l2["F1_minority"],
                "Recall_minority": l2["Recall_minority"],
                "confusion_matrix": l2["confusion_matrix"],
                "cv_folds": l2["cv_folds"],
                "coefficients": coef.astype(np.float32),
                "intercept": intercept,
                "oof_prob": l2["oof_prob"],
            },
            "RF_raw": {
                "AUC": rf["AUC"], "AUC_std": rf["AUC_std"],
                "ACC": rf["ACC"], "F1_minority": rf["F1_minority"],
                "Recall_minority": rf["Recall_minority"],
                "confusion_matrix": rf["confusion_matrix"],
                "cv_folds": rf["cv_folds"],
                "feature_importances": rf_imp.astype(np.float32),
                "oof_prob": rf["oof_prob"],
            },
            "baseline": {
                "single_feature_best_AUC": sfa,
                "dummy_ACC": dum["dummy_ACC"],
                "dummy_AUC": dum["dummy_AUC"],
            },
            "LOOCV": {"AUC": loo_auc},
            "full_AUC": full_a,
            "overfit_delta": full_a - loo_auc,
            "overfit_flag": bool(full_a - loo_auc > 0.1),
            "_y": y,
            "_minority": minority,
        }

        # B2 Soft Voting（条件触发：L2 与 RF AUC 均 ≥0.75）
        if l2["AUC"] >= 0.75 and rf["AUC"] >= 0.75:
            sv_prob = (l2["oof_prob"] + rf["oof_prob"]) / 2.0
            sv_pred = (sv_prob >= 0.5).astype(int)
            sv_auc = roc_auc_score(y, sv_prob)
            best_single = max(l2["AUC"], rf["AUC"])
            entry["soft_voting"] = {
                "AUC": float(sv_auc),
                "ACC": float(accuracy_score(y, sv_pred)),
                "F1_minority": float(f1_score(y, sv_pred, pos_label=minority)),
                "Recall_minority": float(recall_score(y, sv_pred, pos_label=minority)),
                "vs_best_single_delta_AUC": float(sv_auc - best_single),
                "ensemble_beneficial": bool(sv_auc - best_single > 0.02),
            }
        else:
            entry["soft_voting"] = None

        results[name] = entry
        print(f"[{name}] L2 AUC={l2['AUC']:.3f} RF AUC={rf['AUC']:.3f} "
              f"LOOCV={loo_auc:.3f} full={full_a:.3f} overfit_delta={full_a-loo_auc:+.3f} "
              f"baseline_sfa={sfa:.3f}")

    # ---- small_adenoma 四口径敏感性（仅 Zeller）----
    zeller_name = "Zeller_fecal_colorectal_cancer"
    zeller = pre["datasets"][zeller_name]
    z_X_raw = zeller["X_raw"].astype(np.float64)
    z_X_clr = zeller["X_clr"].astype(np.float64)
    adenoma = {}
    for cal, c in pre["adenoma_calibers"].items():
        y_c = c["y"].astype(int)
        folds_c = c["folds"]
        minority_c = int(c["minority"])
        if "keep_mask" in c:
            km = np.asarray(c["keep_mask"], dtype=bool)
            Xr = z_X_raw[km]
            Xc = z_X_clr[km]
        else:
            Xr = z_X_raw
            Xc = z_X_clr
        l2_c = cv_evaluate(Xc, y_c, make_l2, folds_c, minority_c)
        rf_c = cv_evaluate(Xr, y_c, make_rf, folds_c, minority_c)
        out = {"L2_AUC": l2_c["AUC"], "RF_AUC": rf_c["AUC"], "n_samples": int(c["n_samples"])}
        if cal == "CRC_adenoma_separate":
            # 26 例 small_adenoma 丰度画像（第三类）
            ad_idx = np.asarray(c["adenoma_indices"], dtype=int)
            ad_abund = z_X_raw[ad_idx].mean(axis=0)
            top10 = np.argsort(-ad_abund)[:10]
            out["adenoma_profile"] = {
                "n_adenoma": int(len(ad_idx)),
                "mean_abundance": ad_abund.astype(np.float32),
                "top10_features": [
                    {"feature": feat_names[i], "mean_abundance": float(ad_abund[i])}
                    for i in top10
                ],
            }
        adenoma[cal] = out
        print(f"[adenoma {cal}] L2 AUC={l2_c['AUC']:.3f} RF AUC={rf_c['AUC']:.3f} n={c['n_samples']}")

    adenoma["selected_main_caliber"] = "healthy"  # 默认口径①，最终待建模/人类择优
    results["adenoma_sensitivity"] = adenoma

    # ---- B3：metahit class_weight 对比 ----
    mh = pre["datasets"]["metahit"]
    mh_X_clr = mh["X_clr"].astype(np.float64)
    mh_y = mh["y"].astype(int)
    mh_minority = int(mh["minority"])
    mh_folds = mh["folds"]
    l2_bal = cv_evaluate(mh_X_clr, mh_y, lambda: make_l2(class_weight="balanced"), mh_folds, mh_minority)
    l2_none = cv_evaluate(mh_X_clr, mh_y, lambda: make_l2(class_weight=None), mh_folds, mh_minority)
    results["B3_class_weight"] = {
        "balanced": {"AUC": l2_bal["AUC"], "Recall_minority": l2_bal["Recall_minority"],
                     "F1_minority": l2_bal["F1_minority"]},
        "none": {"AUC": l2_none["AUC"], "Recall_minority": l2_none["Recall_minority"],
                 "F1_minority": l2_none["F1_minority"]},
        "delta_Recall": float(l2_bal["Recall_minority"] - l2_none["Recall_minority"]),
    }
    print(f"[B3] metahit class_weight balanced Recall={l2_bal['Recall_minority']:.3f} "
          f"vs none Recall={l2_none['Recall_minority']:.3f} delta={l2_bal['Recall_minority']-l2_none['Recall_minority']:+.3f}")

    # ---- B4：14 离群样本剔除敏感性 ----
    outlier_mask, k_pca = identify_zeller_outliers()
    n_out = int(outlier_mask.sum())
    assert n_out == 14, f"离群样本数异常: {n_out}（预期 14）"
    keep = ~outlier_mask
    z_X_clr_keep = z_X_clr[keep]
    z_X_raw_keep = z_X_raw[keep]
    z_y_keep = zeller["y"].astype(int)[keep]
    # 重新生成剔除后的分层折（样本数变化，需重划）
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds_keep = [{"train": [int(i) for i in tr], "test": [int(i) for i in te]}
                  for tr, te in skf.split(np.zeros(len(z_y_keep)), z_y_keep)]
    z_minority = int(zeller["minority"])
    l2_keep = cv_evaluate(z_X_clr_keep, z_y_keep, make_l2, folds_keep, z_minority)
    rf_keep = cv_evaluate(z_X_raw_keep, z_y_keep, make_rf, folds_keep, z_minority)
    results["B4_outlier_removal"] = {
        "n_outliers_removed": n_out,
        "k_pca_used": k_pca,
        "full_L2_AUC": results[zeller_name]["L2_CLR"]["AUC"],
        "full_RF_AUC": results[zeller_name]["RF_raw"]["AUC"],
        "removed_L2_AUC": l2_keep["AUC"],
        "removed_RF_AUC": rf_keep["AUC"],
        "delta_L2_AUC": float(l2_keep["AUC"] - results[zeller_name]["L2_CLR"]["AUC"]),
        "delta_RF_AUC": float(rf_keep["AUC"] - results[zeller_name]["RF_raw"]["AUC"]),
    }
    print(f"[B4] 剔除 {n_out} 离群样本: L2 AUC {results[zeller_name]['L2_CLR']['AUC']:.3f}→{l2_keep['AUC']:.3f} "
          f"RF AUC {results[zeller_name]['RF_raw']['AUC']:.3f}→{rf_keep['AUC']:.3f}")

    # ---- 探索图 ----
    plot_roc(results, datasets)
    plot_confusion(results, datasets)
    plot_feature_importance(results, datasets, feat_names)
    plot_threshold(results, datasets)

    # ---- 落盘（清理内部字段 _y/_minority 不落盘）----
    for name in datasets:
        results[name].pop("_y", None)
        results[name].pop("_minority", None)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"[OK] S1-results.pkl 已落盘: {OUT_PKL}")


if __name__ == "__main__":
    main()
