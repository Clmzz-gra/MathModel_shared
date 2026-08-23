"""
目的：
    S2 阶段 2.1 正式模型实现：从 S2-preprocessed.pkl（1.4 预处理产物）加载过滤后 264 特征，
    对三病（CRC/IBD/Obesity）独立执行 Lasso+bootstrap 稳定性选择 + 两路信号检验（Fisher/Wilcoxon
    + BH-FDR m=1331）+ 共现分析初探 + RF/VIP 佐证，产出 S2-results.pkl 与探索图。

原理：
    - 稳定性选择（approach §3.2/§3.3）：小样本下单次 L1 选择方差大，用 B 轮分层 bootstrap 重抽样
      聚合出现频率 π̂_j = (1/B)Σ_b 1{β̂_j^(b)≠0}，π̂_j ≥ τ(=0.5) 的特征入选稳定标志物。
      每轮：CLR 后标准化 → LogisticRegression(solver='liblinear', l1_ratio=1.0, C=0.1) 拟合
      （l1_ratio=1.0 等价纯 L1 稀疏，与 math §3.2 的 penalty='l1' 语义一致）→ 非零系数特征记为"入选"。
      C=0.1 为 proxy-replacement-checklist-S2.md P7 当前临时值。
    - 分层 CV 折内选择（防泄漏，approach §3.4）：除全量 bootstrap（乐观）外，另在 5 折训练折内
      各做 B 轮 bootstrap，跨折平均得 CV 内稳定频率（诚实），两套数字并列。
    - 两路信号（approach §3.5/§3.6）：(a) Fisher 精确检验（存在/缺失 2×2 列联表，超几何分布）；
      (b) 非零样本 Wilcoxon 秩和检验（CLR 后丰度）。两路均 BH-FDR 校正，m=1331 全特征规模
      （人类裁定：医疗宁可严格不可虚报——检验对 264 特征执行，多重比较按全 1331 计数）。
    - 共现分析初探（approach §1.3b）：入选标志物两两 Spearman 相关（非零样本 CLR 丰度）+
      Fisher 精确检验（存在/缺失独立性），OR>1 且显著=cooccur 边，OR<1 且显著=exclude 边。
    - 佐证层（approach §1.4）：RF permutation importance（原始丰度，免 CLR）+ PLS-DA VIP（>1.5），
      与 Lasso 稳定特征 Top-N 一致性作稳健性佐证。

性能：
    并行策略：bootstrap 每轮独立可并行（joblib Parallel, loky 后端, n_jobs=8），
    全量 B=100×3 病 + CV 折内 5×B=50×3 病 ≈ 1050 次 Lasso 拟合并行化，预计墙钟 <1 分钟；
    RF permutation importance 用 n_jobs=-1 并行。无 GPU 方案（liblinear/树模型 CPU 即可）。

输入数据：
    - S2-preprocessed.pkl (处理后/1.4 预处理) — per_disease{CRC/IBD/Obesity}: X_raw(过滤后原始丰度
      n×264), X_clr(CLR 后丰度), y(二分类标签 患病=1/健康=0), cv_folds(5 折索引)；
      feature_names(264 全分类学名), feature_taxonomy(7 级层级)

输出：
    - outputs/data/S2-results.pkl — 每病稳定特征+频率+两路信号+标志物表+共现+RF/VIP+一致性
      + 跨疾病对比 + τ 敏感性 + meta（含 field_semantics）
    - outputs/figures/_explore/S2-2.1-stability-frequency-explore.pdf — 频率直方图
    - outputs/figures/_explore/S2-2.1-tau-sensitivity-explore.pdf — τ 敏感性曲线
    - outputs/figures/_explore/S2-2.1-cooccurrence-heatmap-explore.pdf — 共现 Spearman 热图

对应论文章节：
    §2.1 特征选择与生物标志物（正式模型实现）
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 中文字体（探索图标题/标签用中文）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_IN = ROOT / "outputs" / "data" / "S2-preprocessed.pkl"
DATA_OUT = ROOT / "outputs" / "data" / "S2-results.pkl"
FIG_DIR = ROOT / "outputs" / "figures" / "_explore"

# 参数（proxy-replacement-checklist-S2.md 当前临时值，无占位符）
TAU = 0.5                 # P1：稳定特征频率阈值（范围 0.5~0.6，暂定 0.5）
B_FULL = 100              # P4：全量 bootstrap 轮数
B_CV = 50                 # CV 折内 bootstrap 轮数（诚实估计，5 折 × 50）
C_LASSO = 0.1             # P7：Lasso 正则 C（V6 用值，当前临时值）
FDR_ALPHA = 0.05          # P3：BH-FDR 显著性水平
FDR_M = 1331              # P3：多重比较规模（全特征，人类裁定）
VIP_THRESHOLD = 1.5       # P2：PLS-DA VIP 阈值
TOP_N = 20                # P8：标志物表 Top-N 上限
N_JOBS = 8                # 并行 worker 数
SEED = 0                  # 全局随机种子
TAU_GRID = [0.4, 0.5, 0.6, 0.7]  # τ 敏感性网格

# 已知标志物锚点（domain-knowledge.md / verify-S2-v4）
KNOWN_BIOMARKERS = {
    "Fusobacterium nucleatum": "nucleatum",
    "Faecalibacterium prausnitzii": "prausnitzii",
    "Bifidobacterium (属)": "bifidobacterium",
    "Peptostreptococcus stomatis": "stomatis",
    "Parvimonas micra": "micra",
    "Porphyromonas (属)": "porphyromonas",
    "Bacteroides fragilis": "fragilis",
}


def short_name(feature: str) -> str:
    """取分类学名最后一段（s__ 种名）作可读短名。"""
    return feature.split("|")[-1]


def is_known_biomarker(feature: str) -> bool:
    """判断特征是否匹配已知标志物（大小写不敏感子串匹配）。"""
    low = feature.lower()
    return any(key in low for key in KNOWN_BIOMARKERS.values())


def bh_qvalues(pvals: np.ndarray, m: int | None = None) -> np.ndarray:
    """BH-FDR 校正 q 值（m 为多重比较总规模，缺省=len(pvals)）。"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    m = m if m is not None else n
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    q = np.empty(n, dtype=float)
    q[order[-1]] = sorted_p[-1] * m / n
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i + 1]], sorted_p[i] * m / (i + 1))
    return q


def _fit_lasso_bootstrap(Xs: np.ndarray, y: np.ndarray, C: float, seed: int) -> np.ndarray:
    """单轮分层 bootstrap Lasso：返回非零系数特征 mask（int8）。"""
    rng = np.random.default_rng(seed)
    idx_d = np.where(y == 1)[0]
    idx_h = np.where(y == 0)[0]
    bd = rng.choice(idx_d, size=len(idx_d), replace=True)
    bh = rng.choice(idx_h, size=len(idx_h), replace=True)
    idx = np.concatenate([bd, bh])
    # L1 稀疏 Logistic（Lasso）：sklearn 1.9 起 penalty 参数弃用，l1_ratio=1.0 即等价
    # penalty='l1'（纯 L1 稀疏，与 math-S2.tex §3.2 语义一致）。已实证验证：
    # l1_ratio=1.0（无 penalty）与 penalty='l1' 选中特征集完全一致（13/264 非零，稀疏解），
    # 且无弃用警告；penalty='l2' 则 264/264 全非零（对照，证明 l1_ratio=1.0 确为 L1 非 L2）。
    l1 = LogisticRegression(solver="liblinear", C=C, max_iter=2000, random_state=seed, l1_ratio=1.0)
    l1.fit(Xs[idx], y[idx])
    return (np.abs(l1.coef_[0]) > 1e-8).astype(np.int8)


def bootstrap_frequency(X_clr: np.ndarray, y: np.ndarray, B: int, C: float,
                        base_seed: int, n_jobs: int) -> np.ndarray:
    """分层 bootstrap 聚合频率 π̂_j（并行）。X_clr 先标准化一次。"""
    Xs = StandardScaler().fit_transform(X_clr)
    masks = Parallel(n_jobs=n_jobs)(
        delayed(_fit_lasso_bootstrap)(Xs, y, C, base_seed + b) for b in range(B)
    )
    return np.mean(np.vstack(masks), axis=0)


def cv_foldin_frequency(X_clr: np.ndarray, y: np.ndarray, folds: list, B: int, C: float,
                        base_seed: int, n_jobs: int) -> np.ndarray:
    """分层 CV 折内 bootstrap 频率（诚实估计）：每折训练集内 B 轮，跨折平均。"""
    n_feat = X_clr.shape[1]
    fold_freqs = []
    for k, (tr, _te) in enumerate(folds):
        tr = np.asarray(tr)
        X_tr = X_clr[tr]
        y_tr = y[tr]
        Xs = StandardScaler().fit_transform(X_tr)
        masks = Parallel(n_jobs=n_jobs)(
            delayed(_fit_lasso_bootstrap)(Xs, y_tr, C, base_seed + k * B + b) for b in range(B)
        )
        fold_freqs.append(np.mean(np.vstack(masks), axis=0))
    return np.mean(np.vstack(fold_freqs), axis=0)


def vip_scores(X_std: np.ndarray, y: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PLS-DA VIP 得分（math-S2.tex §6.2 公式）。"""
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_std, y)
    t = pls.x_scores_
    w = pls.x_weights_
    p, h = w.shape
    s = np.diag(t.T @ t).reshape(h, -1)
    total_s = np.sum(s)
    return np.sqrt(p * (s.T @ (w ** 2).T) / total_s).flatten()


def two_path_tests(X_raw: np.ndarray, X_clr: np.ndarray, y: np.ndarray) -> dict:
    """对全部 264 特征做两路信号检验，返回 fisher_p/wilcoxon_p 数组。"""
    n_feat = X_raw.shape[1]
    fisher_p = np.ones(n_feat)
    wilcoxon_p = np.ones(n_feat)
    for j in range(n_feat):
        # (a) Fisher 精确检验：存在/缺失 × 病/健
        pres_d = (X_raw[y == 1, j] > 0).sum()
        pres_h = (X_raw[y == 0, j] > 0).sum()
        abs_d = (y == 1).sum() - pres_d
        abs_h = (y == 0).sum() - pres_h
        table = [[pres_d, pres_h], [abs_d, abs_h]]
        fisher_p[j] = fisher_exact(table, alternative="two-sided")[1]
        # (b) Wilcoxon 秩和检验：非零样本 CLR 丰度
        xd = X_clr[y == 1, j]
        xh = X_clr[y == 0, j]
        xd_nz = xd[X_raw[y == 1, j] > 0]
        xh_nz = xh[X_raw[y == 0, j] > 0]
        if len(xd_nz) >= 5 and len(xh_nz) >= 5:
            wilcoxon_p[j] = mannwhitneyu(xd_nz, xh_nz, alternative="two-sided").pvalue
    return {"fisher_p": fisher_p, "wilcoxon_p": wilcoxon_p}


def cooccurrence_analysis(X_raw: np.ndarray, X_clr: np.ndarray, stable_idx: list,
                          feat_names: list, alpha: float = 0.05) -> dict:
    """入选标志物两两 Spearman 相关 + Fisher 共现/互斥检验。"""
    n = len(stable_idx)
    spearman_matrix = {}
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            ia, ib = stable_idx[a], stable_idx[b]
            # 非零样本（两特征均非零）上的 Spearman 相关（CLR 丰度）
            both_nz = (X_raw[:, ia] > 0) & (X_raw[:, ib] > 0)
            if both_nz.sum() < 5:
                rho = np.nan
            else:
                rho = spearmanr(X_clr[both_nz, ia], X_clr[both_nz, ib])[0]
            spearman_matrix[(feat_names[ia], feat_names[ib])] = float(rho) if not np.isnan(rho) else None
            # Fisher 精确检验：存在/缺失独立性
            both = both_nz.sum()
            only_a = ((X_raw[:, ia] > 0) & (X_raw[:, ib] == 0)).sum()
            only_b = ((X_raw[:, ia] == 0) & (X_raw[:, ib] > 0)).sum()
            neither = ((X_raw[:, ia] == 0) & (X_raw[:, ib] == 0)).sum()
            table = [[both, only_a], [only_b, neither]]
            or_val, p_val = fisher_exact(table, alternative="two-sided")
            if p_val < alpha and not np.isnan(or_val):
                etype = "cooccur" if or_val > 1 else "exclude"
                edges.append({
                    "feature_a": feat_names[ia], "feature_b": feat_names[ib],
                    "type": etype, "spearman": float(rho) if not np.isnan(rho) else None,
                    "fisher_p": float(p_val), "odds_ratio": float(or_val),
                })
    return {"spearman_matrix": spearman_matrix, "cooccurrence_edges": edges}


def rf_permutation_importance(X_raw: np.ndarray, y: np.ndarray, feat_names: list,
                               n_estimators: int = 500, n_repeats: int = 10) -> dict:
    """RF permutation importance（原始丰度，免 CLR）。"""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED, n_jobs=-1)
    rf.fit(X_raw, y)
    perm = permutation_importance(rf, X_raw, y, n_repeats=n_repeats,
                                  random_state=SEED, n_jobs=-1, scoring="roc_auc")
    return {feat_names[j]: float(perm.importances_mean[j]) for j in range(len(feat_names))}


def topN_consistency(freq: np.ndarray, rf_imp: dict, vip: dict, stable_idx: list,
                     feat_names: list, N: int = 20) -> dict:
    """RF/VIP 与 Lasso 稳定特征的 Top-N 一致性。"""
    rf_rank = np.argsort([rf_imp[f] for f in feat_names])[::-1]
    vip_rank = np.argsort([vip[f] for f in feat_names])[::-1]
    stable_set = set(stable_idx)
    rf_top = set(rf_rank[:N])
    vip_top = set(vip_rank[:N])
    rf_overlap = len(stable_set & rf_top) / N
    vip_overlap = len(stable_set & vip_top) / N
    # Spearman 秩相关：频率 vs RF 重要性、频率 vs VIP（全 264 特征）
    rf_vals = np.array([rf_imp[f] for f in feat_names])
    vip_vals = np.array([vip[f] for f in feat_names])
    spearman_rf = spearmanr(freq, rf_vals)[0]
    spearman_vip = spearmanr(freq, vip_vals)[0]
    return {
        "rf_overlap": float(rf_overlap), "vip_overlap": float(vip_overlap),
        "spearman_rank_rf": float(spearman_rf), "spearman_rank_vip": float(spearman_vip),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_IN, "rb") as f:
        prep = pickle.load(f)

    feat_names = prep["feature_names"]
    per_disease = prep["per_disease"]
    diseases = list(per_disease.keys())

    results = {"per_disease": {}, "cross_disease": {}, "meta": {}}
    tau_counts = {d: [] for d in diseases}

    # 图 1：频率直方图
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, disease in zip(axes1, diseases):
        dd = per_disease[disease]
        X_raw = dd["X_raw"]
        X_clr = dd["X_clr"]
        y = dd["y"]
        folds = dd["cv_folds"]

        # 1. 全量 bootstrap 频率（乐观）
        freq_full = bootstrap_frequency(X_clr, y, B_FULL, C_LASSO, SEED, N_JOBS)
        # 2. CV 折内 bootstrap 频率（诚实）
        freq_cv = cv_foldin_frequency(X_clr, y, folds, B_CV, C_LASSO, SEED, N_JOBS)

        # 3. 稳定特征（全量频率 ≥ τ）
        stable_idx = np.where(freq_full >= TAU)[0]
        stable_order = stable_idx[np.argsort(freq_full[stable_idx])[::-1]]
        stable_features = [
            {"feature": feat_names[i], "frequency": float(freq_full[i]),
             "cv_frequency": float(freq_cv[i]), "rank": r + 1}
            for r, i in enumerate(stable_order)
        ]

        # 4. 两路信号检验（全 264 特征，FDR m=1331）
        tp = two_path_tests(X_raw, X_clr, y)
        fisher_q = bh_qvalues(tp["fisher_p"], m=FDR_M)
        wilcoxon_q = bh_qvalues(tp["wilcoxon_p"], m=FDR_M)

        # 5. 标志物表（稳定特征，Top-N）
        top_idx = stable_order[:TOP_N]
        biomarker_table = []
        two_path_signals = []
        for i in top_idx:
            mean_d = X_clr[y == 1, i].mean()
            mean_h = X_clr[y == 0, i].mean()
            direction = "up" if mean_d > mean_h else "down"
            presence_strength = -np.log10(max(tp["fisher_p"][i], 1e-300))
            abundance_strength = -np.log10(max(tp["wilcoxon_p"][i], 1e-300))
            dominant = "presence" if presence_strength > abundance_strength else "abundance"
            two_path_signals.append({
                "feature": feat_names[i],
                "fisher_p": float(tp["fisher_p"][i]), "fisher_fdr": float(fisher_q[i]),
                "wilcoxon_p": float(tp["wilcoxon_p"][i]), "wilcoxon_fdr": float(wilcoxon_q[i]),
                "direction": direction, "dominant_signal": dominant,
            })
            biomarker_table.append({
                "feature": feat_names[i], "frequency": float(freq_full[i]),
                "fisher_fdr": float(fisher_q[i]), "wilcoxon_fdr": float(wilcoxon_q[i]),
                "direction": direction, "known_biomarker": is_known_biomarker(feat_names[i]),
            })

        # 6. 共现分析（入选稳定标志物）
        cooc = cooccurrence_analysis(X_raw, X_clr, list(stable_order), feat_names)

        # 7. RF permutation importance + PLS-DA VIP
        rf_imp = rf_permutation_importance(X_raw, y, feat_names)
        Xs = StandardScaler().fit_transform(X_clr)
        vip = vip_scores(Xs, y)
        vip_dict = {feat_names[j]: float(vip[j]) for j in range(len(feat_names))}

        # 8. Top-N 一致性
        consistency = topN_consistency(freq_full, rf_imp, vip_dict, list(stable_order),
                                       feat_names, N=TOP_N)

        # 9. τ 敏感性
        for tau in TAU_GRID:
            tau_counts[disease].append(int((freq_full >= tau).sum()))

        results["per_disease"][disease] = {
            "stable_features": stable_features,
            "two_path_signals": two_path_signals,
            "biomarker_table": biomarker_table,
            "cooccurrence": cooc,
            "rf_importance": rf_imp,
            "vip": vip_dict,
            "topN_consistency": consistency,
            "full_frequency": {feat_names[j]: float(freq_full[j]) for j in range(len(feat_names))},
            "cv_frequency": {feat_names[j]: float(freq_cv[j]) for j in range(len(feat_names))},
            "n_stable": int(len(stable_idx)),
            "n_fisher_sig": int((fisher_q < FDR_ALPHA).sum()),
            "n_wilcoxon_sig": int((wilcoxon_q < FDR_ALPHA).sum()),
        }

        # 图 1：频率直方图
        ax.hist(freq_full, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
        ax.axvline(TAU, color="red", ls="--", lw=1, label=f"τ={TAU}")
        ax.set_xlabel("入选频率")
        ax.set_ylabel("特征数")
        ax.set_title(f"{disease}  (稳定 {len(stable_idx)} 特征)")
        ax.legend(fontsize=8)

    fig1.suptitle(f"S2 2.1 Lasso bootstrap 入选频率分布（B={B_FULL}）", fontsize=13)
    fig1.tight_layout()
    out1 = FIG_DIR / "S2-2.1-stability-frequency-explore.pdf"
    fig1.savefig(out1)
    plt.close(fig1)

    # 图 2：τ 敏感性曲线
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for disease in diseases:
        ax2.plot(TAU_GRID, tau_counts[disease], marker="o", label=disease)
    ax2.set_xlabel("τ（稳定频率阈值）")
    ax2.set_ylabel("入选特征数")
    ax2.set_title("τ 敏感性：入选特征数随 τ 变化")
    ax2.legend()
    fig2.tight_layout()
    out2 = FIG_DIR / "S2-2.1-tau-sensitivity-explore.pdf"
    fig2.savefig(out2)
    plt.close(fig2)

    # 图 3：共现 Spearman 热图（每病）
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, disease in zip(axes3, diseases):
        dd = results["per_disease"][disease]
        stable = dd["stable_features"]
        n = len(stable)
        mat = np.full((n, n), np.nan)
        labels = [short_name(s["feature"]) for s in stable]
        feat_to_idx = {s["feature"]: i for i, s in enumerate(stable)}
        for (fa, fb), rho in dd["cooccurrence"]["spearman_matrix"].items():
            if fa in feat_to_idx and fb in feat_to_idx:
                ia = feat_to_idx[fa]
                ib = feat_to_idx[fb]
                mat[ia, ib] = mat[ib, ia] = rho
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"{disease} 共现 Spearman（{n} 标志物）")
        fig3.colorbar(im, ax=ax, fraction=0.046)
    fig3.suptitle("S2 2.1 入选标志物两两 Spearman 相关（非零样本 CLR 丰度）", fontsize=13)
    fig3.tight_layout()
    out3 = FIG_DIR / "S2-2.1-cooccurrence-heatmap-explore.pdf"
    fig3.savefig(out3)
    plt.close(fig3)

    # 跨疾病对比
    stable_sets = {}
    for disease in diseases:
        stable_sets[disease] = set(s["feature"] for s in results["per_disease"][disease]["stable_features"])
    jaccard = {}
    for i, d1 in enumerate(diseases):
        for d2 in diseases[i + 1:]:
            a, b = stable_sets[d1], stable_sets[d2]
            jaccard[f"{d1}_{d2}"] = len(a & b) / len(a | b) if (a | b) else 0.0
    common = set.intersection(*stable_sets.values()) if stable_sets else set()
    disease_specific = {}
    for disease in diseases:
        others = set.union(*[stable_sets[d] for d in diseases if d != disease])
        disease_specific[disease] = sorted(stable_sets[disease] - others)

    results["cross_disease"] = {
        "jaccard_matrix": jaccard,
        "common_biomarkers": sorted(common),
        "disease_specific": disease_specific,
    }

    # meta
    results["meta"] = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": DATA_IN.name,
        "note": "S2 2.1 正式模型实现：Lasso+bootstrap 稳定性选择 + 两路信号 + 共现 + RF/VIP 佐证",
        "filter_threshold": prep["meta"]["filter_threshold"], "tau": TAU, "B_full": B_FULL, "B_cv": B_CV,
        "C_lasso": C_LASSO, "fdr_alpha": FDR_ALPHA, "fdr_m": FDR_M,
        "vip_threshold": VIP_THRESHOLD, "clr_delta": prep["meta"]["clr_delta"], "top_n": TOP_N,
        "tau_grid": TAU_GRID, "tau_counts": tau_counts,
        "full_vs_cv": {"full": "全量 bootstrap 频率（乐观）", "cv": "CV 折内 bootstrap 频率（诚实）"},
        "field_semantics": {
            "stable_features.frequency": "全量 bootstrap 入选频率 π̂_j（乐观，B=100）",
            "stable_features.cv_frequency": "CV 折内 bootstrap 入选频率（诚实，5 折 × B=50 平均）",
            "two_path_signals.fisher_fdr": "Fisher 精确检验 BH-FDR q 值（m=1331 全特征规模）",
            "two_path_signals.wilcoxon_fdr": "Wilcoxon 秩和检验 BH-FDR q 值（m=1331 全特征规模）",
            "two_path_signals.direction": "up=患病组 CLR 丰度均值高于健康组，down=低于",
            "two_path_signals.dominant_signal": "presence=存在/缺失信号主导，abundance=非零丰度信号主导",
            "cooccurrence.cooccurrence_edges.type": "cooccur=显著共现(OR>1)，exclude=显著互斥(OR<1)",
            "rf_importance": "RF permutation importance（原始丰度，免 CLR，n_repeats=10）",
            "vip": "PLS-DA VIP 得分（2 成分，阈值>1.5）",
            "n_fisher_sig": "Fisher 检验 BH-FDR(m=1331) 显著特征数（全 264 特征口径）",
            "n_wilcoxon_sig": "Wilcoxon 检验 BH-FDR(m=1331) 显著特征数（全 264 特征口径）",
        },
    }

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT, "wb") as f:
        pickle.dump(results, f, protocol=4)

    # stdout 摘要
    print("=" * 70)
    print("S2 2.1 正式模型实现结果摘要")
    print("=" * 70)
    for disease in diseases:
        dd = results["per_disease"][disease]
        print(f"\n[{disease}] 稳定特征数={dd['n_stable']}  "
              f"Fisher显著={dd['n_fisher_sig']}  Wilcoxon显著={dd['n_wilcoxon_sig']}")
        print(f"  共现边数={len(dd['cooccurrence']['cooccurrence_edges'])}  "
              f"RF重叠={dd['topN_consistency']['rf_overlap']:.2f}  "
              f"VIP重叠={dd['topN_consistency']['vip_overlap']:.2f}")
        print("  Top 标志物:")
        for bm in dd["biomarker_table"][:10]:
            print(f"    {short_name(bm['feature']):40s} freq={bm['frequency']:.2f} "
                  f"fisher_q={bm['fisher_fdr']:.2e} wilcoxon_q={bm['wilcoxon_fdr']:.2e} "
                  f"{bm['direction']} {'[已知]' if bm['known_biomarker'] else ''}")
    print("\n[τ 敏感性]")
    for disease in diseases:
        print(f"  {disease}: " + "  ".join(f"τ={t}:{c}" for t, c in zip(TAU_GRID, tau_counts[disease])))
    print(f"\n[跨疾病 Jaccard] {results['cross_disease']['jaccard_matrix']}")
    print(f"[共同标志物] {len(results['cross_disease']['common_biomarkers'])} 个")
    print(f"\n[输出] {DATA_OUT}")
    print(f"[图] {out1}")
    print(f"[图] {out2}")
    print(f"[图] {out3}")


if __name__ == "__main__":
    main()
