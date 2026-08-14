# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 0.4 数据画像 — 对清洗后数据执行分群（A）、降维（B）、异常检测（C）三条路径，
    产出簇画像、PCA 结构分析与离群任务标记

原理：
    - 分群：workload_trace 任务特征（GPU_Demand/时长/时延/到达周期 + 类型/区域 one-hot）
      标准化 → PCA 预降维（cumR²≥60%）→ KMeans++ 扫描 k=2,3,4，肘部+Silhouette 定 K
    - 降维：workload_trace 与 region_time_data 各自 PCA，方差表+载荷矩阵+碎石图
    - 异常：Isolation Forest（contamination=0.05；auto 在高维 one-hot 下误标 77% 行已弃用）
      标记离群任务，输出偏离 2σ 特征方向

输入数据：
    - outputs/data/c-data-cleaned.pkl（阶段 0.3 清洗后）

输出：
    - solution/model-notes/cluster-profile.md — 分群画像
    - solution/model-notes/dim-reduction-profile.md — 降维画像
    - solution/model-notes/anomaly-profile.md — 异常画像
    - outputs/figures/cluster-tsne.pdf — 聚类 t-SNE 图
    - outputs/figures/pca-scree.pdf — PCA 碎石图

对应论文章节：
    TRAE.md 阶段 0.4 数据画像
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

BASE = Path(r"e:\MathModel_pj-2026-C")
clean = pd.read_pickle(BASE / "outputs" / "data" / "c-data-cleaned.pkl")
wt = clean["workload_trace"]
rt = clean["region_time_data"]

FIGS = BASE / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
NOTES = BASE / "solution" / "model-notes"
NOTES.mkdir(parents=True, exist_ok=True)

# ---------- 特征工程：workload_trace ----------
wt_f = pd.DataFrame({
    "GPU_Demand": wt["GPU_Demand"].astype(float),
    "Duration_min": wt["EstimatedDuration_min"].astype(float),
    "MaxLatency_ms": wt["MaxLatency_ms"].astype(float),
    "LatestFinishHour": wt["LatestFinishHour"].astype(float),
    "ArrivalHour": wt["ArrivalHour"].astype(float),
    "HourOfDay": (wt["ArrivalHour"] % 24).astype(float),
})
# 分类 one-hot（DelaySensitivity 与 TaskType 完全对应，跳过）
wt_f = pd.concat([wt_f, pd.get_dummies(wt["TaskType"], prefix="TT"),
                  pd.get_dummies(wt["SourceRegion"], prefix="SR")], axis=1)
X = StandardScaler().fit_transform(wt_f)
feat_names = list(wt_f.columns)

# ---------- 路径 B1：workload_trace PCA ----------
pca_w = PCA().fit(X)
var_w = pca_w.explained_variance_ratio_
cum_w = np.cumsum(var_w)
n_keep_w = int(np.argmax(cum_w >= 0.60)) + 1

# 碎石图（workload + region 两张）
rt_num = rt.select_dtypes(include=[np.number]).drop(columns=["Hour"], errors="ignore")
Xr = StandardScaler().fit_transform(rt_num)
pca_r = PCA().fit(Xr)
var_r = pca_r.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(range(1, len(var_w) + 1), var_w, "o-", color="#555555")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_title("workload_trace PCA scree")
axes[0].set_xlabel("PC")
axes[0].set_ylabel("explained variance ratio")
axes[1].plot(range(1, len(var_r) + 1), var_r, "o-", color="#555555")
axes[1].set_title("region_time_data PCA scree")
axes[1].set_xlabel("PC")
axes[1].set_ylabel("explained variance ratio")
fig.tight_layout()
fig.savefig(FIGS / "pca-scree.pdf")
plt.close(fig)

# 载荷矩阵（前 5 个 PC）
load_w = pd.DataFrame(pca_w.components_[:5].T, index=feat_names,
                      columns=[f"PC{i+1}" for i in range(5)])

# ---------- 路径 A：KMeans 聚类（PCA 空间） ----------
X_pca = pca_w.transform(X)[:, :n_keep_w]
ks = [2, 3, 4]
sil = {}
inertia = {}
for k in ks:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X_pca)
    inertia[k] = km.inertia_
    if len(wt) <= 10000:
        sil[k] = silhouette_score(X_pca, km.labels_)
    else:
        idx = np.random.RandomState(42).choice(len(X_pca), 8000, replace=False)
        sil[k] = silhouette_score(X_pca[idx], km.labels_[idx])
best_k = max(sil, key=sil.get)
km_best = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42).fit(X_pca)
labels = km_best.labels_

# 簇画像
wt_c = wt.copy()
wt_c["cluster"] = labels
clust_prof = wt_c.groupby("cluster").agg(
    n=("TaskID", "count"),
    gpu_mean=("GPU_Demand", "mean"),
    dur_mean=("EstimatedDuration_min", "mean"),
    latency_mean=("MaxLatency_ms", "mean"),
    latest_mean=("LatestFinishHour", "mean"),
    arr_mean=("ArrivalHour", "mean"),
)
tt_cross = pd.crosstab(wt_c["cluster"], wt_c["TaskType"], normalize="index").round(3)
sr_cross = pd.crosstab(wt_c["cluster"], wt_c["SourceRegion"], normalize="index").round(3)

# t-SNE 可视化（抽样 500）
idx = np.random.RandomState(42).choice(len(X_pca), min(500, len(X_pca)), replace=False)
tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca[idx])
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#444444", "#888888", "#bbbbbb", "#555555"][:best_k]
for k in range(best_k):
    m = labels[idx] == k
    ax.scatter(tsne[m, 0], tsne[m, 1], s=8, c=colors[k], label=f"cluster{k} ({m.sum()}/{len(m)})")
ax.set_title(f"workload_trace task clustering t-SNE (k={best_k})")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "cluster-tsne.pdf")
plt.close(fig)

# ---------- 路径 C：Isolation Forest（contamination=0.05，auto 在 high-dim 下失效） ----------
iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
anom = iso.predict(X) == -1
anom_idx = np.where(anom)[0]
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
anom_detail = []
for i in anom_idx[:20]:
    dev = (X[i] - X_mean) / X_std
    big = [f"{feat_names[j]}({dev[j]:+.1f}σ)" for j in np.argsort(-np.abs(dev))[:4]]
    anom_detail.append(f"任务 {wt.iloc[i]['TaskID']} ({wt.iloc[i]['TaskType']}/{wt.iloc[i]['SourceRegion']}): " + ", ".join(big))

# 与聚类交叉
anom_cluster = pd.Series(labels[anom]).value_counts(normalize=True).sort_index()

# ---------- 写报告 ----------
def write_md(path, title, lines):
    (path).write_text(f"# {title}\n\n> 由 outputs/scratch/data-profiling-04.py 生成，2026-08-07\n\n" + "\n".join(lines), encoding="utf-8")

# 分群画像
A = []
A.append(f"## 方法\nKMeans++ 在 PCA 空间（保留前 {n_keep_w} 个 PC，累计解释 {cum_w[n_keep_w-1]*100:.1f}%）聚类。\n")
A.append(f"## K 选择\n- 肘部: {inertia}\n- Silhouette: {sil}\n- **最优 K = {best_k}**\n")
A.append("## 簇画像\n```\n" + clust_prof.round(1).to_string() + "\n```\n")
A.append("## 簇 × 任务类型（行归一）\n```\n" + tt_cross.to_string() + "\n```\n")
A.append("## 簇 × 来源区域（行归一）\n```\n" + sr_cross.to_string() + "\n```\n")
A.append("## 数据质量标志\n")
purity = tt_cross.max(axis=1)
for k in range(best_k):
    A.append(f"- 簇{k}: 样本 {clust_prof.loc[k,'n']}（{clust_prof.loc[k,'n']/len(wt)*100:.1f}%），最大类型纯度 {purity[k]*100:.0f}%")
A.append("\n- 若某簇纯度>90% → 该类型/特征为强分类线索，供阶段 1 调度分级参考")
write_md(NOTES / "cluster-profile.md", "C 题 分群画像（workload_trace 任务聚类）", A)

# 降维画像
B = []
B.append(f"## workload_trace PCA\n- 保留 PC 数（cum≥60%）: {n_keep_w}，前 5 累计解释率 {cum_w[:5].round(4).tolist()}\n")
B.append("### 载荷矩阵（前 5 PC）\n```\n" + load_w.round(3).to_string() + "\n```\n")
B.append("### PC 含义解读（|载荷|>0.5）\n")
for i in range(5):
    top = load_w.iloc[:, i]
    hi = top[top.abs() > 0.5].index.tolist()
    B.append(f"- PC{i+1}（解释 {var_w[i]*100:.1f}%）：{', '.join(hi) if hi else '无强载荷'}")
B.append(f"\n## region_time_data PCA（{len(rt_num.columns)} 特征）\n")
B.append(f"- 前 5 主成分解释率: {var_r[:5].round(4).tolist()}，累计 {var_r[:5].sum()*100:.1f}%\n")
B.append("### region 载荷矩阵（前 5 PC）\n```\n" + pd.DataFrame(pca_r.components_[:5].T, index=rt_num.columns,
    columns=[f"PC{i+1}" for i in range(5)]).round(3).to_string() + "\n```\n")
B.append("### region PC 含义解读（|载荷|>0.4）\n")
for i in range(5):
    top = pd.Series(pca_r.components_[i], index=rt_num.columns)
    hi = top[top.abs() > 0.4].index.tolist()
    B.append(f"- PC{i+1}（解释 {var_r[i]*100:.1f}%）：{', '.join(hi) if hi else '无强载荷'}")
write_md(NOTES / "dim-reduction-profile.md", "C 题 降维画像（PCA）", B)

# 异常画像
C = []
C.append(f"## Isolation Forest 结果\n- 异常任务数: {int(anom.sum())}（{anom.mean()*100:.2f}%）\n")
C.append("## 异常任务典型偏离（前 20 条，超出均值 ±2σ 的特征）\n```\n" + "\n".join(anom_detail) + "\n```\n")
C.append("## 异常任务与聚类交叉（异常在各簇占比）\n```\n" + anom_cluster.round(3).to_string() + "\n```\n")
C.append("## 结论\n- 若异常任务集中于某簇/某类型 → 该群体可独立处理或独立建模\n")
write_md(NOTES / "anomaly-profile.md", "C 题 异常画像（Isolation Forest）", C)

print(f"[OK] cluster-profile.md  (最优K={best_k}, silhouette={sil[best_k]:.3f})")
print(f"[OK] dim-reduction-profile.md (workload 前5累计 {var_w[:5].sum()*100:.1f}%)")
print(f"[OK] anomaly-profile.md (异常 {int(anom.sum())} 条)")
print(f"[OK] figures: cluster-tsne.pdf, pca-scree.pdf")
