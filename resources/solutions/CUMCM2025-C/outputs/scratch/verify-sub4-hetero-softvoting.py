#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异质基分类器 Soft-Voting 实验（子问题4）
挑选 5 种异质分类器：Fisher LDA / Logistic(L2) / 阈值规则 / KNN / 朴素贝叶斯
"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
from scipy import stats

cache = r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data'

# ===== 加载 =====
sub4 = pd.read_pickle(os.path.join(cache, '2025C-sub4-preprocessed.pkl'))
clean = pd.read_pickle(os.path.join(cache, '2025C-female-clean.pkl'))
for orig, new in {'13号染色体的GC含量':'GC_13','18号染色体的GC含量':'GC_18','21号染色体的GC含量':'GC_21'}.items():
    sub4[new] = clean[orig].values
sub4['X_conc'] = clean['X染色体浓度'].values
sub4['孕妇代码'] = clean['孕妇代码'].values

# 去重
dm = clean[['孕妇代码','检测日期_std']].drop_duplicates('孕妇代码')
sub4['检测日期_std'] = dm.set_index('孕妇代码').loc[sub4['孕妇代码'],'检测日期_std'].values
mask = (sub4['AB_异常']==0)|(sub4['AB_异常']==1)
lb = sub4[mask].sort_values('检测日期_std').drop_duplicates('孕妇代码',keep='first')
idx_dedup = lb.index.values
y = lb['AB_异常'].values.astype(float)

# 每条染色体的异常标签
ab_types = clean.loc[idx_dedup, '染色体的非整倍体'].values
def has_type(s,t): return t in str(s) if not pd.isna(s) and s!='' else False
is_t13 = np.array([has_type(s,'T13') for s in ab_types])
is_t18 = np.array([has_type(s,'T18') for s in ab_types])
is_t21 = np.array([has_type(s,'T21') for s in ab_types])

# ===== 特征工程（与 Fisher LDA 一致） =====
Z = sub4[['Z13_corrected','Z18_corrected','Z21_corrected','ZX_corrected']].values
Z_diff = np.zeros((len(sub4),4))
for k in range(4):
    other = [j for j in range(4) if j!=k]
    Z_diff[:,k] = Z[:,k] - np.median(Z[:,other], axis=1)

GC13,GC18,GC21 = sub4['GC_13'].values, sub4['GC_18'].values, sub4['GC_21'].values
GC_18_13 = GC18 - GC13
GC_21_13 = GC21 - GC13
GC_18_21 = GC18 - GC21

# 构造统一特征矩阵（12 维）
F_all = np.column_stack([
    Z[:,0], Z[:,1], Z[:,2], Z[:,3],           # 0-3: Z_corrected
    Z_diff[:,0], Z_diff[:,1], Z_diff[:,2],     # 4-6: Z contrast
    GC13, GC18, GC21,                          # 7-9: GC per chr
    GC_18_13, GC_21_13, GC_18_21,              # 10-12: GC diff
    sub4['被过滤掉读段数的比例'].values,       # 13: filter_rate
    sub4['重复读段的比例'].values,             # 14: dup_rate
    sub4['在参考基因组上比对的比例'].values,   # 15: map_rate
    sub4['X_conc'].values,                     # 16: X_conc
    sub4['孕妇BMI'].values,                    # 17: BMI
    sub4['年龄'].values,                       # 18: age
    sub4['孕周_数值'].values,                  # 19: gw
    sub4['孕妇BMI'].values * Z[:,1],           # 20: BMI*Z18
    sub4['年龄'].values * Z[:,1],              # 21: age*Z18
])

F = F_all[idx_dedup]  # 147 × 22
F_all_nonan = np.nan_to_num(F_all, 0)
F_nonan = np.nan_to_num(F, 0)

# 标准化
F_mean = F_nonan.mean(0); F_std = F_nonan.std(0); F_std[F_std<1e-10]=1.0
F_scaled = (F_nonan - F_mean) / F_std
F_all_scaled = (F_all_nonan - F_mean) / F_std

n,d = F_scaled.shape
n_pos = int(y.sum())
print(f"样本: {n} (正常={n-n_pos}, 异常={n_pos})")
print(f"T13={is_t13.sum()}, T18={is_t18.sum()}, T21={is_t21.sum()}")

# ===== LOOCV 框架 =====
def roc_auc_manual(yt, ys):
    o = np.argsort(-ys); ys_s = yt[o]
    tp, fn = ys_s.sum(), len(ys_s)-ys_s.sum()
    if tp==0 or fn==0: return np.nan
    tpr = np.cumsum(ys_s)/tp; fpr = np.cumsum(1-ys_s)/fn
    return np.trapezoid(tpr,fpr)

# ============================================================
# 基分类器 1: Fisher LDA（现有，按染色体拆分 + max）
# ============================================================
print("\n" + "="*60)
print("基分类器 1: Fisher LDA (按染色体拆分 + max)")
print("="*60)

def fisher_lda_fit(X_pos, X_neg):
    mu1,m0 = X_pos.mean(0), X_neg.mean(0)
    S1 = np.cov(X_pos,rowvar=False) if len(X_pos)>1 else np.eye(X_pos.shape[1])
    S0 = np.cov(X_neg,rowvar=False)
    Sw = ((len(X_pos)-1)*S1 + (len(X_neg)-1)*S0)/(len(X_pos)+len(X_neg)-2)
    reg = 0.01*np.trace(Sw)/Sw.shape[0]
    try: Sw_inv = np.linalg.inv(Sw+reg*np.eye(Sw.shape[0]))
    except: Sw_inv = np.linalg.pinv(Sw+reg*np.eye(Sw.shape[0]))
    w = Sw_inv @ (mu1 - m0)
    return w

# 每条染色体选特征: Z_chr, Z_contrast_chr, GC_chr, GC_diffs, filter,dup,map, X_conc, BMI, age, gw
chr_feat_idx = {
    13: [0,4,7,10,11,13,14,15,16,17,18,19],
    18: [1,5,8,10,12,13,14,15,16,17,18,19,20,21],
    21: [2,6,9,11,12,13,14,15,16,17,18,19],
}

lda_scores_loocv = np.zeros(n)
for k,feat_idx in chr_feat_idx.items():
    y_k = {'13':is_t13,'18':is_t18,'21':is_t21}[str(k)].astype(float)
    for i in range(n):
        mask_train = np.ones(n,bool); mask_train[i]=False
        X_tr = F_scaled[mask_train][:,feat_idx]; y_tr = y_k[mask_train]
        if y_tr.sum()<2: 
            lda_scores_loocv[i] += 0; continue
        w = fisher_lda_fit(X_tr[y_tr==1], X_tr[y_tr==0])
        lda_scores_loocv[i] = max(lda_scores_loocv[i], F_scaled[i,feat_idx]@w)

auc_lda = roc_auc_manual(y, lda_scores_loocv)
print(f"LOOCV AUC = {auc_lda:.4f}")

# ============================================================
# 基分类器 2: Logistic 回归 (L2 正则化, 梯度下降)
# ============================================================
print("\n基分类器 2: Logistic 回归 (L2 正则化)")

# 选 Top-4 单特征 Cohen's d 最大的特征
def cohens_d(x1,x2):
    m1,m2=x1.mean(),x2.mean()
    v1,v2=x1.var(ddof=1),x2.var(ddof=1)
    s=np.sqrt(((len(x1)-1)*v1+(len(x2)-1)*v2)/(len(x1)+len(x2)-2))
    return abs(m1-m2)/s if s>1e-12 else 0

d_scores = [(j, cohens_d(F_scaled[y==1,j], F_scaled[y==0,j])) for j in range(d)]
d_scores.sort(key=lambda x:-x[1])
top4_idx = [j for j,_ in d_scores[:4]]
print(f"Top-4 特征 (Cohen's d): {top4_idx}")

def logistic_fit(X,y,lam=1.0,lr=0.01,n_iter=2000):
    """L2正则化逻辑回归"""
    n,d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(n_iter):
        z = X@w + b
        p = 1/(1+np.exp(-np.clip(z,-50,50)))
        dw = (X.T@(p-y))/n + lam*w/n
        db = (p-y).mean()
        w -= lr*dw; b -= lr*db
    return w,b

log_scores_loocv = np.zeros(n)
for i in range(n):
    mask_tr = np.ones(n,bool); mask_tr[i]=False
    w_l,b_l = logistic_fit(F_scaled[mask_tr][:,top4_idx], y[mask_tr])
    log_scores_loocv[i] = 1/(1+np.exp(-(F_scaled[i,top4_idx]@w_l + b_l)))

auc_log = roc_auc_manual(y, log_scores_loocv)
print(f"LOOCV AUC = {auc_log:.4f}")

# ============================================================
# 基分类器 3: 阈值规则（非学习型，GC差异 + X_conc）
# ============================================================
print("\n基分类器 3: 阈值规则 (GC差异 + X_conc)")

# GC_18-13 越大越异常（Cohen's d 最高 0.92），X_conc 越大越异常
thr_scores_loocv = np.zeros(n)
for i in range(n):
    mask_tr = np.ones(n,bool); mask_tr[i]=False
    g18_13_tr = GC_18_13[idx_dedup][mask_tr]
    xc_tr = sub4['X_conc'].values[idx_dedup][mask_tr]

    # 规则: 分数 = (GC18-13 - 正常组P95) + (X_conc - 正常组P95) 的正部
    g_p95 = np.percentile(g18_13_tr[y[mask_tr]==0], 95) if (y[mask_tr]==0).sum()>0 else 0
    x_p95 = np.percentile(xc_tr[y[mask_tr]==0], 90) if (y[mask_tr]==0).sum()>0 else 0

    s1 = max(0, GC_18_13[idx_dedup][i] - g_p95)
    s2 = max(0, sub4['X_conc'].values[idx_dedup][i] - x_p95)
    thr_scores_loocv[i] = s1 + s2

auc_thr = roc_auc_manual(y, thr_scores_loocv)
print(f"LOOCV AUC = {auc_thr:.4f}")

# ============================================================
# 基分类器 4: KNN (k=3)
# ============================================================
print("\n基分类器 4: KNN (k=3)")

knn_scores_loocv = np.zeros(n)
k_knn = 3
for i in range(n):
    mask_tr = np.ones(n,bool); mask_tr[i]=False
    # 用 Top-4 特征
    X_tr = F_scaled[mask_tr][:,top4_idx]
    y_tr = y[mask_tr]
    X_te = F_scaled[i,top4_idx].reshape(1,-1)

    # 欧氏距离
    dists = np.sqrt(((X_tr - X_te)**2).sum(1))
    nn = np.argsort(dists)[:k_knn]
    # 输出: k 近邻中正样本比例
    knn_scores_loocv[i] = y_tr[nn].mean()

auc_knn = roc_auc_manual(y, knn_scores_loocv)
print(f"LOOCV AUC = {auc_knn:.4f}")

# ============================================================
# 基分类器 5: 高斯朴素贝叶斯
# ============================================================
print("\n基分类器 5: 高斯朴素贝叶斯")

nb_scores_loocv = np.zeros(n)
for i in range(n):
    mask_tr = np.ones(n,bool); mask_tr[i]=False
    X_tr = F_scaled[mask_tr][:,top4_idx]; y_tr = y[mask_tr]
    # 估计每类均值/方差
    mu0 = X_tr[y_tr==0].mean(0); var0 = X_tr[y_tr==0].var(0,ddof=1)
    mu1 = X_tr[y_tr==1].mean(0); var1 = X_tr[y_tr==1].var(0,ddof=1)
    var0[var0<1e-10]=1e-10; var1[var1<1e-10]=1e-10

    # 对数似然比
    x_i = F_scaled[i,top4_idx]
    ll0 = -0.5*np.sum((x_i-mu0)**2/var0 + np.log(2*np.pi*var0))
    ll1 = -0.5*np.sum((x_i-mu1)**2/var1 + np.log(2*np.pi*var1))
    # 先验: n1/n
    prior1 = y_tr.mean()
    # 后验概率 p(y=1|x)
    log_odds = ll1 - ll0 + np.log(prior1/(1-prior1))
    nb_scores_loocv[i] = 1/(1+np.exp(-np.clip(log_odds,-50,50)))

auc_nb = roc_auc_manual(y, nb_scores_loocv)
print(f"LOOCV AUC = {auc_nb:.4f}")

# ============================================================
# 6. Soft-Voting 集成
# ============================================================
print("\n" + "="*60)
print("6. 异质 Soft-Voting 集成")
print("="*60)

scores = {
    'Fisher LDA': lda_scores_loocv,
    'Logistic(L2)': log_scores_loocv,
    '阈值规则': thr_scores_loocv,
    'KNN(k=3)': knn_scores_loocv,
    '朴素贝叶斯': nb_scores_loocv,
}

# 各分类器单独 AUC
print(f"\n{'基分类器':<18s} {'AUC':>8s}")
print("-"*30)
for name, sc in scores.items():
    print(f"{name:<18s} {roc_auc_manual(y,sc):8.4f}")

# 归一化到 [0,1]
def minmax(x):
    mn,mx = x.min(),x.max()
    return (x-mn)/(mx-mn+1e-10)

scores_norm = {k: minmax(v) for k,v in scores.items()}

# 多种 Soft-Voting 组合
print(f"\n{'组合方案':<30s} {'AUC':>8s}")
print("-"*42)

names = list(scores_norm.keys())
all_scores_mat = np.column_stack([scores_norm[n] for n in names])

# 等权平均
sv_mean = all_scores_mat.mean(1)
print(f"{'等权平均 (5个)':<30s} {roc_auc_manual(y,sv_mean):8.4f}")

# 排除表现最差的
sorted_aucs = sorted([(roc_auc_manual(y,v),k) for k,v in scores.items()], reverse=True)
for n_keep in [4,3,2]:
    keep_names = [k for _,k in sorted_aucs[:n_keep]]
    keep_mat = np.column_stack([scores_norm[n] for n in keep_names])
    sv = keep_mat.mean(1)
    print(f"{'等权平均 (Top'+str(n_keep)+'): '+','.join(keep_names):<30s} {roc_auc_manual(y,sv):8.4f}")

# 加权: 按各分类器 AUC 赋权
aucs = np.array([roc_auc_manual(y,v) for v in scores.values()])
w_auc = aucs / aucs.sum()
sv_weighted = (all_scores_mat @ w_auc)
print(f"{'加权 (按AUC)':<30s} {roc_auc_manual(y,sv_weighted):8.4f}")

# max voting
sv_max = all_scores_mat.max(1)
print(f"{'Max Voting':<30s} {roc_auc_manual(y,sv_max):8.4f}")

# ============================================================
# 7. 综合对比（含 Fisher LDA 基线）
# ============================================================
print("\n" + "="*60)
print("7. 综合对比")
print("="*60)

# 找最优组合
all_methods = {
    'Fisher LDA (基线)': lda_scores_loocv,
    'Logistic (L2)': log_scores_loocv,
    '阈值规则': thr_scores_loocv,
    'KNN (k=3)': knn_scores_loocv,
    '朴素贝叶斯': nb_scores_loocv,
    'SV-等权(5)': sv_mean,
    'SV-加权': sv_weighted,
    'SV-max': sv_max,
}

print(f"\n{'方法':<25s} {'AUC':>8s} {'Cohen d':>8s}")
print("-"*45)
for nm, sc in all_methods.items():
    d_val = cohens_d(sc[y==0], sc[y==1])
    print(f"{nm:<25s} {roc_auc_manual(y,sc):8.4f} {d_val:8.4f}")

# ============================================================
# 8. 最优方案的高灵敏度分析
# ============================================================
print("\n" + "="*60)
print("8. 最优方案高灵敏度分析 (TPR≥90%)")
print("="*60)

for nm, sc in [('Fisher LDA', lda_scores_loocv),
               ('SV-max(异质)', sv_max),
               ('SV-加权(异质)', sv_weighted)]:
    order = np.argsort(-sc)
    ys = y[order]; ss = sc[order]
    tp_total = ys.sum()
    tp = np.cumsum(ys); fp = np.cumsum(1-ys)
    tpr = tp/tp_total; fpr = fp/(len(ys)-tp_total)
    idx90 = np.where(tpr>=0.90)[0]
    if len(idx90)>0:
        i90 = idx90[0]
        print(f"  {nm:<22s}: TPR={tpr[i90]:.1%}, FPR={fpr[i90]:.1%}, 特异度={1-fpr[i90]:.1%}")

print("\n完成.")
