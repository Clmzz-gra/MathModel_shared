#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
突破口实验：外部先验 + 半监督（41条未标注Z异常样本）
"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')

cache = r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data'
sub4 = pd.read_pickle(os.path.join(cache, '2025C-sub4-preprocessed.pkl'))
clean = pd.read_pickle(os.path.join(cache, '2025C-female-clean.pkl'))
for orig, new in {'13号染色体的GC含量':'GC_13','18号染色体的GC含量':'GC_18','21号染色体的GC含量':'GC_21'}.items():
    sub4[new] = clean[orig].values
sub4['X_conc'] = clean['X染色体浓度'].values
sub4['孕妇代码'] = clean['孕妇代码'].values

# 去重
dm = clean[['孕妇代码','检测日期_std']].drop_duplicates('孕妇代码')
sub4['检测日期_std'] = dm.set_index('孕妇代码').loc[sub4['孕妇代码'],'检测日期_std'].values
mask_lb = (sub4['AB_异常']==0)|(sub4['AB_异常']==1)
lb = sub4[mask_lb].sort_values('检测日期_std').drop_duplicates('孕妇代码',keep='first')
idx_lb = lb.index.values
y = lb['AB_异常'].values.astype(float)

# ===== 特征工程（精简：只取最有区分力的） =====
GC18,GC13,GC21 = sub4['GC_18'].values, sub4['GC_13'].values, sub4['GC_21'].values
GC_18_13 = GC18 - GC13
GC_21_13 = GC21 - GC13
GC_18_21 = GC18 - GC21

# 核心特征矩阵（全量605条）
F_all = np.column_stack([
    sub4['Z13_corrected'].values,
    sub4['Z18_corrected'].values,
    sub4['Z21_corrected'].values,
    sub4['ZX_corrected'].values,
    GC_18_13, GC_21_13, GC_18_21,
    sub4['被过滤掉读段数的比例'].values,
    sub4['X_conc'].values,
    sub4['孕妇BMI'].values,
    sub4['孕周_数值'].values,
])
F_all = np.nan_to_num(F_all, 0)
F_lb = F_all[idx_lb]
n, d = F_lb.shape
print(f"标注样本: {n} (正常={int((y==0).sum())}, 异常={int(y.sum())})")

# ===== 识别 41 条 Z 异常未标注样本 =====
unlabeled_mask = ~mask_lb  # AB列空白
unlabeled = sub4[unlabeled_mask].sort_values('检测日期_std').drop_duplicates('孕妇代码',keep='first')
idx_ul = unlabeled.index.values

# Z>3 阈值（经典规则）
z_cols_orig = ['13号染色体的Z值','18号染色体的Z值','21号染色体的Z值']
z_has_alert = np.zeros(len(sub4), dtype=bool)
for zc in z_cols_orig:
    z_has_alert |= (np.abs(sub4[zc].values) > 3)

z_alert_ul = z_has_alert[idx_ul]
print(f"未标注样本: {len(idx_ul)} 人, 其中 Z>3: {z_alert_ul.sum()} 人")

# 有 Z 异常且未标注的样本索引
idx_z_ul = idx_ul[z_alert_ul]
n_zul = len(idx_z_ul)
print(f"Z异常未标注 (候选半监督): {n_zul} 人")

# ===== LOOCV 框架 =====
def roc_auc(yt, ys):
    o = np.argsort(-ys); ys_s = yt[o]
    tp = ys_s.sum()
    if tp==0 or tp==len(ys_s): return np.nan
    tpr = np.cumsum(ys_s)/tp; fpr = np.cumsum(1-ys_s)/(len(ys_s)-tp)
    return np.trapezoid(tpr,fpr)

# ============================================================
# 突破口1: 外部先验 — 临床文献 Z 值置信度
# ============================================================
print("\n" + "="*60)
print("突破口1: 外部先验（临床文献 Z 值置信度）")
print("="*60)

# 临床文献：T21检出率>99%, T18>97%, T13>80%
# 对应 Z 值极端程度映射为"先验异常概率"
# 思路：对每条染色体，Z_corrected 的绝对值越大 → 先验异常概率越高
#       用正常组的 Z 分布拟合一个"先验异常度"函数

# 在 LOOCV 中计算先验分数
prior_scores = np.zeros(n)
z_idx = [0, 1, 2, 3]  # Z13,Z18,Z21,ZX_corrected in F_lb

for i in range(n):
    mask_tr = np.ones(n, bool); mask_tr[i] = False
    z_normal_tr = F_lb[mask_tr][y[mask_tr]==0][:, z_idx]

    # 对每条染色体：Z偏离正常组均值的程度 = 先验异常度
    mu_z = z_normal_tr.mean(0)
    std_z = z_normal_tr.std(0); std_z[std_z<1e-10] = 1.0

    # Mahalanobis-like 先验分: max over chromosomes of |Z-mu|/sigma
    z_dev = np.abs(F_lb[i, z_idx] - mu_z) / std_z
    # 映射到 [0,1]: sigmoid(z_dev - 2) → Z>2sigma时先验>0.5
    prior_scores[i] = 1 / (1 + np.exp(-(z_dev.max() - 2.0)))

auc_prior = roc_auc(y, prior_scores)
print(f"纯先验 AUC = {auc_prior:.4f}")

# 先验 + Logistic(L2) 融合
# Logistic(L2) 用 Top-4 特征（来自上一轮实验: GC diffs + X_conc + filter_rate）
top4 = [4, 5, 7, 6]  # GC_18_13, GC_21_13, filter_rate, GC_18_21 (按Cohen's d)
log_scores = np.zeros(n)

def logistic_fit(X,y,lam=1.0,lr=0.01,n_iter=2000):
    nd,dd = X.shape
    w=np.zeros(dd); b=0.0
    for _ in range(n_iter):
        z=X@w+b; p=1/(1+np.exp(-np.clip(z,-50,50)))
        w-=lr*((X.T@(p-y))/nd + lam*w/nd); b-=lr*(p-y).mean()
    return w,b

for i in range(n):
    mask_tr = np.ones(n,bool); mask_tr[i]=False
    # 标准化
    X_tr = F_lb[mask_tr][:, top4]; X_te = F_lb[i, top4]
    mu_tr, std_tr = X_tr.mean(0), X_tr.std(0); std_tr[std_tr<1e-10]=1.0
    w,b = logistic_fit((X_tr-mu_tr)/std_tr, y[mask_tr])
    log_scores[i] = 1/(1+np.exp(-((X_te-mu_tr)/std_tr @ w + b)))

auc_log = roc_auc(y, log_scores)
print(f"Logistic(L2) AUC = {auc_log:.4f}")

# 融合: α * logistic + (1-α) * prior
def minmax(x):
    mn,mx=x.min(),x.max(); return (x-mn)/(mx-mn+1e-10)

log_n = minmax(log_scores); prior_n = minmax(prior_scores)

for alpha in [0.1, 0.2, 0.3, 0.5]:
    fused = alpha*prior_n + (1-alpha)*log_n
    auc_f = roc_auc(y, fused)
    print(f"  α={alpha:.1f} (先验+Logistic): AUC = {auc_f:.4f}")

# 非对称融合: 先验只在 Z 极端时激活
# score = logistic_score + β * max(0, Z_dev - 2)
for beta in [0.05, 0.1, 0.2, 0.5]:
    z_dev_all = np.max(np.abs(F_lb[:, z_idx] - F_lb[y==0][:, z_idx].mean(0)) /
                       F_lb[y==0][:, z_idx].std(0), axis=1)
    boost = np.maximum(0, z_dev_all - 2.0)
    asym_fused = log_n + beta * boost / (boost.max() + 1e-10)
    auc_af = roc_auc(y, asym_fused)
    print(f"  非对称 β={beta:.2f}: AUC = {auc_af:.4f}")

# ============================================================
# 突破口2: 半监督 — 利用41条Z异常未标注样本
# ============================================================
print("\n" + "="*60)
print("突破口2: 半监督 (41条Z异常未标注样本)")
print("="*60)

# 策略: Self-Training — 用标注数据训练初始模型 → 预测未标注Z异常样本
#       → 高置信度预测加入训练集 → 重新训练

# 在 LOOCV 中无法做真正的 self-training (每次只留1个标注)
# 改为: 将 Z 异常未标注样本作为"软正样本"，赋予部分权重

# 构造扩展训练集: 标注正样本(16) + Z异常未标注样本(权重=λ)
# λ 控制对未标注样本的信任度
Z_ul_feat = F_all[idx_z_ul][:, top4]
Z_lb_feat = F_lb[:, top4]  # 标注样本特征

for lam in [0.1, 0.2, 0.3, 0.5]:
    semi_scores = np.zeros(n)
    for i in range(n):
        mask_tr = np.ones(n, bool); mask_tr[i] = False
        X_tr_lb = F_lb[mask_tr][:, top4]
        y_tr_lb = y[mask_tr]

        # 标注数据
        X_combined = X_tr_lb.copy()
        y_combined = y_tr_lb.copy()

        # 加入 Z 异常未标注样本作为正样本(权重=λ)
        # 用样本加权: 复制 λ 份
        n_ul_add = int(len(Z_ul_feat) * lam)
        if n_ul_add > 0:
            rng = np.random.RandomState(42)
            ul_idx_add = rng.choice(len(Z_ul_feat), n_ul_add, replace=True)
            X_combined = np.vstack([X_combined, Z_ul_feat[ul_idx_add]])
            y_combined = np.concatenate([y_combined, np.ones(n_ul_add)])

        # 标准化 + 训练
        mu_c, std_c = X_combined.mean(0), X_combined.std(0)
        std_c[std_c<1e-10] = 1.0
        w_c, b_c = logistic_fit((X_combined-mu_c)/std_c, y_combined)
        semi_scores[i] = 1/(1+np.exp(-((F_lb[i,top4]-mu_c)/std_c @ w_c + b_c)))

    auc_s = roc_auc(y, semi_scores)
    print(f"  Self-Training λ={lam:.1f}: AUC = {auc_s:.4f}")

# ============================================================
# 突破口1+2 联合: 先验 + 半监督
# ============================================================
print("\n" + "="*60)
print("突破口1+2 联合: 先验 + 半监督融合")
print("="*60)

# 用最优λ=0.3做半监督，再融合先验
lam_best = 0.3
semi_best = np.zeros(n)
for i in range(n):
    mask_tr = np.ones(n, bool); mask_tr[i] = False
    X_tr_lb = F_lb[mask_tr][:, top4]; y_tr_lb = y[mask_tr]
    X_combined = X_tr_lb.copy(); y_combined = y_tr_lb.copy()
    n_ul_add = int(len(Z_ul_feat) * lam_best)
    if n_ul_add > 0:
        rng = np.random.RandomState(42)
        ul_idx_add = rng.choice(len(Z_ul_feat), n_ul_add, replace=True)
        X_combined = np.vstack([X_combined, Z_ul_feat[ul_idx_add]])
        y_combined = np.concatenate([y_combined, np.ones(n_ul_add)])
    mu_c, std_c = X_combined.mean(0), X_combined.std(0)
    std_c[std_c<1e-10] = 1.0
    w_c, b_c = logistic_fit((X_combined-mu_c)/std_c, y_combined)
    semi_best[i] = 1/(1+np.exp(-((F_lb[i,top4]-mu_c)/std_c @ w_c + b_c)))

# 非对称融合
z_dev_all_lb = np.max(np.abs(F_lb[:, z_idx] - F_lb[y==0][:, z_idx].mean(0)) /
                      F_lb[y==0][:, z_idx].std(0), axis=1)
boost_lb = np.maximum(0, z_dev_all_lb - 2.0)

semi_n = minmax(semi_best)
for beta in [0.05, 0.1, 0.2, 0.5]:
    joint = semi_n + beta * boost_lb / (boost_lb.max() + 1e-10)
    auc_j = roc_auc(y, joint)
    print(f"  半监督(λ=0.3) + 先验 β={beta:.2f}: AUC = {auc_j:.4f}")

# ============================================================
# 综合对比 + 高灵敏度分析
# ============================================================
print("\n" + "="*60)
print("综合对比")
print("="*60)

# 最优方案
# 取联合方案中 AUC 最高的
best_joint_scores = semi_n + 0.1 * boost_lb / (boost_lb.max() + 1e-10)

all_final = {
    'Logistic(L2) 基线': log_scores,
    'Logistic + 先验(α=0.1)': 0.1*prior_n + 0.9*log_n,
    '半监督(λ=0.3)': semi_best,
    '半监督 + 先验(β=0.1)': best_joint_scores,
}

print(f"\n{'方法':<30s} {'AUC':>8s}")
print("-"*42)
for nm, sc in all_final.items():
    print(f"{nm:<30s} {roc_auc(y,sc):8.4f}")

# 高灵敏度分析
print(f"\n{'方法':<30s} {'TPR90% FPR':>12s} {'TPR90% 特异度':>14s} {'TPR80% FPR':>12s}")
print("-"*72)
for nm, sc in all_final.items():
    order = np.argsort(-sc)
    ys, ss = y[order], sc[order]
    tp_c = np.cumsum(ys); fp_c = np.cumsum(1-ys)
    tpr = tp_c/tp_c[-1]; fpr = fp_c/fp_c[-1]

    def at_tpr(target):
        cand = np.where(tpr >= target)[0]
        if len(cand) > 0:
            return fpr[cand[0]], 1-fpr[cand[0]]
        return 1.0, 0.0

    fpr90, spec90 = at_tpr(0.90)
    fpr80, spec80 = at_tpr(0.80)
    print(f"{nm:<30s} {fpr90:11.1%} {spec90:13.1%} {fpr80:11.1%}")

print("\n完成.")
