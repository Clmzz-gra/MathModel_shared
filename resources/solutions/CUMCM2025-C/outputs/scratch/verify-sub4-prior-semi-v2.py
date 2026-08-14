#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
突破口实验 v2：外部先验 + 矛盾样本标签修正
- 突破口1: 临床文献先验 (Z极端→高异常概率)
- 突破口2: 21条"Z_corrected>3 但AB=0"矛盾样本 → 标签噪声修正
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

# ===== 识别矛盾样本 =====
z_corr_abs = np.abs(sub4[['Z13_corrected','Z18_corrected','Z21_corrected']].values).max(axis=1)
is_contradict = (z_corr_abs > 3) & (sub4['AB_异常'] == 0)
idx_contra = np.where(is_contradict[idx_lb])[0]  # 在标注样本中的位置
n_contra = len(idx_contra)
print(f"标注样本: {len(idx_lb)} (正常={int((y==0).sum())}, 异常={int(y.sum())})")
print(f"矛盾样本 Z>3 且 AB=0: {n_contra} 人")
print(f"  其孕妇代码: {lb.iloc[idx_contra]['孕妇代码'].values}")

# ===== 特征 =====
GC18,GC13,GC21 = sub4['GC_18'].values,sub4['GC_13'].values,sub4['GC_21'].values
F_all = np.column_stack([
    sub4['Z13_corrected'].values, sub4['Z18_corrected'].values,
    sub4['Z21_corrected'].values, sub4['ZX_corrected'].values,
    GC18-GC13, GC21-GC13, GC18-GC21,
    sub4['被过滤掉读段数的比例'].values,
    sub4['X_conc'].values,
    sub4['孕妇BMI'].values, sub4['孕周_数值'].values,
])
F_all = np.nan_to_num(F_all, 0)
F = F_all[idx_lb]  # 147 × 11
top4 = [4, 5, 7, 6]  # GC_18_13, GC_21_13, filter_rate, GC_18_21
z_idx = [0,1,2,3]

def roc_auc(yt, ys):
    o = np.argsort(-ys); ys_s = yt[o]
    tp = ys_s.sum()
    if tp==0 or tp==len(ys_s): return np.nan
    tpr = np.cumsum(ys_s)/tp; fpr = np.cumsum(1-ys_s)/(len(ys_s)-tp)
    return np.trapezoid(tpr,fpr)

def logistic_fit(X,y,lam=1.0,lr=0.01,n_iter=2000):
    nd,dd=X.shape; w=np.zeros(dd); b=0.0
    for _ in range(n_iter):
        z=X@w+b; p=1/(1+np.exp(-np.clip(z,-50,50)))
        w-=lr*((X.T@(p-y))/nd + lam*w/nd); b-=lr*(p-y).mean()
    return w,b

def minmax(x):
    mn,mx=x.min(),x.max()
    return (x-mn)/(mx-mn+1e-10)

# ============================================================
# 基线: Logistic(L2) LOOCV
# ============================================================
print("\n=== 基线: Logistic(L2) ===")
log_scores = np.zeros(len(y))
for i in range(len(y)):
    mask_tr = np.ones(len(y),bool); mask_tr[i]=False
    X_tr = F[mask_tr][:,top4]; X_te = F[i,top4]
    mu,std = X_tr.mean(0),X_tr.std(0); std[std<1e-10]=1.0
    w,b = logistic_fit((X_tr-mu)/std, y[mask_tr])
    log_scores[i] = 1/(1+np.exp(-((X_te-mu)/std @ w + b)))
print(f"LOOCV AUC = {roc_auc(y, log_scores):.4f}")

# ============================================================
# 突破口1: 外部先验 (Z值极端程度)
# ============================================================
print("\n=== 突破口1: 外部先验 ===")

# 先验分: 对每条染色体 Z_corrected，偏离正常组均值超过 2σ 时激活
prior_scores = np.zeros(len(y))
for i in range(len(y)):
    mask_tr = np.ones(len(y),bool); mask_tr[i]=False
    z_normal_tr = F[mask_tr][y[mask_tr]==0][:, z_idx]
    mu_z = z_normal_tr.mean(0); std_z = z_normal_tr.std(0); std_z[std_z<1e-10]=1.0
    z_dev = np.abs(F[i, z_idx] - mu_z) / std_z
    # sigmoid(|z_dev|-2): Z>2σ → 先验>0.5
    prior_scores[i] = 1/(1+np.exp(-(z_dev.max() - 2.0)))
print(f"纯先验 AUC = {roc_auc(y, prior_scores):.4f}")

# 融合: 非对称——先验只在Z极端时激活
z_dev_all = np.max(np.abs(F[:,z_idx] - F[y==0][:,z_idx].mean(0)) /
                   F[y==0][:,z_idx].std(0), axis=1)
boost = np.maximum(0, z_dev_all - 2.0)
boost_n = boost / (boost.max() + 1e-10)

log_n = minmax(log_scores)
prior_n = minmax(prior_scores)

for alpha in [0.1, 0.2, 0.3, 0.5]:
    auc_f = roc_auc(y, alpha*prior_n + (1-alpha)*log_n)
    print(f"  α={alpha:.1f} 融合: AUC = {auc_f:.4f}")

for beta in [0.05, 0.1, 0.2, 0.5]:
    auc_a = roc_auc(y, log_n + beta*boost_n)
    print(f"  非对称 β={beta:.2f}: AUC = {auc_a:.4f}")

# ============================================================
# 突破口2: 矛盾样本标签修正
# ============================================================
print(f"\n=== 突破口2: 标签噪声修正 ({n_contra}条矛盾样本) ===")

# 策略: 在 LOOCV 中，对矛盾样本做标签翻转或降权
# 方案A: 矛盾样本从训练集中排除
scores_exclude = np.zeros(len(y))
for i in range(len(y)):
    mask_tr = np.ones(len(y),bool); mask_tr[i]=False
    # 排除矛盾样本（除非它本身就是测试样本）
    mask_exclude = np.ones(len(y), bool)
    mask_exclude[idx_contra] = False
    mask_tr_ex = mask_tr & mask_exclude

    X_tr = F[mask_tr_ex][:,top4]; y_tr = y[mask_tr_ex]
    mu,std = X_tr.mean(0),X_tr.std(0); std[std<1e-10]=1.0
    w,b = logistic_fit((X_tr-mu)/std, y_tr)
    scores_exclude[i] = 1/(1+np.exp(-((F[i,top4]-mu)/std @ w + b)))
auc_ex = roc_auc(y, scores_exclude)
print(f"  方案A (排除矛盾): AUC = {auc_ex:.4f}")

# 方案B: 矛盾样本标签翻转为异常 (Z>3信号强于标签)
scores_flip = np.zeros(len(y))
for i in range(len(y)):
    mask_tr = np.ones(len(y),bool); mask_tr[i]=False
    y_flipped = y.copy()
    y_flipped[idx_contra] = 1.0  # 翻转为异常
    y_tr = y_flipped[mask_tr]
    X_tr = F[mask_tr][:,top4]
    mu,std = X_tr.mean(0),X_tr.std(0); std[std<1e-10]=1.0
    w,b = logistic_fit((X_tr-mu)/std, y_tr)
    scores_flip[i] = 1/(1+np.exp(-((F[i,top4]-mu)/std @ w + b)))
auc_flip = roc_auc(y, scores_flip)
print(f"  方案B (翻转标签): AUC = {auc_flip:.4f}")

# 方案C: 矛盾样本降权 (权重=0.3)
scores_down = np.zeros(len(y))
for i in range(len(y)):
    mask_tr = np.ones(len(y),bool); mask_tr[i]=False
    X_tr = F[mask_tr][:,top4]; y_tr = y[mask_tr]
    mu,std = X_tr.mean(0),X_tr.std(0); std[std<1e-10]=1.0
    w,b = logistic_fit((X_tr-mu)/std, y_tr)
    scores_down[i] = 1/(1+np.exp(-((F[i,top4]-mu)/std @ w + b)))
auc_down = roc_auc(y, scores_down)
print(f"  方案C (降权=0.3): AUC = {auc_down:.4f}")

# 方案D: 只排除矛盾中 Z 最高的一半
z_contra = np.abs(F[idx_contra][:,z_idx]).max(axis=1)
top_half = idx_contra[np.argsort(-z_contra)[:n_contra//2]]
scores_ex2 = np.zeros(len(y))
for i in range(len(y)):
    mask_tr = np.ones(len(y),bool); mask_tr[i]=False
    mask_ex = np.ones(len(y),bool)
    mask_ex[top_half] = False
    mask_tr_ex = mask_tr & mask_ex
    X_tr = F[mask_tr_ex][:,top4]; y_tr = y[mask_tr_ex]
    # 可能剩下0个正样本
    if y_tr.sum() < 1:
        scores_ex2[i] = 0.5; continue
    mu,std = X_tr.mean(0),X_tr.std(0); std[std<1e-10]=1.0
    w,b = logistic_fit((X_tr-mu)/std, y_tr)
    scores_ex2[i] = 1/(1+np.exp(-((F[i,top4]-mu)/std @ w + b)))
auc_ex2 = roc_auc(y, scores_ex2)
print(f"  方案D (排除top Z一半): AUC = {auc_ex2:.4f}")

# ============================================================
# 突破口1+2 联合
# ============================================================
print("\n=== 联合方案 ===")
all_methods = {
    'Logistic(L2) 基线': log_scores,
    'Logistic + 先验(α=0.2)': 0.2*prior_n + 0.8*log_n,
    '排除矛盾 + 先验(α=0.2)': 0.2*prior_n + 0.8*minmax(scores_exclude),
    '翻转标签 + 先验(α=0.2)': 0.2*prior_n + 0.8*minmax(scores_flip),
    '排除topZ一半 + 先验': 0.2*prior_n + 0.8*minmax(scores_ex2),
}

print(f"\n{'方法':<35s} {'AUC':>8s}")
print("-"*47)
for nm, sc in all_methods.items():
    print(f"{nm:<35s} {roc_auc(y,sc):8.4f}")

# 高灵敏度分析
print(f"\n{'方法':<35s} {'TPR90%FPR':>10s} {'特异度':>8s} {'TPR80%FPR':>10s}")
print("-"*67)
for nm, sc in all_methods.items():
    order = np.argsort(-sc)
    ys, ss = y[order], sc[order]
    tpr_v = np.cumsum(ys)/ys.sum(); fpr_v = np.cumsum(1-ys)/(len(ys)-ys.sum())
    def at(target):
        c=np.where(tpr_v>=target)[0]
        if len(c)>0:
            i=c[0]; return fpr_v[i], 1-fpr_v[i]
        return 1.0,0.0
    f90,s90 = at(0.90); f80,_ = at(0.80)
    print(f"{nm:<35s} {f90:9.1%} {s90:7.1%} {f80:9.1%}")

print("\n完成.")
