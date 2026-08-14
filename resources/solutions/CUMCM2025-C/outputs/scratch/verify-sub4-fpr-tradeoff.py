#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fisher LDA 正则化强度扫描 + TPR=80%/85% 目标下的 FPR 分析
"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')

cache = r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data'
sub4 = pd.read_pickle(os.path.join(cache, '2025C-sub4-preprocessed.pkl'))
clean = pd.read_pickle(os.path.join(cache, '2025C-female-clean.pkl'))
for orig,new in {'13号染色体的GC含量':'GC_13','18号染色体的GC含量':'GC_18','21号染色体的GC含量':'GC_21'}.items():
    sub4[new]=clean[orig].values
sub4['X_conc']=clean['X染色体浓度'].values
sub4['孕妇代码']=clean['孕妇代码'].values

dm=clean[['孕妇代码','检测日期_std']].drop_duplicates('孕妇代码')
sub4['检测日期_std']=dm.set_index('孕妇代码').loc[sub4['孕妇代码'],'检测日期_std'].values
mask=(sub4['AB_异常']==0)|(sub4['AB_异常']==1)
lb=sub4[mask].sort_values('检测日期_std').drop_duplicates('孕妇代码',keep='first')
idx=lb.index.values; y=lb['AB_异常'].values.astype(float)

# 特征
GC18,GC13,GC21=sub4['GC_18'].values,sub4['GC_13'].values,sub4['GC_21'].values
F_all=np.column_stack([
    sub4['Z13_corrected'].values,sub4['Z18_corrected'].values,
    sub4['Z21_corrected'].values,sub4['ZX_corrected'].values,
    GC18-GC13,GC21-GC13,GC18-GC21,
    sub4['被过滤掉读段数的比例'].values,sub4['X_conc'].values,
    sub4['孕妇BMI'].values,sub4['孕周_数值'].values,
])
F_all=np.nan_to_num(F_all,0); F=F_all[idx]
n,d=F.shape
top4=[4,5,7,6]; z_idx=[0,1,2,3]

def roc_auc(yt,ys):
    o=np.argsort(-ys); ys_s=yt[o]
    tp=ys_s.sum()
    if tp==0 or tp==len(ys_s): return np.nan
    return np.trapezoid(np.cumsum(ys_s)/tp, np.cumsum(1-ys_s)/(len(ys_s)-tp))

def fpr_at_tpr(yt,ys,target):
    o=np.argsort(-ys); ys_s=yt[o]
    tpr_v=np.cumsum(ys_s)/ys_s.sum()
    fpr_v=np.cumsum(1-ys_s)/(len(ys_s)-ys_s.sum())
    c=np.where(tpr_v>=target)[0]
    if len(c)>0:
        i=c[0]; return fpr_v[i],1-fpr_v[i]
    return 1.0,0.0

def fisher_lda_fit(X_pos,X_neg,reg_strength=0.01):
    mu1,m0=X_pos.mean(0),X_neg.mean(0)
    S1=np.cov(X_pos,rowvar=False) if len(X_pos)>1 else np.eye(X_pos.shape[1])
    S0=np.cov(X_neg,rowvar=False)
    Sw=((len(X_pos)-1)*S1+(len(X_neg)-1)*S0)/(len(X_pos)+len(X_neg)-2)
    reg=reg_strength*np.trace(Sw)/Sw.shape[0]
    try: Sw_inv=np.linalg.inv(Sw+reg*np.eye(Sw.shape[0]))
    except: Sw_inv=np.linalg.pinv(Sw+reg*np.eye(Sw.shape[0]))
    return Sw_inv@(mu1-m0)

def logistic_fit(X,y,lam=1.0,lr=0.01,n_iter=2000):
    nd,dd=X.shape; w=np.zeros(dd); b=0.0
    for _ in range(n_iter):
        z=X@w+b; p=1/(1+np.exp(-np.clip(z,-50,50)))
        w-=lr*((X.T@(p-y))/nd+lam*w/nd); b-=lr*(p-y).mean()
    return w,b

# ============================================================
# 1. Fisher LDA 正则化强度扫描 (reg = 0.001 ~ 10.0)
# ============================================================
print("="*60)
print("1. Fisher LDA 正则化强度扫描 (LOOCV)")
print("="*60)

reg_values=[0.001,0.01,0.05,0.1,0.3,0.5,0.7,1.0,3.0,5.0,10.0]
print(f"\n{'reg':>8s} {'AUC':>8s} {'TPR80%FPR':>10s} {'特异度80':>10s} {'TPR90%FPR':>10s}")
print("-"*52)

best_auc=0; best_reg=0.01; best_scores=None
for reg in reg_values:
    fish_loocv=np.zeros(n)
    for i in range(n):
        mask_tr=np.ones(n,bool); mask_tr[i]=False
        # 在训练集中找正负样本
        y_tr=y[mask_tr]
        X_pos=F[mask_tr][y_tr==1]; X_neg=F[mask_tr][y_tr==0]
        if len(X_pos)<2 or len(X_neg)<2:
            fish_loocv[i]=0; continue
        w=fisher_lda_fit(X_pos,X_neg,reg)
        fish_loocv[i]=F[i]@w

    auc_f=roc_auc(y,fish_loocv)
    f80,s80=fpr_at_tpr(y,fish_loocv,0.80)
    f90,_=fpr_at_tpr(y,fish_loocv,0.90)
    print(f"{reg:8.3f} {auc_f:8.4f} {f80:9.1%} {s80:9.1%} {f90:9.1%}")
    if auc_f>best_auc:
        best_auc=auc_f; best_reg=reg; best_scores=fish_loocv.copy()

print(f"\n最优: reg={best_reg}, AUC={best_auc:.4f}")

# ============================================================
# 2. Logistic(L2) λ 扫描
# ============================================================
print("\n"+"="*60)
print("2. Logistic(L2) λ 扫描 (LOOCV, Top-4特征)")
print("="*60)

lam_values=[0.1,0.3,0.5,1.0,2.0,3.0,5.0,10.0]
print(f"\n{'λ':>8s} {'AUC':>8s} {'TPR80%FPR':>10s} {'特异度80':>10s} {'TPR85%FPR':>10s}")
print("-"*52)

best_log_auc=0; best_log_scores=None; best_log_lam=1.0
for lam in lam_values:
    log_s=np.zeros(n)
    for i in range(n):
        mask_tr=np.ones(n,bool); mask_tr[i]=False
        X_tr=F[mask_tr][:,top4]; X_te=F[i,top4]
        mu,std=X_tr.mean(0),X_tr.std(0); std[std<1e-10]=1.0
        w,b=logistic_fit((X_tr-mu)/std,y[mask_tr],lam)
        log_s[i]=1/(1+np.exp(-((X_te-mu)/std@w+b)))

    auc_l=roc_auc(y,log_s)
    f80,s80=fpr_at_tpr(y,log_s,0.80)
    f85,s85=fpr_at_tpr(y,log_s,0.85)
    print(f"{lam:8.3f} {auc_l:8.4f} {f80:9.1%} {s80:9.1%} {f85:9.1%}")
    if auc_l>best_log_auc:
        best_log_auc=auc_l; best_log_scores=log_s.copy(); best_log_lam=lam

print(f"\n最优: λ={best_log_lam}, AUC={best_log_auc:.4f}")

# ============================================================
# 3. 先验融合 + 多TPR目标分析
# ============================================================
print("\n"+"="*60)
print("3. 先验融合 + TPR多目标分析")
print("="*60)

# 先验分
prior_s=np.zeros(n)
for i in range(n):
    mask_tr=np.ones(n,bool); mask_tr[i]=False
    zn=F[mask_tr][y[mask_tr]==0][:,z_idx]
    mu_z,std_z=zn.mean(0),zn.std(0); std_z[std_z<1e-10]=1.0
    zd=np.abs(F[i,z_idx]-mu_z)/std_z
    prior_s[i]=1/(1+np.exp(-(zd.max()-2.0)))

def minmax(x):
    mn,mx=x.min(),x.max(); return (x-mn)/(mx-mn+1e-10)

log_n=minmax(best_log_scores); prior_n=minmax(prior_s)

print(f"\n{'方案':<35s} {'AUC':>8s}",end='')
for t in [0.75,0.80,0.85,0.90]:
    print(f" {'TPR'+str(int(t*100))+'FPR':>10s}",end='')
print(f" {'TPR80特异度':>12s}")
print("-"*110)

methods={
    'Fisher LDA(reg={:.2f})'.format(best_reg): best_scores,
    'Logistic(L2,λ={:.1f})'.format(best_log_lam): best_log_scores,
    'Logistic+先验(α=0.2)': 0.2*prior_n+0.8*log_n,
    'Logistic+先验(α=0.3)': 0.3*prior_n+0.7*log_n,
    'Logistic+先验(α=0.5)': 0.5*prior_n+0.5*log_n,
}

for nm,sc in methods.items():
    au=roc_auc(y,sc)
    print(f"{nm:<35s} {au:8.4f}",end='')
    for t in [0.75,0.80,0.85,0.90]:
        f,_=fpr_at_tpr(y,sc,t)
        print(f" {f:9.1%}",end='')
    f80,s80=fpr_at_tpr(y,sc,0.80)
    print(f" {s80:11.1%}")

# ============================================================
# 4. 最优方案混淆矩阵 (TPR=80%阈值)
# ============================================================
print("\n"+"="*60)
print("4. 最优方案在 TPR=80% 下的混淆矩阵")
print("="*60)

best_nm="Logistic+先验(α=0.3)"
best_sc=0.3*prior_n+0.7*log_n

# 找 TPR=80% 阈值
o=np.argsort(-best_sc); ys=y[o]; ss=best_sc[o]
tpr_v=np.cumsum(ys)/ys.sum()
i80=np.where(tpr_v>=0.80)[0][0]
thr80=ss[i80]
pred=(best_sc>=thr80).astype(int)
tp=((y==1)&(pred==1)).sum()
fp=((y==0)&(pred==1)).sum()
fn=((y==1)&(pred==0)).sum()
tn=((y==0)&(pred==0)).sum()

print(f"\n阈值 = {thr80:.4f}")
print(f"\n          预测异常  预测正常")
print(f"实际异常     {tp:3d}       {fn:3d}")
print(f"实际正常     {fp:3d}       {tn:3d}")
print(f"\n召回率(TPR) = {tp/(tp+fn):.1%}")
print(f"特异度(TNR) = {tn/(tn+fp):.1%}")
print(f"精确率      = {tp/(tp+fp):.1%}")
print(f"FPR         = {fp/(fp+tn):.1%}")
print(f"漏检(FN)    = {fn} 人")

# TPR=85%
i85=np.where(tpr_v>=0.85)[0][0]
thr85=ss[i85]
pred85=(best_sc>=thr85).astype(int)
tp85=((y==1)&(pred85==1)).sum()
fp85=((y==0)&(pred85==1)).sum()
fn85=((y==1)&(pred85==0)).sum()
tn85=((y==0)&(pred85==0)).sum()

print(f"\n--- TPR=85% ---")
print(f"          预测异常  预测正常")
print(f"实际异常     {tp85:3d}       {fn85:3d}")
print(f"实际正常     {fp85:3d}       {tn85:3d}")
print(f"召回率={tp85/(tp85+fn85):.1%}  特异度={tn85/(tn85+fp85):.1%}  FPR={fp85/(fp85+tn85):.1%}  漏检={fn85}人")

print("\n完成.")
