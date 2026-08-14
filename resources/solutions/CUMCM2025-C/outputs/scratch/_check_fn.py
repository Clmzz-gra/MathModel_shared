import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
cache=r'E:\MathModel\problems\2025\C题\2025C题测试\outputs\data'
sub4=pd.read_pickle(os.path.join(cache,'2025C-sub4-preprocessed.pkl'))
clean=pd.read_pickle(os.path.join(cache,'2025C-female-clean.pkl'))
for o,n in {'13号染色体的GC含量':'GC_13','18号染色体的GC含量':'GC_18','21号染色体的GC含量':'GC_21'}.items(): sub4[n]=clean[o].values
sub4['X_conc']=clean['X染色体浓度'].values
sub4['孕妇代码']=clean['孕妇代码'].values
dm=clean[['孕妇代码','检测日期_std']].drop_duplicates('孕妇代码')
sub4['dt']=dm.set_index('孕妇代码').loc[sub4['孕妇代码'],'检测日期_std'].values
m=(sub4['AB_异常']==0)|(sub4['AB_异常']==1)
lb=sub4[m].sort_values('dt').drop_duplicates('孕妇代码',keep='first')
idx=lb.index.values; y=lb['AB_异常'].values.astype(float)

gc=sub4[['GC_13','GC_18','GC_21']].values
F=np.column_stack([sub4[['Z13_corrected','Z18_corrected','Z21_corrected','ZX_corrected']].values,
    gc[:,1]-gc[:,0],gc[:,2]-gc[:,0],gc[:,1]-gc[:,2],
    sub4['被过滤掉读段数的比例'].values,sub4['X_conc'].values,sub4['孕妇BMI'].values,sub4['孕周_数值'].values])
F=np.nan_to_num(F,0)[idx]
n=len(y); top4=[4,5,7,6]; z_idx=[0,1,2,3]

log_s=np.zeros(n)
for i in range(n):
    mt=np.ones(n,bool); mt[i]=False
    Xt=F[mt][:,top4]; yt=y[mt]; Xi=F[i,top4]
    mu,std=Xt.mean(0),Xt.std(0); std[std<1e-10]=1.0
    w=np.zeros(4); b=0.0
    for _ in range(2000):
        zz=(Xt-mu)/std@w+b; p=1/(1+np.exp(-np.clip(zz,-50,50)))
        w-=0.01*((((Xt-mu)/std).T@(p-yt))/len(yt)+1.0*w/len(yt)); b-=0.01*(p-yt).mean()
    log_s[i]=1/(1+np.exp(-((Xi-mu)/std@w+b)))

prior_s=np.zeros(n)
for i in range(n):
    mt=np.ones(n,bool); mt[i]=False
    zn=F[mt][y[mt]==0][:,z_idx]; mz,sz=zn.mean(0),zn.std(0); sz[sz<1e-10]=1.0
    zd=np.abs(F[i,z_idx]-mz)/sz; prior_s[i]=1/(1+np.exp(-(zd.max()-2.0)))

def mm(x): mn,mx=x.min(),x.max(); return (x-mn)/(mx-mn+1e-10)
ln=mm(log_s); pn=mm(prior_s)
sc=0.5*pn+0.5*ln

o=np.argsort(-sc); ys=y[o]
tpr=np.cumsum(ys)/ys.sum()
i80=np.where(tpr>=0.80)[0][0]
thr80=sc[o][i80]
pred=(sc>=thr80).astype(int)

fn_idx=np.where((y==1)&(pred==0))[0]
fn_codes=lb.iloc[fn_idx]['孕妇代码'].values
fn_types=clean.loc[clean['孕妇代码'].isin(fn_codes),['孕妇代码','染色体的非整倍体']].drop_duplicates('孕妇代码')

print('=== 漏诊的{}人 (TPR=80%阈值={:.4f}) ==='.format(len(fn_idx), thr80))
for i,ii in enumerate(fn_idx):
    c=fn_codes[i]
    at=fn_types[fn_types['孕妇代码']==c]['染色体的非整倍体'].values
    at_str=str(at[0]) if len(at)>0 else '?'
    print('孕妇{}: 异常={}, 分={:.4f} | Z13={:.2f} Z18={:.2f} Z21={:.2f} ZX={:.2f} | GC18-13={:.4f} GC21-13={:.4f} filter={:.4f} X_conc={:.4f} | BMI={:.0f} GW={:.0f}'.format(
        c,at_str,sc[ii],F[ii,0],F[ii,1],F[ii,2],F[ii,3],F[ii,4],F[ii,5],F[ii,7],F[ii,8],F[ii,9],F[ii,10]))

print()
print('=== 正常组参考 ===')
for j,nm in enumerate(['Z13','Z18','Z21','ZX','GC18-13','GC21-13','filter','X_conc','BMI','GW']):
    v=F[y==0,j]
    print('{:>10s}: mean={:.3f} std={:.3f} [{:.3f},{:.3f}]'.format(nm,v.mean(),v.std(),v.min(),v.max()))

# 漏诊者的 Z 值是否在正常范围内
print()
print('=== 漏诊者相对正常组的位置 ===')
for i,ii in enumerate(fn_idx):
    c=fn_codes[i]
    flags=[]
    for j,nm in enumerate(['Z13','Z18','Z21','ZX','GC18-13','GC21-13','filter','X_conc','BMI','GW']):
        v=F[y==0,j]
        zscore=(F[ii,j]-v.mean())/v.std()
        if abs(zscore)>2:
            flags.append('{}={:.1f}sigma'.format(nm,zscore))
    if flags:
        print('孕妇{}: 异常信号: {}'.format(c,', '.join(flags)))
    else:
        print('孕妇{}: 所有特征均在正常2sigma内 -- 与正常人无异'.format(c))
