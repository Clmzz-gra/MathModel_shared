"""
阶段 2.1 v3: FA 纯连续特征 (去品类)
输入: 供货总量, 供货周数, 供货满足率, 供订CV差, 可靠性趋势
"""
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings("ignore")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def calc_fulfill(ord_mat, sup_mat, start, end):
    o = ord_mat[:, start:end]; s = sup_mat[:, start:end]
    o_act = o > 0
    num = ((s >= o) & o_act).sum(axis=1).astype(float)
    den = o_act.sum(axis=1).astype(float)
    return np.divide(num, den, where=den>0, out=np.zeros(len(ord_mat)))

# ── 加载 ──
df_o = pd.read_pickle(os.path.join(DATA_DIR, "order-raw.pkl"))
df_s = pd.read_pickle(os.path.join(DATA_DIR, "supply-raw.pkl"))
wc = [c for c in df_o.columns if c.startswith("W")]
om = df_o[wc].values.astype(float); sm = df_s[wc].values.astype(float)
cats = df_o["材料分类"].values; ids = df_o["供应商ID"].values
n, nw = sm.shape; h = nw//2

# ── 特征 ──
F = {}
F["供货总量"]   = sm.sum(axis=1)
F["供货周数"]   = (sm > 0).sum(axis=1)
oa = om > 0
F["供货满足率"] = np.divide(((sm >= om) & oa).sum(axis=1), oa.sum(axis=1),
                          where=oa.sum(axis=1)>0, out=np.zeros(n))
sc = np.zeros(n); oc = np.zeros(n)
for i in range(n):
    s_nz = sm[i][sm[i]>0]; o_nz = om[i][om[i]>0]
    if len(s_nz)>1: sc[i] = s_nz.std()/s_nz.mean()
    if len(o_nz)>1: oc[i] = o_nz.std()/o_nz.mean()
F["供订CV差"]   = sc - oc
F["可靠性趋势"] = calc_fulfill(om,sm,h,nw) - calc_fulfill(om,sm,0,h)

fa_cols = ["供货总量","供货周数","供货满足率","供订CV差","可靠性趋势"]
X_raw = np.column_stack([F[c] for c in fa_cols])
print("="*60)
print("FA v3: 纯连续特征")
print(f"特征: {fa_cols}")

# ── FA ──
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_raw)
R = np.corrcoef(X.T)
ev, evec = np.linalg.eigh(R)
idx = np.argsort(-ev); ev=ev[idx]; evec=evec[:,idx]
m = (ev>=1).sum()
print(f"特征值: {np.round(ev,3)} | m={m}")

L = evec[:,:m]*np.sqrt(ev[:m])

# Varimax
def varimax(L,tol=1e-6):
    p,m=L.shape; Lr=L.copy()
    for it in range(200):
        old=Lr.copy(); h2=(Lr**2).sum(1,keepdims=True)
        u=Lr/np.sqrt(np.maximum(h2,1e-10))
        for j in range(m):
            for k in range(j+1,m):
                uj,uk=u[:,j],u[:,k]; A=uj**2-uk**2; B=2*uj*uk
                C,D=A.sum(),B.sum()
                phi=np.arctan2(D-2*C*D/p,C-(C**2-D**2)/p)/4.0
                cp,sp=np.cos(phi),np.sin(phi)
                Lr[:,[j,k]]=old[:,[j,k]]@np.array([[cp,-sp],[sp,cp]])
                u[:,[j,k]]=u[:,[j,k]]@np.array([[cp,-sp],[sp,cp]])
                old=Lr.copy()
        if np.abs(Lr-old).max()<tol: break
    return Lr

Lr = varimax(L)
print(f"\n载荷:")
for j,nm in enumerate(fa_cols):
    s=" ".join(f"F{k+1}={Lr[j,k]:+.3f}{'*' if abs(Lr[j,k])>.5 else ''}" for k in range(m))
    print(f"  {nm:>8s}: {s}  h²={(Lr[j]**2).sum():.3f}")

w = ev[:m]/ev[:m].sum()
Fs = X@np.linalg.inv(R)@Lr
I_tmp = Fs@w
I = (I_tmp-I_tmp.min())/(I_tmp.max()-I_tmp.min())

# ── Top 50 ──
rank = np.argsort(-I); t50 = rank[:50]
print(f"\n{'='*60}")
print(f"Top 50 (品类: A={(cats[t50]=='A').sum()} B={(cats[t50]=='B').sum()} C={(cats[t50]=='C').sum()})")
print(f"{'排名':>4s} {'ID':>6s} {'品':>2s} {'I':>8s} {'供货':>10s} {'满足率':>8s} {'可靠性趋势':>8s}")
for r,i in enumerate(t50):
    print(f"{r+1:>4d} {ids[i]:>6s} {cats[i]:>2s} {I[i]:>8.4f} {F['供货总量'][i]:>10.0f} {F['供货满足率'][i]:>8.3f} {F['可靠性趋势'][i]:>8.3f}")

baseline = set(np.argsort(-F['供货总量'])[:50])
print(f"\n与仅供货总量基线重叠: {len(baseline & set(t50))}/50")
print(f"含供货前20: {len(set(np.argsort(-F['供货总量'])[:20]) & set(t50))}/20")

# 保存
pd.DataFrame({"供应商ID":ids,"品类":cats,"安全指数":I,"排名":np.argsort(np.argsort(-I))+1})\
  .sort_values("排名").to_csv(os.path.join(DATA_DIR,"sub1-results-fa-v3.csv"),index=False)
print(f"\n完成")
