"""检查进步因子的分布和问题"""
import pandas as pd, numpy as np
d = pd.read_pickle("problems/CUMCM2021/C/outputs/data/sub1-preprocessed.pkl")

pf = d["进步因子"]
print("进步因子分布:")
print(f"  min={pf.min():.3f}, max={pf.max():.3f}, mean={pf.mean():.3f}, median={pf.median():.3f}")
print(f"  >2.0: {(pf>2).sum()}, >5.0: {(pf>5).sum()}, ==10.0(clipped): {(pf==10).sum()}")

top_pf = d.nlargest(10, "进步因子")
print("\n进步因子最高 10 家:")
for _, r in top_pf.iterrows():
    print(f"  {r['供应商ID']} ({r['品类']}): 进步={r['进步因子']:.1f}, 供货={r['供货总量']:.0f}, 周数={r['供货周数']:.0f}, 满足率={r['供货满足率']:.2f}")

top_vol = d.nlargest(10, "供货总量")
print("\n供货量前 10 的进步因子:")
for _, r in top_vol.iterrows():
    print(f"  {r['供应商ID']} ({r['品类']}): 供货={r['供货总量']:.0f}, 进步={r['进步因子']:.3f}")

# 进步因子 vs 供货总量的散点统计
print(f"\n进步因子 vs 供货总量 的 Spearman: {pf.corr(d['供货总量'], method='spearman'):.3f}")
print(f"进步因子 vs 满足率 的 Spearman: {pf.corr(d['供货满足率'], method='spearman'):.3f}")

# FA 排名前5的进步因子
res = pd.read_pickle("problems/CUMCM2021/C/outputs/data/sub1-results-fa.pkl")
merged = res.merge(d[["供应商ID", "进步因子", "供货总量", "供货周数"]], on="供应商ID")
top5 = merged.head(5)
print("\nFA 排名 Top 5 的原始数据:")
for _, r in top5.iterrows():
    print(f"  #{int(r['排名'])} {r['供应商ID']} ({r['品类']}): I={r['安全指数_I']:.4f}, 进步={r['进步因子']:.1f}, 供货={r['供货总量']:.0f}")
