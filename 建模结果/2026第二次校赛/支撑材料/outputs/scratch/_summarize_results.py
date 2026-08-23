# 只读汇总 S2/S3 关键结果（一次性，轻量）
import pickle

d2 = pickle.load(open(r'outputs/data/S2-results.pkl', 'rb'))
d3 = pickle.load(open('outputs/data/S3-results.pkl', 'rb'))

print('=== S2 稳定特征数 ===')
for dis in ['CRC', 'IBD', 'Obesity']:
    print(f'{dis}: {len(d2["per_disease"][dis]["stable_features"])} 个稳定特征')
print('=== S2 Jaccard ===')
print(d2['cross_disease']['jaccard_matrix'])
print('=== S3 四策略均值 ===')
sc = d3['strategy_compare']
for s in ['A_direct', 'B_shared', 'C_genus', 'C_phylum', 'D_calibrated']:
    print(f'{s}: {sc[s]["mean_auc"]:.4f}')
print('=== S3 回退 ===')
fb = d3['fallback']
print(f'triggered={fb["triggered"]}, usable={fb["usable"]}, best={d3["best_strategy"]}')
print('=== S3 衰减 ===')
for dis in ['CRC', 'IBD', 'Obesity']:
    da = d3['decay_attribution'][dis]
    print(f'{dis}: domain={da["domain_auc"]:.4f} cross={da["cross_auc"]:.4f} decay={da["decay"]:.4f} cause={da["dominant_cause"]}')
