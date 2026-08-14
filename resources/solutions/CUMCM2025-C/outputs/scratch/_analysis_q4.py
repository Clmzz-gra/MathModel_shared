"""
问题4：按染色体拆开的判别分析
纯 numpy/scipy/pandas，只读不写，全部打印到控制台
"""
import numpy as np
import pandas as pd
import pickle

# ============================================================
# 0. 加载与去重
# ============================================================
print("=" * 70)
print("0. 数据加载与去重")
print("=" * 70)

with open(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-sub4-preprocessed.pkl', 'rb') as f:
    sub4 = pickle.load(f)
with open(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-female-clean.pkl', 'rb') as f:
    clean = pickle.load(f)

# 按孕妇去重（保留最早检测）
sub4_dedup = sub4.sort_values('检测抽血次数').drop_duplicates(subset='孕妇代码', keep='first').reset_index(drop=True)
clean_dedup = clean.sort_values('检测抽血次数').drop_duplicates(subset='孕妇代码', keep='first').reset_index(drop=True)

# 确保两个数据集的孕妇代码对齐
assert set(sub4_dedup['孕妇代码']) == set(clean_dedup['孕妇代码']), "孕妇代码不匹配！"
clean_dedup = clean_dedup.set_index('孕妇代码').loc[sub4_dedup['孕妇代码']].reset_index()

print(f"去重后总样本数: {len(sub4_dedup)}")
print(f"  正常: {(sub4_dedup['AB_异常'] == 0).sum()}")
print(f"  异常: {(sub4_dedup['AB_异常'] == 1).sum()}")

# ============================================================
# 1. 实验1：各异常类型的Z值模式
# ============================================================
print("\n" + "=" * 70)
print("实验1：各异常类型的Z值模式（12人去重异常样本）")
print("=" * 70)

ab = sub4_dedup[sub4_dedup['AB_异常'] == 1].copy()
ab = ab.sort_values('染色体的非整倍体')

print(f"\n{'孕妇代码':<10} {'异常类型':<10} {'Z13_cor':>8} {'Z18_cor':>8} {'Z21_cor':>8} {'ZX_cor':>8} {'过滤率':>8}")
print("-" * 70)

type_groups = {}
for _, row in ab.iterrows():
    code = row['孕妇代码']
    atyp = row['染色体的非整倍体']
    z13 = row['Z13_corrected']
    z18 = row['Z18_corrected']
    z21 = row['Z21_corrected']
    zx = row['ZX_corrected']
    fr = row['被过滤掉读段数的比例']
    print(f"{code:<10} {atyp:<10} {z13:8.3f} {z18:8.3f} {z21:8.3f} {zx:8.3f} {fr:8.4f}")
    
    if atyp not in type_groups:
        type_groups[atyp] = []
    type_groups[atyp].append((z13, z18, z21, zx))

print("\n--- 各异常类型的Z值均值 ---")
print(f"{'异常类型':<10} {'样本数':>6} {'Z13_mean':>10} {'Z18_mean':>10} {'Z21_mean':>10} {'ZX_mean':>10}")
for atyp in sorted(type_groups.keys()):
    arr = np.array(type_groups[atyp])
    n = len(arr)
    print(f"{atyp:<10} {n:>6} {arr[:,0].mean():10.3f} {arr[:,1].mean():10.3f} {arr[:,2].mean():10.3f} {arr[:,3].mean():10.3f}")

print("\n解读：观察各异常类型在对应染色体Z值上是否有明显升高。")

# ============================================================
# 2. 实验2：按染色体建判别器（核心）
# ============================================================
print("\n" + "=" * 70)
print("实验2：按染色体建判别器")
print("=" * 70)

# 构建特征矩阵
# 从 sub4 取：Z13/18/21_corrected, ZX_corrected, filter_rate, BMI, age
# 从 clean 取：GC含量 for chr13,18,21
sub4_dedup = sub4_dedup.reset_index(drop=True)
clean_dedup = clean_dedup.reset_index(drop=True)

# 构造特征
data = pd.DataFrame({
    '孕妇代码': sub4_dedup['孕妇代码'],
    'Z13': sub4_dedup['Z13_corrected'],
    'Z18': sub4_dedup['Z18_corrected'],
    'Z21': sub4_dedup['Z21_corrected'],
    'ZX': sub4_dedup['ZX_corrected'],
    'filter_rate': sub4_dedup['被过滤掉读段数的比例'],
    'BMI': sub4_dedup['孕妇BMI'],
    'age': sub4_dedup['年龄'],
    'GC13': clean_dedup['13号染色体的GC含量'],
    'GC18': clean_dedup['18号染色体的GC含量'],
    'GC21': clean_dedup['21号染色体的GC含量'],
    'atyp': sub4_dedup['染色体的非整倍体'].fillna('正常'),
    'AB': sub4_dedup['AB_异常'],
})

# 构造对比特征：每条染色体的Z值减去其他三条Z值的中位数
data['Z13_contrast'] = data['Z13'] - data[['Z18', 'Z21', 'ZX']].median(axis=1)
data['Z18_contrast'] = data['Z18'] - data[['Z13', 'Z21', 'ZX']].median(axis=1)
data['Z21_contrast'] = data['Z21'] - data[['Z13', 'Z18', 'ZX']].median(axis=1)

N = len(data)
print(f"总样本数: {N}")

# ============================================================
# 辅助函数
# ============================================================
def roc_auc(y_true, y_score):
    """手动计算 AUC (no sklearn)"""
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tpr = np.cumsum(y_true_sorted) / n_pos
    fpr = np.cumsum(1 - y_true_sorted) / n_neg
    # 梯形积分 (np.trapz -> np.trapezoid in newer numpy)
    try:
        return np.trapezoid(tpr, fpr)
    except AttributeError:
        return np.trapz(tpr, fpr)


def cohens_d(x_pos, x_neg):
    """Cohen's d 效应值"""
    m1, m2 = np.mean(x_pos), np.mean(x_neg)
    v1, v2 = np.var(x_pos, ddof=1), np.var(x_neg, ddof=1)
    n1, n2 = len(x_pos), len(x_neg)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return abs(m1 - m2) / pooled_std


def fisher_lda(X_pos, X_neg):
    """
    Fisher LDA: w ∝ Sw^{-1} (m1 - m0)
    返回权重向量 w 和阈值（按先验等概率，阈值为投影均值的中间点）
    """
    m1 = X_pos.mean(axis=0)
    m0 = X_neg.mean(axis=0)
    S1 = np.cov(X_pos, rowvar=False, bias=False)
    S0 = np.cov(X_neg, rowvar=False, bias=False)
    n1, n0 = X_pos.shape[0], X_neg.shape[0]
    Sw = ((n1 - 1) * S1 + (n0 - 1) * S0) / (n1 + n0 - 2)
    Sw_inv = np.linalg.pinv(Sw)
    w = Sw_inv @ (m1 - m0)
    # 归一化使 w 的 L2 norm = 1，便于跨染色体比较
    w_norm = np.linalg.norm(w)
    if w_norm > 1e-12:
        w = w / w_norm
    threshold = (w @ m1 + w @ m0) / 2.0
    return w, threshold


def loocv_auc(X, y, w):
    """留一法评估 AUC（使用 Fisher 投影分数）"""
    scores = np.zeros(len(y))
    for i in range(len(y)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        
        X_pos = X_train[y_train == 1]
        X_neg = X_train[y_train == 0]
        
        if len(X_pos) < 2 or len(X_neg) < 2:
            scores[i] = (X_test @ w).item() if hasattr(X_test @ w, 'item') else float(X_test @ w)
            continue
        
        w_i, _ = fisher_lda(X_pos, X_neg)
        scores[i] = (X_test @ w_i).item() if hasattr(X_test @ w_i, 'item') else float(X_test @ w_i)
    
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc(y, scores)


def loocv_scores_full(X, y):
    """留一法返回每个样本的 Fisher 投影分数"""
    scores = np.zeros(len(y))
    for i in range(len(y)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        
        X_pos = X_train[y_train == 1]
        X_neg = X_train[y_train == 0]
        
        if len(X_pos) < 2 or len(X_neg) < 2:
            w_i, _ = fisher_lda(X_pos, X_neg)
        else:
            w_i, _ = fisher_lda(X_pos, X_neg)
        scores[i] = (X_test @ w_i).item() if hasattr(X_test @ w_i, 'item') else float(X_test @ w_i)
    return scores


# ============================================================
# 对每条染色体建判别器
# ============================================================
chromosomes = {
    13: {'Z': 'Z13', 'GC': 'GC13', 'Z_contrast': 'Z13_contrast'},
    18: {'Z': 'Z18', 'GC': 'GC18', 'Z_contrast': 'Z18_contrast'},
    21: {'Z': 'Z21', 'GC': 'GC21', 'Z_contrast': 'Z21_contrast'},
}

# 所有候选特征
common_features = ['filter_rate', 'BMI', 'age']

results = {}
all_loocv_scores = {}  # 存储每个染色体的 LOOCV 分数

for k, cols in chromosomes.items():
    print(f"\n{'─' * 60}")
    print(f"染色体 {k} 号判别器")
    print(f"{'─' * 60}")
    
    # 正样本：异常类型中包含该染色体
    positive_mask = data['atyp'].apply(lambda x: f'T{k}' in str(x))
    negative_mask = data['AB'] == 0
    
    pos_data = data[positive_mask]
    neg_data = data[negative_mask]
    
    n_pos = len(pos_data)
    n_neg = len(neg_data)
    print(f"  正样本数: {n_pos}, 负样本数: {n_neg}")
    print(f"  正样本类型: {pos_data['atyp'].value_counts().to_dict()}")
    
    if n_pos < 2:
        print(f"  [WARN] 正样本不足 ({n_pos})，跳过")
        continue
    
    # 专用特征集
    chr_features = [cols['Z'], cols['GC'], cols['Z_contrast']] + common_features
    fname_map = {
        cols['Z']: f'Z{k}_corrected',
        cols['GC']: f'GC含量_chr{k}',
        cols['Z_contrast']: f'Z{k}_contrast(减其他Z中位数)',
        'filter_rate': '被过滤掉读段数的比例',
        'BMI': '孕妇BMI',
        'age': '年龄',
    }
    
    # 单特征筛选：Cohen's d
    print(f"\n  单特征 Cohen's d 筛选：")
    d_scores = {}
    for feat in chr_features:
        x_pos = pos_data[feat].values
        x_neg = neg_data[feat].values
        d = cohens_d(x_pos, x_neg)
        d_scores[feat] = d
        print(f"    {fname_map[feat]:<35s} d = {d:.4f}")
    
    # Top-3
    sorted_features = sorted(d_scores.items(), key=lambda x: x[1], reverse=True)
    top3_feats = [f for f, _ in sorted_features[:3]]
    print(f"\n  Top-3 特征: {[fname_map[f] for f in top3_feats]}")
    
    # 构造特征矩阵
    y_all = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    X_all = np.column_stack([
        np.concatenate([pos_data[f].values, neg_data[f].values]) for f in top3_feats
    ])
    
    # Fisher LDA (全量训练)
    X_pos_mat = X_all[y_all == 1]
    X_neg_mat = X_all[y_all == 0]
    w, threshold = fisher_lda(X_pos_mat, X_neg_mat)
    
    print(f"\n  Fisher LDA 权重 (归一化):")
    for i, f in enumerate(top3_feats):
        print(f"    {fname_map[f]:<35s} w = {w[i]:+.4f}")
    print(f"  阈值: {threshold:.4f}")
    
    # 留一法 CV AUC
    auc_val = loocv_auc(X_all, y_all, w)
    print(f"\n  留一法 CV AUC: {auc_val:.4f}")
    
    # 存储完整分数（对所有样本用全量训练的权重投影到top3特征上）
    X_full_top3 = data[top3_feats].values
    all_w, _ = fisher_lda(X_pos_mat, X_neg_mat)
    full_scores = X_full_top3 @ all_w
    
    results[k] = {
        'n_pos': n_pos,
        'auc': auc_val,
        'top3': top3_feats,
        'fnames': [fname_map[f] for f in top3_feats],
        'w': w,
        'threshold': threshold,
        'chr_features': chr_features,
    }
    all_loocv_scores[k] = full_scores

print("\n" + "=" * 70)
print("实验2 汇总")
print("=" * 70)
print(f"{'染色体':<8} {'正样本数':>8} {'AUC':>8} {'Top-3特征'}")
for k, r in results.items():
    print(f"Chr {k:<4} {r['n_pos']:>8} {r['auc']:>8.4f} {', '.join(r['fnames'])}")

# ============================================================
# 实验3：组合判定
# ============================================================
print("\n" + "=" * 70)
print("实验3：组合判定")
print("=" * 70)

# 收集有结果的染色体的 LOOCV 分数（对所有样本）
n_all = len(data)
y_true = (data['AB'] == 1).values.astype(int)

valid_chrs = sorted(results.keys())
n_valid = len(valid_chrs)
score_matrix = np.zeros((n_all, n_valid))
chr_labels = []

for idx, k in enumerate(valid_chrs):
    score_matrix[:, idx] = all_loocv_scores[k]
    chr_labels.append(f's_{k}')

# 组合方式
if n_valid >= 2:
    combinations = {
        f'max({", ".join(chr_labels)})': np.max(score_matrix, axis=1),
        f'mean({", ".join(chr_labels)})': np.mean(score_matrix, axis=1),
        f'{" + ".join(chr_labels)}': np.sum(score_matrix, axis=1),
    }
elif n_valid == 1:
    combinations = {chr_labels[0]: score_matrix[:, 0]}
else:
    combinations = {}

print(f"\n{'组合方式':<25s} {'AUC':>8}")
print("-" * 40)
best_method = None
best_auc = -1
for name, scores in combinations.items():
    auc_val = roc_auc(y_true, scores)
    print(f"{name:<25s} {auc_val:>8.4f}")
    if auc_val > best_auc:
        best_auc = auc_val
        best_method = name

print(f"\n最优组合方式: {best_method}, AUC = {best_auc:.4f}")

# ============================================================
# 实验4：最终三分类
# ============================================================
print("\n" + "=" * 70)
print("实验4：最终三分类（Youden阈值）")
print("=" * 70)

# 用最优组合方式
best_scores = combinations[best_method]

# 找 Youden 阈值 (max sensitivity + specificity - 1)
def find_youden_threshold(scores, y):
    """找最优 Youden 阈值"""
    sorted_idx = np.argsort(scores)
    scores_sorted = scores[sorted_idx]
    y_sorted = y[sorted_idx]
    
    best_j = -1
    best_t = 0
    
    # 在相邻分数之间扫描
    unique_scores = np.unique(scores_sorted)
    for t in unique_scores:
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        
        if j > best_j:
            best_j = j
            best_t = t
    
    return best_t, best_j

youden_t, youden_j = find_youden_threshold(best_scores, y_true)
print(f"\nYouden 阈值: {youden_t:.4f}, Youden指数 = {youden_j:.4f}")

# 在Youden阈值处计算敏感度和特异度
pred_youden = (best_scores >= youden_t).astype(int)
tp = np.sum((pred_youden == 1) & (y_true == 1))
fn = np.sum((pred_youden == 0) & (y_true == 1))
fp = np.sum((pred_youden == 1) & (y_true == 0))
tn = np.sum((pred_youden == 0) & (y_true == 0))

sens_youden = tp / (tp + fn)
spec_youden = tn / (tn + fp)
print(f"敏感性: {sens_youden:.4f} ({tp}/{tp+fn})")
print(f"特异性: {spec_youden:.4f} ({tn}/{tn+fp})")

# 三分类：高风险、不确定、低风险
# 使用双侧阈值（考虑分数高于或低于正常范围都可能异常）
# 首先观察分数的分布
print(f"\n分数分布统计：")
print(f"  正常组: mean={np.mean(best_scores[y_true==0]):.4f}, std={np.std(best_scores[y_true==0]):.4f}")
print(f"  异常组: mean={np.mean(best_scores[y_true==1]):.4f}, std={np.std(best_scores[y_true==1]):.4f}")
print(f"  正常组 范围: [{np.min(best_scores[y_true==0]):.4f}, {np.max(best_scores[y_true==0]):.4f}]")
print(f"  异常组 范围: [{np.min(best_scores[y_true==1]):.4f}, {np.max(best_scores[y_true==1]):.4f}]")

# 三分类策略：
# 高风险：分数 >= Youden阈值
# 低风险：分数 < 某个低阈值（比如正常组的mean - k*std以下，确保高特异度）
# 不确定：中间

# 用最优组合分数来做三分类
# 低风险阈值：正常组均值 - 0.5 std（保守）
# 或者：用 Youden 阈值和另一个低阈值分割

mu_normal = np.mean(best_scores[y_true == 0])
std_normal = np.std(best_scores[y_true == 0])

# 定义不确定区间
# 下阈值：正常组 mean - 1*std（确保极低风险）
# 实际上，对于 max 组合，分数越高越异常。所以：
# 低风险 = 分数在下阈值以下（很安全）
# 高风险 = 分数在上阈值以上（很危险）
# 不确定 = 中间

# 用 Youden 作为高风险阈值，再找一个低风险阈值（使得正常组中绝大多数被归为低风险）
# 低风险阈值 = 正常组第95百分位（即正常组中95%的人在此以下）
low_threshold = np.percentile(best_scores[y_true == 0], 95)

print(f"\n三分类阈值:")
print(f"  低风险阈值: {low_threshold:.4f} (正常组 P95)")
print(f"  高风险阈值: {youden_t:.4f} (Youden)")

# 三分类
low_risk = best_scores < low_threshold
high_risk = best_scores >= youden_t
uncertain = ~low_risk & ~high_risk

n_low = np.sum(low_risk)
n_uncertain = np.sum(uncertain)
n_high = np.sum(high_risk)

print(f"\n三分类结果 ({best_method}):")
print(f"{'类别':<12} {'人数':>6} {'占比':>8} {'其中真异常':>10} {'异常率':>8}")
print("-" * 55)

for name, mask in [('低风险', low_risk), ('不确定', uncertain), ('高风险', high_risk)]:
    n = np.sum(mask)
    n_ab = np.sum(y_true[mask] == 1)
    rate = n_ab / n if n > 0 else 0
    pct = n / n_all * 100
    print(f"{name:<12} {n:>6} {pct:>7.1f}% {n_ab:>10} {rate:>8.1%}")

# 与之前混合分类器的"不确定"比例对比
# 这里我们无法直接拿到之前的混合分类器结果，但我们可以参考数据中的不确定性
print(f"\n'不确定'类别占比: {n_uncertain/n_all*100:.1f}% ({n_uncertain}/{n_all})")
print(f"注：不确定区间定义为 [P95_正常组={low_threshold:.4f}, Youden={youden_t:.4f})")

# 额外分析：也报告仅用 Youden 做二分类的结果
print(f"\n仅二分类（Youden阈值）:")
print(f"  阳性预测 ({pred_youden.sum()} 人): 其中真异常 {tp}, 假阳性 {fp}")
print(f"  阴性预测 ({(~pred_youden.astype(bool)).sum()} 人): 其中假阴性 {fn}")
print(f"  准确率: {(tp+tn)/n_all:.4f}, F1: {2*tp/(2*tp+fp+fn):.4f}" if (2*tp+fp+fn)>0 else "")

# ============================================================
# 打印各个异常样本的详细分数
# ============================================================
print("\n" + "=" * 70)
print("各异常样本的三染色体分数明细")
print("=" * 70)
# 构建动态表头
chr_header_parts = [f'{"s_"+str(k):>8}' for k in valid_chrs]
for _ in range(len(chr_header_parts), 3):
    chr_header_parts.append(f'{"N/A":>8}')
print(f"{'孕妇代码':<10} {'异常类型':<10} {chr_header_parts[0]} {chr_header_parts[1]} {chr_header_parts[2]} {'综合分数':>10} {'判定':>6}")
print("-" * 65)

for i in range(n_all):
    if y_true[i] == 1:
        score_parts = []
        for j, k in enumerate(valid_chrs):
            score_parts.append(f'{score_matrix[i, j]:8.4f}')
        s_str = ' '.join(score_parts)
        # pad with blanks for missing chromosomes
        for _ in range(len(score_parts), 3):
            s_str += '      N/A'
        combo = best_scores[i]
        if high_risk[i]:
            verdict = '高风险'
        elif low_risk[i]:
            verdict = '低风险'
        else:
            verdict = '不确定'
        print(f"{data.iloc[i]['孕妇代码']:<10} {data.iloc[i]['atyp']:<10} {s_str} {combo:10.4f} {verdict:>6}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
