"""
问题4 新方法：对比特征 + 简单分类器
纯 numpy/scipy/pandas 实现，不依赖 sklearn
"""
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("1. 加载数据")
print("=" * 70)
df = pd.read_pickle(r'e:\MathModel\problems\2025\C题\outputs\data\2025C-sub4-preprocessed.pkl')
print(f"总样本数: {len(df)}")
print(f"孕妇数: {df['孕妇代码'].nunique()}")
print(f"AB_异常分布: 0={ (df['AB_异常']==0).sum()}, 1={ (df['AB_异常']==1).sum()}")

# ============================================================
# 2. 构造对比特征
# ============================================================
print("\n" + "=" * 70)
print("2. 构造对比特征")
print("=" * 70)

Z = df[['Z13_corrected', 'Z18_corrected', 'Z21_corrected', 'ZX_corrected']].values
z13, z18, z21, zx = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]

# 每个染色体与其他三个的中位数之差
df['diff_13'] = z13 - np.median(np.column_stack([z18, z21, zx]), axis=1)
df['diff_18'] = z18 - np.median(np.column_stack([z13, z21, zx]), axis=1)
df['diff_21'] = z21 - np.median(np.column_stack([z13, z18, zx]), axis=1)
df['diff_X']  = zx  - np.median(np.column_stack([z13, z18, z21]), axis=1)

# max_diff 和 which_max
diff_all = df[['diff_13', 'diff_18', 'diff_21', 'diff_X']].values
df['max_diff'] = np.max(diff_all, axis=1)
df['which_max'] = np.argmax(diff_all, axis=1)  # 0=13, 1=18, 2=21, 3=X

# 原始Z值最大值
raw_Z = df[['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']].values
df['max_raw_Z'] = np.max(np.abs(raw_Z), axis=1)

print("新增特征: diff_13, diff_18, diff_21, diff_X, max_diff, which_max, max_raw_Z")
print(f"max_diff 范围: [{df['max_diff'].min():.3f}, {df['max_diff'].max():.3f}]")
print(f"max_raw_Z 范围: [{df['max_raw_Z'].min():.3f}, {df['max_raw_Z'].max():.3f}]")

# ============================================================
# 辅助函数
# ============================================================
def compute_roc_auc(y_true, y_score):
    """手动计算 ROC-AUC (trapezoidal rule)"""
    idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[idx]
    y_score_sorted = y_score[idx]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return float('nan')

    tp, fp = 0, 0
    prev_score = None
    tpr_list, fpr_list = [], []

    for i in range(len(y_true_sorted)):
        if y_score_sorted[i] != prev_score:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
            prev_score = y_score_sorted[i]
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
    tpr_list.append(tp / n_pos)
    fpr_list.append(fp / n_neg)

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)
    sort_idx = np.argsort(fpr)
    fpr = fpr[sort_idx]
    tpr = tpr[sort_idx]

    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0)
    return auc

def confusion_matrix_manual(y_true, y_pred):
    """返回 (TN, FP, FN, TP)"""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp

def print_cm(tn, fp, fn, tp):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    print(f"    混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"    Accuracy={acc:.4f}, Sensitivity={sens:.4f}, Specificity={spec:.4f}")
    print(f"    PPV={ppv:.4f}, NPV={npv:.4f}")

# ============================================================
# 3. 标注样本评估（按孕妇去重）
# ============================================================
print("\n" + "=" * 70)
print("3. 标注样本评估（按孕妇去重，保留最早检测）")
print("=" * 70)

df_dedup = df.loc[df.groupby('孕妇代码')['检测抽血次数'].idxmin()].copy()
print(f"去重后样本数: {len(df_dedup)}")
print(f"去重后孕妇数: {df_dedup['孕妇代码'].nunique()}")
print(f"去重后 AB_异常分布: 0={ (df_dedup['AB_异常']==0).sum()}, 1={ (df_dedup['AB_异常']==1).sum()}")

y = df_dedup['AB_异常'].values

# 3a. max_diff 单特征 ROC-AUC
print("\n--- 3a. 仅用 max_diff 的 ROC-AUC ---")
auc_max_diff = compute_roc_auc(y, df_dedup['max_diff'].values)
print(f"  ROC-AUC = {auc_max_diff:.4f}")

# 3b. max_raw_Z 单特征 ROC-AUC
print("\n--- 3b. 仅用 max_raw_Z 的 ROC-AUC ---")
auc_max_raw_z = compute_roc_auc(y, df_dedup['max_raw_Z'].values)
print(f"  ROC-AUC = {auc_max_raw_z:.4f}")

# 3c. 简单规则: max_diff > 2 → 判异常
print("\n--- 3c. 简单规则: max_diff > 2 → 判异常 ---")
y_pred_rule = (df_dedup['max_diff'].values > 2).astype(int)
tn, fp, fn, tp = confusion_matrix_manual(y, y_pred_rule)
print_cm(tn, fp, fn, tp)

# ============================================================
# 4. Logistic 回归（手动实现梯度下降）
# ============================================================
print("\n" + "=" * 70)
print("4. Logistic 回归（手动梯度下降 + L2 正则化）")
print("=" * 70)

def standardize(X_train, X_test=None):
    """z-score 标准化"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    X_train_scaled = (X_train - mean) / std
    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, mean, std
    return X_train_scaled, mean, std

class LogisticRegressionManual:
    """手动实现 Logistic 回归（梯度下降 + L2 正则化）"""
    def __init__(self, learning_rate=0.01, max_iter=5000, lambda_reg=0.1, tol=1e-6):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for iteration in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = expit(z)

            eps = 1e-15
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            loss += 0.5 * self.lambda_reg * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            dz = y_pred - y
            dw = np.dot(X.T, dz) / n_samples + self.lambda_reg * self.weights
            db = np.mean(dz)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                break

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return expit(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

feature_cols_all = [
    'diff_13', 'diff_18', 'diff_21', 'diff_X',
    'GC含量', 'read_depth_log', '在参考基因组上比对的比例',
    '重复读段的比例', '被过滤掉读段数的比例', '孕妇BMI', '年龄'
]

feature_cols_4 = ['diff_13', 'diff_18', 'diff_21', 'diff_X']

X_all_raw = df_dedup[feature_cols_all].values.astype(np.float64)
X_4_raw   = df_dedup[feature_cols_4].values.astype(np.float64)
y_all     = df_dedup['AB_异常'].values.astype(np.float64)
preg_ids  = df_dedup['孕妇代码'].values

print(f"特征矩阵中 NaN 数量 (all): {np.sum(np.isnan(X_all_raw))}")
print(f"特征矩阵中 NaN 数量 (4):   {np.sum(np.isnan(X_4_raw))}")
X_all_raw = np.nan_to_num(X_all_raw, nan=0.0)
X_4_raw   = np.nan_to_num(X_4_raw, nan=0.0)

def leave_one_pregnant_out_cv(X_raw, y, preg_ids, desc=""):
    """留一法 CV：每次留出一个孕妇"""
    unique_pregs = np.unique(preg_ids)
    all_probas = np.zeros(len(y))

    for i, held_out_preg in enumerate(unique_pregs):
        test_idx = np.where(preg_ids == held_out_preg)[0]
        train_idx = np.where(preg_ids != held_out_preg)[0]

        X_train_raw = X_raw[train_idx]
        y_train = y[train_idx]
        X_test_raw = X_raw[test_idx]

        X_train, mean, std = standardize(X_train_raw)
        X_test = (X_test_raw - mean) / std
        X_test = np.nan_to_num(X_test, nan=0.0)

        model = LogisticRegressionManual(learning_rate=0.01, max_iter=5000, lambda_reg=0.1)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        all_probas[test_idx] = probas

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(unique_pregs)}] folds done...")

    auc = compute_roc_auc(y, all_probas)
    y_pred = (all_probas >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix_manual(y, y_pred)

    print(f"\n  [{desc}]")
    print(f"  CV ROC-AUC = {auc:.4f}")
    print_cm(tn, fp, fn, tp)
    return auc, all_probas, y

# 4a. 全部特征
print("\n--- 4a. 全部特征 Logistic 回归（留一法CV） ---")
auc_all, probas_all, y_cv = leave_one_pregnant_out_cv(
    X_all_raw, y_all, preg_ids, desc="全部特征"
)

# 4b. 仅4个对比特征
print("\n--- 4b. 仅4个对比特征 Logistic 回归（留一法CV） ---")
auc_4, probas_4, _ = leave_one_pregnant_out_cv(
    X_4_raw, y_all, preg_ids, desc="仅4对比特征"
)

# ============================================================
# 5. 汇总
# ============================================================
print("\n" + "=" * 70)
print("5. 结果汇总")
print("=" * 70)
print(f"{'方法':<45} {'ROC-AUC':>10}")
print("-" * 57)
print(f"{'max_diff 单特征 (去重后)':<45} {auc_max_diff:>10.4f}")
print(f"{'max_raw_Z 单特征 (去重后)':<45} {auc_max_raw_z:>10.4f}")
print(f"{'全特征 Logistic CV (11特征+L2)':<45} {auc_all:>10.4f}")
print(f"{'4对比特征 Logistic CV (L2)':<45} {auc_4:>10.4f}")

print("\n简单规则 max_diff > 2:")
print_cm(tn, fp, fn, tp)

print("\n" + "=" * 70)
print("完成。")
print("=" * 70)
