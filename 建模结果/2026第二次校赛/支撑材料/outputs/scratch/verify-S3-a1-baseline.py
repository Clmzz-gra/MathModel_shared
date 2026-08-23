"""
目的：
    S3 A 类验证 A1【简单基线·强制第一步】：直接迁移的 S1 风格模型（正则化 Logistic L2 + CLR）
    做 leave-one-disease-out 三组合跨疾病预测，建立"直接迁移"性能下界。

原理：
    - 对每个组合，仅用训练疾病样本拟合 LogisticRegression(L2, C=1.0, class_weight=balanced)，
      物种级特征；CLR 前置（成分数据零值乘法替换 δ，见 verify-S3-common）。
    - 防泄漏：CLR 为逐样本变换无泄漏；StandardScaler 均值/方差仅从训练集估计后应用到测试集。
    - 主指标 AUC（阈值无关）；辅指标 = 训练集 Youden J 最优阈值（max 灵敏度+特异度-1）迁移到
      测试集，报告 ACC / 灵敏度 / 特异度 / F1。禁止测试集重定阈值（评估泄漏）。
    - 输出 3 组合 AUC + 阈值迁移指标 + 3 组合 AUC 均值，并标注测试集正类占比。

性能：
    轻量-不适用（484×1331 小数据，Logistic 秒级；无并行需求）。

输入数据：
    - B-raw.pkl (处理后) — dataset_name, disease, 1331 物种级相对丰度特征

输出：
    - outputs/figures/_explore/S3-ldo-baseline-auc.pdf — 3 组合 AUC 柱状图 + 阈值迁移指标
    - stdout — 各组合 AUC / ACC / 灵敏度 / 特异度 / F1 / 正类占比 / AUC 均值

对应论文章节：
    §S3 跨疾病预测模型（A 类验证 A1，探索图不入论文）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_S3_common import (  # noqa: E402
    load_data, clr_transform, get_combo_data, FIG_DIR,
)

FIG_DIR.mkdir(parents=True, exist_ok=True)


def youden_threshold(y_true, y_score):
    """在给定标签上求 Youden J 最优阈值（max 灵敏度+特异度-1）。返回阈值。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx]


def run_combo(df, feature_cols, combo_name):
    X_train, y_train, X_test, y_test, pos_frac = get_combo_data(
        df, feature_cols, combo_name
    )
    # CLR（逐样本，无泄漏）
    Xtr = clr_transform(X_train)
    Xte = clr_transform(X_test)
    # 标准化：仅训练集估计
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    # 模型
    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=42,
    )
    clf.fit(Xtr_s, y_train)
    # 训练集得分 → Youden 阈值
    train_score = clf.predict_proba(Xtr_s)[:, 1]
    thr = youden_threshold(y_train, train_score)
    # 测试集预测
    test_score = clf.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(y_test, test_score)
    y_pred = (test_score >= thr).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return dict(auc=auc, acc=acc, f1=f1, sens=sens, spec=spec,
                pos_frac=pos_frac, thr=thr, n_test=len(y_test))


def main():
    df, feature_cols = load_data()
    results = {}
    print("=" * 70)
    print("A1 简单基线：直接迁移 Logistic L2 + CLR（物种级）leave-one-disease-out")
    print("=" * 70)
    for combo in ["C1", "C2", "C3"]:
        r = run_combo(df, feature_cols, combo)
        results[combo] = r
        print(f"\n[{combo}] 测试集 n={r['n_test']} 正类占比={r['pos_frac']:.2%}")
        print(f"  AUC={r['auc']:.4f}  (Youden阈值={r['thr']:.4f})")
        print(f"  ACC={r['acc']:.4f}  F1={r['f1']:.4f}  "
              f"灵敏度={r['sens']:.4f}  特异度={r['spec']:.4f}")
    combos = ["C1", "C2", "C3"]
    mean_auc = np.mean([results[c]["auc"] for c in combos])
    print(f"\n3 组合 AUC 均值 = {mean_auc:.4f}")

    # 落盘结果 JSON（供 A3 读取跨疾病 AUC，避免硬编码）
    res_json = {c: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in results[c].items()} for c in combos}
    res_json["mean_auc"] = float(mean_auc)
    json_path = Path(__file__).resolve().parent / "verify-S3-a1-results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res_json, f, indent=2, ensure_ascii=False)
    print(f"结果已落盘: {json_path}")

    # 图：AUC 柱状图 + 阈值迁移指标
    aucs = [results[c]["auc"] for c in combos]
    labels = ["C1\nCRC", "C2\nIBD", "C3\nObesity"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    bars = ax.bar(labels, aucs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="随机 0.5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("直接迁移跨疾病 AUC（物种级）")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.legend()
    # 阈值迁移指标表
    ax2 = axes[1]
    ax2.axis("off")
    rows = [["组合", "AUC", "ACC", "F1", "灵敏", "特异", "正类%"]]
    for c in combos:
        r = results[c]
        rows.append([c, f"{r['auc']:.3f}", f"{r['acc']:.3f}", f"{r['f1']:.3f}",
                     f"{r['sens']:.3f}", f"{r['spec']:.3f}", f"{r['pos_frac']:.0%}"])
    tbl = ax2.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax2.set_title("训练集 Youden J 阈值迁移指标", pad=20)
    fig.suptitle("S3 A1 简单基线：直接迁移性能下界", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "S3-ldo-baseline-auc.pdf"
    fig.savefig(out)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
