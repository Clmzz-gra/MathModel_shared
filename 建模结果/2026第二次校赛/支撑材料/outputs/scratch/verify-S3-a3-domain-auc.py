"""
目的：
    S3 A 类验证 A3【各疾病域内 AUC】：用与 A1 相同的正则化 Logistic L2 + CLR 对每疾病
    数据集做域内分层 CV 评估，得三疾病域内 AUC 参考值，量化"跨疾病 AUC − 域内 AUC"衰减量。

原理：
    - 域内评估：对每个疾病数据集单独做 5 折分层 CV（StratifiedKFold，shuffle，seed=42），
      每折内 CLR（逐样本）+ StandardScaler（仅训练折估计）+ LogisticRegression(L2, balanced)，
      取 5 折 AUC 均值作为该疾病域内 AUC 参考。
    - 衰减量 = 跨疾病 AUC（A1 结果，测试该疾病的那一组合）− 该疾病域内 AUC。
      映射：CRC←C1、IBD←C2、Obesity←C3。
    - 预期：CRC/IBD 信号强（域内 AUC 高），Obesity 信号弱；跨疾病衰减与信号强度相关。

性能：
    轻量-不适用（每疾病 ≤253 样本 × 1331 特征，5 折 CV 秒级）。

输入数据：
    - B-raw.pkl (处理后) — dataset_name, disease, 1331 物种级相对丰度特征

输出：
    - outputs/figures/_explore/S3-domain-vs-cross-auc.pdf — 域内 vs 跨疾病 AUC 对比柱状图
    - stdout — 三疾病域内 AUC 表 + 衰减量表

对应论文章节：
    §S3 跨疾病预测模型（A 类验证 A3，探索图不入论文）
"""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # 修复 GBK 控制台中文/减号打印崩溃
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_S3_common import (  # noqa: E402
    load_data, clr_transform, binary_label, FIG_DIR, DATASET_DISEASE,
)

FIG_DIR.mkdir(parents=True, exist_ok=True)

# 跨疾病 AUC 映射：CRC←C1、IBD←C2、Obesity←C3（从 A1 结果 JSON 读取）
COMBO_TO_DISEASE = {"C1": "CRC", "C2": "IBD", "C3": "Obesity"}


def domain_auc(df, feature_cols, dataset, n_splits=5):
    sub = df[df["dataset_name"] == dataset]
    X = sub[feature_cols].astype(float)
    y = binary_label(sub["disease"])
    Xc = clr_transform(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, te_idx in skf.split(Xc, y):
        Xtr = Xc.iloc[tr_idx]
        Xte = Xc.iloc[te_idx]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=42)
        clf.fit(scaler.transform(Xtr), y.iloc[tr_idx])
        score = clf.predict_proba(scaler.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(y.iloc[te_idx], score))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    df, feature_cols = load_data()
    print("=" * 70)
    print("A3 各疾病域内 AUC（5 折分层 CV，Logistic L2 + CLR）")
    print("=" * 70)
    domain = {}
    for ds, disease in DATASET_DISEASE.items():
        auc, std = domain_auc(df, feature_cols, ds)
        domain[disease] = auc
        print(f"  {disease} ({ds}): 域内 AUC = {auc:.4f} ± {std:.4f}")

    # 跨疾病 AUC（从 A1 结果 JSON 读取）
    json_path = Path(__file__).resolve().parent / "verify-S3-a1-results.json"
    cross = {"CRC": None, "IBD": None, "Obesity": None}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            a1 = json.load(f)
        for combo, disease in COMBO_TO_DISEASE.items():
            cross[disease] = a1[combo]["auc"]
        print("\n  跨疾病 AUC（来自 A1）:")
        for d in ["CRC", "IBD", "Obesity"]:
            print(f"    {d}: 跨疾病 AUC = {cross[d]:.4f}")
        print("\n  衰减量（跨疾病 − 域内）:")
        for d in ["CRC", "IBD", "Obesity"]:
            print(f"    {d}: {cross[d]:.4f} − {domain[d]:.4f} = "
                  f"{cross[d] - domain[d]:+.4f}")
    else:
        print("\n  [警告] 未找到 A1 结果 JSON，请先运行 verify-S3-a1-baseline.py")

    # 图：域内 vs 跨疾病 AUC 对比
    diseases = ["CRC", "IBD", "Obesity"]
    dom_vals = [domain[d] for d in diseases]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(diseases))
    ax.bar(x, dom_vals, 0.4, label="域内 AUC（5折CV）", color="#4C72B0")
    if all(cross[d] is not None for d in diseases):
        cross_vals = [cross[d] for d in diseases]
        ax.bar(x + 0.4, cross_vals, 0.4, label="跨疾病 AUC（A1）", color="#DD8452")
        for xi, cv in zip(x + 0.4, cross_vals):
            ax.text(xi, cv + 0.02, f"{cv:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x, dom_vals):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("各疾病域内 AUC vs 跨疾病 AUC")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "S3-domain-vs-cross-auc.pdf"
    fig.savefig(out)
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
