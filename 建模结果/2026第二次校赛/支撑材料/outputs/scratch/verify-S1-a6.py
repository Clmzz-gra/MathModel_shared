"""
目的：
    S1 A 类验证 #6：small_adenoma 敏感性（剔除 26 例 small_adenoma 重训 Zeller CRC 模型对比 AUC）。

原理：
    Zeller CRC 健康对照含 26 例 small_adenoma（癌前病变，介于癌与健康间），可能污染健康对照纯度。
    对比「cancer vs (n + small_adenoma)」全口径 vs「cancer vs n」剔除口径下 Logistic(L2)+CLR 与 RF 的 5 折 CV AUC，
    量化口径影响（registry B 级销项证据）。若 AUC 变化 > 0.05 → 报告注明口径影响。

性能：
    轻量-不适用（Zeller 121/95 样本 × 5 折 × 2 模型，秒级）。

输入数据：
    - B-raw.pkl（原始）— dataset_name, disease, 1331 物种丰度

输出：
    - outputs/figures/_explore/S1-adenoma-explore.pdf — 剔除前后 AUC 对比
    - stdout — 两口径两模型 CV AUC

对应论文章节：
    §1.1 A 类验证（探索，不入论文）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import utils


def main():
    df = utils.load_data()
    name = "Zeller_fecal_colorectal_cancer"
    sub = df[df["dataset_name"] == name].copy()
    # 全口径：cancer=1, n+small_adenoma=0
    full = sub.copy()
    Xf = full.drop(columns=["dataset_name", "disease"]).values.astype(np.float64)
    yf = full["disease"].isin(["cancer"]).astype(int).values
    # 剔除口径：cancer=1, n=0（去掉 small_adenoma）
    rem = sub[sub["disease"].isin(["cancer", "n"])].copy()
    Xr = rem.drop(columns=["dataset_name", "disease"]).values.astype(np.float64)
    yr = rem["disease"].isin(["cancer"]).astype(int).values
    print(f"full: n={len(yf)} pos={int(yf.sum())} | removed: n={len(yr)} pos={int(yr.sum())}")

    Xfc = utils.clr_transform(Xf)
    Xrc = utils.clr_transform(Xr)
    r_l2_full = utils.cv_evaluate(Xfc, yf, utils.make_logistic, k=5, minority=1)
    r_l2_rem = utils.cv_evaluate(Xrc, yr, utils.make_logistic, k=5, minority=1)
    r_rf_full = utils.cv_evaluate(Xf, yf, utils.make_rf, k=5, minority=1)
    r_rf_rem = utils.cv_evaluate(Xr, yr, utils.make_rf, k=5, minority=1)
    print(f"L2(CLR): full auc={r_l2_full['auc']:.3f} (+-{r_l2_full['auc_std']:.3f}) | "
          f"removed auc={r_l2_rem['auc']:.3f} (+-{r_l2_rem['auc_std']:.3f}) | diff={r_l2_rem['auc']-r_l2_full['auc']:+.3f}")
    print(f"RF(raw): full auc={r_rf_full['auc']:.3f} (+-{r_rf_full['auc_std']:.3f}) | "
          f"removed auc={r_rf_rem['auc']:.3f} (+-{r_rf_rem['auc_std']:.3f}) | diff={r_rf_rem['auc']-r_rf_full['auc']:+.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["L2(CLR) full", "L2(CLR) removed", "RF(raw) full", "RF(raw) removed"]
    vals = [r_l2_full["auc"], r_l2_rem["auc"], r_rf_full["auc"], r_rf_rem["auc"]]
    errs = [r_l2_full["auc_std"], r_l2_rem["auc_std"], r_rf_full["auc_std"], r_rf_rem["auc_std"]]
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#55A868"]
    ax.bar(labels, vals, yerr=errs, capsize=4, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("CV AUC")
    ax.set_ylim(0, 1)
    ax.set_title("S1 small_adenoma sensitivity (Zeller CRC)")
    fig.tight_layout()
    out = utils.ensure_fig_dir() / "S1-adenoma-explore.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
