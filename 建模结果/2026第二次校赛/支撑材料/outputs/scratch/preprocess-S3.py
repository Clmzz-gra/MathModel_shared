"""
目的：
    S3 跨疾病预测模型（leave-one-disease-out 泛化评估）的 1.4 数据预处理：
    从共享清洗数据 c-data-cleaned.pkl 生成 S3-preprocessed.pkl，为 2.1 正式建模
    （四策略 A/B/C/D + 紧急回退 R1-R4）准备 LODO 三组合划分、二分类标签、
    近全零过滤特征集、CLR 变换、分类学层级聚合、共享特征交集与特征名元数据。

原理：
    - 标签映射（题面口径，与 S1/S2 一致）：患病=1（cancer / ibd_ulcerative_colitis /
      ibd_crohn_disease / obesity），健康=0（n / small_adenoma / leaness）。
      small_adenoma 未选定 S1 主口径前按题面口径归健康（见 handoff §三 R4）。
    - 近全零过滤（三病并集统一口径）：对全部 484 样本计算每特征零值占比，
      零值占比 > 95% 的特征剔除，1331 → 264（与 S1/S2 一致，规则见 S2
      verify-S2-v2-zerobin.py 的 zero_fraction<=0.95 口径）。
    - CLR（中心对数比变换）：对定和成分数据（每行丰度和≈100）解除定和约束。
      零值用乘法替换 δ=0.65×检出限（检出限=全局最小非零值 1e-5，故 δ=6.5e-6），
      再逐样本 log 后减去行均值（等价于 log(x_i/g(x))，g=几何均值，即几何均值中心化）。
      CLR 是逐样本变换、无跨样本参数，故不引入训练/测试泄漏。
    - 分类学层级聚合：特征名形如 k__..|p__..|c__..|o__..|f__..|g__X|s__Y，
      属级=按 g__ 段聚合（同属物种丰度求和，264→106），门级=按 p__ 段聚合（264→11）。
    - 共享特征交集（策略 B 用）：某特征在某数据集内"存在"=平均丰度>0（至少一个样本
      非零）；三数据集（CRC/IBD/Obesity）都存在的特征为共享交集。在过滤后 264 特征集
      内按特征名交集重算（A 类验证在 1331 全集测得 344，正式实现基于 264 重算得 252）。
      仅用特征存在性，绝不用测试集标签（转导式边界，B5 裁定）。
    - LODO 三组合（leave-one-disease-out）：C1 训练{metahit,Chatelier}测Zeller(CRC)；
      C2 训练{Zeller,Chatelier}测metahit(IBD)；C3 训练{Zeller,metahit}测Chatelier(Obesity)。
      样本索引（0-based 行号）预生成存 pkl，测试疾病在训练阶段完全不可见。

性能：
    轻量-不适用（484×1331 小数据，纯向量化过滤/聚合/集合运算，秒级，无并行需求）。

输入数据：
    - c-data-cleaned.pkl (处理后) — dataset_name, disease, 1331 个物种级相对丰度特征
      （484 样本 × 1333 列；dataset_name ∈ {Zeller_fecal_colorectal_cancer=121,
      metahit=110, Chatelier_gut_obesity=253}；disease ∈ {obesity=164, n=132, leaness=89,
      cancer=48, small_adenoma=26, ibd_ulcerative_colitis=21, ibd_crohn_disease=4}）

输出：
    - outputs/data/S3-preprocessed.pkl — 预处理中间产物（结构见下）
    - outputs/data/preprocess-report-S3.txt — 预处理关键数字报告

对应论文章节：
    §S3 跨疾病预测模型（1.4 数据预处理，中间产物，不入论文正文）
"""
from pathlib import Path
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "outputs" / "data" / "c-data-cleaned.pkl"
OUT_PKL = ROOT / "outputs" / "data" / "S3-preprocessed.pkl"
OUT_REPORT = ROOT / "outputs" / "data" / "preprocess-report-S3.txt"

# 检出限 = 全局最小非零丰度（inventory-B.txt: min=1e-05）
DETECTION_LIMIT = 1e-5
CLR_DELTA = 0.65 * DETECTION_LIMIT  # 乘法替换 δ = 6.5e-6

# 数据集 → 疾病名
DATASET_DISEASE = {
    "Zeller_fecal_colorectal_cancer": "CRC",
    "metahit": "IBD",
    "Chatelier_gut_obesity": "Obesity",
}

# 患病标签映射（1=患病，0=健康）
POSITIVE_LABELS = {"cancer", "ibd_ulcerative_colitis", "ibd_crohn_disease", "obesity"}

# leave-one-disease-out 三组合：训练数据集列表 → 测试数据集
COMBOS = {
    "C1": (["metahit", "Chatelier_gut_obesity"], "Zeller_fecal_colorectal_cancer"),
    "C2": (["Zeller_fecal_colorectal_cancer", "Chatelier_gut_obesity"], "metahit"),
    "C3": (["Zeller_fecal_colorectal_cancer", "metahit"], "Chatelier_gut_obesity"),
}

# 分类学层级段前缀（7 级）
TAX_LEVELS = ["k", "p", "c", "o", "f", "g", "s"]


def binary_label(disease_series):
    """disease 列 → 二分类标签（1=患病，0=健康）。"""
    return disease_series.map(lambda x: 1 if x in POSITIVE_LABELS else 0).astype(int)


def clr_transform(X):
    """CLR 变换（逐样本）：零值乘法替换 δ → log → 逐样本减去行均值（几何均值中心化）。

    接受 DataFrame 或 ndarray，返回同类型。无跨样本参数，不引入训练/测试泄漏。
    """
    is_df = isinstance(X, pd.DataFrame)
    cols = X.columns if is_df else None
    idx = X.index if is_df else None
    arr = X.to_numpy(dtype=float) if is_df else np.asarray(X, dtype=float)
    arr = np.where(arr == 0.0, CLR_DELTA, arr)
    logX = np.log(arr)
    out = logX - logX.mean(axis=1, keepdims=True)
    if is_df:
        return pd.DataFrame(out, index=idx, columns=cols)
    return out


def taxonomy_aggregate(X, level):
    """按分类学层级聚合特征（同层丰度求和）。level ∈ {'species','genus','phylum'}。

    species=原特征名；genus=按 g__ 段；phylum=按 p__ 段。返回聚合后 DataFrame。
    """
    if level == "species":
        return X.copy()
    seg = "g__" if level == "genus" else "p__"

    def key(col):
        parts = col.split("|")
        for p in parts:
            if p.startswith(seg):
                return p
        return col  # 兜底：无该段则用原名

    # pandas 3.0 移除 groupby(axis=1)，改用转置后按行分组再转回
    return X.T.groupby(key).sum().T


def parse_taxonomy(feature_name):
    """拆分特征名 k__p__c__o__f__g__s__ 七级，返回 dict（缺段为 None）。"""
    parsed = {lv: None for lv in TAX_LEVELS}
    for part in feature_name.split("|"):
        for lv in TAX_LEVELS:
            prefix = lv + "__"
            if part.startswith(prefix):
                parsed[lv] = part[len(prefix):]
                break
    return parsed


def present_features(X, dataset_mask):
    """某数据集内"存在"（平均丰度>0，即至少一个样本非零）的特征集合。"""
    sub = X[dataset_mask]
    return set(sub.columns[sub.mean(axis=0) > 0])


def main():
    df = pd.read_pickle(DATA)
    feature_cols = [c for c in df.columns if c not in ("dataset_name", "disease")]
    X_all = df[feature_cols].astype(float)

    # 1. 近全零过滤（三病并集统一口径：零值占比>95% 剔除）
    zero_frac = (X_all == 0).mean(axis=0)
    keep_mask = zero_frac <= 0.95
    keep_cols = [c for c, k in zip(feature_cols, keep_mask) if k]
    X_filtered = X_all[keep_cols]

    # 2. 标签
    y = binary_label(df["disease"]).to_numpy()

    # 3. LODO 三组合样本索引（0-based 行号）
    lodo_combos = {}
    for combo_name, (train_ds, test_ds) in COMBOS.items():
        train_idx = np.where(df["dataset_name"].isin(train_ds).to_numpy())[0]
        test_idx = np.where((df["dataset_name"] == test_ds).to_numpy())[0]
        lodo_combos[combo_name] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_datasets": train_ds,
            "test_dataset": test_ds,
            "test_disease": DATASET_DISEASE[test_ds],
        }

    # 4. 共享特征交集（过滤后 264 内，三病按特征名交集）
    ds = df["dataset_name"].to_numpy()
    present_sets = {
        d: present_features(X_filtered, ds == d) for d in DATASET_DISEASE
    }
    shared_features = sorted(set.intersection(*present_sets.values()))

    # 5. 分类学层级聚合后的特征名（供策略 C 用）
    genus_agg = taxonomy_aggregate(X_filtered, "genus")
    phylum_agg = taxonomy_aggregate(X_filtered, "phylum")
    genus_features = list(genus_agg.columns)
    phylum_features = list(phylum_agg.columns)

    # 6. 特征名元数据（k__p__c__o__f__g__s__ 拆分）
    feature_taxonomy = {c: parse_taxonomy(c) for c in keep_cols}

    # 组装 pkl
    meta = {
        "sub": "S3",
        "stage": "1.4",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "outputs/data/c-data-cleaned.pkl",
        "seed": 42,
        "clr_delta": CLR_DELTA,
        "detection_limit": DETECTION_LIMIT,
        "filter_rule": "zero_fraction > 0.95 removed (union across 3 diseases), 1331 -> 264",
        "label_rule": "患病=1(cancer/ibd_ulcerative_colitis/ibd_crohn_disease/obesity), "
                      "健康=0(n/small_adenoma/leaness); small_adenoma 按题面口径归健康",
        "note": "预处理中间产物，供 2.1 正式建模（四策略 A/B/C/D + 回退 R1-R4）使用；"
                "clr_transform / taxonomy_aggregate 函数定义见 outputs/scratch/preprocess-S3.py，"
                "2.1 通过 importlib 导入复用（pkl 不存函数对象，防 pickle 跨模块引用失效）",
        "taxonomy_levels": ["species", "genus", "phylum"],
        "field_semantics": {
            "lodo_combos.<C>.train_idx": "训练集样本 0-based 行号（2 疾病并集）",
            "lodo_combos.<C>.test_idx": "测试集样本 0-based 行号（1 疾病，训练阶段完全不可见）",
            "shared_features": "过滤后 264 内三病按特征名交集（存在=平均丰度>0），供策略 B",
            "genus_features": "属级(g__)聚合后特征名，供策略 C",
            "phylum_features": "门级(p__)聚合后特征名，供策略 C",
        },
    }

    payload = {
        "meta": meta,
        "X_filtered": X_filtered,          # 484 × 264 过滤后物种级丰度（原始，未 CLR）
        "y": y,                            # (484,) 二分类标签
        "dataset_name": ds,                # (484,) 数据集名
        "disease": df["disease"].to_numpy(),  # (484,) 原始 disease 标签
        "feature_names": keep_cols,        # 264 过滤后特征名
        "feature_taxonomy": feature_taxonomy,  # 特征名 → 七级分类学拆分
        "lodo_combos": lodo_combos,        # 三组合样本索引
        "shared_features": shared_features,  # 252 共享特征交集
        "genus_features": genus_features,  # 106 属级聚合特征名
        "phylum_features": phylum_features,  # 11 门级聚合特征名
    }

    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 报告
    lines = []
    lines.append("=" * 70)
    lines.append("S3 1.4 数据预处理报告")
    lines.append("=" * 70)
    lines.append(f"输入: {DATA.name}  (484 样本 × 1331 特征)")
    lines.append(f"近全零过滤: 零值占比>95% 剔除，1331 -> {len(keep_cols)} 特征")
    lines.append(f"CLR δ: {CLR_DELTA:.2e} (0.65 × 检出限 {DETECTION_LIMIT:.0e})")
    lines.append("")
    lines.append("LODO 三组合样本数:")
    for combo_name in ["C1", "C2", "C3"]:
        c = lodo_combos[combo_name]
        n_train = len(c["train_idx"])
        n_test = len(c["test_idx"])
        test_pos = int(y[c["test_idx"]].sum())
        test_neg = n_test - test_pos
        lines.append(
            f"  {combo_name}: 训练 {n_train} (2 疾病) / 测试 {n_test} "
            f"({c['test_disease']}, 患病 {test_pos} / 健康 {test_neg})"
        )
    lines.append("")
    lines.append(f"共享特征交集 (过滤后 264 内三病按特征名交集): {len(shared_features)}")
    lines.append(f"属级(g__)聚合后维度: {len(genus_features)}")
    lines.append(f"门级(p__)聚合后维度: {len(phylum_features)}")
    lines.append("")
    lines.append(f"输出: {OUT_PKL.name}")
    lines.append(f"报告: {OUT_REPORT.name}")
    report_text = "\n".join(lines) + "\n"

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    # stdout 关键数字
    print(report_text)


if __name__ == "__main__":
    main()
