# 交接：S2 建模 → 代码（正式交接，1.4 预处理）

> handoff_type: `code-agent`
> sub: S2（特征选择与生物标志物）
> stage: 1.3（方案确认，门禁 1 材料）
> from: 建模对话 | to: 代码对话
> 日期：2026-08-21
> source_docs: [`approach-S2-confirmed.md`, `decision-tree-S2.md`, `debate-S2.md`]
> next_action: 代码对话 1.4 预处理（与建模 2.0 并行），说「继续」
> status: ready

---

## 0. 交接目标

按 `approach-S2-confirmed.md` 已确认方案，实现 S2 的**数据预处理**（近全零过滤 + CLR 前置 + 分层 CV 折内 Lasso+bootstrap 稳定性选择 + 两路信号检验 + **共现分析初探** + RF/VIP 佐证），产出 `S2-results.pkl`。本交接为**正式实现**（非 A 类验证的只读探索），产出正式结果文件。

---

## 1. 规格（实现约束）

### 1.1 过滤规则

- **近全零过滤**：零值占比 >95% 的特征直接剔除（1067 个），保留 264 维。
- 每病独立计算零值占比；过滤口径需与 A 类验证 V2 一致（`verify-S2-v2-zerobin.py` 参考）。

### 1.2 CLR 口径

- **乘法替换**：零值替换为伪计数 δ（**与 S1 一致**，见 `proxy-replacement-checklist-S2.md` P6），再重归一化到和为 1。
- **几何均值中心化**：$\text{clr}(x)_j=\ln x_j-\frac{1}{D}\sum_k\ln x_k$。
- 参考实现：`outputs/scratch/utils.py`（CLR 函数）。

### 1.3 Lasso 参数

- `LogisticRegression(penalty='l1', solver='liblinear' 或 'saga', C=...)`。
- **C 范围**：V6 用 C=0.1；正式实现给 C 范围（如 0.01~1.0），bootstrap 内每折的 C 选择方式（固定 vs 交叉验证）见 `proxy-replacement-checklist-S2.md` P7。
- **bootstrap 轮数**：建议 **50~100**（P4），每轮分层重抽样（按病/健比例）。

### 1.4 稳定性选择

- 频率 $\hat{\pi}_j=\frac{1}{B}\sum_b\mathbb{1}\{\hat{\beta}_j^{(b)}\neq 0\}$。
- 阈值 **τ=0.5~0.6**（暂定 0.5，P1）。
- **分层 CV 折内选择**（防泄漏）：特征选择在训练折内做，报告同时给「全量（乐观）」与「CV 内稳定频率（诚实）」。

### 1.5 两路信号检验

- **(a) 存在/缺失**：Fisher 精确检验（2×2 列联表：存在/缺失 × 病/健）。
- **(b) 非零丰度**：非零样本上 Wilcoxon 秩和检验（CLR 后丰度）。
- 两路均做 **BH-FDR 校正（α=0.05，P3；m=1331 全特征规模——2026-08-21 人类裁定：医疗宁可严格不可虚报。检验对过滤后 264 特征执行，多重比较按全 1331 计数，报告仅展示稳定特征校正后显著性）**。

### 1.5b 共现分析（协同效应初探，2026-08-21 人类裁定采纳）

- **范围**：对每病入选的稳定标志物（10~20 个）做二阶初探——Lasso 只建模边际效应，显式建模不了微生物协同效应；全特征两两交互（~88 万对）在小样本下不可行，故以入选标志物为限。
- **(a) 两两 Spearman 相关**：非零样本上、CLR 后丰度，输出相关矩阵（每病）。
- **(b) 共现/互斥检验**：相关显著的标志物对，用 Fisher 精确检验（存在/缺失口径）验证同现/互斥是否超出独立期望，输出共现网络（节点=标志物，边=显著共现/互斥，标方向）。
- **边界声明（报告必须含）**：「小样本下仅对入选标志物做二阶探索，无法全特征交互建模；标志物筛选主口径仍为边际信号」。

### 1.6 RF 重要性 + PLS-DA VIP 佐证

- **RF 重要性**：树模型直接跑原始丰度（免 CLR），多轮取平均重要性。
- **PLS-DA VIP**：VIP 阈值 **>1.5**（或分位数，P2）。
- 两者与 Alpha 稳定特征的 Top-N 排名一致性（Spearman 相关或交集比例）。

---

## 2. 数据接口

- **输入**：`outputs/data/c-data-cleaned.pkl`（阶段 0.3 清洗产物，主会话产出）
  - 字段：`dataset_name`、`disease` + 1331 物种特征列（相对丰度，0-100 量级，行和 ≈100）。
  - **口径差异说明**：A 类验证用的 `B-raw.pkl` 是**未清洗**的原始缓存（484 样本 × 1331 特征）；`c-data-cleaned.pkl` 是清洗后产物，样本数/特征数可能因去重/领域排除而略有变化。**正式实现以 `c-data-cleaned.pkl` 为准**，若清洗导致样本数/特征数与 A 类验证数字（F1~F9）显著偏离，需回报建模对话复核。
  - 标签口径（`solution/problem-statement.md`）：
    - CRC：`cancer`=患病，`n`+`small_adenoma`=健康（**small_adenoma 四口径沿 S1 裁定执行**：S1 全做择优（归健康/归病变/剔除/单开一类），本问跟随 S1 最终选定主口径；未选定前按题面口径归健康，见 R4）
    - IBD：`ibd_ulcerative_colitis`/`ibd_crohn_disease`=患病，`n`=健康
    - Obesity：`obesity`=患病，`leaness`=健康

---

## 3. 预期输出：`S2-results.pkl` 结构

```
S2-results.pkl = {
  "per_disease": {                       # 每病一个键
    "CRC": {
      "stable_features": [               # 稳定特征清单（频率 ≥ τ）
        {"feature": str, "frequency": float, "rank": int}
      ],
      "two_path_signals": [              # 两路信号统计量
        {"feature": str,
         "fisher_p": float, "fisher_fdr": float,      # 存在/缺失
         "wilcoxon_p": float, "wilcoxon_fdr": float,  # 非零丰度
         "direction": "up|down",          # 患病组相对健康组升高/降低
         "dominant_signal": "presence|abundance"}     # 主导信号（F2 语义）
      ],
      "biomarker_table": [               # 标志物表（Top 10~20）
        {"feature": str, "frequency": float, "fisher_fdr": float,
         "wilcoxon_fdr": float, "direction": str, "known_biomarker": bool}
      ],
      "cooccurrence": {                  # 共现分析（§1.5b）
        "spearman_matrix": {("f1","f2"): float},      # 两两 Spearman 相关
        "cooccurrence_edges": [                        # 显著共现/互斥边
          {"feature_a": str, "feature_b": str, "type": "cooccur|exclude",
           "spearman": float, "fisher_p": float}]
      },
      "rf_importance": {"feature": float},   # RF 重要性
      "vip": {"feature": float},             # PLS-DA VIP
      "topN_consistency": {              # 佐证一致性
        "rf_overlap": float, "vip_overlap": float, "spearman_rank": float}
    },
    "IBD": {...}, "Obesity": {...}
  },
  "cross_disease": {                     # 跨疾病对比
    "jaccard_matrix": {"CRC_IBD": float, "CRC_Obesity": float, "IBD_Obesity": float},
    "common_biomarkers": [str],
    "disease_specific": {"CRC": [str], "IBD": [str], "Obesity": [str]}
  },
  "meta": {
    "filter_threshold": 0.95, "tau": 0.5, "B": 50, "fdr_alpha": 0.05,
    "vip_threshold": 1.5, "clr_delta": 6.5e-06,
    "full_vs_cv": {"full": {...}, "cv": {...}}   # 全量(乐观) vs CV内(诚实)
  }
}
```

> 结构为**建议**，代码对话可按实现调整字段名，但须覆盖：每病稳定特征清单 + 频率 + 两路信号统计量 + 标志物表 + 交叉一致性。

---

## 4. 参考实现

- A 类验证脚本（`outputs/scratch/`，只读探索，可复用逻辑）：
  - `verify-S2-v1-baseline.py`（Wilcoxon + FDR + AUC）
  - `verify-S2-v2-zerobin.py`（零值分箱 + 过滤）
  - `verify-S2-v5-clr.py`（CLR）
  - `verify-S2-v6-stability.py`（Lasso bootstrap 频率）
  - `utils.py`（公共工具：标签映射/CLR/BH-FDR/零值占比）

---

## 5. 已知风险

- **bootstrap 计算量（C8 并行提示）**：B=50~100 轮 × 3 病 × Lasso 拟合，每轮独立可并行（`joblib`/`multiprocessing`）。预计单病单轮 <1 秒，总时长可控；若超 2 分钟按 TRAE-规范 C4 交主会话后台执行并心跳。
- **τ 敏感性**：τ 在 0.5~0.6 间每病入选数可能变化较大（尤其 Obesity 信号分散），实现时输出 τ 敏感性曲线（τ=0.4/0.5/0.6/0.7 的入选数），供 B 类验证回填。
- **CLR 伪计数 δ 口径**：必须与 S1 一致，避免口径分裂（P6）。

---

## 6. 约束

- **No Placeholders**：所有参数（τ、VIP 阈值、FDR α、B、过滤阈值、δ）在实现时取 `proxy-replacement-checklist-S2.md` 的当前临时值，不得留 TODO/占位符。
- **诚实标注**：区分「全量（乐观）」与「CV 内稳定频率（诚实）」。
- **C1 代码头注释**：正式实现脚本按 TRAE-规范 C1 写头注释。
- **回报**：写 `handoff-S2-model-agent.md`（结果摘要 + 关键数字 + 待裁定项），git commit。
