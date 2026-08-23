# S3 建模→报告交接（handoff_type=report-agent）

> 建模对话（S3）→ 报告对话（S3）| 阶段 2.2 结果分析完成（门禁 N 材料）
> 运行模式：auto | 日期：2026-08-21
> 数据源：`outputs/data/S3-results.pkl`（2.1 正式实现，seed=42，budget_limited=False）
> **数字只取 pkl 实际值**；报告写作数字须从 pkl 只读提取核对，不抄本交接（本交接数字已与 pkl 核对一致）。

---

```yaml
---
handoff_type: report-agent
sub: S3
from: 建模对话（S3）
to: 报告对话（S3）
stage: "2.2"
source_docs:
  - solution/model-notes/result-analysis-S3.md
  - solution/model-notes/approach-S3-confirmed.md
  - solution/model-notes/math-S3.tex
  - outputs/data/S3-results.pkl
status: ready
next_action: 报告对话 2.3 内容段（说「继续」）
---
```

---

## 一、章节映射

| 论文章节 | 内容来源 | 关键数字（pkl 字段） |
|---|---|---|
| 方法节 · 跨疾病预测模型 | approach §1.2-1.6 + math-S3.tex | 四策略构建方式、LODO 协议、回退协议 |
| 结果节 · 四策略对比 | result-analysis §2.1 | `strategy_compare.<S>.mean_auc` |
| 结果节 · 紧急回退 | result-analysis §2.2 | `fallback.*`、`exhausted_evidence.*` |
| 结果节 · 衰减归因 | result-analysis §2.3 | `decay_attribution.<D>.*` |
| 结果节 · 深度迁移 | result-analysis §2.4 | `migration_analysis.*` |
| 结果节 · C3 阈值漂移 | result-analysis §2.5 | `threshold_drift.*` |
| 结论节 · 最优可达与部署建议 | result-analysis §0/§四 | `exhausted_evidence.conclusion` |

---

## 二、关键数字（来源可溯到 pkl）

### 2.1 四策略对比（`strategy_compare`）

| 策略 | C1(CRC) | C2(IBD) | C3(Obesity) | 均值 | pkl 字段 |
|---|---|---|---|---|---|
| A 直接迁移 | 0.5674 | 0.5882 | 0.5253 | **0.5603** | `A_direct.<C>.auc` / `A_direct.mean_auc` |
| B 共享标志物 | 0.5417 | 0.6080 | 0.5218 | 0.5572 | `B_shared.<C>.auc` / `B_shared.mean_auc`（`shared_feature_count=252`） |
| C 属级聚合 | 0.3616 | 0.4861 | 0.5440 | 0.4639 | `C_genus.<C>.auc` / `C_genus.mean_auc`（`n_features=106`） |
| C 门级聚合 | 0.4141 | 0.5261 | 0.5999 | 0.5134 | `C_phylum.<C>.auc` / `C_phylum.mean_auc`（`n_features=11`） |
| D 部署校正 | 0.5674 | 0.5882 | 0.5253 | 0.5603 | `D_calibrated.<C>.auc` / `D_calibrated.mean_auc`（`base_strategy=A_direct`） |

**四策略 3 组合 AUC 均值全部 < 0.60 → 触发紧急回退**（`fallback.triggered=True`）。

### 2.2 紧急回退（`fallback`）

| 回退层级 | 模型族 | 均值 | 达可用线？ | pkl 字段 |
|---|---|---|---|---|
| R1 树模型 | RandomForest 500 树 | 0.5092 | 否 | `R1_tree.mean_auc` |
| R2 样本合并 | Logistic（≡策略 A） | 0.5603 | 否 | `R2_pooled.mean_auc` |
| R3 密度比重加权 | importance weighting | **0.6068** | 否（提升 +0.0465 <0.10） | `R3_weighted.mean_auc` |
| R4 对抗式域适应 | DANN | 0.5947 | 否 | `R4_dann.mean_auc` |

- `fallback.usable=False`、`delivered_strategy=None`。
- **最优可达**：`exhausted_evidence.best_strategy=R3_weighted`、`best_mean_auc=0.6068`、`usable_line=0.65`。
- **结论句**（`exhausted_evidence.conclusion`）：在现有数据与协议下，跨疾病预测模型最优可达 AUC 0.6068（R3_weighted），低于可用线 0.65。

### 2.3 衰减归因（`decay_attribution`，264 特征域内口径）

| 疾病 | 域内 AUC | 跨疾病 AUC | 衰减量 | 主导归因 | pkl 字段 |
|---|---|---|---|---|---|
| CRC | 0.7811 | 0.5674 | −0.2138 | 疾病特异信号 | `decay_attribution.CRC.*` |
| IBD | 0.8588 | 0.5882 | **−0.2706** | 疾病特异信号（最强） | `decay_attribution.IBD.*` |
| Obesity | 0.6638 | 0.5253 | −0.1384 | 标签语义漂移 | `decay_attribution.Obesity.*` |

### 2.4 深度迁移（`migration_analysis`）

- 方向一致 387 / 方向翻转 369，`n_valid=756`，一致占比 **51.2%**（`consistent_fraction=0.5119`），符号检验 **p=0.5364**（不显著）。
- 结论：共享物种方向跨疾病接近随机，存在性不承载可迁移信号，信号藏在疾病特异方向里。

### 2.5 C3 阈值漂移（`threshold_drift`）

- 训练基线 0.3160 vs 测试 0.6482，Δ=+0.3322。
- Youden τ\*=0.9205 落在测试分布 **96.0% 分位**（`boundary_position=0.9605`），灵敏度 **0.0244**。
- 诊断：标签语义漂移；**H4 证伪**（Platt 单调校准无法修复，0.5 阈值灵敏度仅 0.1646）。

---

## 三、口径声明（报告必须声明）

1. **特征集口径**：正式实现基于近全零过滤后 **264 物种级特征**（三病并集统一口径，与 S1/S2 一致），非 A 类验证的 1331 全集。域内 AUC 亦为 264 特征重算（`domain_auc`），与跨疾病 264 特征同口径对比。
2. **主指标 AUC**：阈值无关，最诚实；辅指标（ACC/灵敏/特异/F1）为训练集 Youden J 阈值迁移到测试集（**禁测试集重定阈值**，防泄漏）。
3. **Platt 符号约定**：$P_{\text{cal}}=1/(1+\exp(A\cdot f+B))$，$A<0$ 才单调递增；实现按 sklearn 形式 $A=-w,\ B=-b$，校验 $w>0$（≡A<0）。三处规格已修正。
4. **R3 密度比方法**：域分类器法（Logistic 区分 train/test，$w=\exp(\text{logit})\times n_{\text{train}}/n_{\text{test}}$，裁剪上界 10），转导式边界（用测试特征、不用测试标签）。
5. **R2 ≡ 策略 A**：LODO 已合并 2 训练疾病，R2 样本合并与策略 A 数学恒等，不提供增量（规格冗余）。
6. **C3 不可直接部署**：AUC 0.5253 接近随机而可部署指标崩溃（灵敏度 0.0244），报告须标注「AUC 与可部署指标分离」。

---

## 四、图表清单规格（数据源 pkl + 图名 + 论文位置）

| 图 | 数据源（pkl 字段） | 图名 | 论文位置 | 说明 |
|---|---|---|---|---|
| 四策略 AUC 对比 | `strategy_compare.<S>.<C>.auc` | `S3-strategy-compare-auc` | 结果节 · 四策略对比 | 四策略 × 3 组合分组柱状图；A/B 近随机，C 属级/门级更低 |
| 域内 vs 跨疾病 | `decay_attribution.<D>.domain_auc/cross_auc` | `S3-decay-attribution` | 结果节 · 衰减归因 | 三疾病域内 vs 跨疾病 AUC 对比，IBD 衰减最大 |
| C3 阈值漂移 | `threshold_drift.*` | `S3-threshold-drift` | 结果节 · C3 阈值漂移 | 训练/测试分数分布 + Youden 阈值位置，可视化标签语义漂移 |

> 探索图已存在于 `outputs/figures/_explore/`（S3-strategy-compare-auc.pdf / S3-decay-attribution.pdf / S3-threshold-drift.pdf），正式出图按「图表两级制」由代码 Preset 子代理按本规格出正式图。

---

## 五、AI 标注

- 本问建模（方案/推导/结果分析）由建模对话（AI）产出，代码实现由代码对话（AI）产出，两遍审核由建模/代码子代理（AI）执行。
- 报告写作须按 `ai-usage-report` skill 规范在正文/附录标注 AI 贡献（`[AI-X-Y]` 编号唯一），并写入 `writing-material-S3.tex` 的 AI 标注节。
- 关键数字来源：`outputs/data/S3-results.pkl`（meta.generated=2026-08-21T17:33:01，seed=42，budget_limited=False）。

---

## 六、No Placeholders

本交接所有关键数字均取自 `S3-results.pkl` 实测，无 `TODO`/`TBD`/`待定` 占位符。报告写作数字须从 pkl 只读提取核对，不抄本交接。
