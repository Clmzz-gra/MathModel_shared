# review-3.0-跨问.md — 阶段 3.0 跨问审查结论

> 阶段：3.0 跨子问题审查（人工关卡）| 日期：2026-08-21 | 运行模式：auto
> 审查代理角色：批判阅读审查代理（modeling preset）| 模型：deepseek-v4-pro:0813
> 审查对象：S1/S2/S3 三子问题方案确认书、结果分析、内部报告、讲解包、结果 pkl
> 数字口径：**只取 pkl 实际值**（`S1/S2/S3-results.pkl` python 只读提取）

---

## 一、必读清单已读汇报

| # | 必读项 | 状态 |
|:--|:--|:--|
| 1 | `TRAE-建模.md`（阶段 3.0 跨问审查规范）+ `TRAE-规范.md`（A/B/C 相关节） | ✅ 已读 |
| 2 | `solution/domain-knowledge.md` | ✅ 已读 |
| 3 | `approach-S1/S2/S3-confirmed.md` | ✅ 已读 |
| 4 | `result-analysis-S1/S2/S3.md` | ✅ 已读 |
| 5 | `iter-02-sub1-disease-prediction.tex` / `iter-02-sub2-biomarker.tex` / `iter-02-sub3-cross-disease.tex` | ✅ 已读 |
| 6 | `02-S1-疾病预测模型分析与思路.md` / `02-问题二分析与思路.md` / `02-问题三分析与思路.md` | ✅ 已读 |
| 7 | `S1-results.pkl` / `S2-results.pkl` / `S3-results.pkl`（python 只读提取关键字段） | ✅ 已读 |

---

## 二、判定内容逐项结论

### 1. 过滤口径统一性 —— ✅ 一致

- **结论**：三子问题均用「近全零过滤 1331→264，三病并集统一口径」。
- **依据**：S1 报告 §2.2「剔除零值占比 >95% 的特征，1331→264，与 S2 统一（三病并集）」；S2 报告 §2.2「过滤在三病并集（全 484 样本）上统一计算，1331→264，与 S1/S3 口径一致」；S3 报告 §2.1「近全零过滤（三病并集统一口径，与 S1/S2 一致），过滤后 264」。
- **证据路径**：`iter-02-sub1-disease-prediction.tex` §2.2 / `iter-02-sub2-biomarker.tex` §2.2 / `iter-02-sub3-cross-disease.tex` §2.1；pkl `S2-results.pkl meta.filter_threshold=0.95`。
- **备注**：S2 approach §4 步骤 1 曾写「取三病并集或各病独立过滤（见 handoff 规格）」未锁定，但 S2 报告已锁定三病并集，实际一致，无实质漂移。

### 2. CLR 口径统一性 —— ✅ 一致

- **结论**：三子问题均用 δ=6.5e-06 乘法替换 + 几何均值中心化。
- **依据**：S1 报告 §2.2「δ=6.5×10⁻⁶」；S2 报告 §2.2「伪计数 δ=6.5×10⁻⁶ 与 S1 一致」；S3 报告 §2.1「替换常数 6.5×10⁻⁶」。
- **证据路径**：pkl `S1-results.pkl meta.note`（CLR δ=6.5e-06）、`S2-results.pkl meta.clr_delta=6.5e-06`、`S3-results.pkl meta.clr_delta=6.5e-06`。

### 3. small_adenoma 口径 —— ⚠️ 结果一致、文档时序矛盾

- **结论**：三子问题**最终结果均归健康**（S1 主口径①、S2 题面口径、S3 C1 测试 CRC n=121 含 small_adenoma），结果一致；但 S2 文档残留「S1 主口径未选定」的过时表述。
- **依据**：S1 result-analysis §3.3「推荐维持①（归健康）为主口径」，pkl `selected_main_caliber='healthy'`；S2 result-analysis T3「S1 主口径未选定（S1 2.2 未产出）」、S2 报告 §2.2「未选定前按题面口径归健康」；S3 C1 测试集 `n_test=121`、`test_pos_frac=0.3967`（48/121，即 small_adenoma 归健康）。
- **证据路径**：`result-analysis-S1.md` §3.3 / `result-analysis-S2.md` T3 / `iter-02-sub2-biomarker.tex` §2.2 / pkl `S1-results.pkl adenoma_sensitivity.selected_main_caliber`、`S3-results.pkl strategy_compare.A_direct.C1.n_test`。

### 4. 标签映射统一性 —— ✅ 一致

- **结论**：三数据集患病=1/健康=0 一致。
- **依据**：S1 报告 §2.2「患病=1/健康=0」；S2 报告 §2.2「患病=cancer/ibd_*/obesity，健康=n/leaness」；S3 报告 LODO 协议（C1/C2/C3 正类占比 39.7%/22.7%/64.8% 与 S1 少数类比例自洽）。
- **证据路径**：三报告 §2 数据节；pkl `S3-results.pkl strategy_compare.*.test_pos_frac`。

### 5. 数字一致性（跨问引用）—— ❌ 不一致

- **结论**：S3 引用的「域内 AUC」与 S1 报告的「域内 AUC」不一致，且 S3 未解释差异；S3 approach 引用 S1 数字时混用 A 类验证值与正式值。
- **依据**：S1 报告 L2(CLR) AUC = 0.791/0.887/0.650（pkl 0.7907/0.8871/0.6496）；S3 报告 domain_auc = 0.7811/0.8588/0.6638（pkl 同）。两者均为「264 特征 L2+CLR 域内 AUC」，但数字不同（Δ −0.010/−0.028/+0.014）。S3 报告 §6.1 仅声明「264 特征 5 折 CV 重算」，未解释与 S1 的差异。
- **证据路径**：`iter-02-sub1-disease-prediction.tex` §4.1 vs `iter-02-sub3-cross-disease.tex` §6.1；pkl `S1-results.pkl <ds>.L2_CLR.AUC` vs `S3-results.pkl domain_auc`。

### 6. 接口/假设冲突 —— ❌ 存在冲突

- **结论**：S3 声称「沿用 S1 口径」但实际多加了 StandardScaler（S1/S2 无），导致域内 AUC 与 S1 不一致；S3 报告 silhouette「未单独量化」与 approach/result-analysis 的 0.070 矛盾。
- **依据**：S3 approach §5 预处理含 `StandardScaler`，S1 approach §3 无；S3 approach §4.5 给 silhouette=0.070，S3 报告 §6.1 写「未单独量化」。
- **证据路径**：`approach-S3-confirmed.md` §5 vs `approach-S1-confirmed.md` §3；`iter-02-sub3-cross-disease.tex` §6.1 vs `approach-S3-confirmed.md` §4.5 / `result-analysis-S3.md` §2.3。

### 7. 报告/讲解包一致性 —— ⚠️ 基本一致、少量漂移

- **结论**：三问报告与讲解包数字基本一致（四舍五入差异可接受）；S3 silhouette 矛盾（见第 6 项）、S3 approach 灵敏度 0.006 vs 报告 0.024（A 类验证 vs 正式实现）为少量漂移。
- **依据**：S1/S2/S3 报告与讲解包关键数字（AUC、频率、FDR、衰减量、阈值漂移）逐项核对一致。
- **证据路径**：各报告 vs 各讲解包（`02-*.md`）。

---

## 三、问题清单（问题 | 严重度 | 证据路径）

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| P1 | S3 域内 AUC（0.7811/0.8588/0.6638）与 S1 域内 AUC（0.7907/0.8871/0.6496）不一致，S3 未解释差异 | **高** | `iter-02-sub3-cross-disease.tex` §6.1 vs `iter-02-sub1-disease-prediction.tex` §4.1；pkl `S3-results.pkl domain_auc` vs `S1-results.pkl <ds>.L2_CLR.AUC` |
| P2 | S3 报告 silhouette「未单独量化」与 approach §4.5/result-analysis §2.3 的 0.070 矛盾 | 中 | `iter-02-sub3-cross-disease.tex` §6.1 vs `approach-S3-confirmed.md` §4.5 |
| P3 | S3 声称「沿用 S1 口径」但多加了 StandardScaler（S1/S2 无） | 中 | `approach-S3-confirmed.md` §5 vs `approach-S1-confirmed.md` §3 |
| P4 | S2 文档「S1 主口径未选定」与 S1 已选定①归健康（时序矛盾） | 低 | `result-analysis-S2.md` T3 / `iter-02-sub2-biomarker.tex` §2.2 vs `result-analysis-S1.md` §3.3 |
| P5 | S3 approach 引用 S1 L2 0.812（A 类验证值）vs S1 正式 0.7907 | 低 | `approach-S3-confirmed.md` §1.6 R1 vs `iter-02-sub1-disease-prediction.tex` §4.1 |
| P6 | S3 approach 衰减归因表用 A3 参考值 0.814/0.885/0.644 vs 报告 264 口径 | 低 | `approach-S3-confirmed.md` §6.2 vs `iter-02-sub3-cross-disease.tex` §6.1 |
| P7 | S3 approach 灵敏度 0.006 vs 报告 0.024（A 类验证 vs 正式实现） | 低 | `approach-S3-confirmed.md` §4.5/§6.4 vs `iter-02-sub3-cross-disease.tex` §6.3 |

---

## 四、pkl 数字抽核结果

| 项 | pkl 实际值 | 报告/讲解包引用 | 一致 |
|:--|:--|:--|:--:|
| S1 L2 AUC（Zeller/metahit/Chatelier） | 0.7907 / 0.8871 / 0.6496 | 0.791 / 0.887 / 0.650 | ✅ |
| S1 RF AUC | 0.8454 / 0.9035 / 0.6602 | 0.845 / 0.904 / 0.660 | ✅ |
| S1 单特征基线 | 0.7581 / 0.8153 / 0.6395 | 0.758 / 0.815 / 0.639 | ✅ |
| S1 LOOCV | 0.8042 / 0.8748 / 0.6270 | 0.804 / 0.875 / 0.627 | ✅ |
| S1 adenoma 四口径 L2 | 0.7907 / 0.6112 / 0.8022 / 0.8022 | 0.791 / 0.611 / 0.802 / 0.802 | ✅ |
| S1 selected_main_caliber | 'healthy' | ①归健康 | ✅ |
| S2 clr_delta / filter_threshold / fdr_m / C_lasso / tau | 6.5e-06 / 0.95 / 1331 / 0.1 / 0.5 | 同 | ✅ |
| S2 n_stable（CRC/IBD/Obesity） | 4 / 4 / 20 | 4 / 4 / 20 | ✅ |
| S2 Jaccard | 0.0 / 0.0 / 0.0 | 全 0 | ✅ |
| S3 clr_delta | 6.5e-06 | 6.5×10⁻⁶ | ✅ |
| S3 四策略 mean_auc（A/B/C属/C门/D） | 0.5603 / 0.5572 / 0.4639 / 0.5134 / 0.5603 | 同 | ✅ |
| S3 回退 R1/R2/R3/R4 | 0.5092 / 0.5603 / 0.6068 / 0.5947 | 同 | ✅ |
| S3 domain_auc（CRC/IBD/Obesity） | 0.7811 / 0.8588 / 0.6638 | 报告同 | ✅（但与 S1 不一致，见 P1） |
| S3 domain_auc_reference_A3 | 0.814 / 0.885 / 0.644 | approach §6.2 引用 | ✅（A3 参考值） |
| S3 迁移方向一致/翻转 | 387 / 369（n_valid 756，51.2%，p=0.5364） | 同 | ✅ |
| S3 阈值漂移（基线差/阈值/分位/灵敏度） | +0.332 / 0.9205 / 96.0% / 0.0244 | 同 | ✅ |

**抽核结论**：三问报告/讲解包引用的数字与 pkl 实际值**逐项一致**（四舍五入差异可接受）；唯一跨问不一致是 S3 的 `domain_auc` 与 S1 的 `L2_CLR.AUC` 不同（P1），属跨问数字矛盾而非单问取数错误。

---

## 五、结论：不通过

**跨问审查结论：不通过。**

- **通过项**：过滤口径（1331→264 三病并集）、CLR 口径（δ=6.5e-06）、标签映射（患病=1/健康=0）、small_adenoma 结果口径（均归健康）三问一致；三问报告/讲解包数字与 pkl 逐项一致。
- **不通过项**：存在 1 项高严重度跨问数字不一致（P1：S3 域内 AUC 与 S1 域内 AUC 不一致且未解释）与 2 项中严重度口径矛盾（P2：silhouette 未量化 vs 0.070；P3：S3 静默增加 StandardScaler 却声称「沿用 S1 口径」）。
- **根因**：P1/P3 同源——S3 复用 S1 的「L2+CLR」口径时静默增加 StandardScaler，导致域内 AUC 与 S1 不一致，且报告未声明该差异。
- **修正建议**：P1/P3 合并处理（S3 报告显式声明 StandardScaler 与域内 AUC 口径差异，或统一口径重算）；P2 补回 silhouette=0.070；P4-P7 阶段 3.1 逐条销项。修正执行主体为 report Preset 子代理（内容生产禁则，主建模不代写）。

---

## 六、交接收尾

跨问审查结论已落盘（本文件 + `critical-reading.md` + `review-notes.md`）。**next_action**：主建模自检核实（重点核 P1 的 S3 domain_auc 与 S1 L2 AUC 差异根因）→ 裁决 → 写 `gate-3.0.md` → 阶段 3.1 报告修正（report Preset 子代理执行）。
