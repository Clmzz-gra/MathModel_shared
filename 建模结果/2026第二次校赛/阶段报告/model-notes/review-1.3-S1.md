# 门禁 1 审查结论：S1 建模方案定稿（review-1.3-S1）

> **审查代理角色**：建模对话审查代理（门禁 1 审查，审查/裁决分离协议——本文件为审查结论，裁决由主建模执行）
> **模型**：deepseek-v4-pro:0813
> **思考强度**：max（最高强度，子代理审查固定）
> **日期**：2026-08-21 | 子问题：S1 疾病预测模型 | 分支：experiment/2026sim2B-S1

---

## 0. 必读清单已读汇报

已完整读取并核对：

- **门禁协议**：`TRAE-建模.md`（门禁协议：审查代理职责 + 结论文件格式 + 空洞审查检出）、`TRAE.md`（门禁 1 判定内容 + 四门禁总览 + 交接协议 + 待裁定项分级）。
- **检验要点 skill**：`modeling-decision-tree`（分类评估路径 + 不平衡专项检查 PF-007 + 小样本泛化 PF-019 + 成分数据 PR-006/AL-007）。
- **讲解质量约束 skill**：`math-explainer`（讲解质量约束全量 6 条）。
- **审查对象**：`approach-S1-confirmed.md`、`handoff-S1-code-agent.md`、`proxy-replacement-checklist-S1.md`。
- **对照材料**：`decision-tree-S1.md`（1.1）、`debate-S1.md`（1.2）、`handoff-S1-model-agent-verify.md`（A 类验证 F1-F8）、`handoff-S1-code-agent-verify.md`（A 类验证交接）、`knowledge/method-graph.md`（方法体系核对）。
- **数据文件实况核对**：`outputs/data/`（S1 worktree 与赛题 worktree 两处）、`clean-report-B.txt`（0.3 清洗报告）。

---

## 1. 判定内容逐项结论

### 1.1 方案合理性

**结论**：通过。主模型 Logistic(L2)+CLR+class_weight 严格基于 A 类验证共享事实推导，对照/基线齐备，评估协议诚实。

**依据**：
- 主模型三要素均有共享事实支撑：CLR 前置 ← F4（CLR 增益 +0.146/+0.109/+0.099，三数据集一致）；L2 收缩 ← 小样本高维（n=110~253 vs p=1331，PR-011 协方差收缩）；class_weight ← F5（metahit ACC 0.809 虚高 vs 少数类 Recall 0.400）。
- 对照 RF 齐备：F3（RF 原始丰度 AUC 0.846/0.876/0.669，Zeller/Chatelier 略优），定位为非线性对照 + permutation importance 复用 S2，不进入主口径但正式实现保留。
- 基线齐备：F1（单特征最佳 AUC 0.758/0.815/0.639 + Dummy=0.5），建立性能下界防过度设计。
- 评估协议诚实：F5 实证 ACC 误导 → AUC 主指标（阈值无关、跨数据集可比）；LOOCV 兜底（全量乐观上界 vs LOOCV 诚实估计，差距 >0.1 判过拟合，PF-019）；Chatelier 弱信号（AUC 0.643 ≈ 基线 0.639）诚实标注「接近领域下界 0.65-0.75」，不包装成「合理选择」。

**证据路径**：`approach-S1-confirmed.md` §1/§6；`debate-S1.md` §0 共享事实 F1-F8；`handoff-S1-model-agent-verify.md` §2 #1-#4。

### 1.2 推导自洽

**结论**：通过。CLR 公式、Logistic 损失 + L2、class_weight 公式符号定义与读法齐全（math-explainer 6 条），数学框架与 A 类验证口径一致（δ=6.5e-06）。

**依据**：
- **CLR 公式**：乘法替换 $x_{ij}\leftarrow\max(x_{ij},\delta)$，$\delta=0.65\times10^{-5}=6.5\times10^{-6}$；几何均值中心化 $\mathrm{clr}(x_{ij})=\ln x_{ij}-\frac{1}{p}\sum_k\ln x_{ik}=\ln\frac{x_{ij}}{g_i}$。δ 值与 A 类验证口径（`handoff-S1-code-agent-verify.md` §3「检出限取非零最小值 1e-05 近似」）一致。
- **Logistic 损失 + L2**：交叉熵 + $\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2$，sklearn 对应 $C=1/\lambda$（C=1.0 即 λ=1.0）表述正确。
- **class_weight 公式**：$w_c=n/(n_{\text{classes}}\times n_c)$，代入 metahit $w_1=110/(2\times25)=2.2$、$w_0=110/(2\times85)=0.647$，比值 3.4 倍，计算正确。
- **math-explainer 6 条**：①先定义再使用（§2.1 符号表 15 个符号均含读法）②每步有因为（§2.2/2.3/2.4 均标注）③多视角（CLR「全班总分→相对偏离」、L2「橡皮筋」、class_weight「难题扣分多」类比）④核心洞察收束（§8）⑤直觉解释（每核心公式配类比）⑥防跳跃（§8 末尾列 5 个概念点）。6 条齐全。

**证据路径**：`approach-S1-confirmed.md` §2.1-§2.4、§8；`handoff-S1-code-agent-verify.md` §3。

### 1.3 handoff 可执行性

**结论**：通过（附 1 项中等问题，见问题清单 #1）。数据接口、预期输出、参考实现、已知风险齐全，No Placeholders 满足。

**依据**：
- **数据接口**：`outputs/data/c-data-cleaned.pkl` 路径 + 字段（2 元数据列 `dataset_name`/`disease` + 1331 特征列 float32）与 `clean-report-B.txt` 实测一致（484×1333，1331 特征 + 2 元数据，float32）；与 B-raw.pkl 口径差异已注明（float64 484×1333 vs float32，A 类验证数字仅参考）。
- **预期输出**：`S1-results.pkl` 结构完整（L2_CLR/RF_raw/baseline/LOOCV 四块，含 cv_folds 每折明细 + coefficients + confusion_matrix）。
- **参考实现**：`verify-S1-a1.py`~`a6.py` + `utils.py`（CLR 函数/标签映射/评估协议可复用）。
- **已知风险**：§5 四类风险（Chatelier 弱信号/14 离群/metahit class_weight/小样本 AUC 方差）均带处置。
- **No Placeholders**：grep 全目录无 TBD/待补充/待定占位符（proxy 清单中「待定」均为「非待定」否定表述）。

**证据路径**：`handoff-S1-code-agent.md` §2-§5；`clean-report-B.txt`；`proxy-replacement-checklist-S1.md`。

### 1.4 待裁定项处置

**结论**：通过。四项待裁定项处置均有依据、级别清晰。

**依据**：
- **small_adenoma 销项**：F6 实证剔除 26 例 ΔAUC +0.022~+0.024 < 0.05 阈值，建议销项维持全口径，registry 改 `done`（由主建模在赛题 worktree 执行）。
- **metahit 不平衡**：评估统一 AUC 主指标 + F1/Recall(少数类) 为辅，ACC 仅参考；class_weight 提升 Recall 走 B3 验证。
- **Chatelier 诚实标注**：报告明确「增益有限，接近领域下界 0.65-0.75」，不包装。
- **14 离群样本两轨**：纳入主口径（默认）+ 剔除敏感性（B4 验证）。

**证据路径**：`approach-S1-confirmed.md` §7；`debate-S1.md` §7；`handoff-S1-model-agent-verify.md` §2 #6。

### 1.5 B 类分歧

**结论**：通过。B2/B3/B4 标记清晰，均说明验证方法；B1 已决断不需验证。

**依据**：
- B1（L2 vs RF）：A 类 F2/F3 已决断，不需 B 类验证。
- B2（集成 Gamma）：Soft Voting 增益 < 0.02 则放弃，验证方法明确。
- B3（metahit class_weight）：class_weight='balanced' vs None 的 Recall head-to-head 对比，验证方法明确。
- B4（14 离群样本）：剔除 14 样本重训 Zeller 对比 AUC，验证方法明确。
- 三者均标注「轻量验证，不阻断主方案推进，1.3 后由建模自写 verify 或并入 2.1」。

**证据路径**：`approach-S1-confirmed.md` §7 B 类分歧处置表；`debate-S1.md` §6；`handoff-S1-code-agent.md` §6。

---

## 2. 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| 1 | **c-data-cleaned.pkl 不在 S1 worktree**：handoff §2 引用 `outputs/data/c-data-cleaned.pkl` 作为数据源，但该文件仅存在于赛题 worktree（`E:\MathModel_pj-2026-sim2-B\outputs\data\c-data-cleaned.pkl`，0.3 清洗产出 commit 68d6121）；S1 worktree 的 `outputs/data/` 仅有 B-raw.pkl + inventory-B.txt。S1 worktree 在 commit f7d58e1（0.2 待裁定项登记）处切出，早于 0.3 清洗。代码对话 1.4 开工前需先 merge 赛题分支（problems/2026-sim2-B）获取 c-data-cleaned.pkl，handoff 未显式标注此前置步骤。 | 中（B 级，可逆/可并行，不影响方案正确性，但影响 1.4 即时可执行性） | `handoff-S1-code-agent.md` §2；`git log`（S1 worktree 末 [S0] commit=f7d58e1，赛题 worktree 0.3=68d6121）；`outputs/data/` 两处目录实况 |
| 2 | **符号表 g_i 定义未注明「替换后」**：§2.1 定义 $g_i=(\prod_j x_{ij})^{1/p}$ 用原始丰度，但 CLR 实际用 δ 替换后的值计算几何均值（0 无法取对数）。不影响实现（handoff §1.2 顺序「先替换→再中心化」明确），但符号严谨性可改进。 | 低 | `approach-S1-confirmed.md` §2.1 符号表 vs §2.2 第二步 |
| 3 | **Chatelier class_weight 方向未展开**：class_weight='balanced' 对 Chatelier 会提升健康类（少数类，35.2%）召回，而「预测疾病」目标与「提升少数类召回」在 Chatelier 上存在轻微张力。已由 AUC 主指标 + F1/Recall 正类显式定义化解，但 §2.4 代入示例仅给 metahit，未给 Chatelier 说明。 | 低 | `approach-S1-confirmed.md` §2.4；`handoff-S1-code-agent.md` §1.1（Chatelier 少数类=健康，方向特殊） |
| 4 | **LOOCV 兜底耗时未显式提示**：LOOCV 对 RF（500 棵树）在 Chatelier（253 样本）需训练 253×500 棵树，很可能超 2 分钟，属 C4 高耗时脚本。handoff 未显式提示需后台执行（代码对话会读 C4 规范自行判断，但可显式提示）。 | 低 | `handoff-S1-code-agent.md` §1.5 评估协议（LOOCV 兜底） |

---

## 3. 结论

**通过**（有条件通过）。

- 判定内容 5 项全部通过：方案合理性、推导自洽、handoff 可执行性、待裁定项处置、B 类分歧处置均达标，无空洞审查项。
- 问题清单 4 项，均为 **B 级/低严重度**，不阻断门禁 1 放行：
  - **#1（中，B 级）** 建议主建模裁决时补充说明：代码对话 1.4 开工前先 merge 赛题分支（problems/2026-sim2-B）获取 c-data-cleaned.pkl，或在 handoff §2 补一句前置步骤。
  - **#2/#3/#4（低）** 为符号严谨性/表述补充/耗时提示，不要求返工，可在 2.0 推导或 2.1 实现时顺带完善。

**裁决建议**：默认按推荐方案（主 L2(CLR)+class_weight / 对照 RF / 基线单特征+Dummy / 评估协议分层 5 折 CV+AUC 主指标+LOOCV 兜底）放行；B 类分歧 B2/B3/B4 按标记执行轻量验证，B1 已决断。
