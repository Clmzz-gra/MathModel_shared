# 交接：S1 A 类共享事实验证回报（代码 → 建模）

> **handoff_type**: `model-agent-verify`（A 类验证回报，代码子代理 → 建模侧）
> **sub**: S1 疾病预测模型
> **日期**: 2026-08-21 | 运行模式: auto（门禁 1 呈递人类拍板）
> **⚠️ 本文件为 `-verify` 后缀回报，与正式代码→建模交接 `handoff-S1-model-agent.md` 区分，禁止同名覆盖。**

---

## 0. 必读清单已读汇报

已完整读取：`TRAE.md`（管线骨架/交接协议/执行主体分工）、`TRAE-代码.md`（A 类验证执行）、`TRAE-规范.md`（C1 代码头注释 / C2 技术栈 / C8 代码加速决策树）、`handoff-S1-code-agent-verify.md`（验证规格）、`decision-tree-S1.md`（§5 A 类验证清单）、`domain-knowledge.md`（标签映射）。

---

## 1. 数据接口与口径

- **数据源**：`outputs/data/B-raw.pkl`（484×1333，2 元数据列 + 1331 float64 特征；`c-data-cleaned.pkl` 尚未就绪，按 handoff 以 B-raw.pkl 为准）。
- **标签映射**（患病=1 / 健康=0）：Zeller `cancer`=1，`n`+`small_adenoma`=0；metahit `ibd_ulcerative_colitis`+`ibd_crohn_disease`=1，`n`=0；Chatelier `obesity`=1，`leaness`=0。
- **少数类（F1/Recall 正类）**：Zeller=患病(39.7%)、metahit=患病(22.7%)、Chatelier=健康(35.2%)。
- **CLR 口径**：零值乘法替换（AL-007）δ=0.65×检出限=0.65×1e-05=**6.5e-06**，再逐行 `log(x)-mean(log(x))` 几何均值中心化。
- **评估协议**：分层 5 折 CV（shuffle，seed=42），报告 AUC（阈值无关）+ ACC + 少数类 F1/Recall；AUC 为 CV 诚实估计。

---

## 2. 六项验证结论（量化摘要，保留三位有效数字）

### #1 简单基线下界（第一步强制）

| 数据集 | Dummy ACC | Dummy AUC | 单特征最佳 AUC（样本内） |
|:--|:--:|:--:|:--:|
| Zeller CRC | 0.603 | 0.500 | **0.758** |
| metahit IBD | 0.773 | 0.500 | **0.815** |
| Chatelier Obesity | 0.648 | 0.500 | **0.639** |

**结论**：三数据集单特征最佳 AUC 均显著高于 0.5（0.64~0.82），存在真实信号；Chatelier 地板最低（0.639），与领域基准「肥胖信号弱」一致。正式模型须显著高于此地板方有增益。

### #2 类别不平衡对 AUC vs ACC 影响

| 数据集 | 少数类比例 | ACC | 少数类 Recall | 少数类 F1 | AUC |
|:--|:--:|:--:|:--:|:--:|:--:|
| Zeller CRC | 39.7% | 0.777 | 0.604 | 0.674 | 0.812 |
| metahit IBD | 22.7% | 0.809 | **0.400** | 0.464 | **0.885** |
| Chatelier Obesity | 35.2% | 0.604 | 0.438 | 0.440 | 0.643 |

**结论**：metahit 最不平衡（22.7%），ACC=0.809 虚高（Dummy 多数类即 0.773），但少数类 Recall 仅 0.400、AUC=0.885——**ACC 严重误导，AUC 才是诚实主指标**。确认决策树「AUC 主指标 + F1/Recall(少数类)」口径正确。

### #3 零值 92% 对树 vs 线性模型影响

| 数据集 | RF(原始丰度) AUC | Logistic L2(CLR) AUC |
|:--|:--:|:--:|
| Zeller CRC | **0.846** | 0.812 |
| metahit IBD | 0.876 | **0.885** |
| Chatelier Obesity | **0.669** | 0.643 |

**结论**：RF 直接吃原始零值丰度（免 CLR）在 Zeller/Chatelier 略优于 L2(CLR)，metahit 相当。树模型对零值/单调变换天然鲁棒，验证了决策树「树模型无需 CLR」的判断；但 RF 无系数可解释性，主模型仍推荐 L2(CLR)。

### #4 CLR 必要性

| 数据集 | L2(原始) AUC | L2(CLR) AUC | ΔAUC |
|:--|:--:|:--:|:--:|
| Zeller CRC | 0.666 | 0.812 | **+0.146** |
| metahit IBD | 0.776 | 0.885 | **+0.109** |
| Chatelier Obesity | 0.544 | 0.643 | **+0.099** |

**结论**：CLR 对线性模型带来 **+0.10~+0.15 AUC 的显著增益**（三数据集一致），定和成分数据下线性模型**必须 CLR**（PR-006 铁律得到实证）。Chatelier 原始丰度 AUC=0.544 接近随机，CLR 后回升至 0.643。

### #5 三数据集类内可分性 / 批次差异

- PCA 前 2 主成分解释方差比 = **[0.0838, 0.0505]**（合计仅 13.4%）。
- 探索图 `S1-pca-tsne-explore.pdf`（2×2：PCA/t-SNE × dataset/disease 着色）。

**结论**：前 2 主成分仅解释 13.4% 方差 → 高维稀疏数据无主导低维结构，降维可视化只能定性参考；三数据集来自不同研究，低维投影预期按 dataset 明显分簇（批次效应强），跨疾病差异分析须按「样本量/类别平衡/信号强度/批次」四重来源归因，不可直接断言「哪类更可预测」。

### #6 small_adenoma 敏感性（registry B 级销项）

| 模型 | 全口径 AUC | 剔除口径 AUC | ΔAUC |
|:--|:--:|:--:|:--:|
| L2(CLR) | 0.812 | 0.836 | +0.024 |
| RF(原始) | 0.846 | 0.868 | +0.022 |

**结论**：剔除 26 例 small_adenoma 后 AUC 仅 +0.022~+0.024（**< 0.05 阈值**），口径影响**不显著**。B 级待裁定项可销项：维持「small_adenoma 归健康」全口径即可，论文注明口径即可，无需改标签。

---

## 3. 探索图清单 + 解读

> 全部输出至 `outputs/figures/_explore/`（探索用，禁止进论文）。

| 图 | 路径 | 解读 |
|:--|:--|:--|
| 基线 | `S1-baseline-explore.pdf` | 三数据集单特征最佳 AUC 0.64~0.82 均高于 Dummy(0.5)，信号真实存在；Chatelier 地板最低，与「肥胖信号弱」一致。 |
| 不平衡 | `S1-imbalance-explore.pdf` | metahit ACC(0.809) 与少数类 Recall(0.400) 严重背离，ACC 被多数类主导虚高；AUC 三数据集可比，确认主指标。 |
| 树 vs 线性 | `S1-tree-vs-linear-explore.pdf` | RF(原始) 与 L2(CLR) 性能相当（0.64~0.89），树模型免 CLR 天然吃零值；线性模型需 CLR 才达同等水平。 |
| CLR | `S1-clr-explore.pdf` | CLR 对线性模型 +0.10~+0.15 AUC 一致增益，定和成分数据线性建模必须 CLR。 |
| PCA/t-SNE | `S1-pca-tsne-explore.pdf` | 前 2 PC 仅 13.4% 方差，高维无主导结构；按 dataset 分簇（批次效应），按 disease 分离度与 AUC 强弱一致（CRC/IBD 可分、Obesity 重叠）。 |
| adenoma | `S1-adenoma-explore.pdf` | 剔除 small_adenoma 后 AUC 仅 +0.02，口径影响不显著，B 级销项。 |

---

## 4. 对决策树推荐的修正建议

1. **主模型 L2(CLR) 维持**：CLR 增益实证（+0.10~+0.15），可解释性 + 小样本稳健性，推荐不变。
2. **RF 作为非线性对照价值确认**：RF(原始) 在 Zeller/Chatelier 略优于 L2(CLR)，且免 CLR，建议正式实现保留 RF 对照（特征重要性复用 S2）。
3. **Chatelier 信号弱需诚实标注**：AUC 0.64~0.67，接近领域基准下界（0.65-0.75），且单特征基线(0.639)与全模型(0.643)几乎持平——**肥胖模型增益有限**，正式报告须诚实讨论，不包装成「合理选择」。
4. **metahit 少数类 Recall 低**：AUC 0.885 但 Recall 0.400，正式实现需考虑 class_weight 或阈值调优提升少数类召回。

---

## 5. 异常信号

1. **Chatelier 全模型 ≈ 单特征基线**（0.643 vs 0.639）：肥胖信号弱，多特征建模增益极小，需警惕过拟合包装。
2. **metahit ACC 与 Recall 严重背离**（0.809 vs 0.400）：最不平衡数据集，默认阈值下少数类召回差，评估必须用 AUC+F1(少数类)。
3. **PCA 前 2 主成分仅 13.4% 方差**：高维稀疏，降维可视化仅定性参考，不可作为分离度定量证据。
4. **Zeller L2(原始) AUC 方差大**（±0.138）：小样本下线性模型对 CLR 依赖强，未 CLR 时性能不稳定。

---

## 6. 待裁定项

- **small_adenoma 口径 [B级] → 已销项**：剔除敏感性 ΔAUC=+0.022~+0.024 < 0.05，维持全口径，论文注明即可（见 §2 #6）。

---

## 7. 交接收尾

验证脚本（`outputs/scratch/verify-S1-a*.py` + `utils.py`）+ 探索图（`outputs/figures/_explore/`）+ 本回报文档均已 commit。建模侧读取本文件进入 **1.2 方案辩论**。

**next_action**: 建模对话 1.2 方案辩论（基于 A 类验证结论裁定最终方案），说「继续」。
