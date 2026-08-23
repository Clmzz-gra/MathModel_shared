# 审查结论：S2 阶段 2.1 正式模型代码 — 代码逻辑审查（第二遍）

> 角色+模型：coding 子代理 / deepseek-v4-pro:0813（max）
> 审查类型：② 代码逻辑（变量/索引/数据版本/输出路径/并行度/C1 性能声明）
> 待审代码：`outputs/scratch/S2-model.py`（492 行）
> 对照文档：`handoff-S2-code-agent.md`（§1 规格、§3 输出结构）、`proxy-replacement-checklist-S2.md`（P1-P8）
> 日期：2026-08-21

---

## ① 已读清单汇报

- `TRAE-代码.md`（重点「代码审核规则」）
- `TRAE-规范.md`（C1 代码头注释、C4 高耗时、C8 性能与并行）
- `outputs/scratch/S2-model.py`（全文）
- `handoff-S2-code-agent.md`（§1 规格、§3 输出结构）
- `proxy-replacement-checklist-S2.md`（P1-P8）
- 附加核验：`S2-preprocessed.pkl` 实际字段结构（python 读取）

---

## ② 聚焦点逐项结论

| # | 聚焦点 | 结论 | 证据 |
|---|---|---|---|
| 1 | C1 头注释「性能」字段 | ✅ 通过 | L23-26 含「性能：并行策略（joblib Parallel, loky, n_jobs=8）+ RF n_jobs=-1 + 无 GPU 方案理由」 |
| 2 | C8 单核红线 | ✅ 通过 | bootstrap 用 `Parallel(n_jobs=8)`（L143-145、L159-161）；RF permutation `n_jobs=-1`（L237、L240） |
| 3 | 数据版本 | ✅ 通过 | L68 `DATA_IN = S2-preprocessed.pkl`；全文无 B-raw.pkl / xlsx；字段实测匹配（feature_names=264、per_disease 含 X_raw/X_clr/y/cv_folds） |
| 4 | 索引/变量 | ✅ 通过 | 分层重抽样病/健分别 `rng.choice`（L125-129）；频率 `np.mean(vstack)`（L146）；BH-FDR m=1331 实现正确（L108-119，q=p·m/rank 单调回填）；共现 Fisher 2×2 表 `[[both,only_a],[only_b,neither]]` 方向正确，OR>1=cooccur（L222-225） |
| 5 | 输出路径 | ✅ 通过 | `S2-results.pkl → outputs/data/`（L69）；探索图 → `outputs/figures/_explore/`（L70） |
| 6 | 代理值核销 | ✅ 通过（附 1 待裁定项） | τ=0.5(P1)、C=0.1(P7)、δ=6.5e-06(P6)、VIP=1.5(P2)、FDR m=1331(P3)、B=100(P4)、Top-N=20(P8) 均取当前临时值，无 TODO/占位符 |
| 7 | pkl meta field_semantics | ✅ 通过 | L445-457 含 field_semantics，覆盖 frequency/cv_frequency/fisher_fdr/wilcoxon_fdr/direction/dominant_signal/cooccurrence_edges.type/rf_importance/vip/n_*_sig |

---

## ③ 问题清单

| 问题 | 严重度 | 证据路径 |
|---|---|---|
| C 范围未实现：handoff §1.3 要求「正式实现给 C 范围（如 0.01~1.0）」，代码固定 `C_LASSO=0.1`，未给范围亦未做交叉验证。P7 状态「待定」、当前临时值 0.1，代码取临时值符合核销，但与 handoff「给范围」要求存在规格矛盾 | 中（待裁定项） | `handoff-S2-code-agent.md` §1.3 L36；`S2-model.py` L76；`proxy-replacement-checklist-S2.md` P7 |
| meta 硬编码 `filter_threshold=0.95`、`clr_delta=6.5e-06`，未从 `prep["meta"]` 读取（prep meta 已含同名字段），存在口径漂移风险 | 低 | `S2-model.py` L440/L442；`S2-preprocessed.pkl` meta 实测含 filter_threshold/clr_delta |
| 共现热图用 `labels.index(short_name(fa))` 定位，若两特征种名相同（不同属）会索引错位 | 低（仅探索图，不影响正式结果） | `S2-model.py` L398-399 |

---

## ④ 结论

**通过（附 1 个待裁定项 + 2 个低严重度提示）。**

核心聚焦点（C1 性能声明 / C8 并行度 / 数据版本 / 索引变量 / 输出路径 / 代理值核销 / field_semantics）全部合规。唯一需上报的待裁定项为 **C 范围规格矛盾**：handoff §1.3 要求给 C 范围，P7 未定 C 选择方式，代码取固定 C=0.1（V6 验证值）——建议建模对话裁定「固定 C=0.1 是否可接受」或回填 C 范围实现。两个低严重度提示（meta 硬编码、热图 short_name 索引）不阻塞放行，可随修复一并处理。
