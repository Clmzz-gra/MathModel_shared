# 门禁 M 审查结论：S2 2.0 数学推导（math-S2.tex）

> 审查代理角色：建模对话（modeling Preset）· 门禁 M 审查代理（2.0 数学推导部分）
> 模型：deepseek-v4-pro:0813
> 审查日期：2026-08-21
> 审查对象：`solution/model-notes/math-S2.tex`（12 节 437 行）+ 编译 PDF `math-S2.pdf`
> 对照材料：`approach-S2-confirmed.md`、`handoff-S2-code-agent.md`、`proxy-replacement-checklist-S2.md`、`E:\MathModel_pj\latex\templates\internal-report.cls`
> 门禁 M 判定内容：数学推导自洽 / 可实现性 / 共现分析公式 / 假设检验计划 + 方案讲解质量 / 编译质量

## 必读清单已读汇报

- [x] `TRAE-建模.md`（2.0 数学推导门禁 + 方案讲解质量约束 6 条）
- [x] `TRAE.md`（门禁 M 判定内容 + 门禁协议审查/裁决分离）
- [x] skill `math-explainer`（讲解质量约束 6 条，审查对照）
- [x] 管线进度确认：`git log --oneline -10` → 1.4 已完成 `37a9b6b`，2.0 已完成 `44612e0`，当前处于门禁 M 审查
- [x] 审查对象与对照材料全文读取（math-S2.tex 437 行 / approach 293 行 / handoff 152 行 / proxy 清单 23 行 / internal-report.cls 32 行）

---

## 判定内容逐项结论

### 1. 数学推导自洽

**结论：通过。** 全部核心公式代数正确、与 approach §3 一致、无代数错误。

| 公式 | 位置 | 自洽性核验 | 与 approach 一致性 |
|:--|:--|:--|:--|
| Lasso 目标函数 | `eq:lasso`（L126-133） | 交叉熵 + L1 惩罚，惩罚项 $\lambda\sum_{j=1}^{p}\lvert\beta_j\rvert$ 正确排除截距 $\beta_0$（$j$ 从 1 到 $p$）；$\hat{p}_i=\sigma(\beta_0+\mathbf{x}_i^\top\beta)$ 正确 | 与 approach §3.2 逐字一致 |
| bootstrap 频率 | `eq:freq`（L146-151） | $\hat{\pi}_j=\frac{1}{B}\sum_b\mathbf{1}\{\hat{\beta}_j^{(b)}\neq 0\}$ 正确 | 与 approach §3.3 一致 |
| 稳定特征集 | `eq:stable`（L155-160） | $\hat{S}_\tau=\{j:\hat{\pi}_j\ge\tau\}$，$\tau=0.5$ 正确 | 与 approach §1.1/§3.3 一致 |
| Fisher 超几何 | `eq:fisher`（L211-215） | $P=\frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}$ 为标准 2×2 超几何概率；自检项 5 用 Vandermonde 恒等式证归一性，正确 | 与 approach §3.6 一致 |
| Wilcoxon 秩和 | `eq:wilcoxon`（L227-234） | $U_1=n_1n_2+\frac{n_1(n_1+1)}{2}-R_1$，$U_2=n_1n_2-U_1$，$U=\min(U_1,U_2)$；自检项 6 证 $U_1+U_2=n_1n_2$，正确（$U_1/U_2$ 命名与标准 Mann-Whitney 互换，但取 $\min$ 对称，无方向偏置） | 与 approach §3.5 一致 |
| BH-FDR | `eq:fdr`（L247-252） | $p_{(k)}\le\frac{k}{m}\alpha$ 为标准 BH 步骤；**$m=1331$ 全特征规模**（人类裁定，L245 显式标注） | 与 approach §3.7 一致 |
| Spearman | `eq:spearman`（L270-277） | $\rho_{jk}=1-\frac{6\sum d_i^2}{n'(n'^2-1)}$ 为标准无结 Spearman 公式 | 与 approach §1.3b 一致 |
| 共现 Fisher | §5.2（L282-285） | 2×2 列联表（j 存在/缺失 × k 存在/缺失）复用 `eq:fisher` 超几何，正确 | 与 approach §1.3b 一致 |
| CLR | `eq:clr`（L106-112） | $\mathrm{clr}(x_{ij})=\ln x_{ij}-\frac{1}{p}\sum_k\ln x_{ik}=\ln\frac{x_{ij}}{g_i}$ 正确；自检项 4 证 $\sum_j\mathrm{clr}(x_{ij})=0$ 恒成立 | 与 approach §3.4 一致 |
| 零值替换 | `eq:zeroreplace`（L94-99） | $\delta=0.65\times10^{-5}=6.5\times10^{-6}$ 正确 | 与 approach §3.4 / proxy P6 一致 |
| 零值占比 | `eq:zerofraq`（L68-73） | $z_j=\frac{1}{n}\sum_i\mathbf{1}[x_{ij}=0]$ 正确 | 与 approach §1.1 一致 |
| RF permutation | `eq:permimp`（L300-305） | $\mathrm{PI}_j=\mathrm{AUC}_{\text{orig}}-\frac{1}{R}\sum_r\mathrm{AUC}_{\text{perm},j}^{(r)}$ 正确 | 与 approach §1.4 一致 |
| PLS-DA VIP | `eq:vip`（L315-322） | $\mathrm{VIP}_j=\sqrt{p\cdot\frac{\sum_a\mathrm{SS}_a(w_{ja}/\lVert w_a\rVert)^2}{\sum_a\mathrm{SS}_a}}$ 为标准 VIP 公式，$\mathrm{SS}_a=\sum_i t_{ia}^2$ 正确 | 与 approach §1.4 一致 |

**证据路径**：`solution/model-notes/math-S2.tex` L18-437（§1-§9 全部公式）；`approach-S2-confirmed.md` §3.1-§3.7。

---

### 2. 可实现性（每个公式可无歧义实现 + 与 handoff 规格一致）

**结论：通过（附 3 条 B 级表述澄清）。** 符号表（§1.1，L22-58）覆盖全部核心符号并给读法；关键参数与 handoff 规格一致。

| 参数 | math-S2.tex | handoff | 一致性 |
|:--|:--|:--|:--|
| τ | 0.5（范围 0.5~0.6） | τ=0.5~0.6（暂定 0.5） | ✓ |
| VIP 阈值 | >1.5 | >1.5 | ✓ |
| B | 50~100 | 50~100 | ✓ |
| δ | 6.5e-06 | 6.5e-06 | ✓ |
| m（FDR） | 1331 全特征 | 1331 全特征 | ✓ |
| α | 0.05 | 0.05 | ✓ |
| 过滤阈值 | 0.95 | 0.95 | ✓ |
| K（CV 折数） | 5 | 分层 CV（未显式给 K，math 补 5） | ✓（math 更明确） |

符号定义与读法齐全：符号表含 $n,p_0,p,x_{ij},y_i,z_j,\delta,g_i,\mathrm{clr},\beta_0,\beta_j,\lambda,C,\sigma,\hat{p}_i,B,\hat{\pi}_j,\tau,\mathbf{1},\binom{n}{k},m,\alpha,U,R_1,\rho,\mathrm{VIP}_j$ 共 24 项，均给「读法 + 定义」。公式内联补充定义 $\hat{\beta}_j^{(b)}$（L149）、$n_1,n_2$（L236）、$n',d_i$（L273/277）、$A,w_{ja},t_{ia},\mathrm{SS}_a$（L318/322）、$R,\mathrm{AUC}$（L305）。

**证据路径**：`math-S2.tex` §1.1 符号表 + §7 参数汇总表（L332-353）；`handoff-S2-code-agent.md` §1.1-§1.6。

---

### 3. 共现分析公式（2026-08-21 人类裁定新增）

**结论：通过。** 数学表述完整，三要素齐全。

- **Spearman 相关**（§5.1，L265-280）：`eq:spearman` 给出完整公式 + $n'$（两特征均非零样本数）+ $d_i$（秩差）定义 + 正/负相关语义（共现/互斥）。✓
- **共现/互斥检验**（§5.2，L282-285）：2×2 列联表（两者均存在/仅 j/仅 k/均缺失）复用 `eq:fisher` 超几何，p 值 = 更极端表概率之和，输出 cooccur/exclude 边 + 共现网络（节点/边/方向）。✓
- **边界声明**（§5.3，L287-290）：显式声明「小样本下仅对入选标志物做二阶探索，无法全特征交互建模；标志物筛选主口径仍为边际信号」，定位讨论节、不改变主选择口径。✓

与 approach §1.3b 逐项一致（范围=入选标志物 10~20 个、Spearman 非零样本 CLR 后丰度、Fisher 存在/缺失口径、边界声明原文）。

**证据路径**：`math-S2.tex` §5（L260-290）；`approach-S2-confirmed.md` §1.3b（L49-57）。

---

### 4. 假设检验计划 + 方案讲解质量（math-explainer 6 条）

**结论：通过。** 假设检验计划完整，讲解质量 6 条全满足。

**假设检验计划**（§8，L406-427）：H1-H8 逐条给检验方法与结果/状态——H1-H7 已证（引用 F2-F7 实证数字），H8/B1/B2/B3 标注「待验证」（2.1 实现后回填），无「因模型类型跳过」的遗漏。核心假设（H1 成分数据、H2 过滤、H3 两路信号、H4 低冗余、H5 频率可分、H6 已知锚点）均有实证依据。

**math-explainer 6 条逐条核验**：

| 约束 | 核验 | 证据 |
|:--|:--|:--|
| 1 先定义再使用 | 符号表 24 项全给读法+定义；$\forall/\exists$ 类符号未出现（本推导无全称/存在量词），指示函数 $\mathbf{1}[\cdot]$ 给读法 | §1.1（L22-58） |
| 2 每步有「因为」 | 每个公式后附「因为」段，代入处标数值（如 δ=6.5e-06、m=1331、τ=0.5、264 维） | §2-§6 各公式后 |
| 3 多视角 | 几何（CLR 单纯形投影 L116）、代数（L1 尖角置零 L137）、统计（bootstrap 方差量化 L163） | §2.2/§3.1/§3.2 |
| 4 核心洞察收束 | §10 三问收束（解决什么/怎么做/为什么） | L429-433 |
| 5 直觉解释 | §9 无公式直觉段（L400-404）+ 每公式「直觉类比」（问卷题/预算约束/靠谱员工/店有没有卖/两班身高/1331 门课） | §9 + 各公式后 |
| 6 防跳跃 | 末尾列 5 个可展开概念点（BH-FDR vs Bonferroni、CLR 伪计数、折内泄漏、λ 选择、RF 有偏机制） | L435 |

**证据路径**：`math-S2.tex` §8（L406-427）、§9（L400-404）、§10（L429-435）。

---

### 5. 编译质量

**结论：通过。** xelatex 编译通过，无 `??` 未解析引用。

- 编译产物 `math-S2.pdf`（284067 字节，2026-08-21 12:10:40）存在且与 `.tex`（12:10:14）时间戳一致。
- `.log` 检查：**0 个错误**（`^!` 计数 0）、**0 个未定义引用**（`Reference ... undefined` 计数 0）、**0 个 `??`**。
- 仅 3 条 xeCJK 字体重定义警告（SimSun/Microsoft YaHei/FangSong），属 `internal-report.cls` 与 `ctex` 包正常交互，非错误。
- 所有 `\ref{sec:clr}`、`\ref{sec:leakage}` 等交叉引用均解析成功。

**证据路径**：`solution/model-notes/math-S2.pdf`、`math-S2.log`（0 error / 0 undefined / 0 `??`）。

---

## 问题清单

| # | 问题 | 严重度 | 证据路径 |
|:--|:--|:--|:--|
| Q1 | **CLR 重归一化步骤表述不一致**：approach §3.4 与 handoff §1.2 均写「零值替换 δ 后**再重归一化到和为 1**」，而 math-S2.tex §2.2 只写「乘法替换 → 几何均值中心化」，未提重归一化。数学上 CLR 尺度不变（$\mathrm{clr}(cx)=\mathrm{clr}(x)$），重归一化是 no-op，不影响正确性；但为实现无歧义，建议统一表述（明确「重归一化可省略，因 CLR 尺度不变」或补入 math-S2.tex） | B级（表述） | `math-S2.tex` §2.2（L92-117）vs `approach-S2-confirmed.md` §3.4（L137）vs `handoff-S2-code-agent.md` §1.2（L29） |
| Q2 | **过滤口径「并集 vs 独立」表述不完全一致**：math-S2.tex §2.1 写「三病并集同一 264 特征集」，handoff §1.1 写「每病独立计算零值占比」，approach §4 写「取三病并集或各病独立过滤（见 handoff 规格）」。三者对「264 特征集是单一公共集还是各病独立」未完全对齐，可能影响实现。建议明确：三病数据合并计算 $z_j$ 取并集得单一 264 特征集（与 math-S2.tex 一致），并回 V2 确认 | B级（口径） | `math-S2.tex` L84 vs `handoff-S2-code-agent.md` §1.1（L24-25）vs `approach-S2-confirmed.md` §4（L184） |
| Q3 | **符号 $\mathbf{x}_i$（第 i 样本行向量）未单列**：`eq:lasso` 用 $\mathbf{x}_i^\top\beta$，但符号表仅列元素 $x_{ij}$，未列行向量 $\mathbf{x}_i$。可从上下文推断，但建议符号表补一行「$\mathbf{x}_i$：第 i 样本的 CLR 后特征向量（p 维）」 | B级（表述） | `math-S2.tex` L131 vs §1.1 符号表（L22-58） |
| Q4 | **Spearman 公式为无结（no-ties）版本**：`eq:spearman` 用 $1-\frac{6\sum d_i^2}{n'(n'^2-1)}$，未提并列秩（ties）修正。非零样本上仍可能存在并列丰度，建议实现时用 tie-corrected Spearman（scipy `spearmanr` 默认处理），与公式语义一致 | B级（实现细节） | `math-S2.tex` L270-277 |

> 无 A 级（阻断）问题。4 条均为 B 级（表述/口径/实现细节），不影响数学正确性与下游按现有规格执行，可并行销项。

---

## 结论

**通过。**

数学推导自洽（13 个核心公式代数正确、与 approach §3 一致、无代数错误）、可实现（符号定义与读法齐全、关键参数 τ=0.5/VIP>1.5/B=50~100/δ=6.5e-06/m=1331 与 handoff 一致）、共现分析公式完整（Spearman + Fisher + 边界声明三要素齐全）、假设检验计划完整 + 讲解质量 6 条全满足、编译通过（0 error / 0 undefined / 0 `??`）。

4 条 B 级问题（Q1 CLR 重归一化表述、Q2 过滤口径并集 vs 独立、Q3 符号 $\mathbf{x}_i$ 未单列、Q4 Spearman 无结版本）建议在 2.1 实现前由建模对话澄清或代码对话按拟议口径执行，不阻断门禁 M 放行。
