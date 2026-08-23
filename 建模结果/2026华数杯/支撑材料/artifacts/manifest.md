# manifest.md — Artifact 登记清单

> 随阶段 2 逐步填充 | 2026-08-08 S1 阶段 2.1 首填

## 图表（outputs/figures/）

| 图名 | 文件 | 类型 | 对应论文 | 状态 |
|------|------|------|---------|------|
| 逐时 GPU 需求 + ACF | sub1-demand-acf.pdf | 时序图 + ACF 子图 | S1 统计段（白噪声证明） | ✅ v1 |
| 预测 vs 实际（4 基线） | sub1-forecast-baselines-v2.pdf | 预测对比图 | S1 预测段 | 🗑 已删（2026-08-10 压缩，数据见预测基线表） |
| 最后 24h 调度甘特图 | sub1-gantt-last24h-v2.pdf | 甘特图（按类型 3 分面，含收尾） | S1 调度段 | ✅ v2（分面修正）→ 移至附录 |
| 6 区域逐时 GPU 利用率 | sub1-utilization-v2.pdf | 6 子图折线（Alpha/Beta） | S1 调度段 | ✅ **v2（范围确认）** |
| S3 SOC+充放（6 区域） | sub3-soc-charge-discharge-v2.pdf | 2×3 子图（SOC + 充放功率） | S3 结果段 | ✅ v2 |
| S3 净购电时序对比 | sub3-net-import-compare-v2.pdf | 折线（基准 vs 最优，聚合削峰） | S3 储能价值段 | 🗑 已删（2026-08-10 压缩） |
| S3 碳排压降极限 | sub3-carbon-floor-v2.pdf | 分组柱状（基准 vs C_min，标注 ε_min） | S3 碳排分析 | 🗑 已删（2026-08-10，与 ε_min 表重复） |
| S3 四指标对比 | sub3-four-metrics-v2.pdf | 2×2 分组柱状 | S3 四指标段 | 🗑 已删（2026-08-10 压缩） |
| S3 区域峰值削峰 | sub3-peak-shaving-v2.pdf | 6 区域分组柱状 | S3 削峰段 | ✅ v2 |
| S4 场景对比 | sub4-scenario-compare.pdf | 双轴柱状（成本/碳排，部分可行标注） | S4 场景段（问题四核心） | ✅ v1 |
| S4 区域成本-碳排 | sub4-region-cost-carbon.pdf | 6 区域双轴柱状 | S4 多区域协同段 | ✅ v1 |
| S4 储能 SOC+充放（D） | sub4-soc-storage.pdf | 2 子图（SOC + 充放功率） | S4 储能协同段 | 🗑 已删（2026-08-10 压缩） |
| S4 新能源利用率双口径 | sub4-utilization.pdf | 6 区域分组柱状 + S3 基准线 | S4 利用率段（衔接 E7） | ✅ v1（未入正文） |
| S4 峰值净购电 | sub4-peak-net.pdf | 6 区域柱状 | S4 峰值净购电段 | ✅ v1（未入正文） |
| S4 基荷策略贡献 | sub4-baseload-contribution-v2.pdf | 任务覆盖柱状（0-1 变量压缩） | S4 基荷创新段 | 🗑 已删（2026-08-10 压缩，数字在正文） |
| S4 算-储-电机理（A/D） | sub4-mechanism-net.pdf | 净购电+SOC 时序（A 东部高载 vs D 算力中心） | S4 协同机理段 | ✅ v1 |
| S4 场景权衡热力图 | sub4-tradeoff-heatmap.pdf | 场景×指标向好变化% 热力图 | S4 场景权衡段（替代原 scenario-compare） | 🗑 已删（2026-08-10 压缩） |
| S4 碳约束灵敏度 | sub4-pareto-carbon.pdf | 逐区域 ε_min 与降碳空间 | S4 灵敏度段 | 🗑 已删（2026-08-10 压缩） |
| S4 峰谷价差灵敏度 | sub4-sens-price.pdf | 成本线性/碳不变/储能套利饱和 | S4 灵敏度段（§8.4） | ✅ v1 |
| S4 新能源波动灵敏度 | sub4-sens-renew.pdf | 成本碳排线性/策略不变 | S4 灵敏度段（§8.4） | ✅ v1 |
| S4 D 区边际成本 | sub4-d-marginal-cost.pdf | 双面板（D 区 ε-Pareto） | S4 碳约束分析段 | 🗑 已删（2026-08-10 压缩） |
| S4 LNS 收敛曲线 | sub4-lns-convergence.pdf | E/F 区 50 轮成本收敛 + EDF 基线对照 | S4 大规模可解性段（LNS） | ✅ v1 |
| 跨模型架构图 | 00-architecture.pdf | 架构图（S1–S4 数据流/控制流，全部已定稿） | 阶段 3 跨子问题架构 | ✅ v1（S4 已集成） |
| BDS 非线性检验 p 值 | sub1-whitenoise-bds-v1.pdf | 分组柱状（4 序列 × m=2/3/4，标 0.05 线） | S1 预测段四重检验（附录） | ✅ v1（2026-08-09） |
| Granger 因果 p 值热力图 | sub1-whitenoise-granger-v1.pdf | 1×2 热力图（类型间 6 组 + 区域间 4 组） | S1 预测段四重检验（附录） | ✅ v1（2026-08-09） |
| 测试窗预测对比（基线/Prophet/LSTM） | sub1-whitenoise-forecast-v1.pdf | 5 线对比（实际 + 4 模型，标 RMSE） | S1 预测段四重检验（附录） | ✅ v1（2026-08-09） |
| 四模型 RMSE 对比 | sub1-whitenoise-rmse-v1.pdf | 水平柱状（标 RMSE + 相对均值提升%） | S1 预测段四重检验（附录） | ✅ v1（2026-08-09） |

> 2026-08-10 字数压缩：正文删图 13 张（标记 🗑，含未登记于上表的 S2 四图 region-load/reachability/epsilon/threshold）、甘特图移至附录，另删 minutil 与 s1-tail 两张表；正文剩余 11 图 + 附录 6 图，正文图清单以各 chapter*.tex 为准。

> S3 v2 说明（2026-08-08）：基于**受限消纳 ε=1.00 主解**（D4 修订：ε=0.90/0.95 不可行，ε_min 实测 A 0.9935/B 0.9938/C 0.9941/D 0.9693/E 0.9574/F 0.9616，碳排作评价指标）。原 `sub3-pareto-frontier.pdf`（自由消纳退化口径）已删除替换为碳排压降极限图；v1 四图（自由消纳退化口径，G≈0）作废仅存档，勿用于论文。

> v1 保留（不覆盖）；v2 修正：甘特图按类型分面（538 任务→3 子图）、含 [2400,2406) 收尾任务 168 条；利用率纵轴 [0,1] 确认（max 0.8528 ≤1）。图表副本已同步 `solution/artifacts/charts/`。

## 结果表

| 表 | 文件 | 说明 |
|----|------|------|
| S1 head-to-head 对比 | `outputs/data/s1-schedule-test.pkl` | alpha/beta 开工表（验证基准） |
| S1 结果表（LaTeX 片段） | `solution/artifacts/tables/s1-results.tex` | 3 表：head2head / 预测基线 / 收尾占比（阶段 2.2 登记，iter 报告引用） |
| S4 结果表（LaTeX 片段） | `solution/artifacts/tables/s4-results.tex` | 6 表：六指标 / 分区域 / 成本归因 / 场景 / Clean-test / LNS（阶段 2.2 登记，iter-04 报告引用） |

## 代码

| 代码 | 文件 | 说明 |
|------|------|------|
| S1 正式模型 | `outputs/scratch/sub1-model.py` | 统计+预测+调度 Alpha/Beta（阶段 2.1） |
| S1 增量验证 notebook | `outputs/notebooks/verify-sub1.ipynb` | 缓存执行版（阶段 1.1/1.2） |
| S1 预处理 | `outputs/scratch/preprocess-sub1.py` | 阶段 1.4 |
| S2 正式模型 | `outputs/scratch/sub2-model.py` | 基线+容量感知分配+滚动窗 MILP（K3v2）+ε 三档（阶段 2.1） |
| S2 ε 紧档位扫描 | `outputs/scratch/scan-sub2-epsilon.py` | E_min + 紧档位扫描（任务 A：--emin/--quick/--tight 分段可续） |
| S2 最低利用率灵敏度 | `outputs/scratch/sensitivity-minutil-sub2.py` | A/B 保底 0/5/10/15% vs 成本（建模裁定：外部预分配不改主线，0% 档与现有分配逐任务一致校验） |
| S2 增量验证 notebook | `outputs/notebooks/verify-sub2-model.ipynb` | 分段缓存执行版（基线/ε 三档指纹缓存+断点） |
| S2 预处理 | `outputs/scratch/preprocess-sub2.py` | 阶段 1.4 |
| S2 A 类验证 | `outputs/scratch/verify-sub2.py` / `verify-sub2-capacity.py` | F1-F4 共享事实（阶段 1.1） |
| S2 notebook 生成/执行 | `outputs/scratch/gen-sub2-model-notebook.py` / `run-verify-sub2-model-notebook.py` | 增量执行基础设施 |
| S3 数据预处理 | `outputs/scratch/preprocess-sub3.py` | 阶段 1.4（LP 输入面板，已审查通过） |
| S3 正式模型（LP） | `outputs/scratch/sub3-model.py` | 阶段 2.1（受限消纳 B1，主解 ε=1.00；含 ε_min/撞顶/B3 诊断；`--eps`/`--free` 参数化） |
| S3 出图 | `outputs/scratch/charts-sub3.py` | 阶段 2.1（v2 五图：Pareto 已换碳排压降极限） |
| S3 LP 核心片段 | `solution/artifacts/code-snippets/sub3-lp-core.py` | 阶段 2.1（≤15 行） |
| S1 四重检验出图 | `outputs/scratch/charts-sub1-whitenoise.py` | 阶段 3.2（BDS/Granger/预测对比/RMSE 四图 v1，数值与 verify-sub1-b2.py 一致，先算后画） |
| S1 B 类四重检验 | `outputs/scratch/verify-sub1-b2.py` | 阶段 3.2（BDS/Prophet/LSTM/Granger 对抗检验，可复跑） |
| S4 正式模型 | `outputs/scratch/sub4-model.py` | 半耦合 MILP：层 1 贪心分配 + 基荷预填 + 层 2 每区 MILP 含储能（阶段 2.1） |
| S4 预处理 | `outputs/scratch/preprocess-sub4.py` | 阶段 1.4（基荷预填 + 改派 + EDF） |
| S4 出图 v2 | `outputs/scratch/charts-sub4-v2.py` | 5 图：机理 / 权衡热力图 / 碳约束 / 峰谷价差 / 新能源波动（阶段 2.1.6） |
| S4 D 区边际成本图 | `outputs/scratch/chart-d-marginal.py` | D 区 ε-Pareto 双面板（阶段 2.1.6） |
| S4 成本归因 | `outputs/scratch/decompose-sub4-cost.py` | 逐项关闭储能/卖电的 MILP 重跑（阶段 2.1.6） |
| S4 灵敏度扫描 | `outputs/scratch/sens-scan-sub4.py` | ε/峰谷价差/新能源网格 + ε_min 二分 |
| S4 Clean-test | `outputs/scratch/preprocess-sub4-clean.py` / `sub4-clean-model.py` / `dryrun-sub4-ratio.py` | 禁改派 + 配对选时 + GH 分层抽样（阶段 2.2 实验） |
| S4 LNS 启发式 | `outputs/scratch/lns-sub4.py` / `inspect-lns-progress.py` | 大邻域搜索 + 检查点断点续跑（阶段 2.2 实验） |
| S4 LNS 收敛图 | `outputs/scratch/charts-sub4-lns.py` | E/F 区 50 轮收敛曲线（阶段 2.2，先算后画） |
