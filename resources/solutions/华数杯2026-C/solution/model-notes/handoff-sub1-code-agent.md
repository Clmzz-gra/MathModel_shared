# S1 代码智能体交接说明（handoff-sub1-code-agent）

> 交接方：建模智能体 | 接收方：代码智能体 | 2026-08-08 | 分支：experiment/sub1
> 用途：代码智能体按本说明实现 S1 正式模型代码（`outputs/scratch/sub1-model.py`）+ 出图 + 代理值核销 + Artifact 登记。
> 建模依据（只读，勿改）：`solution/model-notes/approach-sub1-confirmed.md`（方案确认书）、`solution/model-notes/verify-sub1-20260807.md`（A 类共享事实）、`solution/model-notes/preprocess-sub1-20260807.md`（预处理清单）。

---

## 0. 交接概览

| 项 | 内容 |
|----|------|
| 子问题 | S1：GPU 需求统计 + 短期预测 + 基础算力调度 |
| 主方案 | Alpha：时间索引 0-1 MILP 精确调度（决策目标：逐时 GPU 利用率极差最小化） |
| 对照方案 | Beta：动态权重贪心 + 局部改进（B 类对比基准，head-to-head） |
| 预测口径 | 白噪声证明（ACF/泊松拟合）+ 简单基线（常数均值/Last-Hour/季节朴素/线性回归） |
| 模型代码 | `outputs/scratch/sub1-model.py`（本交接的交付物） |
| 已存在参考实现 | `outputs/notebooks/verify-sub1.ipynb`（增量验证 notebook，口径正确、已缓存，**代码实现须与其中逻辑一致**） |
| 调度结果已落盘 | `outputs/data/s1-schedule-test.pkl`（alpha/beta 开工表，**验证基准**） |

---

## 1. 数据接口（输入）

统一从 `outputs/data/s1-preprocessed.pkl` 读取（阶段 1.4 产物），结构：

| 键 | 内容 |
|----|------|
| `series` | dict：`Total/AITraining/BatchInference/RealTimeInference` 各 2400 长逐时 GPU 需求序列（float ndarray） |
| `feat_df` | DataFrame：hour/sin24/cos24/sin168/cos168/lag1/lag24/lag168/ma24/y_total/split（train/val/test/pre） |
| `split_idx` | dict：train/val/test/retrain 四段索引 |
| `schedule_input` | dict：tasks（538）/rt_fixed（160）/free（378）/base（6×30 实时固定占用）/hours/hidx/regions/cap/pue/max_it_power/max_facility_power |
| `power_mapping` | dict：AITraining=0.16 / BatchInference=0.10 / RealTimeInference=0.08 MW/等效GPU |

其他输入：`outputs/data/c-data-cleaned.pkl`（原始清洗数据，`workload_trace` 用于统计段）。

## 2. 统计段（GPU 需求分析）

实现 `outputs/scratch/sub1-model.py` 的统计模块，输出并打印：
- 三类任务数量/占比（538 窗内：194 训练 / 184 批量 / 160 实时；全量 50k：16724/16717/16559）
- GPU_Demand 分布（min/max/mean/median，长尾右偏 1–127）
- 逐时需求序列统计（Total mean≈614, std≈208, max≈1313）
- 白噪声证明：ACF(lag1..200) 各序列 ≈0；泊松拟合到达计数（mean≈20.8/h）

**统计量打印规范**：每数组打印 `min/max/mean/std`（PR-014 先算后画）。

## 3. 预测段（简单基线 + 白噪声证明）

按赛题协议三段式，实现基线预测并打印 RMSE/MAPE：

| 基线 | 协议 | 参考值（测试窗 RMSE） |
|------|------|----------------------|
| 常数均值 | 训练窗均值外推测试窗 | **221.0** |
| Last-Hour | 上小时值 | 293.5 |
| 季节朴素 lag24 | 昨日同时刻 | 331.5 |
| 线性回归（周期特征） | hour+sin/cos24+sin/cos168 | **215.4** |

> ⚠️ 口径修正（2026-08-08 代码实现核对）：交接稿原写"常数均值 189.3 最优"系测试窗事后均值（含数据泄漏），已弃用。诚实口径下线性回归 RMSE=215.4 略优于常数均值 221.0（提升 2.5%），结论不变：**序列近乎不可预测，用"白噪声证明 + 简单基线"叙事**。

**硬约束**：
- 训练/重训/测试切分严格按 `split_idx`（train 0–2351 / val 2352–2375 / retrain 0–2375 / test 2376–2399）
- 特征预热：有效样本从 h=168 起（lag168 需要）
- **结论口径**：线性回归与常数均值差距 <2.5% → 序列不可预测 → 论文用"白噪声证明 + 简单基线"叙事，**不引入 Prophet/LSTM**（B 类已核销）

## 4. 调度段（Alpha MILP 主方案）

数学框架见确认书 §3，代码实现要点（**与 notebook cell 5 逐位一致**）：

- 决策变量：`x[i,k]∈{0,1}`，任务 i 在候选小时 k 开工；实时任务固定 `x=arrive`
- 候选窗口：`w = [h for h in hours if lo<=h<min(latest,2406)-dur+1e-9 and h+dur<=min(latest,2406)+1e-9]`，空则 `w=[lo]`
- 约束：
  1. 每任务恰好开工一次 `Σ_k x[i,k]=1`
  2. GPU 容量（跨小时重叠精确折算）`base[r,t]+Σ a·x ≤ Cap_r`，`a=dem·|[h,h+dur)∩[t,t+1)|`
  3. U 上下界线性化（引入连续变量 Umax/Umin）
- 目标：`min(Umax−Umin)`（逐时利用率极差）
- 求解器：`scipy.optimize.milp`（HiGHS），`options={'time_limit':1800, 'mip_rel_gap':0.01}`
- **必须检查 `res.status`**：`status!=0` 时打印警告并标注"近似最优"（阶段 1.5 Major 修复项，必须保留）

## 5. 调度段（Beta 贪心对照）

- EDF 变体排序：`key=(latest-arrive-dur, -dem)`
- 评分：`score = np.var(放置后全矩阵利用率) - 0.1*spare`（spare=区域空余率）
- 容量检查 + 空窗 fallback（同 notebook cell 6）

## 6. 输出与出图（chart-generator skill）

模型代码先算后画（PR-014），图形统一经 chart-generator skill 生成：

| 图 | 类型 | 文件名 | 对应论文 |
|----|------|--------|---------|
| GPU 需求逐时序列 + ACF | 时序图 + ACF 子图 | `figures/sub1-demand-acf.pdf` | 统计段 |
| 预测 vs 实际（4 基线） | 预测对比图（竖虚线分隔） | `figures/sub1-forecast-baselines.pdf` | 预测段 |
| 最后 24h 调度甘特图 | 甘特图（按区域/类型分面聚合） | `figures/sub1-gantt-last24h.pdf` | 调度段 |
| 6 区域逐时 GPU 利用率 | 6 子图折线 | `figures/sub1-utilization.pdf` | 调度段 |
| Alpha vs Beta 对比 | 方法对比表（LaTeX） | 入文，不单独出图 | B 类验证 |

**格式硬约束**（chart-generator）：PDF 矢量、SimHei 中文字体、去饱和配色、线宽 1.5–2pt、`outputs/figures/` 输出、图表副本 + `manifest.md` 登记（`solution/artifacts/charts/`）。

## 7. 验证基准（必须复现，防口径漂移）

代码运行后，调度段数值须与以下基准一致（来源：`outputs/data/s1-schedule-test.pkl` + 确认书 §7）：

| 指标 | Alpha | Beta |
|------|-------|------|
| 逐时利用率极差 | **0.6283** | 0.8398 |
| 利用率方差 | 0.043992 | **0.037166** |
| 超容量小时 | 0 | 0 |
| 求解时间 | ~472–612s | <1s |

预测段（诚实口径，2026-08-08 代码实现复现）：线性回归 RMSE=215.4 为最优基线，常数均值 221.0 次之；若偏差 >0.5%，暂停并回查口径（不要自行"修正"）。

## 8. 代理值核销（阶段 1.5 强制）

| @PROXY | 本次任务 |
|--------|---------|
| 甘特图聚合绘制 | ✅ 本任务核销：538 任务全量甘特图 + 6 区域利用率子图 |
| HiGHS → PuLP+CBC 复现 | 可选验证：若环境有 PuLP，用 CBC 复跑小实例对比解一致性；无则标注"未复现" |

## 9. Artifact 登记 + 附录归档

- `solution/artifacts/manifest.md`：登记本任务所有图/表/代码片段
- `solution/appendix/supporting-materials-list.md`：登记支撑材料
- 附录代码归档：`solution/appendix/code/` 放最终可复现代码（`sub1-model.py` 定稿副本）
- 提交前检查：`outputs/data/cache/` 指纹缓存存在；`s1-schedule-test.pkl` 不随附录提交（属中间产物）

## 10. 代码自检清单（交付前）

- [ ] 与 notebook cell 4/5/6/7 逻辑逐位一致（候选窗口、重叠折算、目标、求解器配置）
- [ ] `res.status` 检查存在（Major 修复项）
- [ ] 预测切分严格按 split_idx，无测试窗泄漏
- [ ] 统计量全部打印（min/max/mean/std）
- [ ] 出图走 chart-generator 规范（SimHei/PDF/去饱和/manifest 登记）
- [ ] 调度数值复现 §7 基准（偏差 ≤0.5%）
- [ ] 代理值甘特图项核销
- [ ] 代码头注释：数据来源、字段含义、筛选条件
- [ ] 未修改任何 `solution/model-notes/` 下建模文档（只读）
