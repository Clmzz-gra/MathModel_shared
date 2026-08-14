# S2 代码审查清单（阶段 1.5）

> 2026-08-08 | 分支：experiment/sub2 | 方法：TRAE-code-review skill + 2 验证子代理交叉确认
> 审查对象：`outputs/scratch/verify-sub2.py`、`outputs/scratch/preprocess-sub2.py`、`outputs/scratch/verify-sub2-capacity.py`
> 对照基准：`solution/model-notes/approach-sub2-confirmed.md`（方案确认书）、`solution/model-notes/verify-sub2-20260808.md`（共享事实）

## 审查范围与结论

- 代码逻辑与确认书一致性：✅ 一致（可行域裁剪 $\mathcal{M}_j=\{r:\tau(\text{source}_j,r)\le L_j\}$、容量感知 90% 阈值 + 退路、成本/碳公式 `g·power·PUE·(price|carbon)`）
- 关键公式实现：✅ 正确（GPU-hour = `GPU_Demand × EstimatedDuration_min/60`；`reach_by_type` 按 `lat≤MaxLatency` 数据驱动，E↔F=18ms 实时互迁天然含入）
- 共享数据处理逻辑：✅ 一致（verify-sub2.py 与 preprocess-sub2.py 的候选判定同构；verify-sub2-capacity.py 消费 `s2-preprocessed.pkl`，均值口径与 verify-sub2.py 一致）
- 注释与代码匹配：✅ 已核对（头注释均按项目记忆五段式规范）
- 代理值核销：见下方"代理值状态"

## 审查清单（本轮共修复 4 项，2 项排除）

| No. | 问题 | 严重度 | 处置 | 位置 |
|-----|------|--------|------|------|
| 1 | **收益上界控制台打印未携带"全时域均值近似"口径**：头注释/行内注释已注明均值近似，但 L120 打印仅提示无容量约束，下游 model-notes 以打印为数据基础，存在引用口径传播缺口 | Minor | ✅ 已修复：打印补「电价/碳用全时域均值近似（不含时间维时移收益）」 | verify-sub2.py L117 |
| 2 | **可达性打印不完整**：仅打印 Batch 的 A/B 来源、RealTime 的 A/D 来源，AITraining 无打印；`reachable` 字典本已算全 6 区 | Minor | ✅ 已修复：改为全 6 区逐行打印（含 E↔F 可视化） | verify-sub2.py L71-73 |
| 3 | **verify-sub2-capacity.py 退路分支死代码**：`demand[r] += t["gh"] if False else 0` 恒等于 +=0，属混淆性占位语句（引用的 `r` 无意义） | Minor | ✅ 已修复：删除该行 | verify-sub2-capacity.py L65 |
| 4 | **verify-sub2-capacity.py 头注释不合规**：仅 6 行简版，缺原理/输入映射/论文章节（项目记忆强制五段式） | Minor | ✅ 已修复：重写为规范五段式 | verify-sub2-capacity.py L1-28 |
| 5 | **防御性断言缺失（隐性数据不变量）**：候选 `cand` 恒非空（自环 5ms ≤ 20ms）、`region_time_data` 按 Hour 单调递增均依赖外部数据，当前数据下不会出错，但未来数据源变更会静默错位/崩溃 | 可接受 | ✅ 已补：P2 加 `assert len(cand)>0`、P3 加 `assert Hour 单调递增` | preprocess-sub2.py L84/L105 |

## 验证器排除项（false_positive）

| 候选 | 排除理由 |
|------|---------|
| P3 逐小时电价数组与小时错位 | region_time_data 每区域 2407 行按 Hour 0-2406 单调递增（原始 CSV Hour 主键布局，清洗只 astype 不重排），布尔索引保序，`Hour<2400` 过滤后 `.values` 第 k 元素恰对应 Hour=k；P3 八 key 与消费方 verify-sub2-capacity.py 一致（2/2 排除） |
| `cand` 可能为空导致崩溃 | network_latency 36 条完整、6 条自环均 5ms ≤ 实时推理 MaxLatency=20ms 下限，`reach_by_type[tt][i,i]` 恒 True；三脚本实际运行成功为铁证（2/2 排除） |

## 关键事实核验（修复后重跑，exit 0）

- **E↔F 实时互迁已生效**：实时推理 14/36 条可达 = A/B/C 互迁 9 + D 本地 1 + **E↔F 互迁 2** + E/F 本地 2；锁死仅 D 来源 488（D→E/F 22/25ms 超 SLA）
- **F1/F2 无回归**：锁死 2.9%、可迁移 76.8%、成本降 20.2%（上界）、碳降 41.0%（上界）、E 超容量 198%
- **F3/F4 复现成功**：容量感知 90% 阈值 → 成本降 16.6%、碳降 30.4%、退路 117（0.23%）、E/F 各 90.0% 贴阈值、无超容量、自检 True

## 代理值状态（proxy-replacement-checklist-sub2 对照）

| @PROXY | 状态 |
|--------|------|
| 容量感知贪心 90% 阈值 | ✅ 已核销（阶段 2.1 正式实现 + 阈值 80/85/90/95% 敏感性） |
| 碳排 ε 三档 / 平均迁移时延 / D 滚动窗 / S1 全时域回测 | ✅ 已核销（ε 三档 + 任务 A 紧档位扫描扩展 / 时延 35.3ms / Q2 裁定统一 24h 窗 / C₀=333.3M E₀=374.28kt） |
| 新能源间接贡献 | ⏳ 移交 S3/S4（sub2-model.py 已报告各区域承接 GPU-hour + 可再生可用） |

## 结论

- 无 Critical / Major；4 项 Minor 全部修复，2 项候选经子代理交叉验证排除
- 三个脚本（A 类验证 ×2 + 阶段 1.4 预处理）审查通过，可进入阶段 2.0（模型构建）
