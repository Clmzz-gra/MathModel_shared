# S1 数据预处理操作清单（阶段 1.4）

> 2026-08-07 | 分支：experiment/sub1 | 脚本：`outputs/scratch/preprocess-sub1.py`
> 输出：`outputs/data/s1-preprocessed.pkl`
> 依据方案确认书：Alpha 精确 MILP 调度 + 简单基线/线性回归预测

## 操作清单（每项 | 对应模型需求 | 参数理由）

| # | 操作 | 对应模型需求 | 参数选择理由 |
|---|------|-------------|-------------|
| P1 | 逐时 GPU 需求聚合（Total + 3 类型，2400h） | 预测目标序列（决策点 2 双粒度） | 与 verify-sub1.py 口径一致；按到达小时聚合 GPU_Demand |
| P2 | 特征工程：hour + 24h/168h 周期 sin/cos + lag1/lag24/lag168 + ma24 | 线性回归外生变量 + 简单基线 | 周期特征取自 ACF 诊断候选周期；滞后特征覆盖 F2 基线逻辑 |
| P2' | 滞后特征预热期：有效样本从 h=168 起 | 避免 lag168 含 NaN | 前 168h 丢弃（预热），训练集 2184 样本 [168,2351] |
| P3 | 赛题协议切分 | 预测模型训练/验证/测试协议（赛题说明 5） | train 0-2351 / val 2352-2375 / retrain 0-2375 / test 2376-2399 |
| P4 | 调度输入：测试窗 538 任务 → 标准结构 | Alpha MILP 输入 | 与 notebook cell 4/5 口径一致（region/arrive/dur/dem/latest/cand + 实时 base） |
| P4' | 自由任务候选窗口计算 | MILP 决策变量域 | 窗口 = [max(arrive,2376), min(latest,2406)-dur]，实时固定 |

## 切分结果

| 集合 | 样本数 | 时域 |
|------|--------|------|
| train | 2184 | [168, 2351] |
| val | 24 | [2352, 2375] |
| test | 24 | [2376, 2399] |
| retrain | 2208 | [168, 2375] |

## 调度输入核验

- 测试窗 538 任务：160 实时（固定开工）+ 378 自由（训练/批量）
- 实时 base 峰值占用：东部 37–41.6 GPU / 西部 8–10.8 GPU，全部远低于 Available_GPU（F3 复验通过）
- 候选窗口：与 notebook 口径一致，无空窗任务

## 一致性备注

- P4 结构与 notebook 的 `s1_test_tasks` 缓存同源（同一过滤条件 ArrivalHour ≥ 2376），避免口径漂移
- 预测序列 `series` 与 notebook cell 2 的 `s1_series_acf` 同源
