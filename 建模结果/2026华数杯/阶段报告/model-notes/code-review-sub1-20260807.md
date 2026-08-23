# S1 代码审查清单（阶段 1.5）

> 2026-08-07 | 分支：experiment/sub1 | 方法：TRAE-code-review skill + 2 验证子代理交叉确认
> 审查对象：`outputs/scratch/preprocess-sub1.py`、`outputs/scratch/gen-verify-sub1-notebook.py`、`outputs/scratch/run-verify-sub1-notebook.py`
> 对照基准：`solution/model-notes/approach-sub1-confirmed.md`（数学框架）

## 审查范围与结论

- 代码逻辑与确认书一致性：✅ 一致（时间索引 0-1 MILP 公式、GPU 容量重叠折算 a_{i,h,t}、两阶段目标）
- 关键公式实现：✅ 正确（容量约束 Σa·x ≤ Cap−base；U 上下界约束线性化无误）
- 共享数据处理逻辑：✅ 一致（preprocess-sub1.py 与 notebook cell 4/5 同源同口径）
- 代理值核销：见下方"代理值状态"

## 审查清单（交叉验证后保留 2 项）

| No. | 问题 | 严重度 | 建议 | 代码位置 |
|-----|------|--------|------|---------|
| 1 | **Alpha MILP 未显式处理 time-limit 状态（status=1）**：`milp()` 仅当 `res.x is None` 抛异常，未对 status=1（时间限制内返回可行次优解）标注警告。若未来重跑超时，将静默采用非严格最优解，论文"精确最优"口径存在失真风险 | **Major**（实测本次未触发：res_status=0, dt=612s < 1800s） | 补充 status/gap 检查：若 status≠0 打印警告并记录"近似最优（gap=X%）"；论文注明 mip_rel_gap=0.01 的近似口径 | gen-verify-sub1-notebook.py L228-232 |
| 2 | **Beta 贪心全矩阵方差评分稀释单区域贡献**：`np.var(tmp/caps)` 覆盖 6×30 全矩阵，任务只改本区域行，大容量区域（D/E/F）均衡性对评分不敏感；属合理设计（与 cell 7 评估口径一致），非 bug | Minor（可接受） | 可选优化：改为仅本区域行方差；不阻塞，维持现状即可 | gen-verify-sub1-notebook.py L273 |

## 验证器排除项（false_positive）

| 候选 | 排除理由 |
|------|---------|
| I3 retrain 从 h=168 起 | 滞后特征 lag168 预热的必要工程截断，注释充分；区间上限 2375 与协议一致，无泄漏（2/2 排除） |
| I4 空窗口强制 w=[lo] | 实测 free 任务 latest 全部=2406、arrive+dur 最大 2405.15<2406，窗口永非空，死代码防御（2/2 排除） |

## 代理值状态

| @PROXY | 状态 |
|--------|------|
| 利用率均衡逐时极差口径 | ✅ 已实现（MILP 目标 min(Umax−Umin)，fun=0.6283） |
| 简单基线预测协议 | ✅ 已实现（preprocess 切分 train/val/retrain/test） |
| HiGHS 求解器 | ✅ 已确认（scipy 1.17.1，status=0 最优收敛） |
| GPU 容量重叠折算 | ✅ 已实现（a_{i,h,t} 系数） |
| 甘特图聚合绘制 | ⏳ 待阶段 2.1 |

## 结论

- 无 Critical 问题；1 项 Major（建议修复：补 status/gap 标注）；1 项 Minor（可接受）
- 审查通过，可进入阶段 2.0（模型构建）
