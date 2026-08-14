# S1 Alpha MILP 性能剖析（perf-alpha-milp）

> 2026-08-08 | 分支：experiment/sub1 | 脚本：`outputs/scratch/profile-alpha-milp.py`
> 用途：供阶段 2（模型构建/代码实现）参考——求解瓶颈定位与缓存复用策略

## 1. 472s 时间构成

| 环节 | 耗时 | 占比 | 说明 |
|------|------|------|------|
| 候选窗构建（5768 变量索引） | 0.002s | ~0% | Python 侧 |
| 开工约束 Σx=1（378 条） | 0.008s | ~0% | |
| 容量+U 约束矩阵（180×3 组，21368 非零元） | 0.074s | ~0% | |
| **HiGHS 分支定界求解** | **~472s** | **99.98%** | 唯一瓶颈 |

**结论**：瓶颈 100% 在求解器（HiGHS 证明最优性），Python 建模侧开销可忽略（合计 0.08s）。任何"提速"必须作用于求解器配置或问题规模，而非 Python 代码。

## 2. 求解收敛曲线（探测值，未覆盖正式缓存）

| time_limit | 实际耗时 | 目标值（极差） | 相对最优差距 |
|-----------|---------|--------------|-------------|
| 5s | 5.0s | 0.6595 | +5.0% |
| 30s | 30.1s | 0.6392 | +1.7% |
| 60s | 60.0s | 0.6373 | +1.4% |
| 472s（正式，status=0） | 472.4s | **0.6283** | 0%（最优证明） |

**洞察**：
- 目标值改善集中在前 30s（0.6595→0.6392，提升 3.1%）
- 之后 ~440s 用于分支定界**证明最优性**（gap→0），目标仅再降 1.7%
- 即：若可接受"近似最优"，30s 已得 gap≈1.7% 的解

## 3. 阶段 2 缓存复用策略

### 3.1 现有缓存（可直接复用，0 重建）

| 缓存 | 内容 | 阶段 2 用途 |
|------|------|------------|
| `outputs/data/cache/s1_alpha_milp.pkl` | Alpha 最优解（0.6283, 378/378） | **直接引用调度结果，无需重解** |
| `outputs/data/cache/s1_test_tasks.pkl` | 测试窗任务/实时 base | 甘特图/利用率图绘制输入 |
| `outputs/data/cache/s1_series_acf.pkl` | 逐时序列 + ACF | 预测段复用 |
| `outputs/data/s1-preprocessed.pkl` | 预处理全量 | 2.1 模型代码加载源 |
| `outputs/data/s1-schedule-test.pkl` | alpha/beta 开工表 | 验证基准 |

### 3.2 若阶段 2 需重解（改参数/目标），三个加速选项

1. **热启动（推荐）**：注入 `s1_alpha_milp.pkl` 的解作为 HiGHS MIP start → 大幅缩短证明时间（HiGHS 支持通过 `options` 传递初始解或 LP 文件 warm start）
2. **降 gap 早停**：`mip_rel_gap` 0.01→0.02 → 约 30s 出解，差距 <2%；论文注明"近似最优（gap≤2%）"
3. **按区域分解**：容量约束按区域解耦（无跨区域耦合），拆 6 个独立子 MILP 并行 → 总耗时 ≈ 最慢区域；注意跨区域任务时延约束需额外处理（S1 零迁移下天然成立）

### 3.3 不建议的路径

- **GPU 加速**：瓶颈是 HiGHS（纯 CPU 求解器），GPU 无效；数组规模太小，CuPy 传输开销 > 收益
- **Python 侧优化**：建模已 0.08s，优化无意义

## 4. 求解器配置参考

```
scipy.optimize.milp(c, constraints, integrality, bounds,
    options={'time_limit': 1800, 'mip_rel_gap': 0.01})
```

- status=0 最优；status=1 时间/迭代限制内的可行次优（**必须检查并标注近似口径**，阶段 1.5 Major 修复项）
- 正式结果：status=0，fun=0.6283，dt=472.4s

## 5. 关联文件

- 剖析脚本：`outputs/scratch/profile-alpha-milp.py`（只读测量，不写缓存）
- 求解实现：`outputs/notebooks/verify-sub1.ipynb` cell 5（正式缓存源）
- 方案确认书：`solution/model-notes/approach-sub1-confirmed.md` §4/§7
