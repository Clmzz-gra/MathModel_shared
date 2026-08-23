# S1 代理值替换清单（proxy-replacement-checklist-sub1）

> 阶段 1.3 | 2026-08-07 | 分支：experiment/sub1
> 规则：方案中以临时替代值顶替正式指标/数值的，登记 `@PROXY`，阶段 2.1 核销。

| @PROXY 代理值 | 正式实现方式 | 替换状态 |
|---------------|-------------|---------|
| @PROXY: 利用率均衡用逐时极差（max−min） | 正式口径：零迁移下区域平均利用率恒定，逐时极差即削峰量；论文需定义"逐时利用率"口径 | ✅ 已核销（MILP 目标 min(Umax−Umin)，Alpha 极差=0.6283；B 类确认极差口径 Alpha 胜） |
| @PROXY: 简单基线预测（均值/Last-Hour/季节朴素 + 线性回归） | 正式：赛题协议三段式（0–2351 训练 / 2352–2375 调参 / 0–2375 重训 / 2376–2399 测试）+ 白噪声证明（ACF + 泊松拟合） | ✅ 已核销（诚实口径：线性回归 RMSE=215.4 最优、常数均值 221.0；白噪声证明 + 简单基线为最终口径；**189.3 系测试窗事后均值含泄漏，已弃用**） |
| @PROXY: HiGHS (scipy.milp) 求解 | 正式：论文需注明求解器与版本（scipy 1.17.1 HiGHS）；评审环境用 PuLP+CBC 复现需验证解一致性 | ✅ 已核销（scipy.milp status=0 最优收敛；PuLP+CBC 复现留作阶段 2.1 验证项） |
| @PROXY: GPU 容量约束按精确小数重叠折算 | 正式：a_{i,h,t} 系数矩阵（与赛题 AI_IT_Load 重叠逻辑一致） | ✅ 已建模（outputs/notebooks/verify-sub1.ipynb；旧版 outputs/scratch/archive/verify-sub1-b.py） |
| @PROXY: 甘特图聚合绘制（按类型/区域分面） | 正式：538 任务全量甘特图 + 6 区域利用率子图（阶段 2.1.5 图表门禁审查） | ✅ 已核销（sub1-model.py 输出 outputs/figures/sub1-gantt-last24h.pdf + sub1-utilization.pdf） |
