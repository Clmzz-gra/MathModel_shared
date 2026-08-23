# S3 数据预处理设计（阶段 1.4，LP 输入面板）

> 2026-08-08 | 分支：experiment/sub3 | 设计：建模智能体 | 实现：代码智能体（见 `handoff-sub3-preprocess-agent.md`）
> 依据：`solution/model-notes/approach-sub3-confirmed.md`（方案确认书 §2 假设 / §3 数学框架）
> 数据源（只读缓存，禁止重解析原始 xlsx）：`outputs/data/csv/region_time_data/region_time_data.csv`、`outputs/data/csv/storage_information/storage_information.csv`、`outputs/data/csv/GPU_information/GPU中心基础情况.csv`

## 1. 目标

构造储能协同优化 LP 的统一输入面板 `outputs/data/s3-preprocessed.pkl`：**6 区域 × 0–2405h 全时域**的逐时电价 / 卖电价 / 碳强度 / 设施负荷 / 可用新能源 + 储能参数 + 区域购售电边界 + 碳基准（ε 档分母），并附带口径自检结果。

> 时域口径（确认书 A9）：0–2399 主时域 + 2400–2405 结清段（均入模）；2406 仅状态结算（E_2406 = E_2405），不建变量、不需单独行。

## 2. 预处理操作清单

| # | 操作 | 对应模型需求 | 参数选择理由 |
|---|------|-------------|-------------|
| 1 | 读取三份 csv（仅读 `outputs/data/csv/`） | 输入唯一来源 | 阶段 0.2 缓存产物，禁重解析 xlsx |
| 2 | 列选择与重命名（中文→英文映射，见 §4） | LP 变量/系数索引 | 统一命名防跨智能体混淆 |
| 3 | 时域裁剪：保留 Hour 0–2405 全时域长表 | 全时域建模（A9） | 结清段纳入结算，不做滑窗 |
| 4 | 负荷核验：`Total_Load_MW == (Baseline_AI_IT_Load + NonAI_IT_Load) × PUE`（全时域逐行） | Load 输入正确性 | 数据接口核验（A10，RegionA PUE=1.35 实测精确） |
| 5 | 缺失/NaN 检查；`RegionE Hour0` SOC 异常记录（不修改） | 输入完整性 | F3 数据异常仅记录（A6） |
| 6 | 碳基准计算：`CarbonBase_r = Σ(GridPurchase×CarbonIntensity)`（主时域 0–2399）并与数据列 `CarbonEmission` 及 F2 值核对 | ε 档约束分母 | 口径自洽（F2 完全一致） |
| 7 | 储能参数与边界提取；核对 `SellLimit_MW` vs `MaxGridExport_MW` 关系 | 约束系数 | F3/F8：A/B/C SellLimit=0，D/E/F 卖电 2400h 全覆盖 |
| 8 | 输出 pkl + 控制台自检打印（含 §5 基准核对） | 可复现 | 与 verify-sub3 口径一致 |

## 3. 输出结构（s3-preprocessed.pkl）

```python
{
  "meta": {
    "generated": "2026-08-08",
    "source": "outputs/data/csv/**（三文件）",
    "hours": [0..2405],                 # 全时域（2406 状态结算 = E_2405，不建变量）
    "units": {"power": "MW", "energy": "MWh", "price": "CNY/MWh",
              "carbon": "tCO2/MWh", "cost": "CNY", "carbon_total": "tCO2"},
    "pue": {"RegionA": 1.35, ...},      # 记录用（Load 已含 PUE 折算，LP 不再乘）
  },
  "panel": pd.DataFrame,                # 长表，index=(Region, Hour)，列为：
    #   Hour, Region, Price_CNY_per_MWh, SellPrice_CNY_per_MWh,
    #   CarbonIntensity_tCO2_per_MWh, Total_Load_MW, AvailableRenewable_MW,
    #   UsedRenewable_MW(参考), GridPurchase_base_MW(参考), NetGridImport_base_MW(参考)
  "storage": {                          # 每区域储能参数
    "RegionA": {"Capacity_MWh": ..., "MinSOC_MWh": ..., "InitialSOC_MWh": ...,
                "MaxChargePower_MW": ..., "MaxDischargePower_MW": ...,
                "ChargeEfficiency": ..., "DischargeEfficiency": ...,
                "SellLimit_MW": ..., "MaxGridImport_MW": ..., "MaxGridExport_MW": ...},
    ...
  },
  "carbon_base_kt": {"RegionA": 581.27, "RegionB": 519.70, "RegionC": 483.82,
                     "RegionD": 262.96, "RegionE": 84.56,  "RegionF": 113.06},
  "epsilon": [0.90, 0.95, 1.00],        # ε 档位（2.0 可按 Pareto 可行性微调）
  "check": {                            # 预处理自检结果
    "load_equals_IT_times_PUE": bool,
    "power_balance_resid_max_MW": float,     # 全时域 F5 复算
    "carbon_base_matches_F2": bool,
    "soc_recur_resid_max": float,
    "regionE_hour0_note": "...",
  }
}
```

## 4. 列名映射（中文指标 ↔ 代码变量）

| 数据列（csv） | 面板列 | 说明 |
|--------------|--------|------|
| Hour | Hour | 0–2405 |
| Region | Region | RegionA–F |
| ElectricityPrice_CNY_per_MWh | Price_CNY_per_MWh | 购电电价 |
| SellPrice_CNY_per_MWh | SellPrice_CNY_per_MWh | 卖电电价（A/B/C=0） |
| CarbonIntensity_tCO2_per_MWh | CarbonIntensity_tCO2_per_MWh | 购电碳强度 |
| Total_Load_MW | Total_Load_MW | 设施负荷（=IT×PUE，LP 输入） |
| AvailableRenewable_MW | AvailableRenewable_MW | 可用新能源（消纳上限） |
| UsedRenewable_MW | UsedRenewable_MW | 基准直消纳（对照参考） |
| GridPurchase_MW | GridPurchase_base_MW | 基准购电（对照参考） |
| NetGridImport_MW | NetGridImport_base_MW | 基准净购电（对照参考） |

## 5. 验证基准（代码实现须复现，偏差 >0.5% 暂停回查）

| 项 | 基准值 | 来源 |
|----|--------|------|
| 碳基准 6 区域（kt） | A 581.27 / B 519.70 / C 483.82 / D 262.96 / E 84.56 / F 113.06 | F2 |
| 合计碳排（主时域） | 2045.36 kt | F2 |
| 基准成本（主时域） | 1802.34 M 元 | F2 |
| 峰值净购电 | 497.0 MW（RegionA） | F2 |
| 功率平衡残差 max | ≤0.0002 MW | F5 |
| Init/容量 | 45.0%（E 45.1%）、Min/容量 10.0% | F3 |
| 卖电上限 | A/B/C=0；D/E/F=SellLimit_MW | F8 |
| Load = IT×PUE | 全时域精确 | A10 核验 |

## 6. 交付物与下游接口

- 代码智能体交付：`outputs/scratch/preprocess-sub3.py` + `outputs/data/s3-preprocessed.pkl`（exit 0，2026-08-08）
- 下游：阶段 2.1 `outputs/scratch/sub3-model.py`（LP 求解）直接读取本 pkl；阶段 2.0 数学推导据此定系数符号
- 代理值关联：@PROXY ε 档位（2.0 核销）；@PROXY RegionE Hour0（已核销，见 §7 裁定 2）

## 7. 建模裁定（2026-08-08，代码智能体回报 2 项）

**裁定 1 — 负荷核验容差：放宽至 1e-3 MW → `load_equals_IT_times_PUE=True`**
- 依据：1066 行超 1e-6、max|Δ|=5.04e-05 MW、相对 ~1e-7，纯为 Total_Load 4 位小数存储的 ×PUE 舍入，非结构性偏差（IT_Load==Baseline+NonAI 精确至 3e-7）
- 核验语义为结构正确性（折算关系成立），非数值精度；放宽后全过，对 LP 无影响
- 落实：代码智能体将 `check_load` 阈值由 1e-6 改为 1e-3 并重跑

**裁定 2 — SOC 递推口径：check 取链式口径；@PROXY RegionE Hour0 核销；建模无改动**
- 链式口径（E(−1)=InitialSOC 全程递推）：非 E 区 max 0.0005–0.0045 MWh = 2406 步舍入累积（自洽）；RegionE 1.0039 由 Hour0 异常 0.9999 主导
- F3 逐时口径（前值时点锚定）：非 E 区 0.0001 MWh 不变 → 论文采用逐时口径说明数据自洽，双口径并列打印保留
- 建模结论维持 A6：以赛题递推式为约束，数据异常仅记录不修改

**附带确认**：SellLimit == MaxGridExport 全区域一致（A/B/C=0、D=180、E/F=220 MW），卖电边界无待确认项。
