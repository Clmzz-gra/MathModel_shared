# C 题 降维画像（PCA）

> 由 outputs/scratch/data-profiling-04.py 生成，2026-08-07

## workload_trace PCA
- 保留 PC 数（cum≥60%）: 5，前 5 累计解释率 [0.283, 0.4065, 0.4891, 0.5693, 0.6489]

### 载荷矩阵（前 5 PC）
```
                        PC1    PC2    PC3    PC4    PC5
GPU_Demand            0.395 -0.278 -0.005  0.007 -0.009
Duration_min          0.001  0.001  0.017 -0.004  0.017
MaxLatency_ms         0.473 -0.038 -0.004 -0.002 -0.014
LatestFinishHour      0.366  0.377  0.037 -0.056  0.019
ArrivalHour           0.041  0.139  0.082 -0.103  0.072
HourOfDay            -0.005  0.021  0.045  0.030  0.032
TT_AITraining         0.410 -0.361 -0.006  0.009 -0.008
TT_BatchInference    -0.000  0.708  0.005 -0.025 -0.009
TT_RealTimeInference -0.409 -0.348  0.000  0.015  0.017
SR_RegionA           -0.175 -0.009  0.730 -0.012 -0.299
SR_RegionB           -0.151  0.044 -0.665  0.039 -0.451
SR_RegionC           -0.130  0.001 -0.111 -0.049  0.834
SR_RegionD            0.192  0.035  0.035  0.775  0.045
SR_RegionE            0.140 -0.031 -0.011 -0.474 -0.019
SR_RegionF            0.151 -0.049 -0.017 -0.394 -0.046
```

### PC 含义解读（|载荷|>0.5）

- PC1（解释 28.3%）：无强载荷
- PC2（解释 12.3%）：TT_BatchInference
- PC3（解释 8.3%）：SR_RegionA, SR_RegionB
- PC4（解释 8.0%）：SR_RegionD
- PC5（解释 8.0%）：SR_RegionC

## region_time_data PCA（21 特征）

- 前 5 主成分解释率: [0.4148, 0.2514, 0.1583, 0.0734, 0.0293]，累计 92.7%

### region 载荷矩阵（前 5 PC）
```
                                PC1    PC2    PC3    PC4    PC5
ElectricityPrice_CNY_per_MWh -0.178 -0.308 -0.037  0.084 -0.144
SellPrice_CNY_per_MWh         0.311 -0.035 -0.099 -0.063  0.209
CarbonIntensity_tCO2_per_MWh -0.301 -0.064  0.107  0.203  0.142
AITrainingPower_MW            0.275  0.086 -0.018  0.421  0.053
GPU_Utilization_Percent       0.192  0.060 -0.006  0.588  0.018
AvailableRenewable_MW        -0.051  0.186 -0.445  0.029 -0.289
UsedRenewable_MW              0.251  0.226 -0.090 -0.210 -0.025
RenewableCharge_MW            0.068  0.170  0.464 -0.020 -0.144
Curtailment_MW               -0.249  0.018 -0.319  0.140 -0.207
IT_Load_MW                   -0.030  0.411 -0.099 -0.018  0.296
Total_Load_MW                -0.084  0.398 -0.094  0.005  0.313
GridPurchase_MW              -0.266  0.244  0.043  0.159  0.021
GridCharge_MW                 0.091  0.226  0.362 -0.047 -0.171
GridSell_MW                   0.316  0.081 -0.111 -0.114  0.018
NetGridImport_MW             -0.315  0.121  0.078  0.154  0.005
CarbonEmission_tCO2          -0.305  0.115  0.063  0.179  0.172
SOC_MWh                       0.190  0.213 -0.266 -0.036 -0.273
ChargePower_MW                0.083  0.205  0.449 -0.033 -0.165
DischargePower_MW             0.133 -0.300  0.010 -0.043  0.594
Baseline_AI_IT_Load_MW        0.268  0.083 -0.016  0.451  0.055
NonAI_IT_Load_MW             -0.159  0.342 -0.085 -0.237  0.249
```

### region PC 含义解读（|载荷|>0.4）

- PC1（解释 41.5%）：无强载荷
- PC2（解释 25.1%）：IT_Load_MW
- PC3（解释 15.8%）：AvailableRenewable_MW, RenewableCharge_MW, ChargePower_MW
- PC4（解释 7.3%）：AITrainingPower_MW, GPU_Utilization_Percent, Baseline_AI_IT_Load_MW
- PC5（解释 2.9%）：DischargePower_MW