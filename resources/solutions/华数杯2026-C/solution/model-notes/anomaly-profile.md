# C 题 异常画像（Isolation Forest）

> 由 outputs/scratch/data-profiling-04.py 生成，2026-08-07

## Isolation Forest 结果
- 异常任务数: 2500（5.00%）

## 异常任务典型偏离（前 20 条，超出均值 ±2σ 的特征）
```
任务 26 (AITraining/RegionB): GPU_Demand(+2.5σ), SR_RegionB(+2.1σ), Duration_min(+1.7σ), HourOfDay(-1.7σ)
任务 45 (RealTimeInference/RegionF): SR_RegionF(+2.5σ), LatestFinishHour(-1.5σ), TT_RealTimeInference(+1.4σ), HourOfDay(-1.4σ)
任务 61 (RealTimeInference/RegionB): LatestFinishHour(-2.6σ), SR_RegionB(+2.1σ), HourOfDay(-1.7σ), ArrivalHour(-1.5σ)
任务 65 (AITraining/RegionA): GPU_Demand(+2.2σ), SR_RegionA(+2.0σ), TT_AITraining(+1.4σ), ArrivalHour(-1.4σ)
任务 91 (RealTimeInference/RegionF): SR_RegionF(+2.5σ), HourOfDay(+1.5σ), TT_RealTimeInference(+1.4σ), Duration_min(+1.3σ)
任务 124 (AITraining/RegionA): SR_RegionA(+2.0σ), TT_AITraining(+1.4σ), ArrivalHour(-1.4σ), Duration_min(-1.3σ)
任务 150 (RealTimeInference/RegionF): SR_RegionF(+2.5σ), ArrivalHour(+1.7σ), TT_RealTimeInference(+1.4σ), Duration_min(-1.3σ)
任务 151 (AITraining/RegionC): GPU_Demand(+2.6σ), SR_RegionC(+2.4σ), ArrivalHour(+1.5σ), TT_AITraining(+1.4σ)
任务 156 (RealTimeInference/RegionD): LatestFinishHour(-2.5σ), SR_RegionD(+2.1σ), TT_RealTimeInference(+1.4σ), ArrivalHour(-1.4σ)
任务 164 (AITraining/RegionC): SR_RegionC(+2.4σ), Duration_min(-1.7σ), HourOfDay(+1.7σ), TT_AITraining(+1.4σ)
任务 183 (AITraining/RegionA): SR_RegionA(+2.0σ), Duration_min(+1.7σ), TT_AITraining(+1.4σ), ArrivalHour(+1.4σ)
任务 189 (RealTimeInference/RegionE): SR_RegionE(+2.5σ), TT_RealTimeInference(+1.4σ), Duration_min(-1.3σ), LatestFinishHour(-1.3σ)
任务 202 (RealTimeInference/RegionD): LatestFinishHour(-2.4σ), SR_RegionD(+2.1σ), TT_RealTimeInference(+1.4σ), ArrivalHour(-1.3σ)
任务 235 (RealTimeInference/RegionE): SR_RegionE(+2.5σ), ArrivalHour(+1.7σ), Duration_min(-1.6σ), TT_RealTimeInference(+1.4σ)
任务 269 (AITraining/RegionC): SR_RegionC(+2.4σ), GPU_Demand(+2.0σ), TT_AITraining(+1.4σ), MaxLatency_ms(+1.3σ)
任务 307 (RealTimeInference/RegionD): SR_RegionD(+2.1σ), TT_RealTimeInference(+1.4σ), Duration_min(+1.4σ), MaxLatency_ms(-1.2σ)
任务 314 (RealTimeInference/RegionF): SR_RegionF(+2.5σ), TT_RealTimeInference(+1.4σ), Duration_min(+1.3σ), MaxLatency_ms(-1.2σ)
任务 328 (AITraining/RegionC): SR_RegionC(+2.4σ), TT_AITraining(+1.4σ), MaxLatency_ms(+1.3σ), HourOfDay(-1.1σ)
任务 374 (AITraining/RegionC): SR_RegionC(+2.4σ), GPU_Demand(+1.7σ), TT_AITraining(+1.4σ), HourOfDay(-1.4σ)
任务 393 (AITraining/RegionA): GPU_Demand(+2.5σ), SR_RegionA(+2.0σ), TT_AITraining(+1.4σ), Duration_min(-1.4σ)
```

## 异常任务与聚类交叉（异常在各簇占比）
```
0    0.014
1    0.034
2    0.565
3    0.387
```

## 结论
- 若异常任务集中于某簇/某类型 → 该群体可独立处理或独立建模
