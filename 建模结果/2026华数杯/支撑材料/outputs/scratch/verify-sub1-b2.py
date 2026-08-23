# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
B 类验证：白噪声结论的对抗性检验（阶段 1.2 head-to-head）
=============================================================
对赛题协议测试窗 2376-2399 做三组对抗检验，检验"GPU 需求序列近乎不可预测"
结论是否可被更复杂模型推翻：

  [T1] BDS 非线性依赖检验 —— ACF≈0 只排除线性自相关；BDS 检验排除任意
       (m 维) 依赖，若拒绝 i.i.d. 则序列存在可被非线性模型利用的结构。
  [T2] Prophet head-to-head —— 与常数均值/线性回归基线同测试窗对比。
       Prophet 内置趋势+日/周/年周期傅里叶分解，若真存在周期结构应显著占优。
  [T3] LSTM head-to-head —— torch 单层 LSTM 滚动预测，若存在非线性时序依赖
       应显著占优。
  [T4] 跨类型/跨区域 Granger 因果 —— 若其他序列能格兰杰原因解释目标序列，
       则存在可利用的横截面信息（外生变量路径）。

口径（与 verify-sub1-20260807.md 及 notebook 一致，诚实无泄漏）：
  - 训练 0-2351 / 调参 2352-2375 / 重训 0-2375 / 测试 2376-2399
  - 测试窗预测只允许使用 0-2375 的信息
  - RMSE 为最终判据；提升 <5% 视为"无实质改善"

输入：outputs/data/c-data-cleaned.pkl
输出：控制台结果表（无文件写入）
"""
import pickle
import warnings
import logging
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# ============================================================
# 0. 数据加载与序列构建（与 notebook cell 3 逐位一致）
# ============================================================
BASE = "outputs/data"
with open(f"{BASE}/c-data-cleaned.pkl", "rb") as f:
    d = pickle.load(f)
wt = d["workload_trace"]
H = 2400
TYPES = ["AITraining", "BatchInference", "RealTimeInference"]
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]

series = {}
for t in TYPES:
    sub = wt[wt["TaskType"] == t]
    ts = np.zeros(H)
    for h, g in zip(sub["ArrivalHour"], sub["GPU_Demand"]):
        if h < H:
            ts[h] += g
    series[t] = ts
series["Total"] = series["AITraining"] + series["BatchInference"] + series["RealTimeInference"]

# 区域序列（用于 T4）
region_series = {r: np.zeros(H) for r in REGIONS}
for _, row in wt.iterrows():
    h = int(row["ArrivalHour"])
    if h < H:
        region_series[row["SourceRegion"]][h] += row["GPU_Demand"]

# 赛题协议分窗
TR = np.arange(0, 2352)      # 训练 0-2351
VA = np.arange(2352, 2376)   # 调参 2352-2375
TE = np.arange(2376, 2400)   # 测试 2376-2399
y_test = series["Total"][TE]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def mape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs((a - b) / np.maximum(a, 1e-6))) * 100)


# 基线（诚实口径，均只使用 0-2375 信息）
pred_mean = np.full(24, series["Total"][:2376].mean())
pred_lin = None  # 由下面线性回归填充


def lin_reg_pred():
    from numpy import sin, cos, pi

    def feats(h):
        return np.array([1, h, sin(2 * pi * h / 24), cos(2 * pi * h / 24),
                         sin(2 * pi * h / 168), cos(2 * pi * h / 168)])

    X = np.array([feats(h) for h in TR])
    y = series["Total"][TR]
    b = np.linalg.solve(X.T @ X + 1e-6 * np.eye(6), X.T @ y)
    return np.array([feats(h) @ b for h in TE])


pred_lin = lin_reg_pred()
baselines = {"常数均值": pred_mean, "线性回归": pred_lin}
print("=" * 72)
print("T0  基线（参考，测试窗 2376-2399）")
print("=" * 72)
for name, p in baselines.items():
    print(f"  {name:<10} RMSE={rmse(y_test, p):7.1f}  MAPE={mape(y_test, p):5.1f}%")

# ============================================================
# T1  BDS 非线性依赖检验
# ============================================================
print("\n" + "=" * 72)
print("T1  BDS 检验（H0: 序列 i.i.d.；拒绝则存在任意维度依赖）")
print("=" * 72)
from statsmodels.tsa.stattools import bds

print("  序列            m=2          m=3          m=4         结论")
for name in ["Total"] + TYPES:
    x = series[name]
    row = []
    for m in [2, 3, 4]:
        try:
            # bds 返回 (max_dim-1,) 数组：各元素对应嵌入维度 2..max_dim 的 p 值
            _, p = bds(x, max_dim=m, epsilon=1.5 * x.std())
            p = float(np.asarray(p).ravel()[-1])  # 取嵌入维度=m 的 p 值
            row.append(p)
        except Exception as e:  # 个别 m 可能数值异常
            row.append(np.nan)
    n_sig = sum(1 for p in row if p < 0.05)
    concl = "存在依赖" if n_sig >= 2 else "≈i.i.d."
    ps = "  ".join(f"{p:8.4f}" if not np.isnan(p) else "     NaN" for p in row)
    print(f"  {name:<12} {ps}   {concl}")

# ============================================================
# T2  Prophet head-to-head
# ============================================================
print("\n" + "=" * 72)
print("T2  Prophet vs 基线（训练 0-2351 / 调参 2352-2375 / 重训 0-2375 / 测试 2376-2399）")
print("=" * 72)
from prophet import Prophet


def prophet_fit_predict(train_h, pred_h):
    """train_h: 训练用小时数组; pred_h: 需预测的小时数组（须在 train_h 之后）"""
    df = pd.DataFrame({"ds": pd.to_datetime(train_h, unit="h", origin="unix"),
                       "y": series["Total"][train_h]})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                yearly_seasonality=False, changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0)
    m.fit(df)
    fut = pd.DataFrame({"ds": pd.to_datetime(pred_h, unit="h", origin="unix")})
    return m.predict(fut)["yhat"].values


# 调参窗验证（可选：比较默认 vs 大季节先验，此处仅报一次结果）
t0 = time.time()
pred_val = prophet_fit_predict(TR, VA)
print(f"  调参窗 2352-2375: RMSE={rmse(series['Total'][VA], pred_val):7.1f}  "
      f"MAPE={mape(series['Total'][VA], pred_val):5.1f}%")

t0 = time.time()
pred_prophet = prophet_fit_predict(np.arange(0, 2376), TE)
print(f"  测试窗 2376-2399: RMSE={rmse(y_test, pred_prophet):7.1f}  "
      f"MAPE={mape(y_test, pred_prophet):5.1f}%  (耗时 {time.time()-t0:.1f}s)")
imp = (rmse(y_test, pred_mean) - rmse(y_test, pred_prophet)) / rmse(y_test, pred_mean) * 100
print(f"  vs 常数均值: {imp:+.1f}%  → {'有实质改善(≥5%)' if imp >= 5 else '无实质改善(<5%)'}")

# ============================================================
# T3  LSTM head-to-head（torch CPU）
# ============================================================
print("\n" + "=" * 72)
print("T3  LSTM vs 基线（滚动一步预测，输入=滞后24h窗口）")
print("=" * 72)
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

LAG = 24


def make_xy(hours, x_series, lag=LAG):
    """滚动监督样本：输入 [t-lag, t-1]，输出 t"""
    X, y = [], []
    for t in hours:
        if t >= lag:
            X.append(x_series[t - lag:t])
            y.append(x_series[t])
    return np.array(X, float), np.array(y, float)


class LSTMModel(nn.Module):
    def __init__(self, lag, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :]).squeeze(-1)


X_tr, y_tr = make_xy(TR, series["Total"])
X_va, y_va = make_xy(VA, series["Total"])
X_te, y_te = make_xy(TE, series["Total"])

# 归一化（只用训练窗统计量，防泄漏）
mu, sd = y_tr.mean(), y_tr.std()
X_tr_n, y_tr_n = (X_tr - mu) / sd, (y_tr - mu) / sd
X_va_n, y_va_n = (X_va - mu) / sd, (y_va - mu) / sd
X_te_n, y_te_n = (X_te - mu) / sd, (y_te - mu) / sd

X_tr_t = torch.tensor(X_tr_n, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_n, dtype=torch.float32)
X_va_t = torch.tensor(X_va_n, dtype=torch.float32)
X_te_t = torch.tensor(X_te_n, dtype=torch.float32)

model = LSTMModel(LAG, hidden=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()

best_val = np.inf
best_state = None
epochs = 60
for ep in range(epochs):
    model.train()
    opt.zero_grad()
    loss = lossf(model(X_tr_t), y_tr_t)
    loss.backward()
    opt.step()
    with torch.no_grad():
        model.eval()
        vp = model(X_va_t).numpy() * sd + mu
        vrmse = rmse(y_va, vp)
    if vrmse < best_val:
        best_val = vrmse
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
with torch.no_grad():
    model.eval()
    pred_lstm = model(X_te_t).numpy() * sd + mu
print(f"  验证窗最优 epoch 处 RMSE={best_val:7.1f}")
print(f"  测试窗 2376-2399: RMSE={rmse(y_test, pred_lstm):7.1f}  "
      f"MAPE={mape(y_test, pred_lstm):5.1f}%")
imp = (rmse(y_test, pred_mean) - rmse(y_test, pred_lstm)) / rmse(y_test, pred_mean) * 100
print(f"  vs 常数均值: {imp:+.1f}%  → {'有实质改善(≥5%)' if imp >= 5 else '无实质改善(<5%)'}")

# ============================================================
# T4  跨类型 / 跨区域 Granger 因果
# ============================================================
print("\n" + "=" * 72)
print("T4  Granger 因果（跨类型：若显著，存在可利用的外生路径）")
print("=" * 72)
from statsmodels.tsa.stattools import grangercausalitytests

PAIRS = [("AITraining", "BatchInference"), ("AITraining", "RealTimeInference"),
         ("BatchInference", "AITraining"), ("BatchInference", "RealTimeInference"),
         ("RealTimeInference", "AITraining"), ("RealTimeInference", "BatchInference")]
print("  H0: X 不格兰杰原因 Y（p<0.05 拒绝）")
print(f"  {'X':<18}{'Y':<20}{'lag1_p':<10}{'lag2_p':<10}{'lag3_p':<10}结论")
for x, y in PAIRS:
    # grangercausalitytests 检验第 2 列 → 第 1 列；故第 2 列放声明的"原因 X"
    df = pd.DataFrame({"x": series[y], "y": series[x]})
    res = grangercausalitytests(df[["x", "y"]], maxlag=3, verbose=False)
    ps = [res[l][0]["ssr_ftest"][1] for l in [1, 2, 3]]
    n_sig = sum(1 for p in ps if p < 0.05)
    concl = "存在Granger因果" if n_sig >= 2 else "无"
    ps_s = "  ".join(f"{p:9.4f}" for p in ps)
    print(f"  {x:<18}{y:<20}{ps_s}   {concl}")

# 区域间：只做几组有代表性的（东部 A->B, 东部->西部 D, 光伏 E->F）
REG_PAIRS = [("RegionA", "RegionB"), ("RegionA", "RegionD"),
             ("RegionB", "RegionC"), ("RegionE", "RegionF")]
print("\n  区域间 Granger（代表性配对）:")
print(f"  {'X':<12}{'Y':<12}{'lag1_p':<10}{'lag2_p':<10}{'lag3_p':<10}结论")
for x, y in REG_PAIRS:
    # 同上：第 2 列放声明的"原因 X"
    df = pd.DataFrame({"x": region_series[y], "y": region_series[x]})
    res = grangercausalitytests(df[["x", "y"]], maxlag=3, verbose=False)
    ps = [res[l][0]["ssr_ftest"][1] for l in [1, 2, 3]]
    n_sig = sum(1 for p in ps if p < 0.05)
    concl = "存在Granger因果" if n_sig >= 2 else "无"
    ps_s = "  ".join(f"{p:9.4f}" for p in ps)
    print(f"  {x:<12}{y:<12}{ps_s}   {concl}")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 72)
print("汇总（测试窗 2376-2399）")
print("=" * 72)
rows = []
for name, p in baselines.items():
    rows.append((name, rmse(y_test, p), mape(y_test, p)))
rows.append(("Prophet", rmse(y_test, pred_prophet), mape(y_test, pred_prophet)))
rows.append(("LSTM", rmse(y_test, pred_lstm), mape(y_test, pred_lstm)))
print(f"  {'模型':<12}{'RMSE':<10}{'MAPE'}")
for name, r, m in sorted(rows, key=lambda z: z[1]):
    print(f"  {name:<12}{r:<10.1f}{m:.1f}%")
best = min(rows, key=lambda z: z[1])
print(f"\n  最优: {best[0]} (RMSE={best[1]:.1f})")
print("\n[T1] BDS → 依赖结构判定")
print("[T2/T3] Prophet/LSTM vs 均值 → 提升 <5% 则不可预测性结论成立")
print("[T4] Granger → 跨类型/跨区域无显著因果则外生路径亦不可用")
