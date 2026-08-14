# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 3.2 四重检验附录图 — 为"GPU 需求序列近乎不可预测"结论生成可放附录的
    4 张图：BDS p 值、Granger p 值热力图、Prophet/LSTM vs 基线预测对比、四模型 RMSE 对比。
    数值与 verify-sub1-b2.py 逐位一致（同一数据、同一口径）。

原理：
    1. 逐时 GPU 需求序列 Y_t = Σ_{A_j=t} g_j（0-2399）
    2. T1 BDS：相关积分 C_m(ε)、统计量 W_m,T（statsmodels.tsa.stattools.bds），
       ε=1.5σ，m=2/3/4，H0: i.i.d.
    3. T2 Prophet：日/周傅里叶季节（p=24/168），训练 0-2351 → 调参 2352-2375 →
       重训 0-2375 → 测试 2376-2399
    4. T3 LSTM：单层 16 隐单元，滞后 24h 滚动一步，调参窗早停（torch CPU）
    5. T4 Granger：VAR(3) F 检验，跨类型 6 组 + 跨区域 4 组（第 2 列为原因）
    6. 汇总：四模型测试窗 RMSE/MAPE 对比

输入数据：
    - outputs/data/c-data-cleaned.pkl（workload_trace / GPU_information）

输出（先算后画，PR-014）：
    - outputs/figures/sub1-whitenoise-bds-v1.pdf — BDS p 值（4 序列 × m=2/3/4，标 0.05 线）
    - outputs/figures/sub1-whitenoise-granger-v1.pdf — Granger p 值热力图（类型间 6 + 区域间 4）
    - outputs/figures/sub1-whitenoise-forecast-v1.pdf — 测试窗实际 vs 基线/Prophet/LSTM（标 RMSE）
    - outputs/figures/sub1-whitenoise-rmse-v1.pdf — 四模型 RMSE 对比（标相对均值提升）
    - 副本同步 solution/artifacts/charts/ + manifest 登记

对应论文章节：
    问题一（S1）预测段 — 白噪声四重对抗检验（附录佐证图）
"""
import pickle
import logging
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 中文字体与负号（chart-generator 强制前置）===
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Source Han Sans CN"]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

BASE = "outputs/data"
FIGS = "outputs/figures"
TYPES = ["AITraining", "BatchInference", "RealTimeInference"]
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
COLORS = ["#333333", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
SIG = 0.05

# ============================================================
# 第一阶段：纯计算
# ============================================================
with open(f"{BASE}/c-data-cleaned.pkl", "rb") as f:
    d = pickle.load(f)
wt = d["workload_trace"]
H = 2400

series = {}
for t in TYPES:
    sub = wt[wt["TaskType"] == t]
    ts = np.zeros(H)
    for h, g in zip(sub["ArrivalHour"], sub["GPU_Demand"]):
        if h < H:
            ts[h] += g
    series[t] = ts
series["Total"] = series["AITraining"] + series["BatchInference"] + series["RealTimeInference"]

region_series = {r: np.zeros(H) for r in REGIONS}
for _, row in wt.iterrows():
    h = int(row["ArrivalHour"])
    if h < H:
        region_series[row["SourceRegion"]][h] += row["GPU_Demand"]

TR = np.arange(0, 2352)
VA = np.arange(2352, 2376)
TE = np.arange(2376, 2400)
y_test = series["Total"][TE]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def mape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs((a - b) / np.maximum(a, 1e-6))) * 100)


def stat(name, arr):
    a = np.asarray(arr, dtype=float)
    print(f"{name}: min={a.min():.3f}, max={a.max():.3f}, mean={a.mean():.3f}, std={a.std():.3f}")
    return a


# ---- 基线 ----
pred_mean = np.full(24, series["Total"][:2376].mean())


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

# ---- T1 BDS ----
from statsmodels.tsa.stattools import bds

bds_p = {}
for name in ["Total"] + TYPES:
    x = series[name]
    ps = []
    for m in [2, 3, 4]:
        _, p = bds(x, max_dim=m, epsilon=1.5 * x.std())
        ps.append(float(np.asarray(p).ravel()[-1]))
    bds_p[name] = ps
print("BDS p 值:")
for k, v in bds_p.items():
    print(f"  {k}: {[round(p, 4) for p in v]}")

# ---- T2 Prophet ----
from prophet import Prophet


def prophet_fit_predict(train_h, pred_h):
    df = pd.DataFrame({"ds": pd.to_datetime(train_h, unit="h", origin="unix"),
                       "y": series["Total"][train_h]})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                yearly_seasonality=False, changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0)
    m.fit(df)
    fut = pd.DataFrame({"ds": pd.to_datetime(pred_h, unit="h", origin="unix")})
    return m.predict(fut)["yhat"].values


pred_val = prophet_fit_predict(TR, VA)
pred_prophet = prophet_fit_predict(np.arange(0, 2376), TE)
print(f"Prophet 调参窗 RMSE={rmse(series['Total'][VA], pred_val):.1f} / "
      f"测试窗 RMSE={rmse(y_test, pred_prophet):.1f} MAPE={mape(y_test, pred_prophet):.1f}%")

# ---- T3 LSTM ----
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
LAG = 24


def make_xy(hours, x_series, lag=LAG):
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
best_val, best_state = np.inf, None
for ep in range(60):
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
print(f"LSTM 验证窗最优 RMSE={best_val:.1f} / 测试窗 RMSE={rmse(y_test, pred_lstm):.1f} "
      f"MAPE={mape(y_test, pred_lstm):.1f}%")

# ---- T4 Granger ----
from statsmodels.tsa.stattools import grangercausalitytests

TYPE_PAIRS = [("AITraining", "BatchInference"), ("AITraining", "RealTimeInference"),
              ("BatchInference", "AITraining"), ("BatchInference", "RealTimeInference"),
              ("RealTimeInference", "AITraining"), ("RealTimeInference", "BatchInference")]
REG_PAIRS = [("RegionA", "RegionB"), ("RegionA", "RegionD"),
             ("RegionB", "RegionC"), ("RegionE", "RegionF")]


def granger_pvalues(x_series, y_series, pairs):
    out = {}
    for x, y in pairs:
        df = pd.DataFrame({"x": x_series[y], "y": x_series[x]})  # 第 2 列为原因 X
        res = grangercausalitytests(df[["x", "y"]], maxlag=3, verbose=False)
        out[(x, y)] = [res[l][0]["ssr_ftest"][1] for l in [1, 2, 3]]
    return out


g_type = granger_pvalues(series, series, TYPE_PAIRS)
g_reg = granger_pvalues(region_series, region_series, REG_PAIRS)
print("Granger 类型间 p 值:")
for k, v in g_type.items():
    print(f"  {k}: {[round(p, 4) for p in v]}")
print("Granger 区域间 p 值:")
for k, v in g_reg.items():
    print(f"  {k}: {[round(p, 4) for p in v]}")

# ---- 汇总 ----
models = {
    "Prophet": (pred_prophet, rmse(y_test, pred_prophet), mape(y_test, pred_prophet)),
    "线性回归": (pred_lin, rmse(y_test, pred_lin), mape(y_test, pred_lin)),
    "常数均值": (pred_mean, rmse(y_test, pred_mean), mape(y_test, pred_mean)),
    "LSTM": (pred_lstm, rmse(y_test, pred_lstm), mape(y_test, pred_lstm)),
}
print("\n汇总（测试窗 2376-2399）:")
for name, (_, r, m) in sorted(models.items(), key=lambda z: z[1][1]):
    print(f"  {name}: RMSE={r:.1f} MAPE={m:.1f}% vs均值 {((models['常数均值'][1]-r)/models['常数均值'][1]*100):+.1f}%")
stat("y_test", y_test)

# ============================================================
# 第二阶段：绘图
# ============================================================
import os
os.makedirs(FIGS, exist_ok=True)
os.makedirs("solution/artifacts/charts", exist_ok=True)

# ---- 图1：BDS p 值 ----
fig, ax = plt.subplots(figsize=(8, 4.2))
seqs = ["Total"] + TYPES
xpos = np.arange(len(seqs))
width = 0.25
for j, m in enumerate([2, 3, 4]):
    vals = [bds_p[s][j] for s in seqs]
    ax.bar(xpos + (j - 1) * width, vals, width=width, color=COLORS[j + 1],
           label=f"嵌入维度 m={m}", alpha=0.85)
    # 柱顶 p 值注解已删：p 值信息由图注与 0.05 参考线传达（审查用户指令）
ax.axhline(SIG, color="#d62728", ls="--", lw=1.2)
ax.text(len(seqs) - 1 + width, SIG + 0.02, f"显著性水平 {SIG}", color="#d62728", ha="right", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(["Total", "AI 训练", "批量推理", "实时推理"])
ax.set_ylabel("p 值")
ax.set_ylim(0, 1.05)
ax.set_title("BDS 非线性依赖检验：全部 p ≥ 0.30，不拒绝 i.i.d.（白噪声）")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIGS}/sub1-whitenoise-bds-v1.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {FIGS}/sub1-whitenoise-bds-v1.pdf")

# ---- 图2：Granger p 值热力图 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (pairs, gd, title) in zip(
        axes, [(TYPE_PAIRS, g_type, "跨类型 Granger 因果（VAR(3)）"),
               (REG_PAIRS, g_reg, "跨区域 Granger 因果（VAR(3)）")]):
    labels = [f"{x}→{y}" for x, y in pairs]
    mat = np.array([gd[p] for p in pairs])
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=0.2, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="#333333" if v > 0.05 else "#ffffff")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["lag=1", "lag=2", "lag=3"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("滞后阶（F 检验 p 值，红=显著<0.05）")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("p 值")
fig.suptitle("Granger 因果检验：全部 p > 0.05，无跨序列因果路径", y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGS}/sub1-whitenoise-granger-v1.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {FIGS}/sub1-whitenoise-granger-v1.pdf")

# ---- 图3：测试窗预测对比（实际 vs 基线/Prophet/LSTM）----
fig, ax = plt.subplots(figsize=(10.5, 5))
test_h = TE
ax.plot(test_h, y_test, color="#333333", lw=1.8, label="实际")
ax.plot(test_h, pred_mean, color="#1f77b4", ls="--", lw=1.5,
        label=f"常数均值 (RMSE={models['常数均值'][1]:.0f})")
ax.plot(test_h, pred_lin, color="#2ca02c", ls="-.", lw=1.5,
        label=f"线性回归 (RMSE={models['线性回归'][1]:.0f})")
ax.plot(test_h, pred_prophet, color="#d62728", ls=":", lw=1.8,
        label=f"Prophet (RMSE={models['Prophet'][1]:.0f})")
ax.plot(test_h, pred_lstm, color="#9467bd", ls=(0, (1, 1)), lw=1.5,
        label=f"LSTM (RMSE={models['LSTM'][1]:.0f})")
ax.set_title("测试窗 2376–2399：复杂模型未显著优于均值基线")
ax.set_xlabel("小时")
ax.set_ylabel("GPU 需求")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGS}/sub1-whitenoise-forecast-v1.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {FIGS}/sub1-whitenoise-forecast-v1.pdf")

# ---- 图4：四模型 RMSE 对比 ----
fig, ax = plt.subplots(figsize=(7.5, 3.6))
order = sorted(models.items(), key=lambda z: z[1][1], reverse=True)
names = [n for n, _ in order]
rmses = [v[1] for _, v in order]
improve = [(models["常数均值"][1] - r) / models["常数均值"][1] * 100 for r in rmses]
bars = ax.barh(names, rmses, color=[COLORS[3], COLORS[1], COLORS[0], COLORS[4]], alpha=0.88, height=0.62)
XR = 240  # 图右侧注解列（xlim 至 350，右缘留空不贴图框）
for b, r, imp in zip(bars, rmses, improve):
    ax.text(XR, b.get_y() + b.get_height() / 2,
            f"RMSE={r:.1f}  ({imp:+.1f}%)", va="center", ha="left", fontsize=9)
ax.axvline(models["常数均值"][1], color="#333333", ls=":", lw=1.0)
ax.text(models["常数均值"][1] + 2, -0.35, "常数均值基线", fontsize=7, color="#333333")
ax.set_xlim(0, 350)
ax.set_xlabel("RMSE（越小越好；括号内为相对常数均值提升）")
ax.set_title("测试窗四模型 RMSE：最优模型仅优于均值 2.9%（<5% 判据）")
fig.tight_layout()
fig.savefig(f"{FIGS}/sub1-whitenoise-rmse-v1.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {FIGS}/sub1-whitenoise-rmse-v1.pdf")

# ---- 副本同步 artifacts ----
import shutil
for f in ["sub1-whitenoise-bds-v1.pdf", "sub1-whitenoise-granger-v1.pdf",
          "sub1-whitenoise-forecast-v1.pdf", "sub1-whitenoise-rmse-v1.pdf"]:
    shutil.copy2(f"{FIGS}/{f}", f"solution/artifacts/charts/{f}")
print("[OK] 副本已同步 solution/artifacts/charts/")
