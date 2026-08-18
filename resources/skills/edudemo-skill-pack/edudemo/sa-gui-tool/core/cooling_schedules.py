"""降温策略集合：返回温度序列或给出下一步温度。

每个 schedule 以生成器函数实现，接收初始温度 T0 和参数，yield 每一步的温度。
另有辅助函数按名称获取调度器。
"""

import math
import numpy as np


def geometric(T0, alpha=0.95, **kwargs):
    """几何降温 (最常用): T_{k+1} = alpha * T_k

    alpha ∈ [0.8, 0.99] 最常用。alpha 越小降温越快，越容易过早收敛。
    """
    T = T0
    while T > 0:
        yield T
        T *= alpha


def linear(T0, T_end=0.01, max_iter=1000, **kwargs):
    """线性降温: T_k = T0 - k * (T0 - T_end) / max_iter

    降温速率恒定。max_iter 取 0 时使用默认值。
    """
    if max_iter <= 1:
        max_iter = 1000
    delta = (T0 - T_end) / max_iter
    T = T0
    for _ in range(max_iter + 1):
        yield T
        T -= delta
        if T < T_end:
            yield T_end
            return


def logarithmic(T0, c=1.0, **kwargs):
    """对数降温: T_k = T0 / log(2 + k/c)

    降温极慢，适合需要长时间探索的问题。
    """
    k = 0
    while True:
        T = T0 / math.log(2 + k / c)
        if T < 1e-16:
            return
        yield T
        k += 1


def exponential(T0, beta=0.01, **kwargs):
    """指数降温: T_k = T0 * exp(-beta * k)

    比几何降温更快地下降。beta 控制速率，越大降温越快。
    """
    k = 0
    while True:
        T = T0 * math.exp(-beta * k)
        if T < 1e-16:
            return
        yield T
        k += 1


def adaptive(T0, alpha=0.95, adapt_trigger=0.02, **kwargs):
    """自适应降温: 接受率过低时加速降温，接受率正常时按几何降温。

    adapt_trigger: 接受率阈值。低于此值认为搜索收敛，加速降温。
    实现方式：yield 的是当前温度；需外部每步后调用 update_accept_rate(rate)。
    """
    T = T0
    step = 0
    window_accept = []
    window_size = 50

    while T > 1e-16:
        yield T

        step += 1
        # 每 window_size 步检查一次接受率
        if len(window_accept) >= window_size:
            accepted_rate = sum(window_accept) / len(window_accept)
            if accepted_rate < adapt_trigger:
                T *= alpha * 0.8  # 加速降温
            else:
                T *= alpha
            window_accept = []
        else:
            T *= alpha

    def update_accept_rate(self, accepted: bool):
        """由 SA 引擎每步调用，记录是否接受新解。"""
        window_accept.append(1 if accepted else 0)


# ── 调度器注册表 ────────────────────────────────────────────

SCHEDULES = {
    "Geometric": {
        "fn": geometric,
        "params": {"alpha": (0.95, "降温系数，0.8-0.99。越接近1降温越慢")},
        "description": "应用最广的降温方式。每步温度乘以常数 α。简单有效，适合多数问题。",
    },
    "Linear": {
        "fn": linear,
        "params": {
            "T_end": (0.01, "终止温度，到达此温度算法停止"),
            "max_iter": (1000, "总迭代次数，用于计算每步降温量"),
        },
        "description": "温度线性下降。降温速率恒定，适合已知大致迭代次数的情况。",
    },
    "Logarithmic": {
        "fn": logarithmic,
        "params": {"c": (1.0, "缩放常数，越大降温越慢")},
        "description": "对数降温，下降极慢。适合需要长搜索时间、容易陷入局部最优的问题。",
    },
    "Exponential": {
        "fn": exponential,
        "params": {"beta": (0.01, "指数衰减系数，越大降温越快。典型 0.001-0.1")},
        "description": "指数快速降温。比几何降温更快，适合搜索空间较小的问题。",
    },
    "Adaptive": {
        "fn": adaptive,
        "params": {
            "alpha": (0.95, "基础降温系数"),
            "adapt_trigger": (0.02, "接受率触发阈值。当近期接受率低于此值，加速降温退出搜索"),
        },
        "description": "自适应降温：根据近期解的接受率动态调整降温速率。接受率高时慢降，接受率低时加速。",
    },
}


def get_schedule(name, **params):
    """获取调度器生成器。name 必须是 SCHEDULES 中的键。"""
    if name not in SCHEDULES:
        raise ValueError(f"未知的降温策略: {name}。可选: {list(SCHEDULES.keys())}")
    info = SCHEDULES[name]
    # 合并默认参数与用户参数
    merged = {k: v[0] for k, v in info["params"].items()}
    merged.update(params)
    return info["fn"](**merged)
