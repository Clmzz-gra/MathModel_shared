"""邻域生成策略：在当前解附近生成候选解。

返回 (new_x, step_size_info) — new_x 为候选解，step_size_info 为调试信息。
"""

import numpy as np


def gaussian(current_x, bounds, sigma=0.1, **kwargs):
    """高斯扰动：N(0, sigma^2 * 搜索范围) 的随机扰动。

    适合连续空间中的平滑目标函数。sigma 控制扰动幅度。
    """
    range_width = bounds[:, 1] - bounds[:, 0]
    perturbation = np.random.normal(0, sigma * range_width, len(current_x))
    new_x = current_x + perturbation
    new_x = np.clip(new_x, bounds[:, 0], bounds[:, 1])
    return new_x, {"type": "gaussian", "sigma": sigma, "perturbation_norm": np.linalg.norm(perturbation)}


def uniform(current_x, bounds, scale=0.1, **kwargs):
    """均匀扰动：在 [-scale * 范围, +scale * 范围] 内均匀采样。

    比高斯扰动有更厚的尾部，偶尔会产生大幅跳跃。
    """
    range_width = bounds[:, 1] - bounds[:, 0]
    perturbation = np.random.uniform(-scale * range_width, scale * range_width, len(current_x))
    new_x = current_x + perturbation
    new_x = np.clip(new_x, bounds[:, 0], bounds[:, 1])
    return new_x, {"type": "uniform", "scale": scale, "perturbation_norm": np.linalg.norm(perturbation)}


def cauchy(current_x, bounds, gamma=0.1, **kwargs):
    """柯西扰动：从 Cauchy(0, gamma * 范围) 采样。

    柯西分布有极厚的尾部，偶尔产生大幅跳跃，有助于跳出局部最优。
    """
    range_width = bounds[:, 1] - bounds[:, 0]
    perturbation = np.random.standard_cauchy(len(current_x)) * gamma * range_width
    perturbation = np.clip(perturbation, -range_width, range_width)  # 防止极端值
    new_x = current_x + perturbation
    new_x = np.clip(new_x, bounds[:, 0], bounds[:, 1])
    return new_x, {"type": "cauchy", "gamma": gamma, "perturbation_norm": np.linalg.norm(perturbation)}


def adaptive_step(current_x, bounds, step_size=None, decay=0.999, min_step_ratio=0.001, **kwargs):
    """自适应步长：步长随迭代衰减。越到后期步长越小，精细搜索。

    使用时需在外部维护 step_size 状态，每步传入当前 step_size。
    """
    if step_size is None:
        step_size = (bounds[:, 1] - bounds[:, 0]) * 0.2

    std = np.maximum(step_size, (bounds[:, 1] - bounds[:, 0]) * min_step_ratio)
    perturbation = np.random.normal(0, std, len(current_x))
    new_x = current_x + perturbation
    new_x = np.clip(new_x, bounds[:, 0], bounds[:, 1])

    next_step = step_size * decay
    return new_x, {"type": "adaptive", "step_size": step_size, "next_step": next_step,
                   "perturbation_norm": np.linalg.norm(perturbation)}


# ── 邻域策略注册表 ──────────────────────────────────────────

NEIGHBORHOODS = {
    "Gaussian": {
        "fn": gaussian,
        "params": {"sigma": (0.1, "扰动标准差系数。越大搜索步长越大。典型 0.01-0.5")},
        "description": "最常用的邻域策略。在当前解上加高斯噪声。小幅高概率 + 中幅低概率。",
    },
    "Uniform": {
        "fn": uniform,
        "params": {"scale": (0.1, "均匀扰动范围系数。越大搜索范围越广")},
        "description": "在 [-scale*range, +scale*range] 内均匀采样。尾部比高斯厚，跳跃概率更均匀。",
    },
    "Cauchy": {
        "fn": cauchy,
        "params": {"gamma": (0.1, "柯西尺度参数。越大概率越容易大幅跳跃。典型 0.01-0.3")},
        "description": "柯西分布扰动，厚尾。偶尔大幅跳跃，适合跳出深局部最优。",
    },
    "Adaptive": {
        "fn": adaptive_step,
        "params": {
            "decay": (0.999, "步长衰减系数，每步乘以此值"),
            "min_step_ratio": (0.001, "最小步长相对搜索范围的比率"),
        },
        "description": "自适应步长。初期大步搜索，后期精细收敛。在高维问题中表现良好。",
    },
}


def get_neighborhood(name, **params):
    """按名称获取邻域函数。"""
    if name not in NEIGHBORHOODS:
        raise ValueError(f"未知的邻域策略: {name}。可选: {list(NEIGHBORHOODS.keys())}")
    info = NEIGHBORHOODS[name]
    merged = {k: v[0] for k, v in info["params"].items()}
    merged.update(params)
    return info["fn"], merged
