"""TSP 问题实例生成器。

提供多种城市分布模式的随机实例，以及经典 TSP 测试实例。
"""

import numpy as np

INSTANCE_PATTERNS = {}


def _register(name, fn, params, description):
    INSTANCE_PATTERNS[name] = {"fn": fn, "params": params, "description": description}


# ── 1. 均匀随机 ────────────────────────────────────────────

def generate_uniform(n_cities: int, seed: int = None, **kwargs):
    """城市在 [0,1]² 内均匀随机分布。"""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, (n_cities, 2))


_register("Uniform", generate_uniform, {},
          "均匀随机：城市在 [0,1]² 内均匀分布。最通用的测试场景。")


# ── 2. 圆形分布 ────────────────────────────────────────────

def generate_circle(n_cities: int, seed: int = None, radius: float = 0.45, **kwargs):
    """城市均匀分布在圆周上。已知最优解是一个圆。"""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n_cities)
    angles.sort()
    x = 0.5 + radius * np.cos(angles)
    y = 0.5 + radius * np.sin(angles)
    return np.column_stack([x, y])


_register("Circle", generate_circle,
          {"radius": (0.45, "圆的半径。0-0.5，越大城市离中心越远")},
          "圆形分布：城市在圆周上。最优解显然是按角度顺序走一圈。")


# ── 3. 聚类分布 ────────────────────────────────────────────

def generate_clusters(n_cities: int, seed: int = None, n_clusters: int = 3, spread: float = 0.08, **kwargs):
    """城市围绕几个聚类中心分布。模拟现实中的城市群。"""
    rng = np.random.RandomState(seed)
    # 生成聚类中心
    cluster_centers = rng.uniform(0.2, 0.8, (n_clusters, 2))
    cities = []
    for i in range(n_cities):
        center = cluster_centers[i % n_clusters]
        city = center + rng.normal(0, spread, 2)
        city = np.clip(city, 0, 1)
        cities.append(city)
    return np.array(cities)


_register("Clusters", generate_clusters,
          {"n_clusters": (3, "聚类群数量"), "spread": (0.08, "聚类内部散布程度")},
          "聚类分布：城市围绕几个中心聚集。模拟现实中的城市群分布。")


# ── 4. 网格分布 ────────────────────────────────────────────

def generate_grid(n_cities: int, seed: int = None, noise: float = 0.02, **kwargs):
    """城市在网格上分布，加少量噪声。"""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n_cities)))
    cities = []
    for i in range(n_cities):
        row, col = divmod(i, side)
        x = (col + 0.5) / side + rng.normal(0, noise)
        y = (row + 0.5) / side + rng.normal(0, noise)
        cities.append([np.clip(x, 0, 1), np.clip(y, 0, 1)])
    return np.array(cities)


_register("Grid", generate_grid,
          {"noise": (0.02, "网格抖动幅度。0 = 完美网格")},
          "网格分布：城市均匀分布在网格上，加少量随机噪声。结构清晰。")


def get_instance(name: str, n_cities: int, seed: int = None, **kwargs) -> np.ndarray:
    """生成 TSP 实例的城市坐标矩阵 (n_cities, 2)。"""
    if name not in INSTANCE_PATTERNS:
        raise ValueError(f"未知分布模式: {name}。可选: {list(INSTANCE_PATTERNS.keys())}")
    info = INSTANCE_PATTERNS[name]
    params = {k: v[0] for k, v in info["params"].items()}
    params.update(kwargs)
    return info["fn"](n_cities, seed=seed, **params)


def build_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """从城市坐标构建欧氏距离矩阵。"""
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def list_patterns():
    return list(INSTANCE_PATTERNS.keys())
