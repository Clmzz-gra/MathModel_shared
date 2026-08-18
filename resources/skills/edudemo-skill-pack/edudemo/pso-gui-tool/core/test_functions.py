"""经典优化测试函数集，带已知全局最优解和搜索边界。"""

import numpy as np

# ── 测试函数注册表 ──────────────────────────────────────────

TEST_FUNCTIONS = {}  # name -> {fn, bounds, global_min, global_x, description}


def _register(name, fn, bounds, global_min, global_x_2d, description, dims=None):
    bounds_arr = np.array(bounds, dtype=float)
    if bounds_arr.ndim == 1:
        bounds_arr = bounds_arr.reshape(1, 2)
    TEST_FUNCTIONS[name] = {
        "fn": fn,
        "bounds": bounds_arr,
        "global_min": global_min,
        "global_x_2d": np.array(global_x_2d, dtype=float),
        "description": description,
        "dims": dims,  # None 表示任意维度
    }


# ── 1. Sphere ───────────────────────────────────────────────

def sphere(x):
    return np.sum(x ** 2)

_register("Sphere", sphere, (-5.12, 5.12), 0.0, [0.0, 0.0],
          "最简单的凸函数。全局唯一最小值在原点。适合验证基本收敛性。")

# ── 2. Rastrigin ────────────────────────────────────────────

def rastrigin(x):
    A = 10.0
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

_register("Rastrigin", rastrigin, (-5.12, 5.12), 0.0, [0.0, 0.0],
          "多峰函数，等高的局部最优均匀分布在网格上。SA 的经典测试对象。")

# ── 3. Rosenbrock ───────────────────────────────────────────

def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

_register("Rosenbrock", rosenbrock, (-2.048, 2.048), 0.0, [1.0, 1.0],
          "香蕉谷函数。全局最小值在狭长的抛物线谷底。考验算法沿窄谷下坡的能力。")

# ── 4. Ackley ───────────────────────────────────────────────

def ackley(x):
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

_register("Ackley", ackley, (-32.768, 32.768), 0.0, [0.0, 0.0],
          "平坦的外围 + 中心深谷 + 大量局部最优。考验全局搜索和局部精调能力。")

# ── 5. Griewank ─────────────────────────────────────────────

def griewank(x):
    sum_term = np.sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1

_register("Griewank", griewank, (-600, 600), 0.0, [0.0, 0.0],
          "大范围多峰，但维度越高越像 Sphere。考验高维性能。")

# ── 6. Schwefel ─────────────────────────────────────────────

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

_register("Schwefel", schwefel, (-500, 500), 0.0, [420.9687, 420.9687],
          "全局最小值在边界附近，且有次优解远离全局最优。考验远跳能力。")

# ── 7. Levy ─────────────────────────────────────────────────

def _levy_w(xi):
    return 1 + (xi - 1) / 4

def levy(x):
    x = np.asarray(x)
    w = _levy_w(x)
    d = len(x)
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3

_register("Levy", levy, (-10, 10), 0.0, [1.0, 1.0],
          "多峰函数，全局最小值在 (1,1,...,1)。各维耦合复杂。")

# ── 8. Michalewicz ──────────────────────────────────────────

def michalewicz(x):
    m = 10.0
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m))

_register("Michalewicz", michalewicz, (0, np.pi), -1.8013, [2.20, 1.57],
          "陡峭的山脊和山谷。全局最小值随维度变化。2 维全局最优约 -1.8013。",
          dims=">1")

# ── 9. Zakharov ─────────────────────────────────────────────

def zakharov(x):
    d = len(x)
    i = np.arange(1, d + 1)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2 ** 2 + sum2 ** 4

_register("Zakharov", zakharov, (-10, 10), 0.0, [0.0, 0.0],
          "原点附近的平坦区域 + 外围陡升。考验精细搜索精度。")

# ── 10. Sum of Powers ───────────────────────────────────────

def sum_of_powers(x):
    i = np.arange(1, len(x) + 1)
    return np.sum(np.abs(x) ** (i + 1))

_register("Sum of Powers", sum_of_powers, (-1, 1), 0.0, [0.0, 0.0],
          "各维敏感度不同。高维误差被指数放大。")

# ── 11. Drop-Wave ───────────────────────────────────────────

def drop_wave(x):
    x = np.asarray(x)
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return -(1 + np.cos(12 * r)) / (0.5 * r ** 2 + 2)

_register("Drop-Wave", drop_wave, (-5.12, 5.12), -1.0, [0.0, 0.0],
          "只在 2D 可用。中心全局最小值周围有等高递减的波纹。", dims="=2")

# ── 12. Shubert ─────────────────────────────────────────────

def shubert_2d(x):
    x1, x2 = x[0], x[1]
    s1 = np.sum([i * np.cos((i + 1) * x1 + i) for i in range(1, 6)])
    s2 = np.sum([i * np.cos((i + 1) * x2 + i) for i in range(1, 6)])
    return s1 * s2

_register("Shubert", shubert_2d, (-10, 10), -186.7309, [-7.0835, 4.8580],
          "只在 2D 可用。18 个全局最小值密集分布。极度多峰。", dims="=2")

# ── 13. Easom ───────────────────────────────────────────────

def easom(x):
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))

_register("Easom", easom, (-100, 100), -1.0, [np.pi, np.pi],
          "只在 2D 可用。99.99% 区域值为 0，仅中心极小区域有尖峰。", dims="=2")

# ── 14. Cross-in-Tray ───────────────────────────────────────

def cross_in_tray(x):
    x1, x2 = x[0], x[1]
    t = np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)
    return -0.0001 * (np.abs(np.sin(x1) * np.sin(x2) * np.exp(t)) + 1) ** 0.1

_register("Cross-in-Tray", cross_in_tray, (-10, 10), -2.06261,
          [1.3491, -1.3491],
          "只在 2D 可用。四叶草形状，4 个对称的全局最小值。", dims="=2")

# ── 15. Holder Table ────────────────────────────────────────

def holder_table(x):
    x1, x2 = x[0], x[1]
    t = np.abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)
    return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(t))

_register("Holder Table", holder_table, (-10, 10), -19.2085,
          [8.05502, 9.66459],
          "只在 2D 可用。4 个全局最小值，表面高度起伏剧烈。", dims="=2")

# ── 16. Styblinski-Tang ─────────────────────────────────────

def styblinski_tang(x):
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

_register("Styblinski-Tang", styblinski_tang, (-5, 5),
          -39.16617 * 2,  # 2D 近似值
          [-2.903534, -2.903534],
          "全局最小值约 -39.16617n。大量局部最优中全局最优在 -2.9 附近。")

# ── 17. Beale ───────────────────────────────────────────────

def beale(x):
    x1, x2 = x[0], x[1]
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2 ** 2) ** 2
        + (2.625 - x1 + x1 * x2 ** 3) ** 2
    )

_register("Beale", beale, (-4.5, 4.5), 0.0, [3.0, 0.5],
          "只在 2D 可用。有尖峰和浅谷。全局最小值在 (3, 0.5)。", dims="=2")


# ── 工具函数 ────────────────────────────────────────────────

def get_function(name):
    """按名称获取测试函数字典。"""
    return TEST_FUNCTIONS[name]

def list_functions():
    """列出所有已注册的测试函数名称。"""
    return list(TEST_FUNCTIONS.keys())

def list_functions_for_dim(dim):
    """列出对给定维度可用的函数名称。"""
    available = []
    for name, info in TEST_FUNCTIONS.items():
        dims = info["dims"]
        if dims is None:
            available.append(name)
        elif dims == "=2" and dim == 2:
            available.append(name)
        elif dims == ">1" and dim > 1:
            available.append(name)
    return available
