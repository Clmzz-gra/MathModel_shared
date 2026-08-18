"""变异算子：对个体基因进行小幅度随机扰动。"""

import numpy as np

MUTATION_METHODS = {}


def _register(name, fn, params, description):
    MUTATION_METHODS[name] = {"fn": fn, "params": params, "description": description}


# ── 1. 均匀变异 ────────────────────────────────────────────

def uniform_mutation(individual, bounds, mutation_rate=0.1, scale=0.1, **kwargs):
    """随机选一部分基因，在范围内重新随机生成。"""
    mutant = individual.copy()
    range_width = bounds[:, 1] - bounds[:, 0]
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # 在当前值附近均匀扰动
            delta = np.random.uniform(-scale * range_width[i], scale * range_width[i])
            mutant[i] = np.clip(individual[i] + delta, bounds[i, 0], bounds[i, 1])
    return mutant


_register("Uniform", uniform_mutation,
          {"scale": (0.1, "变异幅度系数。越大变异越剧烈")},
          "均匀变异：以概率 mutation_rate 对每个基因加上均匀随机扰动。最常用。")

# ── 2. 高斯变异 ────────────────────────────────────────────

def gaussian_mutation(individual, bounds, mutation_rate=0.1, sigma=0.1, **kwargs):
    """随机选一部分基因，加高斯噪声。"""
    mutant = individual.copy()
    range_width = bounds[:, 1] - bounds[:, 0]
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            delta = np.random.normal(0, sigma * range_width[i])
            mutant[i] = np.clip(individual[i] + delta, bounds[i, 0], bounds[i, 1])
    return mutant


_register("Gaussian", gaussian_mutation,
          {"sigma": (0.1, "高斯噪声标准差系数")},
          "高斯变异：加正态分布噪声。小幅变化概率高、大幅变化概率低。")

# ── 3. 多项式变异 ──────────────────────────────────────────

def polynomial_mutation(individual, bounds, mutation_rate=0.1, eta=20, **kwargs):
    """多项式变异（Polynomial Mutation），进化计算中标准算子。

    eta 控制变异分布形状：eta 越大变异越集中在原值附近。
    """
    mutant = individual.copy()
    low, high = bounds[:, 0], bounds[:, 1]
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            u = np.random.random()
            delta = (individual[i] - low[i]) / (high[i] - low[i] + 1e-16)
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            mutant[i] = individual[i] + delta_q * (high[i] - low[i])
            mutant[i] = np.clip(mutant[i], low[i], high[i])
    return mutant


_register("Polynomial", polynomial_mutation,
          {"eta": (20, "分布指数。越大变异越集中在原值附近。5-50 常用")},
          "多项式变异：进化计算中的标准算子。变异幅度分布灵活，eta 控制集中度。")


def get_mutator(name):
    if name not in MUTATION_METHODS:
        raise ValueError(f"未知变异算子: {name}。可选: {list(MUTATION_METHODS.keys())}")
    return MUTATION_METHODS[name]["fn"], MUTATION_METHODS[name]["params"]


# ── 4. DE 变异 (来自 AL-012 DEGA) ────────────────────────

def de_mutation(population, fitnesses, best_x, bounds, F=0.8):
    """DE/best/2 变异: 以最优个体为基向量，加两个差分向量。

    来源: AL-012 DEGA卡片
    公式: P_N = N_best + F*(N_1 - N_2) + F*(N_3 - N_4)
    其中 N_1-N_4 是从种群中随机选取的互不相同个体。
    """
    n, d = population.shape
    mutated = np.zeros_like(population)
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        n1, n2, n3, n4 = population[np.random.choice(candidates, size=4, replace=False)]
        mutant = best_x + F * (n1 - n2) + F * (n3 - n4)
        mutated[i] = np.clip(mutant, bounds[:, 0], bounds[:, 1])
    return mutated
