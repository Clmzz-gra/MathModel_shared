"""选择算子：从种群中选出下一代父代。"""

import numpy as np

SELECTION_METHODS = {}


def _register(name, fn, params, description):
    SELECTION_METHODS[name] = {"fn": fn, "params": params, "description": description}


# ── 1. 轮盘赌选择 ──────────────────────────────────────────

def roulette(fitnesses, population, n_select, **kwargs):
    """按适应度比例随机选择。适应度越高被选中概率越大。

    注意：适应度可能为负（最小化问题），用 rank-based 转换后再做轮盘赌。
    """
    f = fitnesses.copy()
    # 如果全为负或最小值为负，转换为 rank
    if np.min(f) <= 0:
        ranks = np.argsort(np.argsort(f)) + 1  # 1-indexed ranks
        f = ranks.astype(float)
    total = np.sum(f)
    if total == 0:
        probs = np.ones(len(f)) / len(f)
    else:
        probs = f / total
    indices = np.random.choice(len(population), size=n_select, p=probs)
    return population[indices].copy()


_register("Roulette", roulette, {},
          "轮盘赌选择：适应度越高，被选中概率越大。需要适应度全为正，否则自动转为 rank 选择。")

# ── 2. 锦标赛选择 ──────────────────────────────────────────

def tournament(fitnesses, population, n_select, tournament_size=3, **kwargs):
    """每次随机选 tournament_size 个体，取其中适应度最高的。"""
    selected = np.zeros((n_select, population.shape[1]))
    for i in range(n_select):
        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
        winner = candidates[np.argmax(fitnesses[candidates])]
        selected[i] = population[winner]
    return selected


_register("Tournament", tournament,
          {"tournament_size": (3, "每轮参赛个体数。越大选择压力越大，2-5 常用")},
          "锦标赛选择：随机抽几个比一比，最强的胜出。简单高效，最常用。")

# ── 3. 排序选择 ────────────────────────────────────────────

def rank_selection(fitnesses, population, n_select, **kwargs):
    """按 fitness 排序后，用排名决定概率。避免了轮盘赌的绝对值敏感问题。"""
    order = np.argsort(fitnesses)  # 从小到大
    ranks = np.zeros(len(fitnesses))
    ranks[order] = np.arange(1, len(fitnesses) + 1)
    probs = ranks / np.sum(ranks)
    indices = np.random.choice(len(population), size=n_select, p=probs)
    return population[indices].copy()


_register("Rank", rank_selection, {},
          "排序选择：不管适应度绝对大小，只按排名分配概率。稳定、不受异常值影响。")

# ── 4. 精英选择 ────────────────────────────────────────────

def elitism(fitnesses, population, n_elites, **kwargs):
    """直接保留适应度最高的 n_elites 个体到下一代。通常配合其他选择算子使用。"""
    elite_idx = np.argsort(fitnesses)[-n_elites:]
    return population[elite_idx].copy()


_register("Elitism", elitism,
          {"n_elites": (2, "精英保留数，通常 1-5")},
          "精英保留：每代最优的几个直接进下一代，保证最优解不会退化。通常与其他选择算子配合。")


def get_selector(name):
    if name not in SELECTION_METHODS:
        raise ValueError(f"未知选择算子: {name}。可选: {list(SELECTION_METHODS.keys())}")
    return SELECTION_METHODS[name]["fn"], SELECTION_METHODS[name]["params"]
