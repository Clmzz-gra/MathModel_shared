"""交叉算子：将两个父代个体组合产生子代。"""

import numpy as np

CROSSOVER_METHODS = {}


def _register(name, fn, params, description):
    CROSSOVER_METHODS[name] = {"fn": fn, "params": params, "description": description}


# ── 1. 单点交叉 ────────────────────────────────────────────

def single_point(parent1, parent2, **kwargs):
    """在随机位置切一刀，交换后半段基因。"""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


_register("Single-Point", single_point, {},
          "单点交叉：随机选一个断点，交换断点后的基因段。简单、经典。")

# ── 2. 两点交叉 ────────────────────────────────────────────

def two_point(parent1, parent2, **kwargs):
    """随机选两个位置，交换中间段的基因。"""
    points = np.sort(np.random.choice(range(1, len(parent1)), size=2, replace=False))
    a, b = points[0], points[1]
    child1 = np.concatenate([parent1[:a], parent2[a:b], parent1[b:]])
    child2 = np.concatenate([parent2[:a], parent1[a:b], parent2[b:]])
    return child1, child2


_register("Two-Point", two_point, {},
          "两点交叉：交换两个断点之间的基因段。比单点交叉保留更多父代信息。")

# ── 3. 均匀交叉 ────────────────────────────────────────────

def uniform(parent1, parent2, bias=0.5, **kwargs):
    """每个基因位独立地以概率 0.5 从父代1或父代2继承。"""
    mask = np.random.random(len(parent1)) < bias
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


_register("Uniform", uniform,
          {"bias": (0.5, "偏向父代1的概率。0.5 表示各一半")},
          "均匀交叉：每个基因独立地从两个父代中选取。适合实数编码，混合最充分。")

# ── 4. 算术交叉 ────────────────────────────────────────────

def arithmetic(parent1, parent2, alpha=0.5, **kwargs):
    """子代 = alpha * parent1 + (1-alpha) * parent2（凸组合）。实数编码专用。"""
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


_register("Arithmetic", arithmetic,
          {"alpha": (0.5, "混合系数。0.5 表示两个父代权重相等")},
          "算术交叉：子代是父代的加权平均。实数编码专用，平滑探索父代之间的空间。")


def get_crossover(name):
    if name not in CROSSOVER_METHODS:
        raise ValueError(f"未知交叉算子: {name}。可选: {list(CROSSOVER_METHODS.keys())}")
    return CROSSOVER_METHODS[name]["fn"], CROSSOVER_METHODS[name]["params"]
