"""粒子群邻域拓扑策略。

拓扑决定每个粒子"看到"谁的最优解作为社会引导。
全局拓扑收敛最快但易早熟，局部拓扑探索更充分。
"""

import numpy as np

TOPOLOGY_METHODS = {}


def _register(name, fn, params, description):
    TOPOLOGY_METHODS[name] = {"fn": fn, "params": params, "description": description}


# ── 1. 全局拓扑 (gBest) ────────────────────────────────────

def global_topology(personal_bests, personal_best_fitnesses,
                    swarm_best_pos, swarm_best_fit, **kwargs):
    """全局拓扑：所有粒子共享同一个全局最优。"""
    n = len(personal_bests)
    return np.tile(swarm_best_pos, (n, 1)), np.full(n, swarm_best_fit)


_register("Global", global_topology, {},
          "全局拓扑：所有粒子共享全局最优。收敛最快，但多样性低，易早熟。")


# ── 2. 环形拓扑 (lBest Ring) ───────────────────────────────

def ring_topology(personal_bests, personal_best_fitnesses,
                  swarm_best_pos, swarm_best_fit, k=2, **kwargs):
    """环形拓扑：每个粒子只看自己和左右 k 个邻居的 pBest。"""
    n = len(personal_bests)
    neighbors_best_pos = np.zeros_like(personal_bests)
    neighbors_best_fit = np.full(n, -np.inf)

    for i in range(n):
        indices = [(i + j) % n for j in range(-k, k + 1)]
        best_local_idx = indices[np.argmax(personal_best_fitnesses[indices])]
        neighbors_best_pos[i] = personal_bests[best_local_idx]
        neighbors_best_fit[i] = personal_best_fitnesses[best_local_idx]

    return neighbors_best_pos, neighbors_best_fit


_register("Ring", ring_topology,
          {"k": (2, "每侧的邻居数。k=1 最局部（慢但稳），k=n/2 接近全局")},
          "环形拓扑：粒子排列成环，每个粒子只看自己和相邻几个邻居。探索更充分。")


# ── 3. Von Neumann 拓扑 ────────────────────────────────────

def von_neumann_topology(personal_bests, personal_best_fitnesses,
                         swarm_best_pos, swarm_best_fit, **kwargs):
    """Von Neumann 拓扑：2D 网格，每个粒子看上/下/左/右邻居。"""
    n = len(personal_bests)
    grid_w = int(np.ceil(np.sqrt(n)))
    grid_h = int(np.ceil(n / grid_w))

    neighbors_best_pos = np.zeros_like(personal_bests)
    neighbors_best_fit = np.full(n, -np.inf)

    for i in range(n):
        row, col = divmod(i, grid_w)
        indices = [i]
        if row > 0:
            indices.append(i - grid_w)
        if row < grid_h - 1 and (row + 1) * grid_w + col < n:
            indices.append(i + grid_w)
        if col > 0:
            indices.append(i - 1)
        if col < grid_w - 1 and i + 1 < n:
            indices.append(i + 1)

        best_local_idx = indices[np.argmax(personal_best_fitnesses[indices])]
        neighbors_best_pos[i] = personal_bests[best_local_idx]
        neighbors_best_fit[i] = personal_best_fitnesses[best_local_idx]

    return neighbors_best_pos, neighbors_best_fit


_register("Von Neumann", von_neumann_topology, {},
          "Von Neumann 拓扑：2D 网格，每粒子连上/下/左/右邻居。信息传播速度适中，标准选择。")


def get_topology(name):
    if name not in TOPOLOGY_METHODS:
        raise ValueError(f"未知拓扑: {name}。可选: {list(TOPOLOGY_METHODS.keys())}")
    return TOPOLOGY_METHODS[name]["fn"], TOPOLOGY_METHODS[name]["params"]
