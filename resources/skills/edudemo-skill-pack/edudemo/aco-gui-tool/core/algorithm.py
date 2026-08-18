"""蚁群算法核心引擎 — 经典 Ant System (AS)。

标准 ACO 求解 TSP 问题：
1. 每只蚂蚁根据信息素 + 启发式信息构建路径
2. 路径越短，遗留信息素越多
3. 信息素挥发避免过早收敛
4. 正反馈引导后续蚂蚁选择好路径

最小化路径总长度。
"""

import time
import threading
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from . import tsp_problems


@dataclass
class IterationResult:
    """一次迭代结果。"""
    iteration: int
    best_tour: np.ndarray         # 当前全局最优路径 (city indices)
    best_distance: float
    mean_distance: float
    worst_distance: float
    all_tours: list               # 本轮所有蚂蚁的路径
    all_distances: np.ndarray
    pheromone_matrix: np.ndarray  # 当前信息素矩阵
    diversity: float              # 路径多样性
    elapsed: float = 0.0


@dataclass
class ACOResult:
    """ACO 完整运行结果。"""
    best_tour: np.ndarray
    best_distance: float
    history: list
    total_iterations: int
    total_evaluations: int
    total_time: float
    stopped_early: bool = False


class AntColonyOptimizer:
    """蚁群优化算法（Ant System）。

    标准 AS 变体，支持 ACS / MMAS / Elitist 改进。
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        cities: np.ndarray,
        n_ants: int = 30,
        max_iterations: int = 200,
        alpha: float = 1.0,        # 信息素重要性
        beta: float = 2.0,         # 启发式重要性
        rho: float = 0.5,          # 信息素挥发率
        q0: float = 0.0,           # ACS 伪随机比例（0 = 标准 AS）
        xi: float = 0.1,           # ACS 局部信息素更新率
        tau0: float = None,        # 初始信息素
        elitist_weight: float = 0.0,  # 精英蚁权重（0 = 标准 AS）
        mmas: bool = False,        # Max-Min Ant System
        tau_max: float = None,
        tau_min: float = None,
        seed: Optional[int] = None,
    ):
        self.distance_matrix = distance_matrix
        self.cities = cities
        self.n_cities = len(distance_matrix)
        self.n_ants = n_ants
        self.max_iterations = max_iterations

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0                        # ACS: 0=标准AS, >0=ACS伪随机
        self.xi = xi                        # ACS 局部挥发率
        self._tau0 = tau0
        self.elitist_weight = elitist_weight  # Elitist: >0 额外加强最优路径
        self.mmas = mmas
        self._tau_max = tau_max
        self._tau_min = tau_min

        # 启发式矩阵 (1/distance)
        with np.errstate(divide="ignore"):
            self.heuristic = np.where(distance_matrix > 0, 1.0 / distance_matrix, 0)
        np.fill_diagonal(self.heuristic, 0)

        if seed is not None:
            np.random.seed(seed)

        # 内部状态
        self._pheromone: Optional[np.ndarray] = None
        self._best_tour: Optional[np.ndarray] = None
        self._best_distance: float = float("inf")
        self._iteration: int = 0
        self._history: list = []
        self._start_time: float = 0.0
        self._done: bool = False
        self._stopped: bool = False
        self._total_evaluations: int = 0

        # 线程控制
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()

    # ── 公共接口 ─────────────────────────────────────────────

    def initialize(self):
        """初始化信息素矩阵。"""
        if self._tau0 is None:
            # 贪心路径长度作为 tau0 的参考
            greedy_dist = self._greedy_tour_length()
            self._tau0 = float(self.n_ants / greedy_dist) if greedy_dist > 0 else 1.0

        self._pheromone = np.full((self.n_cities, self.n_cities), self._tau0)
        np.fill_diagonal(self._pheromone, 0)

        # MMAS 边界
        if self.mmas:
            self._tau_max = self._tau0
            self._tau_min = self._tau_max / (2 * self.n_cities)

        self._best_tour = None
        self._best_distance = float("inf")
        self._iteration = 0
        self._done = False
        self._stopped = False
        self._history = []
        self._total_evaluations = 0

        self._pause_event.set()
        self._stop_event.clear()
        self._start_time = time.time()

    def step(self) -> Optional[IterationResult]:
        """执行一次完整迭代（所有蚂蚁建路径 + 信息素更新）。"""
        if self._done:
            return None

        self._pause_event.wait()

        if self._stop_event.is_set():
            self._done = True
            return None

        if self._iteration >= self.max_iterations:
            self._done = True
            return None

        n = self.n_cities

        # 1. 每只蚂蚁构建路径
        all_tours = []
        all_distances = np.zeros(self.n_ants)

        for ant in range(self.n_ants):
            tour = self._construct_tour()
            distance = self._tour_length(tour)
            all_tours.append(tour)
            all_distances[ant] = distance
            self._total_evaluations += 1

        # 2. 更新全局最优
        best_idx = np.argmin(all_distances)
        if all_distances[best_idx] < self._best_distance:
            self._best_distance = all_distances[best_idx]
            self._best_tour = all_tours[best_idx].copy()

        iter_best_idx = np.argmin(all_distances)
        iter_best_tour = all_tours[iter_best_idx]
        iter_best_dist = all_distances[iter_best_idx]

        # 3. 信息素挥发
        self._pheromone *= (1 - self.rho)

        # 4. 信息素沉积
        if self.mmas:
            # MMAS: 只迭代最优（或全局最优）的蚂蚁沉积
            deposit_tour = self._best_tour if np.random.random() < 0.5 else iter_best_tour
            deposit_dist = self._tour_length(deposit_tour)
            deposit = 1.0 / max(deposit_dist, 1e-10)
            for i in range(len(deposit_tour) - 1):
                a, b = deposit_tour[i], deposit_tour[i + 1]
                self._pheromone[a, b] += deposit
                self._pheromone[b, a] += deposit
        else:
            # 标准 AS: 所有蚂蚁沉积
            for ant in range(self.n_ants):
                deposit = 1.0 / max(all_distances[ant], 1e-10)
                tour = all_tours[ant]
                for i in range(len(tour) - 1):
                    a, b = tour[i], tour[i + 1]
                    self._pheromone[a, b] += deposit
                    self._pheromone[b, a] += deposit

            # Elitist: 最优蚂蚁额外沉积
            if self.elitist_weight > 0 and self._best_tour is not None:
                elite_deposit = self.elitist_weight / max(self._best_distance, 1e-10)
                for i in range(len(self._best_tour) - 1):
                    a, b = self._best_tour[i], self._best_tour[i + 1]
                    self._pheromone[a, b] += elite_deposit
                    self._pheromone[b, a] += elite_deposit

        # 5. MMAS 边界钳制
        if self.mmas:
            self._pheromone = np.clip(self._pheromone, self._tau_min, self._tau_max)

        # 对称化并清除对角线
        self._pheromone = (self._pheromone + self._pheromone.T) / 2
        np.fill_diagonal(self._pheromone, 0)

        self._iteration += 1

        # 多样性: 不同路径在总路径数中的比例
        unique_tours = len(set(tuple(t) for t in all_tours))
        diversity = unique_tours / self.n_ants

        elapsed = time.time() - self._start_time

        result = IterationResult(
            iteration=self._iteration,
            best_tour=self._best_tour.copy(),
            best_distance=self._best_distance,
            mean_distance=float(np.mean(all_distances)),
            worst_distance=float(np.max(all_distances)),
            all_tours=[t.copy() for t in all_tours],
            all_distances=all_distances.copy(),
            pheromone_matrix=self._pheromone.copy(),
            diversity=float(diversity),
            elapsed=elapsed,
        )
        self._history.append(result)
        return result

    def run(self, callback=None) -> ACOResult:
        """完整运行。"""
        if self._pheromone is None:
            self.initialize()
        while True:
            result = self.step()
            if result is None:
                break
            if callback:
                callback(result)
        return self.finalize()

    def finalize(self) -> ACOResult:
        total_time = time.time() - self._start_time
        return ACOResult(
            best_tour=self._best_tour.copy() if self._best_tour is not None else np.array([]),
            best_distance=self._best_distance,
            history=self._history.copy(),
            total_iterations=self._iteration,
            total_evaluations=self._total_evaluations,
            total_time=total_time,
            stopped_early=self._stopped,
        )

    # ── 控制接口 ─────────────────────────────────────────────

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()

    @property
    def done(self):
        return self._done

    @property
    def is_paused(self):
        return not self._pause_event.is_set()

    # ── 内部 ─────────────────────────────────────────────────

    def _construct_tour(self) -> np.ndarray:
        """一只蚂蚁构建完整路径。

        ACS 模式 (q0 > 0): 伪随机比例规则
        标准模式 (q0 = 0): 轮盘赌选择
        """
        n = self.n_cities
        start = np.random.randint(0, n)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)

        while unvisited:
            current = tour[-1]
            unvisited_list = list(unvisited)

            if self.q0 > 0 and np.random.random() < self.q0:
                # ACS: 贪心选择 (exploitation)
                next_city = self._acs_best_next(current, unvisited_list)
            else:
                # 标准: 轮盘赌选择
                next_city = self._roulette_next(current, unvisited_list)

            tour.append(next_city)
            unvisited.remove(next_city)

            # ACS 局部信息素更新
            if self.q0 > 0 and self.xi > 0:
                a, b = current, next_city
                self._pheromone[a, b] = (1 - self.xi) * self._pheromone[a, b] + self.xi * self._tau0
                self._pheromone[b, a] = self._pheromone[a, b]

        # 回到起点
        tour.append(tour[0])
        return np.array(tour)

    def _acs_best_next(self, current: int, unvisited: list) -> int:
        """ACS 贪心选择：选 tau^alpha * eta^beta 最大的城市。"""
        tau = self._pheromone[current, unvisited] ** self.alpha
        eta = self.heuristic[current, unvisited] ** self.beta
        scores = tau * eta
        return unvisited[np.argmax(scores)]

    def _roulette_next(self, current: int, unvisited: list) -> int:
        """轮盘赌选择下一个城市。"""
        tau = self._pheromone[current, unvisited] ** self.alpha
        eta = self.heuristic[current, unvisited] ** self.beta
        probs = tau * eta
        total = probs.sum()
        if total < 1e-16:
            return np.random.choice(unvisited)
        probs /= total
        return np.random.choice(unvisited, p=probs)

    def _tour_length(self, tour: np.ndarray) -> float:
        """计算路径总长度。"""
        dist = 0.0
        for i in range(len(tour) - 1):
            dist += self.distance_matrix[tour[i], tour[i + 1]]
        return float(dist)

    def _greedy_tour_length(self) -> float:
        """贪心算法（最近邻）计算路径长度，作为 tau0 参考。"""
        n = self.n_cities
        start = 0
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        while unvisited:
            current = tour[-1]
            unvisited_list = list(unvisited)
            # 选最近的未访问城市
            dists = self.distance_matrix[current, unvisited_list]
            next_city = unvisited_list[np.argmin(dists)]
            tour.append(next_city)
            unvisited.remove(next_city)
        tour.append(tour[0])
        return self._tour_length(np.array(tour))

    # ── 属性 ─────────────────────────────────────────────────

    @property
    def current_pheromone(self):
        return self._pheromone

    @property
    def current_best_tour(self):
        return self._best_tour

    @property
    def current_best_distance(self):
        return self._best_distance
