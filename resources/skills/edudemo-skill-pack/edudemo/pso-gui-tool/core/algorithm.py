"""粒子群优化核心引擎。

标准 PSO 带惯性权重（Inertia Weight PSO）。
速度更新: v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
位置更新: x = x + v

最小化目标函数，适应度 = -f(x)。
"""

import time
import threading
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from . import topology as topo


@dataclass
class IterationResult:
    """一次迭代结果。"""
    iteration: int
    positions: np.ndarray
    velocities: np.ndarray
    fitnesses: np.ndarray
    personal_bests: np.ndarray
    personal_best_fitnesses: np.ndarray
    best_x: np.ndarray
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    avg_velocity: float
    inertia: float
    elapsed: float = 0.0


@dataclass
class PSOResult:
    """PSO 完整运行结果。"""
    best_x: np.ndarray
    best_energy: float
    history: list
    total_iterations: int
    total_evaluations: int
    total_time: float
    stopped_early: bool = False


class ParticleSwarmOptimizer:
    """标准粒子群优化算法（惯性权重版本）。

    每个粒子有位置 x、速度 v、个人历史最优 pBest。
    全局最优 gBest（或局部最优 lBest）引导搜索方向。
    """

    def __init__(
        self,
        objective_fn: Callable,
        bounds: np.ndarray,
        swarm_size: int = 50,
        max_iterations: int = 300,
        inertia: float = 0.7,
        cognitive: float = 1.5,
        social: float = 1.5,
        topology_name: str = "Global",
        topology_params: Optional[dict] = None,
        boundary: str = "clip",
        v_clamp_ratio: float = 0.2,
        constriction: bool = False,
        adaptive_inertia: bool = False,
        inertia_decay: float = 0.995,
        inertia_min: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.objective_fn = objective_fn
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.topology_name = topology_name
        self.topology_params = topology_params or {}

        self.boundary = boundary
        self.v_clamp_ratio = v_clamp_ratio
        self.constriction = constriction
        self.adaptive_inertia = adaptive_inertia
        self.inertia_decay = inertia_decay
        self.inertia_min = inertia_min
        self._current_inertia = inertia

        if seed is not None:
            np.random.seed(seed)

        # 内部状态
        self._positions: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None
        self._fitnesses: Optional[np.ndarray] = None
        self._personal_bests: Optional[np.ndarray] = None
        self._personal_best_fitnesses: Optional[np.ndarray] = None
        self._swarm_best_pos: Optional[np.ndarray] = None
        self._swarm_best_fit: float = -float("inf")
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
        """随机初始化粒子群。"""
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        span = high - low

        self._positions = np.random.uniform(low, high, (self.swarm_size, self.dim))
        self._velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim)) * span

        self._fitnesses = np.array([self._evaluate(p) for p in self._positions])
        self._total_evaluations = self.swarm_size

        self._personal_bests = self._positions.copy()
        self._personal_best_fitnesses = self._fitnesses.copy()

        best_idx = np.argmax(self._fitnesses)
        self._swarm_best_pos = self._positions[best_idx].copy()
        self._swarm_best_fit = self._fitnesses[best_idx]

        self._iteration = 0
        self._done = False
        self._stopped = False
        self._history = []
        self._current_inertia = self.inertia

        self._pause_event.set()
        self._stop_event.clear()
        self._start_time = time.time()

    def step(self) -> Optional[IterationResult]:
        """执行一次完整迭代（更新所有粒子）。返回 IterationResult 或 None。"""
        if self._done:
            return None

        self._pause_event.wait()

        if self._stop_event.is_set():
            self._done = True
            return None

        if self._iteration >= self.max_iterations:
            self._done = True
            return None

        low, high = self.bounds[:, 0], self.bounds[:, 1]
        span = high - low

        # 获取邻域最优
        topo_fn, _ = topo.get_topology(self.topology_name)
        topo_kwargs = {k: v[0] for k, v in topo.TOPOLOGY_METHODS[self.topology_name]["params"].items()}
        topo_kwargs.update(self.topology_params)

        neighbors_best_pos, neighbors_best_fit = topo_fn(
            self._personal_bests, self._personal_best_fitnesses,
            self._swarm_best_pos, self._swarm_best_fit, **topo_kwargs
        )

        # 更新每个粒子
        for i in range(self.swarm_size):
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            # 速度更新
            cognitive_v = self.cognitive * r1 * (self._personal_bests[i] - self._positions[i])
            social_v = self.social * r2 * (neighbors_best_pos[i] - self._positions[i])
            self._velocities[i] = self._current_inertia * self._velocities[i] + cognitive_v + social_v

            # 收缩因子 (Clerc & Kennedy 2002)
            if self.constriction:
                phi = self.cognitive + self.social
                if phi > 4.0:
                    kappa = 2.0 / abs(2.0 - phi - np.sqrt(phi ** 2 - 4 * phi))
                else:
                    kappa = 1.0
                self._velocities[i] *= kappa

            # 速度钳制
            if self.v_clamp_ratio > 0:
                v_max = self.v_clamp_ratio * span
                self._velocities[i] = np.clip(self._velocities[i], -v_max, v_max)

            # 位置更新
            self._positions[i] = self._positions[i] + self._velocities[i]

            # 边界处理
            self._positions[i], vel_change = self._handle_boundary(
                self._positions[i], self._velocities[i], low, high
            )
            self._velocities[i] = vel_change

            # 评估
            self._fitnesses[i] = self._evaluate(self._positions[i])
            self._total_evaluations += 1

            # 更新个人最优
            if self._fitnesses[i] > self._personal_best_fitnesses[i]:
                self._personal_best_fitnesses[i] = self._fitnesses[i]
                self._personal_bests[i] = self._positions[i].copy()

                # 更新全局最优
                if self._fitnesses[i] > self._swarm_best_fit:
                    self._swarm_best_fit = self._fitnesses[i]
                    self._swarm_best_pos = self._positions[i].copy()

        self._iteration += 1

        # 自适应惯性权重衰减
        if self.adaptive_inertia:
            self._current_inertia = max(
                self._current_inertia * self.inertia_decay,
                self.inertia_min,
            )

        # 多样性
        pos_std = np.mean(np.std(self._positions, axis=0))
        diversity = pos_std / (np.mean(span) + 1e-16)

        # 平均速度
        avg_v = float(np.mean(np.linalg.norm(self._velocities, axis=1)))

        elapsed = time.time() - self._start_time

        result = IterationResult(
            iteration=self._iteration,
            positions=self._positions.copy(),
            velocities=self._velocities.copy(),
            fitnesses=self._fitnesses.copy(),
            personal_bests=self._personal_bests.copy(),
            personal_best_fitnesses=self._personal_best_fitnesses.copy(),
            best_x=self._swarm_best_pos.copy(),
            best_fitness=self._swarm_best_fit,
            mean_fitness=float(np.mean(self._fitnesses)),
            worst_fitness=float(np.min(self._fitnesses)),
            diversity=float(diversity),
            avg_velocity=avg_v,
            inertia=self._current_inertia,
            elapsed=elapsed,
        )
        self._history.append(result)
        return result

    def run(self, callback=None) -> PSOResult:
        """完整运行。"""
        if self._positions is None:
            self.initialize()
        while True:
            result = self.step()
            if result is None:
                break
            if callback:
                callback(result)
        return self.finalize()

    def finalize(self) -> PSOResult:
        total_time = time.time() - self._start_time
        return PSOResult(
            best_x=self._swarm_best_pos.copy() if self._swarm_best_pos is not None else np.array([]),
            best_energy=-self._swarm_best_fit,
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

    def _evaluate(self, individual):
        """适应度 = -目标函数值（最小化 → 最大化适应度）。"""
        return -float(self.objective_fn(individual))

    def _handle_boundary(self, pos, vel, low, high):
        """边界处理。

        返回 (修正后位置, 修正后速度)。
        """
        if self.boundary == "clip":
            return np.clip(pos, low, high), vel

        elif self.boundary == "reflect":
            # 反射：碰到墙反弹
            out_low = pos < low
            out_high = pos > high
            pos_corrected = pos.copy()
            vel_corrected = vel.copy()
            pos_corrected[out_low] = 2 * low[out_low] - pos[out_low]
            pos_corrected[out_high] = 2 * high[out_high] - pos[out_high]
            vel_corrected[out_low] *= -1
            vel_corrected[out_high] *= -1
            # 确保反射后仍在范围内
            pos_corrected = np.clip(pos_corrected, low, high)
            return pos_corrected, vel_corrected

        elif self.boundary == "random":
            # 随机重置：超出边界随机回范围内
            out_low = pos < low
            out_high = pos > high
            pos_corrected = pos.copy()
            vel_corrected = vel.copy()
            pos_corrected[out_low] = np.random.uniform(low[out_low], high[out_low])
            pos_corrected[out_high] = np.random.uniform(low[out_high], high[out_high])
            vel_corrected[out_low | out_high] *= -0.5
            return pos_corrected, vel_corrected

        elif self.boundary == "absorb":
            # 吸收：碰到墙速度归零
            pos_corrected = np.clip(pos, low, high)
            vel_corrected = vel.copy()
            at_boundary = (pos_corrected <= low) | (pos_corrected >= high)
            vel_corrected[at_boundary] = 0
            return pos_corrected, vel_corrected

        else:
            return np.clip(pos, low, high), vel
