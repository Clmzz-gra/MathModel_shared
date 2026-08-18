"""遗传算法核心引擎。实数编码，支持多种选择/交叉/变异算子。"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import selection as sel
from . import crossover as cross
from . import mutation as mut


@dataclass
class GenerationResult:
    """一代进化结果。"""
    generation: int
    population: np.ndarray
    fitnesses: np.ndarray
    best_x: np.ndarray
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    diversity: float
    elite_count: int = 0
    elapsed: float = 0.0


@dataclass
class GAResult:
    """GA 完整运行结果。"""
    best_x: np.ndarray
    best_energy: float  # 原始目标函数值（最小化）
    history: list       # List[GenerationResult]
    total_generations: int
    total_evaluations: int
    total_time: float
    stopped_early: bool = False


class GeneticAlgorithm:
    """实数编码遗传算法。

    chrom_len 直接等于问题维度。每个个体就是一个浮点向量。
    最小化目标函数，适应度 = -f(x)。
    """

    def __init__(
        self,
        objective_fn: Callable,
        bounds: np.ndarray,
        pop_size: int = 100,
        chrom_len: int = 2,
        max_generations: int = 200,
        selection_name: str = "Tournament",
        selection_params: Optional[dict] = None,
        crossover_name: str = "Uniform",
        crossover_params: Optional[dict] = None,
        crossover_rate: float = 0.8,
        mutation_name: str = "Gaussian",
        mutation_params: Optional[dict] = None,
        mutation_rate: float = 0.1,
        elite_count: int = 2,
        # 来自卡片的改进选项
        opposition_init: bool = False,
        de_mutation: bool = False,
        de_f: float = 0.8,
        early_restart: bool = False,
        early_restart_threshold: float = 0.1,
        adaptive_mutation: bool = False,
        adaptive_mut_decay: float = 0.98,
        adaptive_mut_min: float = 0.001,
        seed: Optional[int] = None,
    ):
        self.objective_fn = objective_fn
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]
        self.pop_size = pop_size
        self.chrom_len = chrom_len  # 应等于 self.dim
        self.max_generations = max_generations

        self.selection_name = selection_name
        self.selection_params = selection_params or {}

        self.crossover_name = crossover_name
        self.crossover_params = crossover_params or {}
        self.crossover_rate = crossover_rate

        self.mutation_name = mutation_name
        self.mutation_params = mutation_params or {}
        self.mutation_rate = mutation_rate

        self.elite_count = elite_count

        # 改进选项
        self.opposition_init = opposition_init
        self.de_mutation = de_mutation
        self.de_f = de_f
        self.early_restart = early_restart
        self.early_restart_threshold = early_restart_threshold
        self.adaptive_mutation = adaptive_mutation
        self.adaptive_mut_decay = adaptive_mut_decay
        self.adaptive_mut_min = adaptive_mut_min
        self._current_mutation_rate = mutation_rate

        self._restart_count = 0

        if seed is not None:
            np.random.seed(seed)

        # 内部状态
        self._population: Optional[np.ndarray] = None
        self._fitnesses: Optional[np.ndarray] = None
        self._generation: int = 0
        self._best_x: Optional[np.ndarray] = None
        self._best_fitness: float = -float("inf")  # 适应度，不是能量
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
        """随机初始化种群（可选反向初始化）。"""
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        self._restart_count = 0
        self._current_mutation_rate = self.mutation_rate

        if self.opposition_init:
            # 反向初始化 (AL-012): 生成 2n 个体，取最优 n 个
            n = self.pop_size
            pop = np.random.uniform(low, high, (n, self.dim))
            opp = low + high - pop  # 反向个体
            combined = np.vstack([pop, opp])
            fits = np.array([self._evaluate(ind) for ind in combined])
            best_idx = np.argsort(fits)[-n:]
            self._population = combined[best_idx].copy()
            self._fitnesses = fits[best_idx]
            self._total_evaluations = 2 * n
        else:
            self._population = np.random.uniform(low, high, (self.pop_size, self.dim))
            self._fitnesses = np.array([self._evaluate(ind) for ind in self._population])
            self._total_evaluations = self.pop_size

        self._generation = 0
        self._done = False
        self._stopped = False
        self._history = []

        best_idx = np.argmax(self._fitnesses)
        self._best_x = self._population[best_idx].copy()
        self._best_fitness = self._fitnesses[best_idx]

        self._pause_event.set()
        self._stop_event.clear()
        self._start_time = time.time()

    def step(self) -> Optional[GenerationResult]:
        """执行一代进化。返回 GenerationResult 或 None。"""
        if self._done:
            return None

        self._pause_event.wait()

        if self._stop_event.is_set():
            self._done = True
            return None

        if self._generation >= self.max_generations:
            self._done = True
            return None

        pop = self._population
        fits = self._fitnesses
        n = self.pop_size

        # 1. 精英保留
        elites = np.zeros((0, self.dim))
        if self.elite_count > 0:
            elite_idx = np.argsort(fits)[-self.elite_count:]
            elites = pop[elite_idx].copy()

        # 2. 选择父代
        selector_fn, sel_params = sel.get_selector(self.selection_name)
        sel_kwargs = {k: v[0] for k, v in sel.SELECTION_METHODS[self.selection_name]["params"].items()}
        sel_kwargs.update(self.selection_params)
        n_parents = n - self.elite_count
        # 确保偶数
        if n_parents % 2 != 0:
            n_parents += 1
        parents = selector_fn(fits, pop, n_parents, **sel_kwargs)

        # 3. 交叉
        crossover_fn, cross_params = cross.get_crossover(self.crossover_name)
        cross_kwargs = {k: v[0] for k, v in cross.CROSSOVER_METHODS[self.crossover_name]["params"].items()}
        cross_kwargs.update(self.crossover_params)
        offspring = np.zeros((n_parents, self.dim))
        for i in range(0, n_parents, 2):
            if np.random.random() < self.crossover_rate:
                c1, c2 = crossover_fn(parents[i], parents[i + 1], **cross_kwargs)
                offspring[i] = c1
                offspring[i + 1] = c2
            else:
                offspring[i] = parents[i].copy()
                offspring[i + 1] = parents[i + 1].copy()

        # 4. 变异
        if self.de_mutation:
            # DE/best/2 变异 (AL-012): 用种群差分信息生成全部新个体
            # 在交叉后的 offspring 基础上，对整个子代池进行 DE 式替换
            offspring = mut.de_mutation(
                offspring, np.zeros(len(offspring)), self._best_x,
                self.bounds, F=self.de_f
            )
        else:
            mutator_fn, mut_params = mut.get_mutator(self.mutation_name)
            mut_kwargs = {k: v[0] for k, v in mut.MUTATION_METHODS[self.mutation_name]["params"].items()}
            mut_kwargs.update(self.mutation_params)
            mut_kwargs["mutation_rate"] = self._current_mutation_rate
            for i in range(len(offspring)):
                offspring[i] = mutator_fn(offspring[i], self.bounds, **mut_kwargs)
                offspring[i] = np.clip(offspring[i], self.bounds[:, 0], self.bounds[:, 1])

        # 5. 组合新种群
        new_pop = np.vstack([elites, offspring]) if len(elites) > 0 else offspring
        new_pop = new_pop[:n]  # 确保恰好 pop_size

        # 6. 评估
        self._population = new_pop
        self._fitnesses = np.array([self._evaluate(ind) for ind in new_pop])
        self._total_evaluations += len(new_pop)
        self._generation += 1

        # 自适应变异率衰减 (AL-003)
        if self.adaptive_mutation:
            self._current_mutation_rate = max(
                self._current_mutation_rate * self.adaptive_mut_decay,
                self.adaptive_mut_min,
            )

        # 更新最优
        gen_best_idx = np.argmax(self._fitnesses)
        if self._fitnesses[gen_best_idx] > self._best_fitness:
            self._best_x = self._population[gen_best_idx].copy()
            self._best_fitness = self._fitnesses[gen_best_idx]

        # 计算多样性
        pop_std = np.mean(np.std(self._population, axis=0))
        diversity = pop_std / (np.mean(self.bounds[:, 1] - self.bounds[:, 0]) + 1e-16)

        # 早熟检测与重启 (AL-003 C234): 最优个体占比 > 阈值 → 注入随机个体
        if self.early_restart and self._generation > self.max_generations // 4:
            best_count = np.sum(np.all(np.abs(self._population - self._best_x) < 1e-8, axis=1))
            if best_count / n > self.early_restart_threshold:
                low, high = self.bounds[:, 0], self.bounds[:, 1]
                n_replace = max(n // 3, 10)
                self._population[-n_replace:] = np.random.uniform(low, high, (n_replace, self.dim))
                self._fitnesses[-n_replace:] = [self._evaluate(ind) for ind in self._population[-n_replace:]]
                self._total_evaluations += n_replace
                self._restart_count += 1

        elapsed = time.time() - self._start_time

        result = GenerationResult(
            generation=self._generation,
            population=self._population.copy(),
            fitnesses=self._fitnesses.copy(),
            best_x=self._best_x.copy(),
            best_fitness=self._best_fitness,
            mean_fitness=float(np.mean(self._fitnesses)),
            worst_fitness=float(np.min(self._fitnesses)),
            diversity=float(diversity),
            elite_count=self.elite_count,
            elapsed=elapsed,
        )
        self._history.append(result)
        return result

    def run(self, callback=None) -> GAResult:
        """完整运行。"""
        if self._population is None:
            self.initialize()
        while True:
            result = self.step()
            if result is None:
                break
            if callback:
                callback(result)
        return self.finalize()

    def finalize(self) -> GAResult:
        total_time = time.time() - self._start_time
        # best_energy = -best_fitness（恢复原始目标函数值）
        return GAResult(
            best_x=self._best_x.copy() if self._best_x is not None else np.array([]),
            best_energy=-self._best_fitness,
            history=self._history.copy(),
            total_generations=self._generation,
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
