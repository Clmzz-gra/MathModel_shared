"""模拟退火核心引擎。

支持多种降温策略、邻域策略、接受准则，可暂停/恢复/停止。
"""

import time
import threading
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import cooling_schedules as cs
from . import neighborhood as nb


@dataclass
class StepResult:
    """单步迭代结果。"""
    iteration: int
    temperature: float
    current_x: np.ndarray
    current_energy: float
    best_x: np.ndarray
    best_energy: float
    accepted: bool
    acceptance_rate: float = 0.0
    elapsed: float = 0.0
    step_size: Optional[float] = None


@dataclass
class SAResult:
    """SA 运行完整结果。"""
    best_x: np.ndarray
    best_energy: float
    history: list  # List[StepResult] — 完整迭代历史
    total_iterations: int
    total_time: float
    stopped_early: bool = False
    final_temperature: float = 0.0
    total_acceptance_rate: float = 0.0


class SimulatedAnnealing:
    """模拟退火算法引擎。

    使用方式:
        sa = SimulatedAnnealing(fn, bounds, ...)
        # 方式 1: 完整运行
        result = sa.run(callback=my_callback)

        # 方式 2: 手动步进
        sa.start()
        while not sa.done:
            step = sa.step()
            # process step...
        result = sa.finalize()
    """

    def __init__(
        self,
        objective_fn: Callable,
        bounds: np.ndarray,
        T0: float = 1000.0,
        T_end: float = 0.01,
        max_iter: int = 10000,
        cooling_schedule: str = "Geometric",
        cooling_params: Optional[dict] = None,
        neighborhood: str = "Gaussian",
        neighborhood_params: Optional[dict] = None,
        acceptance: str = "Metropolis",
        acceptance_params: Optional[dict] = None,
        reheating: bool = False,
        reheating_trigger: float = 0.01,
        reheating_factor: float = 0.3,
        markov_chain_len: int = 1,
        seed: Optional[int] = None,
    ):
        self.objective_fn = objective_fn
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]

        self.T0 = T0
        self.T_end = T_end
        self.max_iter = max_iter

        self.cooling_schedule = cooling_schedule
        self.cooling_params = cooling_params or {}

        self.neighborhood_name = neighborhood
        self.neighborhood_params = neighborhood_params or {}

        self.acceptance_name = acceptance
        self.acceptance_params = acceptance_params or {}

        self.reheating = reheating
        self.reheating_trigger = reheating_trigger  # 接受率低于此值触发重加热
        self.reheating_factor = reheating_factor      # 重加热到的温度 = T0 * factor

        self.markov_chain_len = markov_chain_len

        if seed is not None:
            np.random.seed(seed)

        # 内部状态
        self._current_x: Optional[np.ndarray] = None
        self._current_energy: float = 0.0
        self._best_x: Optional[np.ndarray] = None
        self._best_energy: float = float("inf")
        self._iteration: int = 0
        self._temperature: float = T0
        self._history: list = []
        self._start_time: float = 0.0
        self._done: bool = False
        self._stopped: bool = False

        # 线程控制
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始不暂停
        self._stop_event = threading.Event()

        # 接受率追踪
        self._accept_history: list = []  # 最近 N 步的接受记录

        # 冷却调度器
        self._cooling_gen = None
        # 自适应步长的额外状态
        self._step_size = None

        # 重加热计数
        self._reheat_count = 0

    # ── 公共接口 ─────────────────────────────────────────────

    def initialize(self):
        """随机初始化起始点。"""
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        self._current_x = np.random.uniform(low, high, self.dim)
        self._current_energy = float(self.objective_fn(self._current_x))
        self._best_x = self._current_x.copy()
        self._best_energy = self._current_energy
        self._iteration = 0
        self._done = False
        self._stopped = False
        self._history = []
        self._accept_history = []
        self._reheat_count = 0

        # 初始化冷却调度器
        merged_cooling = {k: v[0] for k, v in cs.SCHEDULES[self.cooling_schedule]["params"].items()}
        merged_cooling.update(self.cooling_params)
        merged_cooling["max_iter"] = self.max_iter
        self._cooling_gen = cs.SCHEDULES[self.cooling_schedule]["fn"](
            T0=self.T0, **merged_cooling
        )
        self._temperature = self.T0

        # 重置控制信号
        self._pause_event.set()
        self._stop_event.clear()

        self._start_time = time.time()

    def step(self) -> Optional[StepResult]:
        """执行一次迭代。返回 StepResult 或 None（结束）。

        每次调用先从冷却调度器获取下一个温度。
        如果温度低于 T_end 或达到 max_iter 或 stop 被触发，返回 None。
        pause 被触发时阻塞直到 resume。
        """
        if self._done:
            return None

        # 检查暂停
        self._pause_event.wait()

        # 检查停止
        if self._stop_event.is_set():
            self._done = True
            return None

        # 获取下一个温度
        try:
            self._temperature = next(self._cooling_gen)
        except StopIteration:
            self._done = True
            return None

        # 终止条件
        if self._temperature < self.T_end:
            self._done = True
            return None
        if self._iteration >= self.max_iter:
            self._done = True
            return None

        # Markov 链内循环
        accepted_in_chain = 0
        best_in_chain = float("inf")
        best_x_in_chain = None

        for _ in range(self.markov_chain_len):
            # 生成邻域候选
            neigh_fn, neigh_params = nb.get_neighborhood(self.neighborhood_name, **self.neighborhood_params)
            # 处理自适应步长的额外状态
            extra = {}
            if self.neighborhood_name == "Adaptive":
                extra["step_size"] = self._step_size

            new_x, step_info = neigh_fn(self._current_x, self.bounds, **extra)
            new_energy = float(self.objective_fn(new_x))
            delta = new_energy - self._current_energy

            # 接受判断
            accepted = self._accept(delta, self._temperature)
            self._accept_history.append(accepted)
            if len(self._accept_history) > 500:
                self._accept_history.pop(0)

            if accepted:
                self._current_x = new_x
                self._current_energy = new_energy
                accepted_in_chain += 1

                if new_energy < best_in_chain:
                    best_in_chain = new_energy
                    best_x_in_chain = new_x.copy()

            # 更新自适应步长
            if self.neighborhood_name == "Adaptive" and "next_step" in step_info:
                self._step_size = step_info["next_step"]

        # 更新全局最优
        if best_x_in_chain is not None and best_in_chain < self._best_energy:
            self._best_x = best_x_in_chain.copy()
            self._best_energy = best_in_chain

        self._iteration += 1

        # 重加热检查
        if self.reheating and len(self._accept_history) >= 100:
            recent_rate = sum(self._accept_history[-100:]) / 100
            if recent_rate < self.reheating_trigger and self._reheat_count < 5:
                self._temperature = self.T0 * self.reheating_factor
                self._cooling_gen = cs.SCHEDULES[self.cooling_schedule]["fn"](
                    T0=self._temperature, **merged_cooling
                )
                self._reheat_count += 1

        # 计算接受率
        total_accept_rate = (
            sum(self._accept_history) / len(self._accept_history)
            if self._accept_history else 0.0
        )

        elapsed = time.time() - self._start_time

        return StepResult(
            iteration=self._iteration,
            temperature=self._temperature,
            current_x=self._current_x.copy(),
            current_energy=self._current_energy,
            best_x=self._best_x.copy(),
            best_energy=self._best_energy,
            accepted=accepted_in_chain > 0,
            acceptance_rate=total_accept_rate,
            elapsed=elapsed,
            step_size=(
                np.mean(self._step_size) if self._step_size is not None else None
            ),
        )

    def run(self, callback: Optional[Callable] = None, poll_interval: float = 0.0) -> SAResult:
        """完整运行 SA 算法。

        Args:
            callback: 每步回调 callback(StepResult)
            poll_interval: 每步之间的间隔秒数（用于展示效果）
        Returns:
            SAResult
        """
        if self._current_x is None:
            self.initialize()

        while True:
            result = self.step()
            if result is None:
                break
            self._history.append(result)
            if callback:
                callback(result)
            if poll_interval > 0:
                time.sleep(poll_interval)

        return self.finalize()

    def finalize(self) -> SAResult:
        """从已完成的运行中生成 SAResult。"""
        total_time = time.time() - self._start_time
        total_accept_rate = (
            sum(self._accept_history) / len(self._accept_history)
            if self._accept_history else 0.0
        )
        return SAResult(
            best_x=self._best_x.copy() if self._best_x is not None else np.array([]),
            best_energy=self._best_energy,
            history=self._history.copy(),
            total_iterations=self._iteration,
            total_time=total_time,
            stopped_early=self._stopped,
            final_temperature=self._temperature,
            total_acceptance_rate=total_accept_rate,
        )

    # ── 控制接口 ─────────────────────────────────────────────

    def pause(self):
        """暂停优化。当前迭代完成后阻塞。"""
        self._pause_event.clear()

    def resume(self):
        """恢复优化。"""
        self._pause_event.set()

    def stop(self):
        """请求停止优化。"""
        self._stop_event.set()
        self._pause_event.set()  # 解除暂停以便检查 stop

    @property
    def done(self):
        return self._done

    @property
    def is_paused(self):
        return not self._pause_event.is_set()

    # ── 内部方法 ─────────────────────────────────────────────

    def _accept(self, delta: float, T: float) -> bool:
        """判断是否接受新解。

        delta = new_energy - current_energy
        T = 当前温度
        """
        if delta <= 0:
            return True  # 更优解直接接受

        if self.acceptance_name == "Metropolis":
            # 标准 Metropolis 准则
            if T < 1e-16:
                return False
            return np.random.random() < math.exp(-delta / T)

        elif self.acceptance_name == "Threshold":
            # 阈值接受：只看 delta 是否在阈值内，不依赖温度概率
            threshold = self.acceptance_params.get("threshold", 1.0)
            return delta < threshold

        elif self.acceptance_name == "Tsallis":
            # Tsallis 接受准则（广义统计力学）
            q = self.acceptance_params.get("q", 1.5)
            if q <= 1:
                # q→1 退化为 Metropolis
                return np.random.random() < math.exp(-delta / T) if T > 1e-16 else False
            # P = [1 - (1-q)*delta/T]^(1/(1-q))
            arg = 1 - (1 - q) * delta / max(T, 1e-16)
            if arg <= 0:
                return False
            prob = arg ** (1 / (1 - q))
            return np.random.random() < prob

        else:
            return False


# ── 接受准则注册表 ──────────────────────────────────────────

ACCEPTANCE_CRITERIA = {
    "Metropolis": {
        "params": {},
        "description": "标准 Metropolis 准则: P(accept) = exp(-ΔE/T)。差解以概率接受，概率随温度降低而减小。",
    },
    "Threshold": {
        "params": {"threshold": (1.0, "接受阈值。ΔE 小于此值直接接受，大于此值拒绝")},
        "description": "阈值接受：不依赖温度的概率计算，差解的 ΔE 在阈值内直接接受。简单但不保证收敛到全局最优。",
    },
    "Tsallis": {
        "params": {"q": (1.5, "Tsallis 熵参数。q=1 退化为 Metropolis，q>1 更易接受差解")},
        "description": "基于 Tsallis 广义统计力学的接受准则。q>1 时比 Metropolis 更激进地接受差解。",
    },
}


# ── 初始温度自动标定 ──────────────────────────────────────

def auto_calibrate_T0(objective_fn, bounds, target_rate=0.8, n_samples=100):
    """自动计算使初始接受率约等于 target_rate 的 T₀。

    基于 AL-002 的诊断指标：初始接受率应约 80%。

    方法：随机采样 n_samples 个点，计算每对之间的能量差分布，
    取正差的中位数，解 T₀ = -median(ΔE⁺) / ln(target_rate)。
    """
    low, high = bounds[:, 0], bounds[:, 1]
    dim = bounds.shape[0]

    samples = np.random.uniform(low, high, (n_samples, dim))
    energies = np.array([float(objective_fn(x)) for x in samples])

    # 随机配对，计算能量差
    deltas = []
    for _ in range(n_samples // 2):
        i, j = np.random.choice(n_samples, size=2, replace=False)
        de = energies[j] - energies[i]
        if de > 0:
            deltas.append(de)

    deltas = np.array(deltas)
    if len(deltas) == 0:
        return 1000.0  # fallback

    median_de = np.median(deltas)
    # 解: target_rate = exp(-median_de / T₀) → T₀ = -median_de / ln(target_rate)
    T0 = -median_de / math.log(target_rate) if target_rate > 0 else 1000.0
    return max(T0, 1.0)  # 至少 1.0
