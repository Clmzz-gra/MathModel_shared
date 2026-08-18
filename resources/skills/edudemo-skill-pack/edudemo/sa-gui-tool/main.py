"""模拟退火算法 GUI 工具 — Streamlit 入口。

用法:
    streamlit run main.py
"""

import time
import threading
import copy

import numpy as np
import streamlit as st

# 页面配置必须放在最前面
st.set_page_config(
    page_title="模拟退火算法 GUI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.annealing import SimulatedAnnealing, SAResult, auto_calibrate_T0
from core.test_functions import get_function, TEST_FUNCTIONS
from ui.sidebar import build_sidebar
from ui.visualization import (
    render_convergence_plot,
    render_temperature_plot,
    render_acceptance_plot,
    render_search_path_2d,
    render_search_path_3d,
    render_progress_bar,
)
from ui.analysis import (
    render_result_cards,
    render_convergence_stats,
    render_acceptance_breakdown,
    render_multi_run_stats,
    render_export_section,
)
from ui.help_content import get_help
from utils.experiment import run_batch_experiment, render_experiment_results


# ── 会话状态初始化 ────────────────────────────────────────

def init_session():
    defaults = {
        "sa_engine": None,
        "sa_running": False,
        "sa_paused": False,
        "sa_finished": False,
        "sa_result": None,
        "sa_history": [],
        "sa_thread": None,
        "last_params": None,
        "multi_run_results": [],
        "experiment_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ── 标题 ──────────────────────────────────────────────────

st.title("模拟退火算法 (Simulated Annealing) — 交互式演示")
st.caption("探索参数对全局优化过程的影响 | 实时可视化 | 结果分析 | 批量实验对比")

# ── 侧边栏参数 ────────────────────────────────────────────

params = build_sidebar()

# 判断参数是否变化
params_changed = (st.session_state.last_params != params)
if params_changed and st.session_state.sa_running:
    st.sidebar.warning("参数已更改。下次运行将使用新参数。")

# ── 目标函数构建 ──────────────────────────────────────────

func_info = None
if params["custom_expr"]:
    local_ns = {"np": np}
    obj_fn = lambda x, expr=params["custom_expr"], ns=local_ns: float(eval(expr, ns, {"x": x}))  # noqa
    bounds = np.array([[-10.0, 10.0]] * params["dim"])
else:
    func_info = get_function(params["func_name"])
    obj_fn = func_info["fn"]
    bounds = func_info["bounds"]
    if bounds.shape[0] < params["dim"]:
        bounds = np.tile(bounds[0], (params["dim"], 1))

# ── 主面板 Tabs ──────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "参数概览", "过程可视化", "数值分析", "实验对比", "帮助文档"
])

# ══════════════════════════════════════════════════════════
# Tab 1: 参数概览
# ══════════════════════════════════════════════════════════

with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("当前参数")
        param_summary = f"""
| 参数 | 值 |
|------|-----|
| 目标函数 | {params['func_name']} ({'自定义' if params['custom_expr'] else '内置'}) |
| 维度 | {params['dim']} |
| T₀ (初始温度) | {params['T0']} |
| T_end (终止温度) | {params['T_end']} |
| α (降温速率) | {params['alpha']} |
| N (最大迭代) | {params['max_iter']} |
| 冷却策略 | {params['cooling']} |
| 邻域策略 | {params['neigh']} |
| 接受准则 | {params['accept']} |
| Markov 链长度 | {params['markov_chain_len']} |
| 随机种子 | {params['seed']} |
| 重加热 | {'启用' if params['reheating'] else '关闭'} |
"""
        st.markdown(param_summary)

        if func_info:
            st.markdown(f"**{params['func_name']} 说明:** {func_info['description']}")
            if params["dim"] <= 3:
                gx = func_info["global_x_2d"]
                st.markdown(f"已知全局最优: f({', '.join(f'{v:.4f}' for v in gx[:params['dim']])}) = {func_info['global_min']:.6g}")

    with col_b:
        st.subheader("参数含义")
        explanations = """
| 参数 | 含义 |
|------|------|
| T₀ | 起始温度——越高，初期探索范围越大 |
| T_end | 终止温度——越低，最终解越精确 |
| α | 降温速率——接近1时降温慢，搜索充分 |
| N | 迭代上限——防止无限循环 |
| 冷却策略 | 控制温度如何下降 |
| 邻域策略 | 控制如何从当前解生成候选解 |
| 接受准则 | 控制差解的接受概率 |
| Markov 链长 | 每个温度下的搜索次数 |
| 重加热 | 低温停滞时自动升温 |
"""
        st.markdown(explanations)

    # 按钮区
    st.markdown("---")
    btn_cols = st.columns([2, 1, 1, 1])

    with btn_cols[0]:
        if not st.session_state.sa_running:
            if st.button("开始优化", type="primary", use_container_width=True):
                st.session_state.last_params = copy.deepcopy(params)
                st.session_state.sa_finished = False
                st.session_state.sa_result = None
                st.session_state.sa_history = []

                # 初始温度自动标定（来自 AL-002）
                T0_val = params["T0"]
                if params["auto_T0"]:
                    T0_val = auto_calibrate_T0(obj_fn, bounds)

                engine = SimulatedAnnealing(
                    objective_fn=obj_fn,
                    bounds=bounds,
                    T0=T0_val,
                    T_end=params["T_end"],
                    max_iter=params["max_iter"],
                    cooling_schedule=params["cooling"],
                    cooling_params=params["cooling_params"],
                    neighborhood=params["neigh"],
                    neighborhood_params=params["neigh_params"],
                    acceptance=params["accept"],
                    acceptance_params=params["accept_params"],
                    reheating=params["reheating"],
                    reheating_trigger=params["reheat_trigger"],
                    reheating_factor=params["reheat_factor"],
                    markov_chain_len=params["markov_chain_len"],
                    seed=params["seed"],
                )
                engine.initialize()
                st.session_state.sa_engine = engine
                st.session_state.sa_running = True
                st.session_state.sa_paused = False
                st.session_state.sa_thread = None
                st.rerun()

    if st.session_state.sa_running:
        with btn_cols[1]:
            if st.session_state.sa_paused:
                if st.button("▶ 继续", use_container_width=True):
                    st.session_state.sa_engine.resume()
                    st.session_state.sa_paused = False
                    st.rerun()
            else:
                if st.button("⏸ 暂停", use_container_width=True):
                    st.session_state.sa_engine.pause()
                    st.session_state.sa_paused = True
                    st.rerun()

        with btn_cols[2]:
            if st.button("⏹ 停止", use_container_width=True):
                st.session_state.sa_engine.stop()
                st.session_state.sa_running = False
                st.session_state.sa_paused = False
                st.session_state.sa_finished = True
                if st.session_state.sa_history:
                    st.session_state.sa_result = st.session_state.sa_engine.finalize()
                st.rerun()

    # ══════════════════════════════════════════════════════
    # Tab 2: 过程可视化
    # ══════════════════════════════════════════════════════

    with tab2:
        if st.session_state.sa_running and st.session_state.sa_engine:
            engine = st.session_state.sa_engine

            # 批量执行多步再刷新界面，减少 Streamlit 重跑开销
            if not st.session_state.sa_paused:
                batch_size = max(params["max_iter"] // 50, 20)  # 约 50 次刷新完成
                for _ in range(batch_size):
                    step_result = engine.step()
                    if step_result is None:
                        st.session_state.sa_running = False
                        st.session_state.sa_finished = True
                        st.session_state.sa_result = engine.finalize()
                        st.session_state.sa_history = engine._history
                        st.rerun()
                    engine._history.append(step_result)
                st.session_state.sa_history = engine._history

                # 根据 poll_interval 控制速度（用于教学慢放）
                if params["poll_interval"] > 0:
                    time.sleep(params["poll_interval"])

            history = st.session_state.sa_history

            # 进度条
            render_progress_bar(
                current_iter=engine._iteration,
                max_iter=params["max_iter"],
                temperature=engine._temperature,
                best_energy=engine._best_energy if hasattr(engine, "_best_energy") else 0,
                elapsed=time.time() - engine._start_time,
                is_paused=st.session_state.sa_paused,
            )

            # 图表
            if history:
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.plotly_chart(render_convergence_plot(history), use_container_width=True)
                with col_v2:
                    st.plotly_chart(render_temperature_plot(history), use_container_width=True)

                st.plotly_chart(render_acceptance_plot(history), use_container_width=True)

                # 2D/3D 路径 — 每 3 次刷新才画（省计算）
                if "path_update_count" not in st.session_state:
                    st.session_state.path_update_count = 0
                st.session_state.path_update_count += 1
                if st.session_state.path_update_count % 3 == 0:
                    if params["dim"] == 2:
                        st.plotly_chart(
                            render_search_path_2d(history, obj_fn, bounds, params["func_name"]),
                            use_container_width=True,
                        )
                    elif params["dim"] >= 3:
                        st.plotly_chart(
                            render_search_path_3d(history, obj_fn, bounds, params["dim"], params["func_name"]),
                            use_container_width=True,
                        )

            # 继续轮询
            if st.session_state.sa_running:
                time.sleep(0.02)
                st.rerun()

        elif st.session_state.sa_finished and st.session_state.sa_result:
            history = st.session_state.sa_history
            st.success("优化完成！")

            final_cols = st.columns(4)
            final_cols[0].metric("最优能量", f"{st.session_state.sa_result.best_energy:.8g}")
            final_cols[1].metric("总迭代", f"{st.session_state.sa_result.total_iterations}")
            final_cols[2].metric("总耗时", f"{st.session_state.sa_result.total_time:.2f}s")
            final_cols[3].metric("接受率", f"{st.session_state.sa_result.total_acceptance_rate:.2%}")

            if history:
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.plotly_chart(render_convergence_plot(history), use_container_width=True)
                with col_v2:
                    st.plotly_chart(render_temperature_plot(history), use_container_width=True)

                st.plotly_chart(render_acceptance_plot(history), use_container_width=True)

                if params["dim"] == 2:
                    st.plotly_chart(
                        render_search_path_2d(history, obj_fn, bounds, params["func_name"]),
                        use_container_width=True,
                    )

            # 多次运行按钮
            if st.button("🔄 用相同参数再跑 4 次（用于统计分析）"):
                st.session_state.multi_run_results = []
                for i in range(5):
                    T0_multi = auto_calibrate_T0(obj_fn, bounds) if params["auto_T0"] else params["T0"]
                    engine2 = SimulatedAnnealing(
                        objective_fn=obj_fn,
                        bounds=bounds,
                        T0=T0_multi,
                        T_end=params["T_end"],
                        max_iter=params["max_iter"],
                        cooling_schedule=params["cooling"],
                        cooling_params=params["cooling_params"],
                        neighborhood=params["neigh"],
                        neighborhood_params=params["neigh_params"],
                        acceptance=params["accept"],
                        acceptance_params=params["accept_params"],
                        reheating=params["reheating"],
                        reheating_trigger=params["reheat_trigger"],
                        reheating_factor=params["reheat_factor"],
                        markov_chain_len=params["markov_chain_len"],
                        seed=(params["seed"] or 42) + i * 100,
                    )
                    engine2.initialize()
                    result = engine2.run()
                    st.session_state.multi_run_results.append(result)
                    st.write(f"运行 {i+1}/5: 最优能量 = {result.best_energy:.6g}, 耗时 = {result.total_time:.1f}s")
                st.rerun()

        else:
            st.info("请在「参数概览」Tab 中点击「开始优化」按钮。")

# ══════════════════════════════════════════════════════════
# Tab 3: 数值分析
# ══════════════════════════════════════════════════════════

with tab3:
    if st.session_state.sa_result:
        history = st.session_state.sa_history

        render_result_cards(st.session_state.sa_result, func_info, params)

        st.markdown("---")
        if history:
            render_convergence_stats(history, func_info)

        st.markdown("---")
        if history:
            render_acceptance_breakdown(history)

        # 多次运行统计
        if st.session_state.multi_run_results:
            st.markdown("---")
            render_multi_run_stats(st.session_state.multi_run_results)

        # 导出
        st.markdown("---")
        render_export_section(
            st.session_state.sa_result, history,
            params.get("func_name", "custom"),
        )

    else:
        st.info("尚未运行优化。请先在「参数概览」Tab 中点击「开始优化」。")

# ══════════════════════════════════════════════════════════
# Tab 4: 实验对比
# ══════════════════════════════════════════════════════════

with tab4:
    st.subheader("批量参数对比实验")

    st.markdown("""
    在此 Tab 中，可以同时测试多组参数组合，对比不同策略的效果。
    每组参数将运行多次以评估稳定性。
    """)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        compare_cooling = st.multiselect(
            "对比冷却策略",
            options=["Geometric", "Linear", "Logarithmic", "Exponential"],
            default=["Geometric", "Linear"],
        )
        compare_neigh = st.multiselect(
            "对比邻域策略",
            options=["Gaussian", "Uniform", "Cauchy"],
            default=["Gaussian"],
        )

    with exp_col2:
        compare_accept = st.multiselect(
            "对比接受准则",
            options=["Metropolis", "Threshold", "Tsallis"],
            default=["Metropolis"],
        )
        n_runs = st.slider("每组参数运行次数", 1, 10, 3,
                          help="次数越多统计越可靠，但耗时越长。")

    if st.button("运行批量实验", type="primary") and (compare_cooling or compare_neigh or compare_accept):
        # 构建参数网格
        # 如果某项未选择，使用当前参数面板中的值
        if not compare_cooling:
            compare_cooling = [params["cooling"]]
        if not compare_neigh:
            compare_neigh = [params["neigh"]]
        if not compare_accept:
            compare_accept = [params["accept"]]

        param_grid = {
            "cooling": compare_cooling,
            "neighborhood": compare_neigh,
            "acceptance": compare_accept,
            "T0": [params["T0"]],
            "alpha": [params["alpha"]],
            "max_iter": [params["max_iter"]],
        }

        with st.spinner(f"正在运行批量实验 ({len(compare_cooling)*len(compare_neigh)*len(compare_accept)} 组参数 × {n_runs} 次)..."):
            st.session_state.experiment_results = run_batch_experiment(
                func_name=params["func_name"],
                custom_expr=params["custom_expr"],
                dim=params["dim"],
                param_grid=param_grid,
                n_runs=n_runs,
            )

        st.success("批量实验完成！")
        st.rerun()

    if st.session_state.experiment_results:
        render_experiment_results(st.session_state.experiment_results)
    else:
        st.info("选择要对比的策略并点击「运行批量实验」。")

# ══════════════════════════════════════════════════════════
# Tab 5: 帮助文档
# ══════════════════════════════════════════════════════════

with tab5:
    st.markdown(get_help())
