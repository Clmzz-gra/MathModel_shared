"""遗传算法 GUI 工具 — Streamlit 入口。

用法:
    streamlit run main.py
"""

import time, copy
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="遗传算法 GUI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.algorithm import GeneticAlgorithm, GAResult
from core.test_functions import get_function, TEST_FUNCTIONS
from ui.sidebar import build_sidebar
from ui.visualization import (
    render_fitness_plot, render_diversity_plot,
    render_population_scatter_2d, render_progress_bar,
)
from ui.analysis import (
    render_result_cards, render_convergence_stats,
    render_population_stats, render_multi_run_stats,
    render_export_section,
)
from ui.help_content import get_help
from utils.experiment import run_batch_experiment, render_experiment_results


def init_session():
    defaults = {
        "ga_engine": None, "ga_running": False, "ga_paused": False,
        "ga_finished": False, "ga_result": None, "ga_history": [],
        "last_params": None, "multi_run_results": [],
        "experiment_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

st.title("遗传算法 (Genetic Algorithm) — 交互式演示")
st.caption("种群进化 · 选择/交叉/变异 · 实时可视化 · 实验对比")

params = build_sidebar()

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

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "参数概览", "过程可视化", "数值分析", "实验对比", "帮助文档",
])

# ══════════════════════════════════════════════════════════
# Tab 1: 参数概览
# ══════════════════════════════════════════════════════════

with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("当前参数")
        st.markdown(f"""
| 参数 | 值 |
|------|-----|
| 目标函数 | {params['func_name']} |
| 维度 | {params['dim']} |
| 种群大小 | {params['pop_size']} |
| 最大代数 | {params['max_generations']} |
| 选择策略 | {params['selection']} |
| 交叉策略 | {params['crossover']} (率={params['crossover_rate']}) |
| 变异策略 | {params['mutation']} (率={params['mutation_rate']}) |
| 精英保留 | {params['elite_count']} |
| 随机种子 | {params['seed']} |
""")
        if func_info:
            st.markdown(f"**{params['func_name']}:** {func_info['description']}")

    with col_b:
        st.subheader("参数含义")
        st.markdown("""
| 参数 | 含义 |
|------|------|
| 种群大小 | 每代多少个体。越大搜索越充分 |
| 最大代数 | 进化多少代后停止 |
| 选择 | 如何选出优秀父代 |
| 交叉 | 父代基因如何组合 |
| 变异 | 基因如何随机扰动 |
| 精英保留 | 最优个体直接进下一代 |
""")

    st.markdown("---")
    btn_cols = st.columns([2, 1, 1, 1])

    with btn_cols[0]:
        if not st.session_state.ga_running:
            if st.button("开始进化", type="primary", use_container_width=True):
                st.session_state.last_params = copy.deepcopy(params)
                st.session_state.ga_finished = False
                st.session_state.ga_result = None
                st.session_state.ga_history = []

                engine = GeneticAlgorithm(
                    objective_fn=obj_fn, bounds=bounds,
                    pop_size=params["pop_size"],
                    chrom_len=params["dim"],
                    max_generations=params["max_generations"],
                    selection_name=params["selection"],
                    selection_params=params["selection_params"],
                    crossover_name=params["crossover"],
                    crossover_params=params["crossover_params"],
                    crossover_rate=params["crossover_rate"],
                    mutation_name=params["mutation"],
                    mutation_params=params["mutation_params"],
                    mutation_rate=params["mutation_rate"],
                    elite_count=params["elite_count"],
                    opposition_init=params["opposition_init"],
                    de_mutation=params["de_mutation"],
                    early_restart=params["early_restart"],
                    adaptive_mutation=params["adaptive_mutation"],
                    seed=params["seed"],
                )
                engine.initialize()
                st.session_state.ga_engine = engine
                st.session_state.ga_running = True
                st.session_state.ga_paused = False
                st.rerun()

    if st.session_state.ga_running:
        with btn_cols[1]:
            label = "▶ 继续" if st.session_state.ga_paused else "⏸ 暂停"
            if st.button(label, use_container_width=True):
                if st.session_state.ga_paused:
                    st.session_state.ga_engine.resume()
                    st.session_state.ga_paused = False
                else:
                    st.session_state.ga_engine.pause()
                    st.session_state.ga_paused = True
                st.rerun()

        with btn_cols[2]:
            if st.button("⏹ 停止", use_container_width=True):
                st.session_state.ga_engine.stop()
                st.session_state.ga_running = False
                st.session_state.ga_paused = False
                st.session_state.ga_finished = True
                if st.session_state.ga_history:
                    st.session_state.ga_result = st.session_state.ga_engine.finalize()
                st.rerun()

# ══════════════════════════════════════════════════════════
# Tab 2: 过程可视化
# ══════════════════════════════════════════════════════════

with tab2:
    if st.session_state.ga_running and st.session_state.ga_engine:
        engine = st.session_state.ga_engine

        if not st.session_state.ga_paused:
            step_result = engine.step()
            if step_result is None:
                st.session_state.ga_running = False
                st.session_state.ga_finished = True
                st.session_state.ga_result = engine.finalize()
                st.session_state.ga_history = engine._history
                st.rerun()
            else:
                st.session_state.ga_history = engine._history
                if params["poll_interval"] > 0:
                    time.sleep(params["poll_interval"])

        history = st.session_state.ga_history

        if history:
            render_progress_bar(
                current_gen=engine._generation,
                max_gen=params["max_generations"],
                best_fitness=engine._best_fitness,
                mean_fitness=float(np.mean(engine._fitnesses)) if engine._fitnesses is not None else 0,
                elapsed=time.time() - engine._start_time,
                is_paused=st.session_state.ga_paused,
            )

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_fitness_plot(history), use_container_width=True)
            with col_v2:
                st.plotly_chart(render_diversity_plot(history), use_container_width=True)

            if params["dim"] == 2:
                st.plotly_chart(
                    render_population_scatter_2d(history, obj_fn, bounds, params["func_name"]),
                    use_container_width=True,
                )

        if st.session_state.ga_running:
            time.sleep(0.02)
            st.rerun()

    elif st.session_state.ga_finished and st.session_state.ga_result:
        history = st.session_state.ga_history
        st.success("进化完成！")

        final_cols = st.columns(4)
        final_cols[0].metric("最优能量", f"{st.session_state.ga_result.best_energy:.8g}")
        final_cols[1].metric("进化代数", f"{st.session_state.ga_result.total_generations}")
        final_cols[2].metric("总评估", f"{st.session_state.ga_result.total_evaluations}")
        final_cols[3].metric("耗时", f"{st.session_state.ga_result.total_time:.2f}s")

        if history:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_fitness_plot(history), use_container_width=True)
            with col_v2:
                st.plotly_chart(render_diversity_plot(history), use_container_width=True)
            if params["dim"] == 2:
                st.plotly_chart(
                    render_population_scatter_2d(history, obj_fn, bounds, params["func_name"]),
                    use_container_width=True,
                )

        if st.button("🔄 相同参数再跑 4 次（用于统计）"):
            st.session_state.multi_run_results = []
            for i in range(5):
                eng2 = GeneticAlgorithm(
                    objective_fn=obj_fn, bounds=bounds,
                    pop_size=params["pop_size"], max_generations=params["max_generations"],
                    selection_name=params["selection"],
                    crossover_name=params["crossover"],
                    crossover_rate=params["crossover_rate"],
                    mutation_name=params["mutation"],
                    mutation_rate=params["mutation_rate"],
                    elite_count=params["elite_count"],
                    opposition_init=params["opposition_init"],
                    de_mutation=params["de_mutation"],
                    early_restart=params["early_restart"],
                    adaptive_mutation=params["adaptive_mutation"],
                    seed=(params["seed"] or 42) + i * 100,
                )
                eng2.initialize()
                r = eng2.run()
                st.session_state.multi_run_results.append(r)
                st.write(f"运行 {i+1}/5: 最优 = {r.best_energy:.6g}, 耗时 = {r.total_time:.1f}s")
            st.rerun()

    else:
        st.info("请在「参数概览」Tab 中点击「开始进化」按钮。")

# ══════════════════════════════════════════════════════════
# Tab 3: 数值分析
# ══════════════════════════════════════════════════════════

with tab3:
    if st.session_state.ga_result:
        history = st.session_state.ga_history
        render_result_cards(st.session_state.ga_result, func_info, params)
        st.markdown("---")
        render_convergence_stats(history)
        st.markdown("---")
        if history:
            render_population_stats(history)
        if st.session_state.multi_run_results:
            st.markdown("---")
            render_multi_run_stats(st.session_state.multi_run_results)
        st.markdown("---")
        render_export_section(st.session_state.ga_result, history, params["func_name"])
    else:
        st.info("尚未运行优化。请在「参数概览」Tab 中点击「开始进化」。")

# ══════════════════════════════════════════════════════════
# Tab 4: 实验对比
# ══════════════════════════════════════════════════════════

with tab4:
    st.subheader("批量参数对比实验")
    st.markdown("同时测试多组参数组合，对比不同遗传算子的效果。")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        compare_sel = st.multiselect(
            "对比选择策略", list(["Tournament", "Roulette", "Rank"]),
            default=["Tournament", "Roulette"],
        )
        compare_cross = st.multiselect(
            "对比交叉策略", list(["Uniform", "Single-Point", "Arithmetic"]),
            default=["Uniform"],
        )
    with exp_col2:
        compare_mut = st.multiselect(
            "对比变异策略", list(["Gaussian", "Uniform", "Polynomial"]),
            default=["Gaussian"],
        )
        n_runs = st.slider("每组参数运行次数", 1, 10, 3)

    if st.button("运行批量实验", type="primary"):
        if not compare_sel:
            compare_sel = [params["selection"]]
        if not compare_cross:
            compare_cross = [params["crossover"]]
        if not compare_mut:
            compare_mut = [params["mutation"]]

        param_grid = {
            "selection": compare_sel,
            "crossover": compare_cross,
            "mutation": compare_mut,
            "pop_size": [params["pop_size"]],
            "max_generations": [params["max_generations"]],
            "crossover_rate": [params["crossover_rate"]],
            "mutation_rate": [params["mutation_rate"]],
            "elite_count": [params["elite_count"]],
        }
        total = len(compare_sel) * len(compare_cross) * len(compare_mut)
        with st.spinner(f"批量实验 ({total} 组 × {n_runs} 次)..."):
            st.session_state.experiment_results = run_batch_experiment(
                func_name=params["func_name"],
                custom_expr=params["custom_expr"],
                dim=params["dim"], param_grid=param_grid, n_runs=n_runs,
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
