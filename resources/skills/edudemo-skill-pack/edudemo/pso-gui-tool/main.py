"""粒子群优化 GUI 工具 — Streamlit 入口。

用法:
    streamlit run main.py
"""

import time, copy
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="粒子群优化 GUI",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.algorithm import ParticleSwarmOptimizer, PSOResult
from core.test_functions import get_function, TEST_FUNCTIONS
from ui.sidebar import build_sidebar
from ui.visualization import (
    render_fitness_plot, render_diversity_plot,
    render_particle_scatter_2d, render_progress_bar,
)
from ui.analysis import (
    render_result_cards, render_convergence_stats,
    render_swarm_stats, render_multi_run_stats,
    render_export_section,
)
from ui.help_content import get_help
from utils.experiment import run_batch_experiment, render_experiment_results


def init_session():
    defaults = {
        "pso_engine": None, "pso_running": False, "pso_paused": False,
        "pso_finished": False, "pso_result": None, "pso_history": [],
        "last_params": None, "multi_run_results": [],
        "experiment_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

st.title("粒子群优化 (Particle Swarm Optimization) — 交互式演示")
st.caption("粒子飞行 · 速度更新 · 拓扑邻域 · 实时可视化 · 实验对比")

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
| 粒子数 | {params['swarm_size']} |
| 最大迭代 | {params['max_iterations']} |
| 惯性权重 w | {params['inertia']} |
| 认知系数 c₁ | {params['cognitive']} |
| 社会系数 c₂ | {params['social']} |
| 拓扑结构 | {params['topology']} |
| 边界处理 | {params['boundary']} |
| 速度钳制 | {params['v_clamp_ratio']} |
| 收缩因子 | {'是' if params['constriction'] else '否'} |
| 自适应惯性 | {'是' if params['adaptive_inertia'] else '否'} |
| 随机种子 | {params['seed']} |
""")
        if func_info:
            st.markdown(f"**{params['func_name']}:** {func_info['description']}")

    with col_b:
        st.subheader("参数含义")
        st.markdown("""
| 参数 | 含义 |
|------|------|
| 粒子数 | 搜索空间中飞行的粒子数量 |
| 惯性权重 w | 保持原飞行方向的能力 |
| 认知系数 c₁ | 向自己最佳位置学习的强度 |
| 社会系数 c₂ | 向群体最佳位置学习的强度 |
| 拓扑结构 | 粒子间信息共享的邻域关系 |
| 边界处理 | 粒子飞出搜索空间时的处理 |
| 速度钳制 | 限制最大飞行速度 |
""")

    st.markdown("---")
    btn_cols = st.columns([2, 1, 1, 1])

    with btn_cols[0]:
        if not st.session_state.pso_running:
            if st.button("开始优化", type="primary", use_container_width=True):
                st.session_state.last_params = copy.deepcopy(params)
                st.session_state.pso_finished = False
                st.session_state.pso_result = None
                st.session_state.pso_history = []

                engine = ParticleSwarmOptimizer(
                    objective_fn=obj_fn, bounds=bounds,
                    swarm_size=params["swarm_size"],
                    max_iterations=params["max_iterations"],
                    inertia=params["inertia"],
                    cognitive=params["cognitive"],
                    social=params["social"],
                    topology_name=params["topology"],
                    topology_params=params["topology_params"],
                    boundary=params["boundary"],
                    v_clamp_ratio=params["v_clamp_ratio"],
                    constriction=params["constriction"],
                    adaptive_inertia=params["adaptive_inertia"],
                    inertia_decay=params["inertia_decay"],
                    inertia_min=params["inertia_min"],
                    seed=params["seed"],
                )
                engine.initialize()
                st.session_state.pso_engine = engine
                st.session_state.pso_running = True
                st.session_state.pso_paused = False
                st.rerun()

    if st.session_state.pso_running:
        with btn_cols[1]:
            label = "▶ 继续" if st.session_state.pso_paused else "⏸ 暂停"
            if st.button(label, use_container_width=True):
                if st.session_state.pso_paused:
                    st.session_state.pso_engine.resume()
                    st.session_state.pso_paused = False
                else:
                    st.session_state.pso_engine.pause()
                    st.session_state.pso_paused = True
                st.rerun()

        with btn_cols[2]:
            if st.button("⏹ 停止", use_container_width=True):
                st.session_state.pso_engine.stop()
                st.session_state.pso_running = False
                st.session_state.pso_paused = False
                st.session_state.pso_finished = True
                if st.session_state.pso_history:
                    st.session_state.pso_result = st.session_state.pso_engine.finalize()
                st.rerun()

# ══════════════════════════════════════════════════════════
# Tab 2: 过程可视化
# ══════════════════════════════════════════════════════════

with tab2:
    if st.session_state.pso_running and st.session_state.pso_engine:
        engine = st.session_state.pso_engine

        if not st.session_state.pso_paused:
            step_result = engine.step()
            if step_result is None:
                st.session_state.pso_running = False
                st.session_state.pso_finished = True
                st.session_state.pso_result = engine.finalize()
                st.session_state.pso_history = engine._history
                st.rerun()
            else:
                st.session_state.pso_history = engine._history
                if params["poll_interval"] > 0:
                    time.sleep(params["poll_interval"])

        history = st.session_state.pso_history

        if history:
            render_progress_bar(
                current_iter=engine._iteration,
                max_iter=params["max_iterations"],
                best_fitness=engine._swarm_best_fit,
                mean_fitness=float(np.mean(engine._fitnesses)) if engine._fitnesses is not None else 0,
                elapsed=time.time() - engine._start_time,
                is_paused=st.session_state.pso_paused,
            )

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_fitness_plot(history), use_container_width=True)
            with col_v2:
                st.plotly_chart(render_diversity_plot(history), use_container_width=True)

            if params["dim"] == 2:
                st.plotly_chart(
                    render_particle_scatter_2d(history, obj_fn, bounds, params["func_name"]),
                    use_container_width=True,
                )

        if st.session_state.pso_running:
            time.sleep(0.02)
            st.rerun()

    elif st.session_state.pso_finished and st.session_state.pso_result:
        history = st.session_state.pso_history
        st.success("优化完成！")

        final_cols = st.columns(4)
        final_cols[0].metric("最优能量", f"{st.session_state.pso_result.best_energy:.8g}")
        final_cols[1].metric("迭代次数", f"{st.session_state.pso_result.total_iterations}")
        final_cols[2].metric("总评估", f"{st.session_state.pso_result.total_evaluations}")
        final_cols[3].metric("耗时", f"{st.session_state.pso_result.total_time:.2f}s")

        if history:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_fitness_plot(history), use_container_width=True)
            with col_v2:
                st.plotly_chart(render_diversity_plot(history), use_container_width=True)
            if params["dim"] == 2:
                st.plotly_chart(
                    render_particle_scatter_2d(history, obj_fn, bounds, params["func_name"]),
                    use_container_width=True,
                )

        if st.button("🔄 相同参数再跑 4 次（用于统计）"):
            st.session_state.multi_run_results = []
            for i in range(5):
                eng2 = ParticleSwarmOptimizer(
                    objective_fn=obj_fn, bounds=bounds,
                    swarm_size=params["swarm_size"],
                    max_iterations=params["max_iterations"],
                    inertia=params["inertia"],
                    cognitive=params["cognitive"],
                    social=params["social"],
                    topology_name=params["topology"],
                    boundary=params["boundary"],
                    v_clamp_ratio=params["v_clamp_ratio"],
                    constriction=params["constriction"],
                    adaptive_inertia=params["adaptive_inertia"],
                    seed=(params["seed"] or 42) + i * 100,
                )
                eng2.initialize()
                r = eng2.run()
                st.session_state.multi_run_results.append(r)
                st.write(f"运行 {i+1}/5: 最优 = {r.best_energy:.6g}, 耗时 = {r.total_time:.1f}s")
            st.rerun()

    else:
        st.info("请在「参数概览」Tab 中点击「开始优化」按钮。")

# ══════════════════════════════════════════════════════════
# Tab 3: 数值分析
# ══════════════════════════════════════════════════════════

with tab3:
    if st.session_state.pso_result:
        history = st.session_state.pso_history
        render_result_cards(st.session_state.pso_result, func_info, params)
        st.markdown("---")
        render_convergence_stats(history)
        st.markdown("---")
        if history:
            render_swarm_stats(history)
        if st.session_state.multi_run_results:
            st.markdown("---")
            render_multi_run_stats(st.session_state.multi_run_results)
        st.markdown("---")
        render_export_section(st.session_state.pso_result, history, params["func_name"])
    else:
        st.info("尚未运行优化。请在「参数概览」Tab 中点击「开始优化」。")

# ══════════════════════════════════════════════════════════
# Tab 4: 实验对比
# ══════════════════════════════════════════════════════════

with tab4:
    st.subheader("批量参数对比实验")
    st.markdown("同时测试多组参数组合，对比不同策略的效果。")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        compare_topo = st.multiselect(
            "对比拓扑结构", list(["Global", "Ring", "Von Neumann"]),
            default=["Global", "Ring"],
        )
        compare_cog = st.multiselect(
            "对比 c₁ 值", [1.0, 1.5, 2.0, 2.5],
            default=[1.5],
        )
    with exp_col2:
        compare_social = st.multiselect(
            "对比 c₂ 值", [1.0, 1.5, 2.0, 2.5],
            default=[1.5],
        )
        n_runs = st.slider("每组参数运行次数", 1, 10, 3)

    if st.button("运行批量实验", type="primary"):
        if not compare_topo:
            compare_topo = [params["topology"]]
        if not compare_cog:
            compare_cog = [params["cognitive"]]
        if not compare_social:
            compare_social = [params["social"]]

        param_grid = {
            "topology": compare_topo,
            "cognitive": compare_cog,
            "social": compare_social,
            "swarm_size": [params["swarm_size"]],
            "max_iterations": [params["max_iterations"]],
            "inertia": [params["inertia"]],
            "boundary": [params["boundary"]],
            "v_clamp_ratio": [params["v_clamp_ratio"]],
        }
        total = len(compare_topo) * len(compare_cog) * len(compare_social)
        with st.spinner(f"批量实验 ({total} 组 x {n_runs} 次)..."):
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
