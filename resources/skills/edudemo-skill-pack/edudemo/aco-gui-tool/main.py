"""蚁群算法 GUI 工具 — Streamlit 入口。

ACO 求解 TSP（旅行商问题）：蚂蚁在信息素+启发式引导下构建城市访问路径。

用法:
    streamlit run main.py
"""

import time, copy
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="蚁群算法 GUI",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.algorithm import AntColonyOptimizer, ACOResult
from core.tsp_problems import get_instance, build_distance_matrix, list_patterns
from ui.sidebar import build_sidebar
from ui.visualization import (
    render_tour_plot, render_convergence_plot,
    render_diversity_plot, render_pheromone_heatmap,
    render_progress_bar,
)
from ui.analysis import (
    render_result_cards, render_convergence_stats,
    render_tour_stats, render_multi_run_stats,
    render_export_section,
)
from ui.help_content import get_help
from utils.experiment import run_batch_experiment, render_experiment_results


def init_session():
    defaults = {
        "aco_engine": None, "aco_running": False, "aco_paused": False,
        "aco_finished": False, "aco_result": None, "aco_history": [],
        "cities": None, "last_params": None,
        "multi_run_results": [], "experiment_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

st.title("蚁群算法 (Ant Colony Optimization) — 交互式演示")
st.caption("TSP 求解 · 信息素引导 · 路径优化 · 实时可视化 · 变体对比")

params = build_sidebar()

# ── TSP 实例生成 ──────────────────────────────────────────
cities = get_instance(
    params["pattern"], params["n_cities"],
    seed=params["city_seed"], **params["pattern_params"]
)
st.session_state.cities = cities
dist_matrix = build_distance_matrix(cities)

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
        variant_label = "标准 AS"
        if params["use_acs"]:
            variant_label = f"ACS (q₀={params['q0']})"
        if params["use_mmas"]:
            variant_label = "MMAS"
        if params["use_elitist"]:
            variant_label = f"Elitist AS (e={params['elitist_weight']})"

        st.markdown(f"""
| 参数 | 值 |
|------|-----|
| 城市分布 | {params['pattern']} |
| 城市数量 | {params['n_cities']} |
| 蚂蚁数量 | {params['n_ants']} |
| 最大迭代 | {params['max_iterations']} |
| α (信息素) | {params['alpha']} |
| β (启发式) | {params['beta']} |
| ρ (挥发率) | {params['rho']} |
| ACO 变体 | {variant_label} |
| 城市种子 | {params['city_seed']} |
""")

    with col_b:
        st.subheader("TSP 地图预览")
        # 随机贪心路径作为初始预览
        greedy_tour = None
        try:
            from core.algorithm import AntColonyOptimizer
            tmp = AntColonyOptimizer(dist_matrix, cities, n_ants=1, max_iterations=1)
            greedy_tour = np.arange(len(cities))
            # 简单贪心
            unvisited = set(range(1, len(cities)))
            tour = [0]
            while unvisited:
                cur = tour[-1]
                unvisited_list = list(unvisited)
                dists = dist_matrix[cur, unvisited_list]
                nxt = unvisited_list[np.argmin(dists)]
                tour.append(nxt)
                unvisited.remove(nxt)
            tour.append(0)
            greedy_tour = np.array(tour)
        except:
            pass

        st.plotly_chart(
            render_tour_plot(cities, greedy_tour,
                             float(np.sum([dist_matrix[greedy_tour[i], greedy_tour[i+1]]
                                           for i in range(len(greedy_tour)-1)])) if greedy_tour is not None else 0,
                             f"城市分布 ({len(cities)} 城市) + 贪心路径"),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("参数含义")
    st.markdown("""
| 参数 | 含义 |
|------|------|
| α (信息素重要性) | 蚂蚁"跟风"的强度。α 大 → 走前人的路 |
| β (启发式重要性) | 蚂蚁"贪心"的强度。β 大 → 挑最近的走 |
| ρ (信息素挥发率) | 旧知识遗忘速度。ρ 大 → 探索性更强 |
| 蚂蚁数量 | 每代同时建路径的蚂蚁数 |
| ACS | 伪随机比例规则：贪心+随机混搭，+ 局部信息素更新 |
| MMAS | 信息素上下限约束，防止某条边独大导致停滞 |
""")

    btn_cols = st.columns([2, 1, 1, 1])

    with btn_cols[0]:
        if not st.session_state.aco_running:
            if st.button("开始优化", type="primary", use_container_width=True):
                st.session_state.last_params = copy.deepcopy(params)
                st.session_state.aco_finished = False
                st.session_state.aco_result = None
                st.session_state.aco_history = []

                engine = AntColonyOptimizer(
                    distance_matrix=dist_matrix, cities=cities,
                    n_ants=params["n_ants"],
                    max_iterations=params["max_iterations"],
                    alpha=params["alpha"],
                    beta=params["beta"],
                    rho=params["rho"],
                    q0=params["q0"] if params["use_acs"] else 0.0,
                    xi=params["xi"] if params["use_acs"] else 0.1,
                    elitist_weight=params["elitist_weight"] if params["use_elitist"] else 0.0,
                    mmas=params["use_mmas"],
                )
                engine.initialize()
                st.session_state.aco_engine = engine
                st.session_state.aco_running = True
                st.session_state.aco_paused = False
                st.rerun()

    if st.session_state.aco_running:
        with btn_cols[1]:
            label = "▶ 继续" if st.session_state.aco_paused else "⏸ 暂停"
            if st.button(label, use_container_width=True):
                if st.session_state.aco_paused:
                    st.session_state.aco_engine.resume()
                    st.session_state.aco_paused = False
                else:
                    st.session_state.aco_engine.pause()
                    st.session_state.aco_paused = True
                st.rerun()

        with btn_cols[2]:
            if st.button("⏹ 停止", use_container_width=True):
                st.session_state.aco_engine.stop()
                st.session_state.aco_running = False
                st.session_state.aco_paused = False
                st.session_state.aco_finished = True
                if st.session_state.aco_history:
                    st.session_state.aco_result = st.session_state.aco_engine.finalize()
                st.rerun()

# ══════════════════════════════════════════════════════════
# Tab 2: 过程可视化
# ══════════════════════════════════════════════════════════

with tab2:
    if st.session_state.aco_running and st.session_state.aco_engine:
        engine = st.session_state.aco_engine

        if not st.session_state.aco_paused:
            step_result = engine.step()
            if step_result is None:
                st.session_state.aco_running = False
                st.session_state.aco_finished = True
                st.session_state.aco_result = engine.finalize()
                st.session_state.aco_history = engine._history
                st.rerun()
            else:
                st.session_state.aco_history = engine._history
                if params["poll_interval"] > 0:
                    time.sleep(params["poll_interval"])

        history = st.session_state.aco_history

        if history:
            last = history[-1]

            render_progress_bar(
                current_iter=engine._iteration,
                max_iter=params["max_iterations"],
                best_distance=engine._best_distance,
                mean_distance=float(np.mean(last.all_distances)),
                elapsed=time.time() - engine._start_time,
                is_paused=st.session_state.aco_paused,
            )

            # 最优路径图
            st.plotly_chart(
                render_tour_plot(cities, engine._best_tour, engine._best_distance,
                                 f"最优路径 (第 {engine._iteration} 代)"),
                use_container_width=True,
            )

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_convergence_plot(history), use_container_width=True)
            with col_v2:
                st.plotly_chart(render_diversity_plot(history), use_container_width=True)

            # 信息素热力图
            if engine._pheromone is not None and len(cities) <= 80:
                st.plotly_chart(
                    render_pheromone_heatmap(engine._pheromone),
                    use_container_width=True,
                )

        if st.session_state.aco_running:
            time.sleep(0.02)
            st.rerun()

    elif st.session_state.aco_finished and st.session_state.aco_result:
        history = st.session_state.aco_history
        st.success("优化完成！")

        final_cols = st.columns(4)
        final_cols[0].metric("最优路径", f"{st.session_state.aco_result.best_distance:.4f}")
        final_cols[1].metric("迭代次数", f"{st.session_state.aco_result.total_iterations}")
        final_cols[2].metric("总评估", f"{st.session_state.aco_result.total_evaluations}")
        final_cols[3].metric("耗时", f"{st.session_state.aco_result.total_time:.2f}s")

        if history:
            # 最优路径图
            st.plotly_chart(
                render_tour_plot(cities, st.session_state.aco_result.best_tour,
                                 st.session_state.aco_result.best_distance,
                                 f"最优路径 (最终)"),
                use_container_width=True,
            )

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_convergence_plot(history), use_container_width=True)
            with col_v2:
                st.plotly_chart(render_diversity_plot(history), use_container_width=True)

            # 信息素热力图
            if history and len(cities) <= 80:
                st.plotly_chart(
                    render_pheromone_heatmap(history[-1].pheromone_matrix),
                    use_container_width=True,
                )

        if st.button("🔄 相同参数再跑 4 次（用于统计）"):
            st.session_state.multi_run_results = []
            for i in range(5):
                eng2 = AntColonyOptimizer(
                    distance_matrix=dist_matrix, cities=cities,
                    n_ants=params["n_ants"],
                    max_iterations=params["max_iterations"],
                    alpha=params["alpha"], beta=params["beta"],
                    rho=params["rho"],
                    q0=params["q0"] if params["use_acs"] else 0.0,
                    elitist_weight=params["elitist_weight"] if params["use_elitist"] else 0.0,
                    mmas=params["use_mmas"],
                )
                eng2.initialize()
                r = eng2.run()
                st.session_state.multi_run_results.append(r)
                st.write(f"运行 {i+1}/5: 最优 = {r.best_distance:.4f}, 耗时 = {r.total_time:.1f}s")
            st.rerun()

    else:
        st.info("请在「参数概览」Tab 中点击「开始优化」按钮。")

# ══════════════════════════════════════════════════════════
# Tab 3: 数值分析
# ══════════════════════════════════════════════════════════

with tab3:
    if st.session_state.aco_result:
        history = st.session_state.aco_history
        render_result_cards(st.session_state.aco_result, cities)
        st.markdown("---")
        render_convergence_stats(history)
        st.markdown("---")
        if history:
            render_tour_stats(history)
        if st.session_state.multi_run_results:
            st.markdown("---")
            render_multi_run_stats(st.session_state.multi_run_results)
        st.markdown("---")
        render_export_section(st.session_state.aco_result, history, cities)
    else:
        st.info("尚未运行优化。请在「参数概览」Tab 中点击「开始优化」。")

# ══════════════════════════════════════════════════════════
# Tab 4: 实验对比
# ══════════════════════════════════════════════════════════

with tab4:
    st.subheader("批量参数对比实验")
    st.markdown("同时测试多组参数组合，对比不同 ACO 变体和参数的效果。")

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        compare_variants = st.multiselect(
            "对比 ACO 变体",
            ["AS (标准)", "ACS", "MMAS", "Elitist"],
            default=["AS (标准)", "MMAS"],
        )
        compare_alpha = st.multiselect(
            "对比 α 值", [0.5, 1.0, 1.5, 2.0],
            default=[1.0],
        )
    with exp_col2:
        compare_beta = st.multiselect(
            "对比 β 值", [1.0, 2.0, 3.0, 5.0],
            default=[2.0],
        )
        n_runs = st.slider("每组参数运行次数", 1, 5, 2)

    if st.button("运行批量实验", type="primary"):
        if not compare_variants:
            compare_variants = ["AS (标准)"]
        if not compare_alpha:
            compare_alpha = [params["alpha"]]
        if not compare_beta:
            compare_beta = [params["beta"]]

        param_combos = []
        for v in compare_variants:
            for a in compare_alpha:
                for b in compare_beta:
                    p = {
                        "n_ants": params["n_ants"],
                        "max_iterations": params["max_iterations"],
                        "alpha": a, "beta": b, "rho": params["rho"],
                        "q0": 0.9 if "ACS" in v else 0.0,
                        "elitist_weight": 2.0 if "Elitist" in v else 0.0,
                        "mmas": v == "MMAS",
                        "seed": params["city_seed"] or 42,
                    }
                    param_combos.append(p)

        param_grid = {}
        for key in param_combos[0]:
            param_grid[key] = [p[key] for p in param_combos]

        existing_keys = list(param_combos[0].keys())
        param_grid_single = {}
        for k in existing_keys:
            vals = list(set(p[k] for p in param_combos))
            param_grid_single[k] = vals

        total = len(param_combos)
        # 使用变体做笛卡尔积比较麻烦，直接跑所有组合
        all_results = []
        progress = st.progress(0.0)
        for idx, p in enumerate(param_combos):
            run_rs = []
            for run_idx in range(n_runs):
                seed = (params["city_seed"] or 42) + idx * 100 + run_idx
                eng = AntColonyOptimizer(
                    distance_matrix=dist_matrix, cities=cities,
                    n_ants=p["n_ants"],
                    max_iterations=p["max_iterations"],
                    alpha=p["alpha"], beta=p["beta"], rho=p["rho"],
                    q0=p["q0"], elitist_weight=p["elitist_weight"],
                    mmas=p["mmas"], seed=seed,
                )
                r = eng.run()
                run_rs.append(r)

            best_dists = [r.best_distance for r in run_rs]
            variant_label = "AS"
            if p["q0"] > 0:
                variant_label = "ACS"
            if p["mmas"]:
                variant_label = "MMAS"
            if p["elitist_weight"] > 0:
                variant_label = "Elitist"

            all_results.append({
                "params": p,
                "variant": variant_label,
                "runs": run_rs,
                "mean": np.mean(best_dists),
                "std": np.std(best_dists),
                "best": np.min(best_dists),
                "worst": np.max(best_dists),
                "avg_time": np.mean([r.total_time for r in run_rs]),
                "avg_iters": np.mean([r.total_iterations for r in run_rs]),
            })
            progress.progress((idx + 1) / total,
                              text=f"完成: {idx + 1}/{total} 组参数")
        progress.empty()

        st.session_state.experiment_results = {
            "results": all_results, "cities": cities,
            "dist_matrix": dist_matrix,
        }
        st.success("批量实验完成！")
        st.rerun()

    if st.session_state.experiment_results:
        render_experiment_results(st.session_state.experiment_results)
    else:
        st.info("选择要对比的变体和参数组合，点击「运行批量实验」。")

# ══════════════════════════════════════════════════════════
# Tab 5: 帮助文档
# ══════════════════════════════════════════════════════════

with tab5:
    st.markdown(get_help())
