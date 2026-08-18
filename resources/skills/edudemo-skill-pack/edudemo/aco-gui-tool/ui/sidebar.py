"""侧边栏参数面板 — 蚁群算法（TSP 求解）。"""

import streamlit as st

from core.tsp_problems import INSTANCE_PATTERNS, list_patterns


def build_sidebar() -> dict:
    st.sidebar.title("蚁群算法 · 参数设置")

    # ── TSP 问题 ────────────────────────────────────────────
    st.sidebar.header("TSP 问题配置")
    pattern_name = st.sidebar.selectbox(
        "城市分布", list_patterns(),
        format_func=lambda x: f"{x} — {INSTANCE_PATTERNS[x]['description'][:25]}...",
        help="城市在地图上的分布模式。不同模式难度不同。"
    )
    st.caption(INSTANCE_PATTERNS[pattern_name]["description"])

    n_cities = st.sidebar.slider(
        "城市数量", 10, 200, 40, step=5,
        help="城市数量越多问题越难。建议先用 20-50 体验，再尝试 100+。"
    )
    city_seed = st.sidebar.number_input(
        "城市随机种子", 0, 99999, 42,
        help="固定种子可复现相同的城市布局。"
    )
    city_seed = int(city_seed) if city_seed > 0 else None

    # 收集分布模式的自定义参数
    pattern_params = {}
    for pname, (pdefault, phelp) in INSTANCE_PATTERNS[pattern_name]["params"].items():
        if isinstance(pdefault, float):
            pattern_params[pname] = st.sidebar.number_input(
                pname, value=pdefault, step=pdefault * 0.5,
                format="%.3f", help=phelp, key=f"pat_{pname}"
            )
        else:
            pattern_params[pname] = st.sidebar.number_input(
                pname, value=int(pdefault), step=1,
                help=phelp, key=f"pat_{pname}"
            )

    # ── 蚁群参数 ────────────────────────────────────────────
    st.sidebar.header("蚁群参数")
    n_ants = st.sidebar.slider(
        "蚂蚁数量 m", 5, 200, 30, step=5,
        help="蚂蚁越多搜索越充分，但计算量也越大。通常取城市数量的 0.5-1 倍。"
    )
    max_iters = st.sidebar.slider(
        "最大迭代次数", 20, 2000, 200, step=10,
        help="迭代多少次后停止。简单 TSP 50-100 即可，困难的 200+。"
    )

    # ── 信息素 & 启发式 ─────────────────────────────────────
    st.sidebar.header("信息素 & 启发式")

    with st.sidebar.expander("核心系数", expanded=False):
        alpha = st.slider(
            "信息素重要性 α", 0.0, 5.0, 1.0, step=0.1,
            help="信息素的权重。α 越大蚂蚁越跟着前人走。α=0 退化为贪心。"
        )
        beta = st.slider(
            "启发式重要性 β", 0.0, 10.0, 2.0, step=0.1,
            help="距离倒数（能见度）的权重。β 越大蚂蚁越倾向于选近的城市。"
        )
        rho = st.slider(
            "信息素挥发率 ρ", 0.01, 0.99, 0.5, step=0.01,
            help="每代信息素挥发的比例。ρ 越大旧信息消失越快，探索性越强。"
        )

    # ── ACO 变体 ────────────────────────────────────────────
    st.sidebar.header("ACO 变体改进")

    use_acs = st.sidebar.checkbox(
        "蚁群系统 ACS",
        value=False,
        help="Ant Colony System: 加入伪随机比例规则（贪心+随机平衡）和局部信息素更新。收敛更快。"
    )
    q0 = 0.0
    xi = 0.1
    if use_acs:
        q0 = st.sidebar.slider(
            "ACS 贪心概率 q₀", 0.0, 1.0, 0.9, step=0.05,
            help="以 q₀ 概率贪心选最优下一城，(1-q₀) 概率轮盘赌。q₀ 越大越贪心。"
        )
        xi = st.sidebar.slider(
            "局部挥发率 ξ", 0.0, 0.5, 0.1, step=0.01,
            help="每步局部信息素更新率。蚂蚁走过就让信息素挥发一点，减少后来者重复。"
        )

    use_elitist = st.sidebar.checkbox(
        "精英蚁 Elitist",
        value=False,
        help="全局最优路径额外加强信息素。加速向最优靠拢但可能早熟。"
    )
    elitist_weight = 0.0
    if use_elitist:
        elitist_weight = st.sidebar.slider(
            "精英权重 e", 0.1, 10.0, 2.0, step=0.5,
            help="最优路径的信息素加成倍数。越大精英影响越强。"
        )

    use_mmas = st.sidebar.checkbox(
        "最大最小蚂蚁 MMAS",
        value=False,
        help="Max-Min Ant System: 信息素有上下限 + 只最优蚂蚁沉积。防止早熟和停滞。"
    )

    # ── 高级 ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    poll_interval = st.sidebar.slider(
        "可视化刷新间隔（秒）",
        0.0, 0.5, 0.0, 0.01,
        help="0 = 全速。ACO 每代已经批量处理所有蚂蚁，通常不需要额外间隔。"
    )

    return {
        "pattern": pattern_name, "n_cities": n_cities, "city_seed": city_seed,
        "pattern_params": pattern_params,
        "n_ants": n_ants, "max_iterations": max_iters,
        "alpha": alpha, "beta": beta, "rho": rho,
        "use_acs": use_acs, "q0": q0, "xi": xi,
        "use_elitist": use_elitist, "elitist_weight": elitist_weight,
        "use_mmas": use_mmas,
        "poll_interval": poll_interval,
    }
