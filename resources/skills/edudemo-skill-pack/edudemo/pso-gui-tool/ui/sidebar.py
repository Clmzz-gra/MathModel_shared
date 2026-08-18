"""侧边栏参数面板 — 粒子群优化。"""

import streamlit as st

from core.test_functions import list_functions, list_functions_for_dim
from core.topology import TOPOLOGY_METHODS

BOUNDARY_METHODS = {
    "clip": "钳制：超出边界直接裁剪到边界值。最常用。",
    "reflect": "反射：碰到边界反弹回来，速度反向。",
    "random": "随机重置：超出边界随机回范围内，速度减半反向。",
    "absorb": "吸收：碰到墙速度归零，停在边界上。",
}


def build_sidebar() -> dict:
    st.sidebar.title("粒子群优化 · 参数设置")

    # ── 目标函数 ────────────────────────────────────────────
    st.sidebar.header("目标问题")
    all_funcs = list_functions()
    func_name = st.sidebar.selectbox(
        "目标函数",
        all_funcs,
        index=all_funcs.index("Rastrigin") if "Rastrigin" in all_funcs else 0,
        help="选择要优化的测试函数。"
    )
    dim = st.sidebar.slider(
        "问题维度",
        min_value=1, max_value=10, value=2,
        help="搜索空间维度。维数越高搜索难度越大。"
    )
    available = list_functions_for_dim(dim)
    if func_name not in available:
        st.sidebar.warning(f"⚠️ {func_name} 不支持 {dim} 维")
        if available:
            func_name = available[0]

    custom_expr = st.sidebar.text_area(
        "自定义函数（可选）", value="",
        placeholder="例: x[0]**2 + x[1]**2",
        help="留空使用上方选择的测试函数。"
    )
    if custom_expr.strip():
        func_name = "Custom"

    # ── 粒子群参数 ──────────────────────────────────────────
    st.sidebar.header("粒子群参数")
    swarm_size = st.sidebar.slider(
        "粒子数", 10, 500, 50, step=10,
        help="粒子群规模。粒子越多搜索越充分，但每次迭代评估越多。30-100 常用。"
    )
    max_iters = st.sidebar.slider(
        "最大迭代次数", 20, 2000, 300, step=10,
        help="迭代多少次后停止。简单函数 100-200 即可，复杂多峰需要 300-1000。"
    )

    # ── 速度更新参数 ────────────────────────────────────────
    st.sidebar.header("速度更新参数")

    with st.sidebar.expander("惯性权重 & 加速系数", expanded=False):
        inertia = st.slider(
            "惯性权重 w", 0.1, 1.2, 0.7, step=0.05,
            help="控制粒子保持原有速度的能力。大=全局探索，小=局部精调。0.4-0.9 常用。"
        )
        cognitive = st.slider(
            "认知系数 c₁", 0.0, 4.0, 1.5, step=0.1,
            help="向自己历史最优学习的权重。c₁ > c₂ 偏向独立探索。"
        )
        social = st.slider(
            "社会系数 c₂", 0.0, 4.0, 1.5, step=0.1,
            help="向群体最优学习的权重。c₂ > c₁ 偏向群体趋同。"
        )

    with st.sidebar.expander("邻域拓扑（Topology）", expanded=False):
        topo_name = st.selectbox(
            "拓扑结构", list(TOPOLOGY_METHODS.keys()), index=0,
            help=TOPOLOGY_METHODS["Global"]["description"]
        )
        st.caption(TOPOLOGY_METHODS[topo_name]["description"])
        topo_params = {}
        for pname, (pdefault, phelp) in TOPOLOGY_METHODS[topo_name]["params"].items():
            topo_params[pname] = st.number_input(
                pname, value=int(pdefault), step=1,
                help=phelp, key=f"topo_{pname}"
            )

    with st.sidebar.expander("边界处理（Boundary）", expanded=False):
        boundary = st.selectbox(
            "边界策略", list(BOUNDARY_METHODS.keys()),
            format_func=lambda x: f"{x} — {BOUNDARY_METHODS[x][:20]}...",
            help="粒子飞出搜索空间时的处理方式。"
        )
        st.caption(BOUNDARY_METHODS[boundary])

        v_clamp = st.slider(
            "速度钳制系数", 0.0, 0.5, 0.2, step=0.05,
            help="限制最大速度为搜索范围的该倍数。0=不钳制。0.1-0.3 常用。"
        )

    # ── 算法改进 ────────────────────────────────────────────
    st.sidebar.header("算法改进")
    constriction = st.sidebar.checkbox(
        "收缩因子 (Clerc & Kennedy 2002)",
        value=False,
        help="用收缩因子代替速度钳制，从理论上保证收敛。当 c₁+c₂>4 时自动激活。"
    )
    adaptive_inertia = st.sidebar.checkbox(
        "自适应惯性权重",
        value=False,
        help="惯性权重每代衰减。前期大惯性探索（w≈0.9），后期小惯性精调（w→0.3）。"
    )
    if adaptive_inertia:
        inertia_decay = st.sidebar.slider(
            "惯性衰减率", 0.95, 0.999, 0.995, step=0.001,
            help="每代 w *= decay。越接近 1 衰减越慢。"
        )
        inertia_min = st.sidebar.slider(
            "最小惯性", 0.1, 0.5, 0.3, step=0.05,
            help="惯性衰减的下限。"
        )
    else:
        inertia_decay = 0.995
        inertia_min = 0.3

    # ── 高级 ────────────────────────────────────────────────
    st.sidebar.header("高级选项")
    seed = st.sidebar.number_input(
        "随机种子", 0, 99999, 42,
        help="0 = 不固定。固定种子可复现结果。"
    )

    st.sidebar.markdown("---")
    poll_interval = st.sidebar.slider(
        "可视化刷新间隔（秒）",
        0.0, 0.5, 0.0, 0.01,
        help="0 = 全速。PSO 每代已经批量更新所有粒子，通常不需要额外间隔。"
    )

    return {
        "func_name": func_name, "custom_expr": custom_expr.strip(), "dim": dim,
        "swarm_size": swarm_size, "max_iterations": max_iters,
        "inertia": inertia, "cognitive": cognitive, "social": social,
        "topology": topo_name, "topology_params": topo_params,
        "boundary": boundary, "v_clamp_ratio": v_clamp,
        "constriction": constriction,
        "adaptive_inertia": adaptive_inertia,
        "inertia_decay": inertia_decay, "inertia_min": inertia_min,
        "seed": int(seed) if seed > 0 else None,
        "poll_interval": poll_interval,
    }
