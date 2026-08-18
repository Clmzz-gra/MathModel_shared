"""侧边栏参数面板。"""

import streamlit as st

from core.test_functions import list_functions, list_functions_for_dim
from core.cooling_schedules import SCHEDULES
from core.neighborhood import NEIGHBORHOODS
from core.annealing import ACCEPTANCE_CRITERIA


def build_sidebar() -> dict:
    """构建侧边栏并返回参数字典。"""

    st.sidebar.title("模拟退火 · 参数设置")

    # ── 目标函数选择 ────────────────────────────────────────
    st.sidebar.header("目标问题")

    all_funcs = list_functions()
    func_name = st.sidebar.selectbox(
        "目标函数",
        all_funcs,
        index=all_funcs.index("Rastrigin") if "Rastrigin" in all_funcs else 0,
        help="选择要优化的测试函数。不同函数有不同的难度和特征。"
    )
    dim = st.sidebar.slider(
        "问题维度",
        min_value=1, max_value=10, value=2,
        help="越高维度搜索空间越大，优化难度指数增长。"
    )

    # 动态过滤对当前维度可用的函数
    available_for_dim = list_functions_for_dim(dim)
    if func_name not in available_for_dim:
        st.sidebar.warning(f"⚠️ {func_name} 不支持 {dim} 维。请换一个函数。")
        if available_for_dim:
            func_name = available_for_dim[0]

    # 自定义函数
    st.sidebar.markdown("---")
    custom_expr = st.sidebar.text_area(
        "自定义函数表达式（可选）",
        value="",
        height=68,
        placeholder="例: x[0]**2 + x[1]**2\n留空则使用上方选择的测试函数。",
        help="用 Python 表达式定义自定义优化函数。变量 x 是 numpy 数组。例: np.sum(x**2)"
    )

    if custom_expr.strip():
        func_name = "Custom"

    # ── 基本参数 ────────────────────────────────────────────
    st.sidebar.header("基本参数")

    T0 = st.sidebar.slider(
        "初始温度 T₀",
        min_value=1.0, max_value=10000.0, value=1000.0, step=100.0,
        help="起始温度。越高，初期越容易接受差解，搜索范围越广。过高浪费计算，过低容易过早收敛。"
    )
    T_end = st.sidebar.slider(
        "终止温度 T_end",
        min_value=0.0001, max_value=10.0, value=0.01, step=0.001, format="%.4f",
        help="温度低于此值时算法停止。越小搜索越精细，但耗时越长。"
    )
    alpha = st.sidebar.slider(
        "降温速率 α（用于 Geometric 等）",
        min_value=0.500, max_value=0.999, value=0.950, step=0.001,
        help="温度每步乘以此系数。越接近 1 降温越慢，搜索越充分。0.8-0.99 是常用范围。"
    )
    max_iter = st.sidebar.slider(
        "最大迭代次数 N",
        min_value=100, max_value=50000, value=5000, step=100,
        help="无论温度是否降到 T_end，到达此迭代次数即停止。防止无限循环。"
    )

    # ── 优化方式 ────────────────────────────────────────────
    st.sidebar.header("优化方式")

    with st.sidebar.expander("冷却策略（Cooling Schedule）", expanded=False):
        cooling = st.selectbox(
            "选择冷却策略",
            list(SCHEDULES.keys()),
            index=0,
            help=SCHEDULES["Geometric"]["description"]
        )
        st.caption(SCHEDULES[cooling]["description"])

        cooling_params = {}
        for pname, (pdefault, phelp) in SCHEDULES[cooling]["params"].items():
            if pname == "alpha":
                cooling_params[pname] = alpha  # 与基本面板联动
            elif pname == "T_end":
                cooling_params[pname] = T_end
            elif pname == "max_iter":
                cooling_params[pname] = max_iter
            elif isinstance(pdefault, float):
                cooling_params[pname] = st.number_input(
                    f"{pname}", value=pdefault, step=pdefault * 0.1,
                    format="%.4f", help=phelp, key=f"cool_{pname}"
                )
            elif isinstance(pdefault, int):
                cooling_params[pname] = st.number_input(
                    f"{pname}", value=pdefault, help=phelp, key=f"cool_{pname}"
                )

    with st.sidebar.expander("邻域策略（Neighborhood）", expanded=False):
        neigh = st.selectbox(
            "选择邻域策略",
            list(NEIGHBORHOODS.keys()),
            index=0,
            help=NEIGHBORHOODS["Gaussian"]["description"]
        )
        st.caption(NEIGHBORHOODS[neigh]["description"])

        neigh_params = {}
        for pname, (pdefault, phelp) in NEIGHBORHOODS[neigh]["params"].items():
            if isinstance(pdefault, float):
                neigh_params[pname] = st.number_input(
                    f"{pname}", value=pdefault, step=pdefault * 0.1,
                    format="%.4f", help=phelp, key=f"neigh_{pname}"
                )
            elif isinstance(pdefault, int):
                neigh_params[pname] = st.number_input(
                    f"{pname}", value=pdefault, help=phelp, key=f"neigh_{pname}"
                )

    with st.sidebar.expander("接受准则（Acceptance Criterion）", expanded=False):
        accept = st.selectbox(
            "选择接受准则",
            list(ACCEPTANCE_CRITERIA.keys()),
            index=0,
            help=ACCEPTANCE_CRITERIA["Metropolis"]["description"]
        )
        st.caption(ACCEPTANCE_CRITERIA[accept]["description"])

        accept_params = {}
        for pname, (pdefault, phelp) in ACCEPTANCE_CRITERIA[accept]["params"].items():
            if isinstance(pdefault, float):
                accept_params[pname] = st.number_input(
                    f"{pname}", value=pdefault, step=pdefault * 0.1,
                    format="%.4f", help=phelp, key=f"acc_{pname}"
                )

    # ── 高级选项 ────────────────────────────────────────────
    st.sidebar.header("高级选项")

    auto_T0 = st.sidebar.checkbox(
        "初始温度自动标定",
        value=False,
        help="自动计算使初始接受率约 80% 的 T₀，覆盖手动设定的 T₀。来源：AL-002 诊断指标。"
    )

    markov_len = st.sidebar.number_input(
        "Markov 链长度",
        min_value=1, max_value=100, value=1,
        help="每个温度下执行多少次邻域搜索。>1 表示在当前温度下多次尝试。值越大每个温度层搜索越充分，但总迭代数不变意味着温度层数减少。"
    )
    seed = st.sidebar.number_input(
        "随机种子",
        min_value=0, max_value=99999, value=42,
        help="固定随机种子可以复现实验结果。设为 0 则不固定。"
    )

    reheating = st.sidebar.checkbox(
        "启用重加热（Reheating）",
        value=False,
        help="当近期接受率过低时，自动升温回到较高温度重新搜索。有助于跳出局部最优，但可能增加迭代次数。"
    )
    reheat_trigger = 0.01
    reheat_factor = 0.3
    if reheating:
        reheat_trigger = st.sidebar.number_input(
            "重加热触发阈值",
            min_value=0.0, max_value=0.5, value=0.01, step=0.005,
            help="近期接受率低于此值时触发重加热。"
        )
        reheat_factor = st.sidebar.number_input(
            "重加热因子",
            min_value=0.05, max_value=0.8, value=0.3, step=0.05,
            help="重加热目标温度 = T₀ × 此因子。"
        )

    # ── 调试选项 ────────────────────────────────────────────
    st.sidebar.markdown("---")
    poll_interval = st.sidebar.slider(
        "可视化刷新间隔（秒）",
        min_value=0.0, max_value=0.5, value=0.0, step=0.01,
        help="0 = 全速运行。设为 0.1-0.5 可慢放观察搜索过程。每批约运行 N/50 步后才刷新。"
    )

    return {
        "func_name": func_name,
        "custom_expr": custom_expr.strip(),
        "dim": dim,
        "T0": T0,
        "T_end": T_end,
        "alpha": alpha,
        "max_iter": max_iter,
        "cooling": cooling,
        "cooling_params": cooling_params,
        "neigh": neigh,
        "neigh_params": neigh_params,
        "accept": accept,
        "accept_params": accept_params,
        "markov_chain_len": markov_len,
        "auto_T0": auto_T0,
        "seed": int(seed) if seed > 0 else None,
        "reheating": reheating,
        "reheat_trigger": reheat_trigger,
        "reheat_factor": reheat_factor,
        "poll_interval": poll_interval,
    }
