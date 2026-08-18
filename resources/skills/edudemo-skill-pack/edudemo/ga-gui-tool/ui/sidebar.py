"""侧边栏参数面板 — 遗传算法。"""

import streamlit as st

from core.test_functions import list_functions, list_functions_for_dim
from core.selection import SELECTION_METHODS
from core.crossover import CROSSOVER_METHODS
from core.mutation import MUTATION_METHODS


def build_sidebar() -> dict:
    st.sidebar.title("遗传算法 · 参数设置")

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
        "染色体长度（= 问题维度）",
        min_value=1, max_value=10, value=2,
        help="实数编码，每个基因是一个浮点数。维数越高搜索空间越大。"
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

    # ── 种群参数 ────────────────────────────────────────────
    st.sidebar.header("种群参数")
    pop_size = st.sidebar.slider(
        "种群大小", 20, 500, 100, step=10,
        help="每代个体数。越大搜索越充分，但每代评估次数也越多。50-200 常用。"
    )
    max_gens = st.sidebar.slider(
        "最大代数", 20, 2000, 200, step=10,
        help="进化多少代后停止。简单问题 50-100 代即可，复杂问题需要 300-1000 代。"
    )

    # ── 遗传操作 ────────────────────────────────────────────
    st.sidebar.header("遗传操作")

    with st.sidebar.expander("选择策略（Selection）", expanded=False):
        sel_name = st.selectbox(
            "选择算子", list(SELECTION_METHODS.keys()), index=1,
            help=SELECTION_METHODS["Tournament"]["description"]
        )
        st.caption(SELECTION_METHODS[sel_name]["description"])
        sel_params = {}
        for pname, (pdefault, phelp) in SELECTION_METHODS[sel_name]["params"].items():
            if isinstance(pdefault, int):
                sel_params[pname] = st.number_input(
                    pname, value=pdefault, help=phelp, key=f"sel_{pname}"
                )
            else:
                sel_params[pname] = st.number_input(
                    pname, value=float(pdefault), step=pdefault * 0.1,
                    format="%.2f", help=phelp, key=f"sel_{pname}"
                )

    with st.sidebar.expander("交叉策略（Crossover）", expanded=False):
        cross_name = st.selectbox(
            "交叉算子", list(CROSSOVER_METHODS.keys()), index=2,
            help=CROSSOVER_METHODS["Uniform"]["description"]
        )
        st.caption(CROSSOVER_METHODS[cross_name]["description"])
        cross_rate = st.slider(
            "交叉概率", 0.3, 1.0, 0.8, step=0.05,
            help="每对父代执行交叉的概率。0.7-0.9 常用。"
        )
        cross_params = {}
        for pname, (pdefault, phelp) in CROSSOVER_METHODS[cross_name]["params"].items():
            cross_params[pname] = st.number_input(
                pname, value=float(pdefault), step=pdefault * 0.1,
                format="%.2f", help=phelp, key=f"cross_{pname}"
            )

    with st.sidebar.expander("变异策略（Mutation）", expanded=False):
        mut_name = st.selectbox(
            "变异算子", list(MUTATION_METHODS.keys()), index=1,
            help=MUTATION_METHODS["Gaussian"]["description"]
        )
        st.caption(MUTATION_METHODS[mut_name]["description"])
        mut_rate = st.slider(
            "变异概率", 0.001, 0.5, 0.1, step=0.001,
            help="每个基因独立变异的概率。太高变随机搜索，太低多样性不够。0.01-0.2 常用。"
        )
        mut_params = {}
        for pname, (pdefault, phelp) in MUTATION_METHODS[mut_name]["params"].items():
            if isinstance(pdefault, float):
                mut_params[pname] = st.number_input(
                    pname, value=pdefault, step=pdefault * 0.1,
                    format="%.3f", help=phelp, key=f"mut_{pname}"
                )

    # ── 算法改进 ────────────────────────────────────────────
    st.sidebar.header("算法改进 (来自知识卡片)")
    opposition_init = st.sidebar.checkbox(
        "反向初始化 (AL-012)",
        value=False,
        help="生成 2n 个体（n 个随机 + n 个反向），取最优 n 个作为初始种群。扩大初始搜索范围。"
    )
    de_mutation = st.sidebar.checkbox(
        "DE 差分变异 (AL-012)",
        value=False,
        help="用 DE/best/2 变异替代常规变异。利用种群差分信息产生方向性搜索。来源: DEGA 卡片。"
    )
    early_restart = st.sidebar.checkbox(
        "早熟检测与重启 (AL-003 C234)",
        value=False,
        help="当最优个体占比 > 10% 时，注入随机个体防止早熟收敛。来源: 2024 C234 论文。"
    )
    adaptive_mutation = st.sidebar.checkbox(
        "自适应变异率 (AL-003)",
        value=False,
        help="变异率每代衰减，从初始值逐步降到最小值。前期大变异探索，后期小变异精调。"
    )

    # ── 高级 ────────────────────────────────────────────────
    st.sidebar.header("高级选项")
    elite_count = st.sidebar.number_input(
        "精英保留数", 0, 20, 2,
        help="每代直接保留的最优个体数。保证最优解不退化，1-5 常用。"
    )
    seed = st.sidebar.number_input(
        "随机种子", 0, 99999, 42,
        help="0 = 不固定。固定种子可复现结果。"
    )

    st.sidebar.markdown("---")
    poll_interval = st.sidebar.slider(
        "可视化刷新间隔（秒）",
        0.0, 0.5, 0.0, 0.01,
        help="0 = 全速。GA 每代已天然成批（pop_size 次评估），通常不需要额外间隔。"
    )

    return {
        "func_name": func_name, "custom_expr": custom_expr.strip(), "dim": dim,
        "pop_size": pop_size, "max_generations": max_gens,
        "selection": sel_name, "selection_params": sel_params,
        "crossover": cross_name, "crossover_params": cross_params,
        "crossover_rate": cross_rate,
        "mutation": mut_name, "mutation_params": mut_params,
        "mutation_rate": mut_rate,
        "elite_count": elite_count,
        "opposition_init": opposition_init,
        "de_mutation": de_mutation,
        "early_restart": early_restart,
        "adaptive_mutation": adaptive_mutation,
        "seed": int(seed) if seed > 0 else None,
        "poll_interval": poll_interval,
    }
