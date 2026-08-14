"""
目的：
    验证 charts-sub4.py 图7/图8 修改后图例位置：图框上方、标题下方、横向展开

原理：
    matplotlib 窗口坐标 y 向上。判定条件：
    - 图例在图框上方：legend.y0 > axes.y1
    - 标题在图例上方：title.y0 > legend.y1
    - 图例横向展开：legend._ncols >= 2

输入数据：
    - outputs/data/s4-results.pkl（处理后）— 与 charts-sub4.py 相同

输出：
    - 控制台布局判定报告

对应论文章节：
    问题四（S4）图表布局核查
"""
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location(
    "cs4", r"e:\MathModel_pj-2026-C\outputs\scratch\charts-sub4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 在 savefig 前截获，打印布局量测
orig_savefig = plt.Figure.savefig


def patched(self, *a, **k):
    r = self.canvas.get_renderer()
    for ax in self.axes:
        t = ax.get_title()
        lg = ax.get_legend()
        ab = ax.get_window_extent(renderer=r)
        if t or lg is not None:
            print(f"  axes  y=[{ab.y0:.0f},{ab.y1:.0f}]")
            if t:
                tb = ax.title.get_window_extent(renderer=r)
                print(f"    title  y=[{tb.y0:.0f},{tb.y1:.0f}]  {t.splitlines()[0][:25]!r}")
            if lg is not None:
                lb = lg.get_window_extent(renderer=r)
                ncols = getattr(lg, "_ncols", None)
                print(f"    legend y=[{lb.y0:.0f},{lb.y1:.0f}]  ncols={ncols}  "
                      f"label={lg.get_texts()[0].get_text()[:12]!r}")
                ok_above = lb.y0 > ab.y1 - 1
                ok_below_title = (not t) or (tb.y0 > lb.y1 - 1)
                ok_ncol = (ncols or 1) >= 2
                print(f"    → 图框上方: {ok_above} | 标题下方: {ok_below_title} | "
                      f"横向展开: {ok_ncol}")
    return orig_savefig(self, *a, **k)


plt.Figure.savefig = patched
d = m.load()
print("=== 图8 sub4-scenario-compare ===")
m.fig1_scenario_compare(d)
print("=== 图7 sub4-region-cost-carbon ===")
m.fig2_region_cost_carbon(d)
print("LAYOUT CHECK DONE")
