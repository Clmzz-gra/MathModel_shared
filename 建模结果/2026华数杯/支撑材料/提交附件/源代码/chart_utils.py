"""
图表工具模块 (Chart Utilities)
=================================
提供统一的 matplotlib 配置、路径管理和出图+审查一体化接口。

位于仓库根目录，供所有子项目共享使用。

使用方法（在子项目脚本中）：
  import sys
  sys.path.insert(0, 'E:/MathModel_pj')  # 确保根目录在搜索路径中
  
  from chart_utils import setup_mpl, save_figure, resolve_dirs
  
  setup_mpl()
  FIG_DIR, CHART_DIR = resolve_dirs(__file__)
  
  fig, ax = plt.subplots()
  # ... 绘图代码 ...
  save_figure(fig, "figure-name", fig_dir=FIG_DIR, chart_dir=CHART_DIR,
              context="子问题1：达标时间分析")
  # 自动完成：tight_layout → 保存 PDF/PNG → 多模态审查 → 关闭
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 确保 chart-reviewer 子项目在搜索路径中
_REVIEWER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chart-reviewer')
if _REVIEWER_DIR not in sys.path:
    sys.path.insert(0, _REVIEWER_DIR)

# ============================================================================
# Matplotlib Setup
# ============================================================================

def setup_mpl():
    """统一 matplotlib 配置：非交互式后端 + 中文字体 + 负号支持"""
    if not matplotlib.get_backend().lower().startswith('agg'):
        matplotlib.use('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Path Resolution
# ============================================================================

def resolve_dirs(caller_file: str, subproject_rel: str = "") -> tuple:
    """
    根据调用脚本的位置自动解析 fig_dir 和 chart_dir。

    子项目内部约定：
      {subproject}/outputs/figures/    → 图表输出
      {subproject}/solution/artifacts/charts/ → 方案制品

    参数：
      caller_file:     调用脚本的 __file__（必须传入）
      subproject_rel:  子项目相对于仓库根的路径，如 "problems/2025/C题"
                       留空则自动从 caller_file 向上查找包含 outputs/ 的目录

    返回：
      (fig_dir, chart_dir) 两个绝对路径字符串

    示例：
      FIG_DIR, CHART_DIR = resolve_dirs(__file__)
      FIG_DIR, CHART_DIR = resolve_dirs(__file__, "problems/2025/C题")
    """
    caller = Path(caller_file).resolve()

    if subproject_rel:
        # 从仓库根拼接
        repo_root = _find_repo_root(caller)
        sub_root = repo_root / subproject_rel
    else:
        # 自动检测：从 caller 向上找包含 outputs/ 的目录
        sub_root = _find_subproject_root(caller)

    fig_dir = str(sub_root / 'outputs' / 'figures')
    chart_dir = str(sub_root / 'solution' / 'artifacts' / 'charts')
    return fig_dir, chart_dir


def _find_repo_root(start: Path) -> Path:
    """从 start 向上查找仓库根（包含 .git 或 chart_utils.py 的目录）"""
    for p in [start] + list(start.parents):
        if (p / '.git').exists() or (p / 'chart_utils.py').exists():
            return p
    return start.parents[-1]  # 回退到磁盘根


def _find_subproject_root(start: Path) -> Path:
    """从 caller 脚本位置向上查找包含 outputs/ 的子项目根"""
    for p in [start.parent] + list(start.parents):
        if (p / 'outputs').is_dir():
            return p
    return start.parent  # 回退到脚本所在目录

# ============================================================================
# Save & Review
# ============================================================================

def save_figure(fig, name, *, fig_dir, chart_dir=None, dpi=300,
                context="", close=True, review=True, fmt="pdf"):
    """
    保存图表到指定目录，可选：生成 PNG 并送多模态审查。

    参数：
      fig:       matplotlib Figure 对象
      name:      文件名（不含扩展名），如 "sub1-correlation-heatmap"
      fig_dir:   图表输出目录（必填）
      chart_dir: 方案制品目录，默认与 fig_dir 相同
      dpi:       分辨率，默认 300（与 chart-generator skill 对齐）
      context:   图表上下文描述，用于审查提示词
      close:     保存后是否关闭 Figure，默认 True
      review:    是否调用多模态审查，默认 True
      fmt:       主输出格式，默认 "pdf"

    保存内容：
      - {fig_dir}/{name}.{fmt}      → 主输出
      - {chart_dir}/{name}.{fmt}    → 方案制品
      - {fig_dir}/{name}.png        → 审查用 PNG（如启用审查）
      - {fig_dir}/{name}.review.md  → 审查报告（如启用审查且配置了 API）
    """
    if chart_dir is None:
        chart_dir = fig_dir

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)

    fig.tight_layout()

    fig_path = os.path.join(fig_dir, f"{name}.{fmt}")
    chart_path = os.path.join(chart_dir, f"{name}.{fmt}")
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(chart_path, dpi=dpi, bbox_inches='tight')

    if review:
        png_path = os.path.join(fig_dir, f"{name}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')

        try:
            from chart_reviewer import review_figure
            review_figure(png_path, context=context)
        except ImportError:
            print("[chart_utils] chart_reviewer 模块未找到，跳过审查。")
        except Exception as e:
            print(f"[chart_utils] 审查异常（不阻塞出图）: {e}")

    if close:
        plt.close(fig)

    print(f"[chart_utils] 已保存: {name}.{fmt}")

# ============================================================================
# Batch Review
# ============================================================================

def review_all_figures(fig_dir, glob_pattern="*.pdf", context_map=None):
    """
    批量审查指定目录下的所有图表。

    参数：
      fig_dir:      图表所在目录
      glob_pattern: 文件匹配模式，默认 "*.pdf"
      context_map:  dict，文件名 → 上下文的映射，可选

    示例：
      review_all_figures(FIG_DIR, "sub1-*.pdf", {
          "sub1-correlation-heatmap": "子问题1：相关性分析",
          "sub1-residual-qq": "子问题1：残差诊断",
      })
    """
    import glob as glob_mod
    from chart_reviewer import review_figure

    context_map = context_map or {}
    pattern = os.path.join(fig_dir, glob_pattern)
    files = glob_mod.glob(pattern)

    if not files:
        print(f"[chart_utils] 未找到匹配的图表: {pattern}")
        return

    print(f"[chart_utils] 开始批量审查 {len(files)} 张图表...")
    for f in sorted(files):
        name = os.path.splitext(os.path.basename(f))[0]
        ctx = context_map.get(name, "")
        review_figure(f, context=ctx)
    print(f"[chart_utils] 批量审查完成。")

# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    setup_mpl()

    # 路径解析自检（不写文件）
    print(f"chart_utils 模块位于: {Path(__file__).resolve().parent}")
    print(f"resolve_dirs 示例 (relative): {resolve_dirs(__file__, 'problems/2025/C题')}")
    print(f"_find_repo_root: {_find_repo_root(Path(__file__).resolve())}")
    print("chart_utils 自检完成。")
