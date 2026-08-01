# 2025 C题 — NIPT 时序选择与胎儿异常判定

## 图表工具

本子项目使用仓库根目录的共享图表模块，位于：

```
E:\MathModel_pj\
├── chart_utils.py           # 出图工具：mpl 配置、路径解析、保存+审查
└── chart-reviewer/          # 审查模块子项目（独立，可删除）
    ├── chart_reviewer.py
    └── README.md
```

### 在脚本中导入

```python
import sys
sys.path.insert(0, r'E:\MathModel_pj')

from chart_utils import setup_mpl, save_figure, resolve_dirs

setup_mpl()
FIG_DIR, CHART_DIR = resolve_dirs(__file__)

# --- 绘图代码 ---
fig, ax = plt.subplots()
# ...

# 保存 + 自动审查
save_figure(fig, "sub1-figure-name", fig_dir=FIG_DIR, chart_dir=CHART_DIR,
            context="子问题1：达标时间影响因素分析")
```

### 目录约定

| 路径 | 用途 |
|------|------|
| `outputs/figures/` | 图表原始输出 |
| `solution/artifacts/charts/` | 方案制品 |
| `outputs/figures/*.review.md` | 多模态审查报告（配置 API 后生成） |

### 审查配置

默认跳过审查。启用需设环境变量：

```powershell
$env:CHART_REVIEWER_API_KEY = "sk-..."
$env:CHART_REVIEWER_MODEL = "gpt-4o"
```

`chart_reviewer.py` 是独立模块，DeepSeek 具备多模态能力后可直接删除。
