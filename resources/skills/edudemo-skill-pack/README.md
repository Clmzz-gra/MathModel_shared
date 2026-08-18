# edudemo-skill-pack — 快速开始

> 版本 1.1 | 2026-08-18 | 独立交付件（不并入 pan 分发包）

## 包含内容

- `edudemo/` — 四件套完整源码（aco/ga/pso/sa，可直接运行）
- `algorithm-demo-builder/` — 反推生成 skill（`SKILL.md`）
- `requirements.txt` — 统一依赖清单
- 本文件 — 快速开始

## 一、装依赖

```bash
pip install -r requirements.txt
```

各工具目录内也各有 `requirements.txt`（aco/ga/pso：streamlit/numpy/plotly/pandas；sa 额外 scipy）。

## 二、运行 demo

```bash
cd edudemo/<algo>-gui-tool
python main.py
```

或双击 `edudemo/<algo>-gui-tool/启动*.bat`（Windows）。

## 三、用 skill 重新生成

把 `algorithm-demo-builder/` 放入 AI 工具的 skills 目录（如 `.trae/skills/`、`.claude/skills/`），
然后对 AI 说：

> 用 algorithm-demo-builder 生成 ga-gui-tool，参数面板包含种群规模/迭代次数/交叉概率/变异概率/种子，实时画布展示适应度演化，附带帮助面板。

## 说明

- 源码来源：`C:\Users\Lenovo\Downloads\edudemo.zip`；已排除 `__pycache__` / `.pyc`。
- 不并入 pan 分发包；pan 包若含 `algorithm-demo-builder` 是其原有 `.trae/skills` 内容。
