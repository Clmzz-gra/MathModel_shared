# edudemo — 快速开始

> 版本 1.1 | 2026-08-18 | 四件套完整源码（来自 `edudemo.zip`）

## 包含工具

| 目录 | 算法 | 演示要点 |
|------|------|---------|
| `aco-gui-tool` | 蚁群优化 ACO | TSP 路径、信息素更新、多代蚂蚁演化 |
| `ga-gui-tool` | 遗传算法 GA | 选择/交叉/变异、适应度演化 |
| `pso-gui-tool` | 粒子群 PSO | 粒子群运动、拓扑结构、速度-位置更新 |
| `sa-gui-tool` | 模拟退火 SA | 温度下降、邻域搜索、接受/拒绝跳变 |

统一结构：`main.py` + `core/`（算法）+ `ui/`（界面）+ `utils/`（实验）+ `requirements.txt`。

## 一、装依赖

在包根目录执行：

```bash
pip install -r requirements.txt
```

或分别进入各工具目录安装各自的 `requirements.txt`。

## 二、运行

```bash
cd <algo>-gui-tool
python main.py
```

Windows 也可直接双击各工具目录内的 `启动*.bat`。

## 三、重新生成/扩展

调用本包 `algorithm-demo-builder` skill（`../algorithm-demo-builder/SKILL.md`），
可复现、扩展或重新生成四件套。

## 说明

- 源码来源：`C:\Users\Lenovo\Downloads\edudemo.zip`；已排除 `__pycache__` / `.pyc`。
