# Anaconda3

- **官网下载**：https://www.anaconda.com/products/desktop
- **上传人**：Clmzz-gra

## 是什么

Anaconda 是一个 Python 数据科学发行版，**安装包自带** 500+ 常用数据科学包（NumPy、Pandas、SciPy、Matplotlib、Scikit-learn、Jupyter 等），省去一个个 pip 安装的麻烦。

## 核心组件

- **conda**：包管理器 + 环境管理器。既装 Python 包，也能管理独立环境（不同项目用不同 Python 版本/依赖不互相干扰）
- **Anaconda Navigator**：图形界面，可视化管理环境和启动 Jupyter / Spyder
- **Jupyter Notebook / JupyterLab**：交互式笔记本，适合建模探索、可视化、结果展示

## 数学建模场景常用

1. **环境隔离**：每个赛题建一个独立环境，避免依赖冲突：
   ```bash
   conda create -n mcm2026 python=3.11
   conda activate mcm2026
   ```
2. **常用库**：`numpy`、`pandas`（数据处理）、`matplotlib`、`seaborn`（绘图）、`scipy`（优化/统计）、`scikit-learn`（机器学习）、`pulp`/`ortools`（线性规划）
3. **装新包**：`conda install 包名`；找不到时用 `pip install 包名`

## 注意

- 体积较大（约 3GB+），也可用轻量版 **Miniconda**（仅 conda + Python，按需装包）
- 首次安装记得勾选/配置 PATH（或只用 Navigator 启动，不依赖命令行全局 PATH）
- 已有系统 Python 的机器，建议统一用 conda 环境管理，避免版本混乱
