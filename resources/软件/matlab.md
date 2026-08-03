# MATLAB

- **官网下载**：https://www.mathworks.com/products/matlab.html
- **上传人**：Clmzz-gra

## 是什么

MATLAB（Matrix Laboratory）是 MathWorks 出品的商业数学软件，以**矩阵运算**为核心，集数值计算、符号计算、数据可视化、编程于一体，是数学建模和工程计算中的经典工具。

## 核心组件

- **MATLAB 桌面环境**：命令行窗口 + 脚本编辑器 + 工作区/变量浏览器，交互式调试方便
- **工具箱（Toolbox）**：优化工具箱、统计与机器学习工具箱、曲线拟合工具箱、符号数学工具箱等，建模常用的功能大多开箱即用
- **Simulink**：基于模型的仿真环境，适合动态系统建模（连续/离散、控制、信号处理）

## 数学建模场景常用

1. **矩阵/向量化编程**：建模数据大多可组织成矩阵，MATLAB 天然高效：
   ```matlab
   A = [1 2; 3 4];
   b = [5; 6];
   x = A \ b;        % 解线性方程组
   ```
2. **优化求解**：`linprog`（线性规划）、`fmincon`（约束非线性规划）、`intlinprog`（整数规划）
3. **统计与拟合**：`fitlm`、`polyfit`、`kmeans`、`anova1` 等，配合内置工具箱直接使用
4. **绘图**：`plot`、`surf`、`heatmap` 等，出图快、论文插图美观
5. **数据导入**：`readtable` 可直接读 Excel/CSV，附件数据处理很方便

## 注意

- 商业收费软件，体积大（安装包约数 GB），优先使用**校园版/学生版**授权
- 未安装时可用 **Octave**（开源、语法高度兼容）或 Python 生态（NumPy/SciPy/Matplotlib）平替
- 代码风格建议：脚本开头加 `clear; clc;`，长计算避免大量循环改用向量化写法
