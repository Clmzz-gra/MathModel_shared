# 支撑材料包（2026 华数杯模拟赛 B 题）

> 赛题：基于宏基因组数据的疾病预测模型研究
> 对应终稿：`solution/final-paper/COMP2026-B-final.pdf`（33 页）
> 生成日期：2026-08-23

## 目录结构

```
package/
├── code/       程序文件（三子问题主模型、预处理、绘图、数据导出）
├── data/       数据文件（原始/预处理/结果 pkl，含六轮归因实验）
├── figures/    图表文件（正式结果图 + 数据探索图 + 数据特征图 + 技术路线图）
├── tables/     结果表（LaTeX 源）
└── references.bib  参考文献
```

## 文件说明

### code/（程序）

| 文件 | 说明 |
|---|---|
| S1-model.py / S2-model.py / S3-model.py | 三子问题主模型实现（S1 疾病预测、S2 稳定性选择、S3 跨疾病 LODO） |
| preprocess-S1/S2/S3.py | 三子问题预处理（近全零过滤、CLR、标准化） |
| clean-B.py / convert-B-raw.py | 原始数据清洗与格式转换 |
| plot_s1.py / plot_s2.py / plot_s3.py | 正式图表绘制 |
| plot_s3_butterfly.py / plot_s3_redraw.py | S3 蝴蝶图与重绘辅助 |
| export_matlab_data.py | MATLAB 重绘数据导出（队友 MATLAB 重绘用） |
| profile-B.py | 数据画像（聚类/降维） |
| utils.py / utils-S2.py | 公共工具函数 |

> 其余辅助脚本（S3-* 归因实验、verify-* 验证、extract-* 提取等 70 余个）位于
> `outputs/scratch/`，按需取用。

### data/（数据）

| 文件 | 说明 |
|---|---|
| B-raw.pkl | 原始数据（484 样本 × 1333 列） |
| c-data-cleaned.pkl | 清洗后共享数据（阶段 0.3） |
| S1/S2/S3-preprocessed.pkl | 三子问题预处理数据（过滤 1331→264 + CLR） |
| S1/S2/S3-results.pkl | 三子问题实验结果（正文数字唯一出口） |
| S3-combat-corrected.pkl | E0 ComBat 批次校正实验 |
| E2-bayes-results.pkl | E2 贝叶斯模型族实验 |
| S3-e1-baselines.pkl / S3-e3-fewshot.pkl / S3-e5-source-ensemble.pkl / S3-e6-mlp.pkl | E1/E3/E5/E6 归因实验 |

### figures/（图表）

- 13 张正式结果图（S1-*、S2-*、S3-*）——正文与附录引用版本，已过图审
- 数据探索图：cluster-tsne.pdf、pca-scree.pdf
- 数据特征图：chart-*.pdf（5 张，样本构成/零值稀疏/丰度分布/批次效应/已知标志物）
- 技术路线图：flow-overview.drawio.pdf

### tables/（结果表）

S2-stable-biomarkers.tex、S2-cross-disease.tex、S2-tau-sensitivity.tex、S3-strategy-compare.tex

## 使用

- 正文数字唯一出口：`solution/data-final/data-integration-2026-sim2-B.md`
- 替换表述库（人类定稿，禁止修改）：`knowledge/expression-library.md`
- 依赖：Python 3.13 + numpy/pandas/scikit-learn + matplotlib + ComBat（E0）
