% MCM/ICM 终稿文档类
% 中文支持，xelatex 编译
% 格式参照校赛实际交稿验证版本 (2026第一次模拟赛)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mcm-paper}[2026 MathModel MCM Paper Class]

\LoadClass[12pt,a4paper]{article}

% --- 中文支持（必须，xelatex 编译）---
\RequirePackage[UTF8]{ctex}
\setCJKmainfont{SimSun}[AutoFakeBold=2]
\setCJKsansfont{KaiTi}
\setCJKmonofont{SimHei}

% --- 数学 ---
\RequirePackage{amsmath, amssymb, amsfonts, amsthm}

% --- 图表 ---
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{float}                    % 表格 [H] 精确定位

% --- 列表格式 ---
\RequirePackage{enumitem}                 % 自定义编号/项目符号

% --- 引用 ---
\RequirePackage[backend=biber,style=gb7714-2015]{biblatex}

% --- 排版 ---
\RequirePackage[margin=2.5cm]{geometry}
\RequirePackage{setspace}
\onehalfspacing

% --- 超链接 ---
\RequirePackage[hidelinks]{hyperref}      % 隐藏链接彩色边框

% --- 页眉页脚 ---
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}                                % 清空页眉页脚
\fancyfoot[C]{\thepage}                   % 页号下方居中
\renewcommand{\headrulewidth}{0pt}        % 去掉页眉横线
