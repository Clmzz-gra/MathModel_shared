% MCM/ICM 终稿文档类
% 中文支持，xelatex 编译
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mcm-paper}[2026 MathModel MCM Paper Class]

\LoadClass[12pt,a4paper]{article}

% --- 中文支持（必须，xelatex 编译）---
\RequirePackage[UTF8]{ctex}
\setCJKmainfont{SimSun}[AutoFakeBold=2]
\setCJKsansfont{KaiTi}
\setCJKmonofont{SimHei}

% --- 数学 ---
\RequirePackage{amsmath, amssymb, amsthm}

% --- 图表 ---
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{caption}
\RequirePackage{subcaption}

% --- 引用 ---
\RequirePackage[backend=biber,style=gb7714-2015]{biblatex}

% --- 排版 ---
\RequirePackage[margin=1in]{geometry}
\RequirePackage{setspace}
\onehalfspacing

% --- 超链接 ---
\RequirePackage[hidelinks]{hyperref}
