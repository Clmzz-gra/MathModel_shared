% 内部报告文档类
% 允许冗余、注释、探索性内容
% 中文支持，xelatex 编译
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{internal-report}[2026 MathModel Internal Report Class]

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

% --- 排版 ---
\RequirePackage[margin=1in]{geometry}
\RequirePackage{setspace}
\onehalfspacing

% --- 注释工具（内部报告允许注释）---
\RequirePackage{xcolor}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}
\newcommand{\note}[1]{\textcolor{blue}{[NOTE: #1]}}
\newcommand{\uncertain}[1]{\textcolor{orange}{[待确认: #1]}}
