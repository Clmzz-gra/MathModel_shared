# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    阶段 0.1 数据准备 — 探查 6 个附件 xlsx 的完整结构（sheet 名/行列数/列名/dtype/前 5 行/字段说明 sheet），
    并导出每个数据 sheet 为 CSV，生成 Markdown 结构摘要报告

原理：
    使用 pandas.read_excel(engine=openpyxl) 逐 sheet 读取；用 ExcelFile 枚举 sheet 名。
    字段说明 sheet 单独提取为表格文本。大表 region_time_data.xlsx 先用 openpyxl 读行数估计规模。

输入数据：
    - problems/2026年第七届华数杯数学建模竞赛赛题/C题 面向算电协同的多目标调度优化研究/附件数据/*.xlsx（原始）

输出：
    - outputs/data/csv/<表名>/<sheet名>.csv — 各数据 sheet 的 CSV 导出
    - solution/data-xlsx-structure.md — 全部表的结构摘要报告（易读格式）

对应论文章节：
    阶段 0.1 数据盘点准备
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(r"e:\MathModel_pj-2026-C")
DATA_DIR = BASE / "problems" / "2026年第七届华数杯数学建模竞赛赛题" / "C题 面向算电协同的多目标调度优化研究" / "附件数据"
OUT_CSV = BASE / "outputs" / "data" / "csv"
OUT_CSV.mkdir(parents=True, exist_ok=True)

FILES = [
    "GPU_information.xlsx",
    "workload_trace.xlsx",
    "region_time_data.xlsx",
    "power_mapping.xlsx",
    "network_latency.xlsx",
    "storage_information.xlsx",
]


def main():
    report = []
    report.append("# C 题 附件 xlsx 数据结构摘要\n")
    report.append("> 由 `outputs/scratch/inspect-c-data.py` 自动生成，2026-08-07\n")

    for fname in FILES:
        path = DATA_DIR / fname
        report.append(f"\n## {fname}\n")
        xls = pd.ExcelFile(path, engine="openpyxl")
        report.append(f"**sheet 列表**：{', '.join(xls.sheet_names)}\n")

        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, header=0)
            report.append(f"\n### sheet: {sheet}  （shape={df.shape}）")
            if df.shape[0] == 0:
                report.append("（空）")
                continue
            # 列信息
            col_lines = []
            for c in df.columns:
                col_lines.append(f"{c} ({df[c].dtype})")
            report.append("\n- 列: " + ", ".join(col_lines))
            # 前 5 行（to_csv 竖线分隔，近似 markdown 表格）
            head = df.head(5).to_csv(sep="|", index=False)
            report.append("\n前 5 行：\n```\n" + head + "\n```")

            # 导出 CSV（所有非空 sheet 都导出，含字段说明表）
            safe = sheet.replace("/", "-").replace("\\", "-")
            csv_out = OUT_CSV / fname.replace(".xlsx", "") / f"{safe}.csv"
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_out, index=False, encoding="utf-8-sig")
            report.append(f"\n- 已导出: `{csv_out.relative_to(BASE)}`")

    out_md = BASE / "solution" / "data-xlsx-structure.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] report -> {out_md}")
    print(f"[OK] csvs -> {OUT_CSV}")


if __name__ == "__main__":
    main()
