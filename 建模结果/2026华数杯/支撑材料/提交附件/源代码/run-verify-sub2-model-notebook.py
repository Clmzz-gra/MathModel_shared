# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    用 nbclient 执行 outputs/notebooks/verify-sub2-model.ipynb，验证全部 cell 可运行
    首次运行耗时步骤（基线/ε 三档）实际求解并缓存；二次运行秒级命中缓存

原理：
    NotebookClient.execute 顺序执行所有 cell；缓存 helper（load_or_compute）
    在 Cell 1 定义，指纹 = 版本 VER + 参数，命中即跳过计算

用法：
    python outputs/scratch/run-verify-sub2-model-notebook.py [--quick]
    --quick：只执行前 4 个 cell（加载+自检+基线缓存判定+分配），不跑 ε 三档

输出：
    执行后 notebook 写回（含输出）；打印各 cell 输出摘要
"""
import json
import sys
import time
from pathlib import Path

from nbclient import NotebookClient
import nbformat

BASE = Path(r"e:\MathModel_pj-2026-C")
nb_path = BASE / "outputs" / "notebooks" / "verify-sub2-model.ipynb"

quick = "--quick" in sys.argv

nb = nbformat.read(str(nb_path), as_version=4)

# quick 模式：只保留前 5 个 cell（md + 加载 + 自检 + 基线 + 分配）
if quick:
    keep = 5
    nb.cells = nb.cells[:keep]

t0 = time.time()
client = NotebookClient(nb, timeout=3600 * 4, kernel_name="python3", allow_errors=False)
try:
    client.execute()
    print(f"[OK] 全部 cell 执行成功，耗时 {time.time()-t0:.0f}s")
except Exception as e:
    print(f"[FAIL] cell 执行失败: {type(e).__name__}: {e}")
    print(f"已耗时 {time.time()-t0:.0f}s")

nbformat.write(nb, nb_path)
print(f"已写回 {nb_path}")

# 打印各 cell 输出摘要
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    outs = c.get("outputs", [])
    texts = []
    for o in outs:
        if o.output_type == "stream":
            texts.append("".join(o.text))
        elif o.output_type in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/plain" in d:
                texts.append("".join(d["text/plain"]))
    if texts:
        print(f"\n--- cell[{i}] ---")
        print("".join(texts)[:600])
