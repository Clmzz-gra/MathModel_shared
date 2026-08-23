"""检查 Type3 字体是否真正包含中文字形（判断中文是否渲染）。"""
import fitz
from pathlib import Path

FIG = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\figures")

for name in ["S2-tau-sensitivity"]:
    doc = fitz.open(str(FIG / f"{name}.pdf"))
    page = doc[0]
    # 获取页面所有绘制内容，统计 Type3 字体的字形
    # 通过 xref 检查字体对象
    fonts = page.get_fonts()
    for f in fonts:
        xref = f[0]
        print(f"font xref={xref} name={f[3]} type={f[2]}")
        # 尝试读取字体对象
        try:
            obj = doc.xref_object(xref)
            # 打印前 2000 字符
            print(obj[:2000])
        except Exception as e:
            print("  err", e)
    print()
