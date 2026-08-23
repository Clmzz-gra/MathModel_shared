"""
目的：
    用 PyMuPDF 提取 5 张数据特征图 PDF 的文本元素（标题/轴标签/图例/标注数值），确认图内实际内容。

原理：
    fitz 读取每页文本块，按块打印 (text, x0, y0, font, size)，供确认图元素与标注数值。

输入数据：
    - outputs/figures/chart-*.pdf x5

输出：
    打印每图文本元素

对应论文章节：数据理解节
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / 'outputs/figures'

files = [
    'chart-sample-composition.pdf',
    'chart-zero-sparsity.pdf',
    'chart-abundance-distribution.pdf',
    'chart-batch-effect.pdf',
    'chart-known-biomarker-presence.pdf',
]

for fn in files:
    print("\n" + "=" * 70)
    print("PDF:", fn)
    print("=" * 70)
    doc = fitz.open(str(FIG / fn))
    for page in doc:
        print(f"--- page {page.number} size={page.rect} ---")
        # 文本块
        blocks = page.get_text('dict')['blocks']
        for b in blocks:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    txt = span['text'].strip()
                    if txt:
                        print(f"  [{span['bbox'][0]:.0f},{span['bbox'][1]:.0f}] "
                              f"size={span['size']:.1f} | {txt}")
    doc.close()
