"""
目的：
    只读提取 S1 图解读教学所需的 pkl 字段与 PDF 图元素信息。

原理：
    S1-results.pkl 与 S1-preprocessed.pkl 为训练落盘结果，本脚本仅做递归打印
    结构（键路径）与关键数值，不写入任何内容；PDF 用 PyMuPDF 提取每页文本与
    绘图指令坐标，确认每张图的标题/轴标签/图例/标注数值。

性能：
    轻量-不适用（秒级、只读、小数据）。

输入数据：
    - S1-results.pkl (处理后) — 训练结果（性能表/腺瘤口径/LOOCV/集成/特征重要性）
    - S1-preprocessed.pkl (处理后) — 预处理后特征
    - outputs/figures/S1-*.pdf (处理后) — 五张正式图

输出：
    控制台打印：pkl 键路径+关键标量；每张 PDF 的文本与坐标框信息。

对应论文章节：
    §S1 图解读教学文档（图解读教学-S1.md）
"""
import pickle
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent  # worktree root
DATA = ROOT / "outputs" / "data"
FIG = ROOT / "outputs" / "figures"


def dump_keys(obj, prefix="", depth=0, maxd=4):
    if depth > maxd:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            print("  " * depth + f"[DICT] {kp}  (type={type(v).__name__}, len={len(v) if hasattr(v,'__len__') else '-'})")
            dump_keys(v, kp, depth + 1, maxd)
    elif isinstance(obj, (list, tuple)):
        print("  " * depth + f"[LIST] {prefix or '<root>'} len={len(obj)} elem0_type={type(obj[0]).__name__ if obj else '-'}")
        if obj and depth <= maxd:
            dump_keys(obj[0], prefix + "[0]", depth + 1, maxd)


def extract_pdf(p):
    print(f"\n===== PDF: {p.name} =====")
    import fitz
    doc = fitz.open(str(p))
    print("pages:", len(doc))
    for pno, page in enumerate(doc):
        txt = page.get_text().strip()
        print(f"--- page {pno} text ---")
        print(txt if txt else "(no text)")
        # drawing objects (lines/rects/text spans coords)
        d = page.get_text("dict")
        blocks = d.get("blocks", [])
        print(f"--- page {pno}: {len(blocks)} blocks ---")
        for b in blocks:
            if b.get("type") == 0:
                for line in b.get("lines", []):
                    spans = " | ".join(s.get("text","") for s in line.get("spans",[]))
                    if spans.strip():
                        x0, y0, x1, y1 = line.get("bbox", (0,0,0,0))
                        print(f"  ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}): {spans}")
    doc.close()


if __name__ == "__main__":
    res = DATA / "S1-results.pkl"
    pre = DATA / "S1-preprocessed.pkl"
    if res.exists():
        print(f"\n===== S1-results : {res} =====")
        with open(res, "rb") as f:
            obj = pickle.load(f)
        print("ROOT TYPE:", type(obj).__name__)
        dump_keys(obj, "", 0, 4)
    else:
        print("MISSING", res)
    if pre.exists():
        print(f"\n===== S1-preprocessed : {pre} =====")
        with open(pre, "rb") as f:
            obj = pickle.load(f)
        print("ROOT TYPE:", type(obj).__name__)
        dump_keys(obj, "", 0, 4)
    else:
        print("MISSING", pre)
    for fn in ["S1-roc-curve.pdf", "S1-performance-compare.pdf",
               "S1-adenoma-sensitivity.pdf", "S1-feature-importance.pdf",
               "S1-threshold-analysis.pdf"]:
        p = FIG / fn
        if p.exists():
            extract_pdf(p)
        else:
            print("MISSING FIG", p)
