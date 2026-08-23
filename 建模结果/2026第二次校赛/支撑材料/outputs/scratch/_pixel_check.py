"""分析渲染 PNG 的像素，判断标题/图例区域是否有文字（暗像素）。"""
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(r"E:\MathModel_pj-2026-sim2-B\outputs\scratch\_render_s2")

for name in ["S2-stable-frequency", "S2-tau-sensitivity", "S2-cooccurrence-heatmap"]:
    img = Image.open(OUT / f"{name}-p0.png").convert("L")
    a = np.array(img)
    h, w = a.shape
    dark = a < 128
    print("=" * 60)
    print(name, f"size={w}x{h}")
    # 顶部 15% 区域（标题区）
    top = dark[:int(h*0.15), :]
    print(f"  顶部15%标题区暗像素占比: {top.mean():.4f}")
    # 底部 10% 区域（xlabel 区）
    bot = dark[int(h*0.9):, :]
    print(f"  底部10%轴标签区暗像素占比: {bot.mean():.4f}")
    # 左侧 15% 区域（ylabel 区）
    left = dark[:, :int(w*0.15)]
    print(f"  左侧15%轴标签区暗像素占比: {left.mean():.4f}")
    # 右上角图例区
    legend = dark[:int(h*0.3), int(w*0.6):]
    print(f"  右上图例区暗像素占比: {legend.mean():.4f}")
    # 全图暗像素占比
    print(f"  全图暗像素占比: {dark.mean():.4f}")
