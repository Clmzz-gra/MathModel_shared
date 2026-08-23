"""
目的：
    只读提取 S1/S2/S3-results.pkl 的顶层键结构，供 data-integration 溯源核对。

原理：
    pickle.load 后递归打印 dict 键路径（深度受限），不修改任何文件。

性能：
    轻量-不适用（秒级一次性小数据）。

输入数据：
    - S1-results.pkl / S2-results.pkl / S3-results.pkl (处理后) — 各子问题结果

输出：
    - 控制台打印各 pkl 顶层键与嵌套键路径

对应论文章节：
    data-integration 单文件（溯源标注）
"""
import pickle, sys

def walk(obj, prefix="", depth=0, maxdepth=3):
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}  <{type(v).__name__}>")
            walk(v, prefix + "  ", depth + 1, maxdepth)
    elif isinstance(obj, (list, tuple)):
        if obj:
            print(f"{prefix}[0]  <{type(obj[0]).__name__}>")
            walk(obj[0], prefix + "  ", depth + 1, maxdepth)

for f in [r"outputs/data/S1-results.pkl", r"outputs/data/S2-results.pkl", r"outputs/data/S3-results.pkl"]:
    print("=" * 60)
    print("FILE:", f)
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    walk(d, maxdepth=2)
