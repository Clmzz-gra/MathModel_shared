"""
目的：
    只读检查 S1/S2/S3 结果 pkl 的键结构与关键字段，供出图脚本取数。

原理：
    递归打印 dict 键与标量值，定位每张正式图所需字段路径。

性能：
    轻量-不适用（一次性小数据只读检查）。

输入数据：
    - S1-results.pkl / S2-results.pkl / S3-results.pkl (结果 pkl)

输出：
    - stdout 结构打印

对应论文章节：
    § 出图（chart-generator）
"""
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"


def walk(obj, prefix="", depth=0, maxdepth=6):
    if depth > maxdepth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                print(f"{'  '*depth}{key}: dict({len(v)})")
                walk(v, key, depth + 1, maxdepth)
            elif isinstance(v, (list, tuple)):
                print(f"{'  '*depth}{key}: {type(v).__name__}[{len(v)}]")
                if v and isinstance(v[0], dict):
                    walk(v[0], f"{key}[0]", depth + 1, maxdepth)
            else:
                s = str(v)
                if len(s) > 60:
                    s = s[:60] + "..."
                print(f"{'  '*depth}{key}: {type(v).__name__} = {s}")
    else:
        print(f"{'  '*depth}{prefix}: {type(obj).__name__}")


for name in ["S1-results.pkl", "S2-results.pkl", "S3-results.pkl"]:
    p = DATA / name
    print("=" * 70)
    print(f"### {name}")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    print(f"top-level type: {type(obj).__name__}")
    walk(obj, maxdepth=5)
