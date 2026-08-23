"""
目的：
    只读提取每张正式图所需的具体 pkl 字段值，供出图脚本取数。

原理：
    按图定位字段路径，打印标量/小结构，跳过超大频率字典。

性能：
    轻量-不适用（一次性小数据只读检查）。

输入数据：
    - S1-results.pkl / S2-results.pkl / S3-results.pkl (结果 pkl)

输出：
    - stdout 关键字段值

对应论文章节：
    § 出图（chart-generator）
"""
import pickle
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "outputs" / "data"


def load(name):
    with open(DATA / name, "rb") as f:
        return pickle.load(f)


def show(label, obj, maxlen=2000):
    s = str(obj)
    if len(s) > maxlen:
        s = s[:maxlen] + f"...<truncated len={len(str(obj))}>"
    print(f"--- {label} ---")
    print(s)
    print()


s1 = load("S1-results.pkl")
print("### S1 top-level keys:", list(s1.keys()))
for k in s1:
    if isinstance(s1[k], dict):
        print(f"  {k}: dict keys = {list(s1[k].keys())[:20]}")
    else:
        print(f"  {k}: {type(s1[k]).__name__} = {s1[k]}")
