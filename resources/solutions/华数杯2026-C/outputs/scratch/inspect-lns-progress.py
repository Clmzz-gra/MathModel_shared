# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

"""
目的：
    中途分析 S4 LNS 检查点（s4-lns-checkpoint.pkl）— 只读统计每区
    收敛进度/改进/耗时，供运行期间与完成后报告（PR-014）。

输入数据：
    - outputs/scratch/s4-lns-checkpoint.pkl（LNS 每 5 轮原子写）

输出：
    - 控制台统计
"""
import pickle
from pathlib import Path

SCRATCH = Path(r"e:\MathModel_pj-2026-C\outputs\scratch")
REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]


def main():
    p = SCRATCH / "s4-lns-checkpoint.pkl"
    if not p.exists():
        print("检查点不存在")
        return
    ck = pickle.load(open(p, "rb"))
    print("检查点区域:", list(ck.keys()))
    for r in REGIONS:
        rc = ck.get(r)
        if not rc:
            print(r, ": 未开始")
            continue
        curve = rc["curve"]
        if not curve:
            print(r, ": 无曲线（尚未迭代）")
            continue
        costs = [c["cost"] for c in curve]
        imps = [c["improvement"] for c in curve]
        times = sorted(c["time_s"] for c in curve)
        print(f"{r}: it={rc['it']} current={rc['current_cost']:.2f}M "
              f"done={rc.get('done')}")
        print(f"  曲线: 起点 {costs[0]:.2f} -> 当前 {costs[-1]:.2f} M"
              f" | 累计改进 {sum(imps):.3f} M")
        print(f"  最近5轮改进: {[f'{i:.4f}' for i in imps[-5:]]}")
        print(f"  耗时: median {times[len(times)//2]:.0f}s | max {times[-1]:.0f}s"
              f" | 总 {sum(c['time_s'] for c in curve):.0f}s")


if __name__ == "__main__":
    main()
