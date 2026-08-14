# -*- coding: utf-8 -*-
# 本程序及代码是在AI工具辅助下完成的。
# AI工具名称：Trae，版本/型号：1.0（CN），开发机构/公司：字节跳动，版本发布日期：2026-08-07 到 2026-08-10。

import pickle
d = pickle.load(open('outputs/data/c-data-cleaned.pkl', 'rb'))
nl = d['network_latency']
regions = ['RegionA', 'RegionB', 'RegionC', 'RegionD', 'RegionE', 'RegionF']
m = {}
for _, row in nl.iterrows():
    m[(row.FromRegion, row.ToRegion)] = row.NetworkLatency_ms
print('时延矩阵 (ms):')
print('     ' + '  '.join(f'{r[-1]:>4}' for r in regions))
for s in regions:
    row = []
    for t in regions:
        v = m.get((s, t), 999)
        mark = ' *' if v <= 20 else ''
        row.append(f'{v:>4}{mark}')
    print(f'{s[-1]}: ' + ' '.join(row))
print()
print('=== 实时推理(<=20ms)可达性 ===')
for s in regions:
    cand = [t for t in regions if m.get((s, t), 999) <= 20]
    print(f'{s}: {cand}')
print()
print('=== D 相关时延 ===')
for t in regions:
    print(f'D->{t}={m.get(("RegionD", t), 999)}ms   {t}->D={m.get((t, "RegionD"), 999)}ms')
