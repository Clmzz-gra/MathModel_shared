# -*- coding: utf-8 -*-
"""
sub3-lp-core.py — S3 LP 核心片段（≤15 行，摘自 sub3-model.py solve_region）
约束：C1 功率平衡 / C2 新能源上限 / C3 SOC 递推 / C4 终态 / C5 充电上限 / C7 碳 ε
变量块：G购电 S卖电 R直供 Cg电网充电 Cr新能源充电 D放电 E(SOC)，各 2406 时点
"""
NT, MAIN = 2406, 2400
off = {n: k * NT for k, n in enumerate(["G", "S", "R", "Cg", "Cr", "D", "E"])}
A_ub = lil_matrix((2 * NT + 2, n_vars)); b_ub = np.zeros(2 * NT + 2)
for t in range(NT):                                   # C2 R+Cr<=Avail；C5 Cg+Cr<=MaxCharge
    A_ub[t, off["R"] + t] = A_ub[t, off["Cr"] + t] = 1.0; b_ub[t] = avail[t]
    A_ub[NT + t, off["Cg"] + t] = A_ub[NT + t, off["Cr"] + t] = 1.0; b_ub[NT + t] = max_ch
for t in range(MAIN):                                 # C7 碳 ε：Σ G_t·CI_t <= 1e3·ε·base(kt→tCO2)
    A_ub[2 * NT, off["G"] + t] = ci[t]
b_ub[2 * NT] = 1e3 * eps * carbon_base_kt
A_ub[2 * NT + 1, off["E"] + NT - 1] = -1.0; b_ub[2 * NT + 1] = -init    # C4 终态 E_2405>=Init
A_eq = lil_matrix((2 * NT, n_vars)); b_eq = np.zeros(2 * NT)
for t in range(NT):                                   # C1 G+R+D-Cg-S=Load
    A_eq[t, [off["G"] + t, off["R"] + t, off["D"] + t]] = 1.0
    A_eq[t, off["Cg"] + t] = A_eq[t, off["S"] + t] = -1.0; b_eq[t] = load[t]
for t in range(NT):                                   # C3 E_t-E_{t-1}-ηc(Cg+Cr)+D/ηd=0, E_{-1}=Init
    A_eq[NT + t, off["E"] + t] = 1.0
    if t: A_eq[NT + t, off["E"] + t - 1] = -1.0
    A_eq[NT + t, off["Cg"] + t] = A_eq[NT + t, off["Cr"] + t] = -eta_c
    A_eq[NT + t, off["D"] + t] = 1.0 / eta_d; b_eq[NT + t] = init if t == 0 else 0.0
res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq,
              bounds=bounds, method="highs")          # c: G←Price, S←−SellPrice
