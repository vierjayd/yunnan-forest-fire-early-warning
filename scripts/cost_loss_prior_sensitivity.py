# -*- coding: utf-8 -*-
"""2.9/3.8 敏感性复核：气候学频率参照由评价期实际发生率换为校准期业务先验。

背景：审核意见 A2/A3 指出主稿 2.9 曾写『改用业务先验 0.020363 时 V_max 由
0.4296 变为 0.4231』，但该数字无落盘计算记录。本脚本只读冻结缓存，按三种
明确口径重算，输出 cost_loss_results/prior_sensitivity.csv，供正文引用。

口径定义（V=(E_c-E_f)/(E_c-E_p)，E_f 基于 p>alpha 的决策计数，与 pi 无关）：
  A  pi_clim=pi_perf=pi_test  主结果口径（应复现 V_max=0.4296）
  B  pi_clim=pi_perf=pi_cal   气候学参照与完美预报上界全部改用业务先验
  C  pi_clim=pi_cal, pi_perf=pi_test  仅气候学固定策略切换点改用业务先验
"""
import os
import numpy as np
import pandas as pd

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
OUT = "E:/workbuddyoutput/2026-11-10-10-24-23/analysis_materials/cost_loss_results"
os.makedirs(OUT, exist_ok=True)

z = np.load(BASE + "/results/figure_cache.npz")
y = z["yte"]
p = z["p_xgb_cal"].astype(np.float64)
n = len(y)
pos = int(y.sum())
pi_test = pos / n
pi_cal = 0.0203630841  # split_dist.csv 校准期业务先验（主稿 2.3/2.5/表5注/3.4）
alphas = np.round(np.linspace(0.001, 0.30, 1200), 6)  # 与 cost_loss_analysis.py 一致

def V(alpha, pi_clim, pi_perf):
    flag = p > alpha
    hit = int((y[flag] == 1).sum())
    fa = int(flag.sum() - hit)
    miss = pos - hit
    e_f = (alpha * (hit + fa) + miss) / n
    e_c = min(alpha, pi_clim)
    e_p = alpha * pi_perf
    denom = e_c - e_p
    return np.nan if denom <= 1e-15 else (e_c - e_f) / denom

scenarios = {
    "A_pi_test_main": (pi_test, pi_test),
    "B_pi_cal_all": (pi_cal, pi_cal),
    "C_switch_only": (pi_cal, pi_test),
}
rows = []
for name, (pc, pp) in scenarios.items():
    vv = np.array([V(a, pc, pp) for a in alphas])
    i = int(np.nanargmax(np.where(np.isfinite(vv), vv, -np.inf)))
    rows.append({"scenario": name, "pi_clim": pc, "pi_perf": pp,
                 "V_max": vv[i], "alpha_star": alphas[i]})
df = pd.DataFrame(rows)
df.to_csv(OUT + "/prior_sensitivity.csv", index=False, encoding="utf-8-sig")
print("n=%d pos=%d pi_test=%.10f pi_cal=%.10f" % (n, pos, pi_test, pi_cal))
print(df.to_string(index=False))
