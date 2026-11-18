# -*- coding: utf-8 -*-
"""
NDVI 旬值时间可得性约束下的前向携带（2026-11-09 答辩整改 Q1=A）

背景与整改原因
--------------
MOD13A2 为 16 天合成产品，ndvi_dekad.csv 按合成期**起始日**映射到旬。
原方案（方案 1A）用 np.interp 沿全局旬序做双向线性插值，第 t 旬的值可能由
t 之后的观测内插得到，与「所有解释变量严格先于标签」的口径冲突，构成
时间可得性泄漏。本版改为**仅使用当时已经完成并可获得的最近一期产品**。

口径
----
1) 观测点 t_obs（合成期起始旬）视为在该旬开始采集，完成 16 天合成并可供
   业务使用的最早时刻保守取 **t_obs + 2 旬**（约 20 天，覆盖合成窗口与
   产品处理延迟）。这是比「起始日即入模」更保守的可用性假设。
2) 对目标旬 t，只使用满足 available_t <= t 的最近一期观测值，前向携带，
   绝不使用 t 之后的观测，也不做跨期双向内插。
3) 极早期（t 早于该网格首个可用时刻）以该网格最早一期观测初始化。该情形
   只影响 2009 防火期最初 1—2 旬，而 2009 期已从建模总体剔除，故不进入
   任何样本与结论。
4) 13 个 MODIS 瓦片边缘全期无有效观测的网格，仍以空间最近邻网格的同期
   序列替代；该替代序列同样满足上述时间可得性约束。

输出: data/processed/ndvi_dekad_imputed.csv（gid, fire_year, dekad_seq, ndvi）
"""
import csv
import numpy as np

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
T = 15 * 23  # 345 个旬（2009—2023 防火期；2009 期不进入建模）
AVAIL_LAG = 2  # 合成期起始旬 + 2 旬 = 实际可用时刻（保守）

# 1. 读入网格与 NDVI 观测
grids = {}
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        grids[r["gid"]] = (float(r["lat_c"]), float(r["lon_c"]))

obs = {g: {} for g in grids}
with open(f"{BASE}/data/processed/ndvi_dekad.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        g = r["gid"]
        if g in obs and r["ndvi"] not in ("", None):
            t = (int(r["fire_year"]) - 2009) * 23 + int(r["dekad_seq"])
            obs[g][t] = float(r["ndvi"])

# 2. 仅过去值的前向携带
series = {}
empty = []
init_rows = 0
for g, rec in obs.items():
    if not rec:
        empty.append(g)
        continue
    ts = np.array(sorted(rec.keys()))
    vs = np.array([rec[t] for t in ts])
    avail = ts + AVAIL_LAG
    out = np.empty(T)
    j, cur = 0, float(vs[0])
    for t in range(T):
        while j < len(avail) and avail[j] <= t:
            cur = float(vs[j])
            j += 1
        if t < avail[0]:
            init_rows += 1
        out[t] = cur
    series[g] = out
print(f"前向携带完成网格: {len(series)}；全缺测网格: {len(empty)}；"
      f"起点初始化单元: {init_rows}（仅 2009 期前段，不进入建模）")

# 3. 全缺测网格 -> 空间最近邻同期序列（序列本身亦满足时间可得性）
if empty:
    have = [g for g in grids if g in series]
    for g in empty:
        la, lo = grids[g]
        nearest = min(have, key=lambda h: (grids[h][0] - la) ** 2 + (grids[h][1] - lo) ** 2)
        series[g] = series[nearest].copy()
        print(f"  {g} <- 最近邻 {nearest}")

# 4. 写出全量单元
n = 0
with open(f"{BASE}/data/processed/ndvi_dekad_imputed.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "fire_year", "dekad_seq", "ndvi"])
    for g in grids:
        s = series[g]
        for t in range(T):
            w.writerow([g, 2009 + t // 23, t % 23, round(float(s[t]), 4)])
            n += 1
print(f"写出 {n} 行（全 {len(grids)} 网格 x 345 旬）-> ndvi_dekad_imputed.csv")
