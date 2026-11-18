# -*- coding: utf-8 -*-
"""
数据集装配（第3章数据集第三步）：气象旬聚合 + 标签合并 + 负样本抽样 + 描述统计输出
输入：data/meteo_era5/*.json（ERA5日尺度，见 era5_to_grid_daily.py）、
      data/processed/fire_labels_forest.csv（林地掩膜后正样本23958）、
      grids_yunnan.csv、static_features.csv、dist_road.csv
输出：
  data/processed/model_dataset.csv   建模数据集（正样本全量 + 负样本1:5抽样）
  data/processed/desc_stats.txt      3.4.2 描述性统计结果
变量（均截至该旬的前7日，2026-10-21 口径：数据源 ERA5 日尺度统计量）：
  tmax7_mean 前7日平均日最高气温；rh7_mean 前7日平均相对湿度（日均温/露点推算）；
  wind7_mean 前7日平均风速（日平均风速）；precip7_sum 前7日累计降水；
  dry_days 前7日内连续无降水日数（截至旬末的连续干旱长度）
"""
import csv, glob, json, os, random
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
random.seed(42)

def dekad_seq(dt):
    m, d = dt.month, dt.day
    if m >= 11:
        fy, mm = dt.year, m - 11
    elif m <= 5:
        fy, mm = dt.year - 1, m + 1
    elif m == 6 and d <= 15:
        fy, mm = dt.year - 1, 7
        dk = 0 if d <= 10 else 1
        return fy, mm * 3 + dk, dt
    else:
        return None
    dk = 0 if d <= 10 else (1 if d <= 20 else 2)
    return fy, mm * 3 + dk, dt

def dekad_end_date(fy, seq):
    """旬的结束日期（用于前7日窗口）。"""
    mm, dk = seq // 3, seq % 3
    if mm <= 1:
        y, m = fy, 11 + mm
    elif mm <= 6:
        y, m = fy + 1, mm - 1
    else:
        y, m = fy + 1, 6
    if m == 6:
        return datetime(y, 6, 10) if dk == 0 else datetime(y, 6, 15)
    d = (10, 20, 28)[dk] if dk < 2 else None
    if dk == 2:
        # 下旬最后一天
        if m == 12:
            d = 31
        else:
            d = (datetime(y, m + 1, 1) - timedelta(days=1)).day
    return datetime(y, m, d)

# 1. 载入网格与静态变量
grids = {}
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        grids[row["gid"]] = (float(row["lat_c"]), float(row["lon_c"]))

static = {}
sf_path = f"{BASE}/data/processed/static_features.csv"
if os.path.exists(sf_path):
    with open(sf_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            static[row["gid"]] = row

# 距道路距离（OSM，km）
road_path = f"{BASE}/data/processed/dist_road.csv"
if os.path.exists(road_path):
    with open(road_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            static.setdefault(row["gid"], {})["dist_road_km"] = row["dist_road_km"]

# 距居民点距离（OSM place=*，km）
stl_path = f"{BASE}/data/processed/dist_settlement.csv"
if os.path.exists(stl_path):
    with open(stl_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            static.setdefault(row["gid"], {})["dist_settlement_km"] = row["dist_settlement_km"]

# 1.5 NDVI 旬值（动态植被特征；优先读时间线性插补后的全量文件）
ndvi = {}
ndvi_path = f"{BASE}/data/processed/ndvi_dekad_imputed.csv"
if not os.path.exists(ndvi_path):
    ndvi_path = f"{BASE}/data/processed/ndvi_dekad.csv"
if os.path.exists(ndvi_path):
    with open(ndvi_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ndvi"] not in ("", None):
                ndvi[(row["gid"], int(row["fire_year"]), int(row["dekad_seq"]))] = float(row["ndvi"])
print(f"NDVI 旬值记录: {len(ndvi)}")

# 2. 正样本标签（林地掩膜口径，与第3章一致：正样本23958）
pos = {}
with open(f"{BASE}/data/processed/fire_labels_forest.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        fy = int(row["fire_year"])
        if fy > 2023:  # 2024-2025防火期不完整
            continue
        pos[(row["gid"], fy, int(row["dekad_seq"]))] = int(row["n_fires"])
print(f"正样本单元: {len(pos)}")

# 3. 逐网格读气象JSON，构建 日→指标 字典，再聚合旬特征
def load_meteo(gid):
    fp = f"{BASE}/data/meteo_era5/{gid}.json"
    if not os.path.exists(fp):
        return None
    j = json.load(open(fp, encoding="utf-8"))
    d = j["daily"]
    out = {}
    for i, ds in enumerate(d["time"]):
        tmax = d["temperature_2m_max"][i]
        rh = d["relative_humidity_2m_mean"][i]
        wnd = d["wind_speed_10m_mean"][i]
        pr = d["precipitation_sum"][i]
        if None in (tmax, rh, wnd, pr):
            continue
        out[ds] = (tmax, rh, wnd, pr)
    return out

def features_for(met, end_dt):
    """截至 end_dt 的前7日（含当日向前7天：end-6..end）指标。"""
    days = [(end_dt - timedelta(days=k)).strftime("%Y-%m-%d") for k in range(6, -1, -1)]
    vals = [met.get(ds) for ds in days]
    if any(v is None or any(x is None for x in v) for v in vals):
        return None
    tmax = [v[0] for v in vals]; rh = [v[1] for v in vals]
    wind = [v[2] for v in vals]; pr = [v[3] for v in vals]
    # 连续无降水日数（降水<1mm视为无降水，自旬末向前计）
    dry = 0
    for v in reversed(pr):
        if v < 1.0:
            dry += 1
        else:
            break
    return (np.mean(tmax), np.mean(rh), np.mean(wind), sum(pr), dry)

# 4. 装配样本：正样本全量；负样本 = 防火期全部网格×旬中未起火者，按1:5随机抽样
units_all = []
meteo_cache = {}
t0_count = 0
for gid in grids:
    met = load_meteo(gid)
    if met is None:
        continue
    meteo_cache[gid] = met

print(f"有气象数据的网格: {len(meteo_cache)}")

rows = []
for gid, met in meteo_cache.items():
    for fy in range(2009, 2024):
        for seq in range(23):  # 0..22（最后旬为6.11-15）
            end = dekad_end_date(fy, seq)
            if end > datetime(2024, 6, 15):
                continue
            key = (gid, fy, seq)
            y = 1 if key in pos else 0
            rows.append((gid, fy, seq, end, y))

print(f"防火期总单元数: {len(rows)}")
pos_rows = [r for r in rows if r[4] == 1]
neg_rows = [r for r in rows if r[4] == 0]
print(f"其中正样本 {len(pos_rows)}，负样本池 {len(neg_rows)}")
neg_sample = random.sample(neg_rows, min(5 * len(pos_rows), len(neg_rows)))
sample = pos_rows + neg_sample
random.shuffle(sample)
print(f"装配样本量: {len(sample)}")

# 5. 计算特征并写出
out_fields = ["gid", "fire_year", "dekad_seq", "y",
              "tmax7_mean", "rh7_mean", "wind7_mean", "precip7_sum", "dry_days",
              "ndvi",
              "forest_frac", "elev_mean", "slope_mean", "south_aspect_frac", "pop_density",
              "dist_road_km", "dist_settlement_km"]
written = 0
skipped = 0
with open(f"{BASE}/data/processed/model_dataset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(out_fields)
    for gid, fy, seq, end, y in sample:
        feat = features_for(meteo_cache[gid], end)
        if feat is None:
            skipped += 1
            continue
        st = static.get(gid, {})
        nv = ndvi.get((gid, fy, seq), "")
        row = [gid, fy, seq, y] + [round(v, 3) for v in feat] + [
            round(nv, 4) if nv != "" else "",
            st.get("forest_frac", ""), st.get("elev_mean", ""), st.get("slope_mean", ""),
            st.get("south_aspect_frac", ""), st.get("pop_density", ""),
            st.get("dist_road_km", ""), st.get("dist_settlement_km", "")]
        w.writerow(row)
        written += 1
print(f"写出 {written} 行，因气象缺失跳过 {skipped} 行")

# 6. 3.4.2 描述性统计
import statistics
cols = {k: [] for k in out_fields[4:]}
with open(f"{BASE}/data/processed/model_dataset.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        for k in cols:
            v = row[k]
            if v not in ("", None):
                cols[k].append(float(v))
lines = ["变量,均值,标准差,最小值,中位数,最大值"]
for k, vs in cols.items():
    if not vs:
        continue
    a = np.array(vs)
    lines.append(f"{k},{a.mean():.3f},{a.std():.3f},{a.min():.3f},{np.median(a):.3f},{a.max():.3f}")
with open(f"{BASE}/data/processed/desc_stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n描述性统计已写出 desc_stats.txt")
print("\n".join(lines))
