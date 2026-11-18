# -*- coding: utf-8 -*-
"""
构建云南省"网格×旬"火灾标签数据集（第3章正式数据集第一步）
输入：GADM 云南省界 + data/firms/ MODIS CSV
输出：
  data/processed/grids_yunnan.csv      网格清单（gid, lat_min, lon_min, 中心点）
  data/processed/fire_labels.csv       防火期"网格×旬"正样本标签（gid, fire_year, dekad_seq, 火点数）
  控制台输出统计数字（供第3章核对）
"""
import csv, glob, json, os
from collections import defaultdict
from datetime import datetime, date, timedelta
from shapely.geometry import shape, box, Point
from shapely.prepared import prep

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
GRID = 0.1

# 1. 云南省界
gj = json.load(open(f"{BASE}/data/geo/gadm41_CHN_1.json", encoding="utf-8"))
yn_geom = None
for feat in gj["features"]:
    if feat["properties"].get("NAME_1") == "Yunnan":
        yn_geom = shape(feat["geometry"])
        break
assert yn_geom is not None, "未找到云南省界"
yn_prep = prep(yn_geom)
print("云南省界已加载，bounds:", [round(v, 2) for v in yn_geom.bounds])

# 2. 网格清单：网格中心在省界内即保留
grids = {}
lon0, lat0 = 97.5, 21.1
nx = int(round((106.3 - lon0) / GRID))
ny = int(round((29.4 - lat0) / GRID))
for iy in range(ny):
    for ix in range(nx):
        la = lat0 + iy * GRID
        lo = lon0 + ix * GRID
        cx, cy = lo + GRID / 2, la + GRID / 2
        if yn_prep.contains(Point(cx, cy)):
            gid = (iy, ix)
            grids[gid] = (round(cy, 4), round(cx, 4))
print(f"云南省内网格数（中心点在省界内）：{len(grids)}")

with open(f"{BASE}/data/processed/grids_yunnan.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "lat_c", "lon_c", "lat_min", "lon_min"])
    for (iy, ix), (cy, cx) in sorted(grids.items()):
        w.writerow([f"{iy}_{ix}", cy, cx, round(lat0 + iy * GRID, 3), round(lon0 + ix * GRID, 3)])

# 3. 防火期旬序定义：每防火年度（Y年11月1日—Y+1年6月15日）的旬序
def dekad_seq(dt):
    """返回 (fire_year, dekad_seq)；fire_year 为防火期起始年份；不在防火期返回 None。"""
    m, d = dt.month, dt.day
    if m >= 11:
        fy = dt.year
        mm = m - 11  # 11月→0, 12月→1
    elif m <= 5:
        fy = dt.year - 1
        mm = m + 1   # 1月→2, ..., 5月→6
    elif m == 6 and d <= 15:
        fy = dt.year - 1
        mm = 7
        dk = 0 if d <= 10 else 1  # 6月：1-10为旬21，11-15为旬22
        return fy, mm * 3 + dk
    else:
        return None
    dk = 0 if d <= 10 else (1 if d <= 20 else 2)
    return fy, mm * 3 + dk

# 4. 火点 → 正样本标签
labels = defaultdict(int)  # (gid, fy, seq) -> 火点数
fire_pts = 0
in_pts = 0
for path in sorted(glob.glob(f"{BASE}/data/firms/modis_*_China.csv")):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        h = next(r)
        i_lat, i_lon = h.index("latitude"), h.index("longitude")
        i_date, i_conf = h.index("acq_date"), h.index("confidence")
        for row in r:
            try:
                lat = float(row[i_lat]); lon = float(row[i_lon])
                dt = datetime.strptime(row[i_date], "%Y-%m-%d")
                conf = float(row[i_conf])
            except Exception:
                continue
            if conf < 50:
                continue
            fire_pts += 1
            if not (97.5 <= lon <= 106.3 and 21.1 <= lat <= 29.4):
                continue
            ix = int((lon - lon0) / GRID)
            iy = int((lat - lat0) / GRID)
            gid = (iy, ix)
            if gid not in grids:
                continue
            # 精确省界：中心点法建网格后，火点需落在省界内
            if not yn_prep.contains(Point(lon, lat)):
                continue
            ds = dekad_seq(dt)
            if ds is None:
                continue
            in_pts += 1
            labels[(gid[0], gid[1], ds[0], ds[1])] += 1

print(f"MODIS置信度>=50火点总数（全国文件）: {fire_pts}")
print(f"落入云南省界且属防火期的火点: {in_pts}")
print(f"防火期网格×旬正样本单元: {len(labels)}")

with open(f"{BASE}/data/processed/fire_labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "fire_year", "dekad_seq", "n_fires"])
    for (iy, ix, fy, seq), n in sorted(labels.items()):
        w.writerow([f"{iy}_{ix}", fy, seq, n])

# 按防火年度统计
by_fy = defaultdict(int)
for (_, _, fy, _), n in labels.items():
    by_fy[fy] += 1
print("\n按防火年度（Y年11月—Y+1年6月15日）正样本单元数：")
for fy in sorted(by_fy):
    print(f"  {fy}-{fy+1}防火期: {by_fy[fy]}")
