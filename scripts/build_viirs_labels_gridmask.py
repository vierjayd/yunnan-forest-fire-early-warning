# -*- coding: utf-8 -*-
"""
VIIRS 标签构建（网格级林地掩膜，与既有 MODIS fire_labels_forest.csv 同口径）
口径：置信度剔除 low -> 省界精确裁剪 -> 防火期旬映射 -> 仅保留林地占比>=0.5 的林区网格
说明：经对照验证，既有 MODIS 林地标签系"网格级林地占比>=0.5 掩膜"产物，
     而非火点级 WorldCover 采样（后者仅得 21738 单元，与既有 23958 不符）。
     为保持双源一致，VIIRS 采用同一网格级掩膜。
输出：data/processed/fire_labels_forest_viirs.csv
"""
import csv, glob, json
from collections import defaultdict
from datetime import datetime
from shapely.geometry import shape, Point
from shapely.prepared import prep

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
GRID = 0.1
lon0, lat0 = 97.5, 21.1

gj = json.load(open(f"{BASE}/data/geo/gadm41_CHN_1.json", encoding="utf-8"))
yn_geom = None
for feat in gj["features"]:
    if feat["properties"].get("NAME_1") == "Yunnan":
        yn_geom = shape(feat["geometry"]); break
yn_prep = prep(yn_geom)

grids = set()
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        grids.add(r["gid"])

forest_frac = {}
with open(f"{BASE}/data/processed/static_features.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        forest_frac[r["gid"]] = float(r["forest_frac"])

def dekad_seq(dt):
    m, d = dt.month, dt.day
    if m >= 11:
        fy, mm = dt.year, m - 11
    elif m <= 5:
        fy, mm = dt.year - 1, m + 1
    elif m == 6 and d <= 15:
        fy, mm = dt.year - 1, 7
        dk = 0 if d <= 10 else 1
        return fy, mm * 3 + dk
    else:
        return None
    dk = 0 if d <= 10 else (1 if d <= 20 else 2)
    return fy, mm * 3 + dk

labels = defaultdict(int)
tot = inprov = forest_pts = 0
for path in sorted(glob.glob(f"{BASE}/data/firms/viirs-snpp_*_China.csv")):
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f); h = next(r)
        i_lat, i_lon = h.index("latitude"), h.index("longitude")
        i_date, i_conf = h.index("acq_date"), h.index("confidence")
        for row in r:
            try:
                lat = float(row[i_lat]); lon = float(row[i_lon])
                dt = datetime.strptime(row[i_date], "%Y-%m-%d")
            except Exception:
                continue
            if row[i_conf].strip().lower() == "l":
                continue
            tot += 1
            if not (97.5 <= lon <= 106.3 and 21.1 <= lat <= 29.4):
                continue
            ix = int((lon - lon0) / GRID); iy = int((lat - lat0) / GRID)
            gid = f"{iy}_{ix}"
            if gid not in grids:
                continue
            if not yn_prep.contains(Point(lon, lat)):
                continue
            ds = dekad_seq(dt)
            if ds is None:
                continue
            inprov += 1
            if forest_frac.get(gid, 0) < 0.5:
                continue
            forest_pts += 1
            labels[(gid, ds[0], ds[1])] += 1

with open(f"{BASE}/data/processed/fire_labels_forest_viirs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "fire_year", "dekad_seq", "n_fires"])
    for (gid, fy, seq), n in sorted(labels.items()):
        w.writerow([gid, fy, seq, n])

print(f"VIIRS 置信度过滤后总火点: {tot}")
print(f"省界内防火期火点: {inprov}")
print(f"林区网格内火点: {forest_pts}")
print(f"正样本单元: {len(labels)}；涉及网格: {len(set(g for g,_,_ in labels))}")
by_fy = defaultdict(int)
for (g, fy, s), n in labels.items():
    by_fy[fy] += 1
print("各防火期正样本单元:", dict(sorted(by_fy.items())))
