# -*- coding: utf-8 -*-
"""
距居民点距离提取（第3章静态变量补全）
数据源: data/geo/china-latest.osm.pbf（已下载）
方法: pyosmium 流式解析，收集云南省域及周边的居民点要素：
  - place = city / town / village / hamlet / suburb / neighbourhood / isolated_dwelling（节点）
  - landuse = residential（面要素取质心）
  对每个 0.1° 网格中心，用 cKDTree 求最近居民点距离（等距圆柱近似，km）。
输出: data/processed/dist_settlement.csv (gid, dist_settlement_km, nearest_place_type)
"""
import csv, math, os
import osmium
import numpy as np
from scipy.spatial import cKDTree

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
PBF = f"{BASE}/data/geo/china-latest.osm.pbf"
# 云南区域加 0.5° 缓冲（边界网格的最近居民点可能在省外）
LAT0, LAT1, LON0, LON1 = 21.0 - 0.5, 29.4 + 0.5, 97.4 - 0.5, 106.4 + 0.5

PLACE_TYPES = {"city", "town", "village", "hamlet", "suburb", "neighbourhood", "isolated_dwelling"}

pts = []   # (lat, lon)
ptypes = []

class Handler(osmium.SimpleHandler):
    def node(self, n):
        if not (LAT0 <= n.location.lat <= LAT1 and LON0 <= n.location.lon <= LON1):
            return
        p = n.tags.get("place")
        if p in PLACE_TYPES:
            pts.append((n.location.lat, n.location.lon))
            ptypes.append(p)

class AreaHandler(osmium.SimpleHandler):
    def area(self, a):
        if a.tags.get("landuse") != "residential":
            return
        try:
            # 面质心：取外环节点均值（osmium Area 含多边形环）
            ring = next(iter(a.outer_rings()), None)
            if ring is None:
                return
            lats = [nd.lat for nd in ring if nd.valid()]
            lons = [nd.lon for nd in ring if nd.valid()]
            if not lats:
                return
            la, lo = sum(lats) / len(lats), sum(lons) / len(lons)
            if LAT0 <= la <= LAT1 and LON0 <= lo <= LON1:
                pts.append((la, lo)); ptypes.append("residential_area")
        except Exception:
            pass

print("解析 OSM pbf（节点 place=*）...")
Handler().apply_file(PBF, locations=False)
print(f"place 节点: {len(pts)}")

print("解析 OSM pbf（面 landuse=residential）...")
AreaHandler().apply_file(PBF, locations=True)
print(f"居民点总数（含居住面质心）: {len(pts)}")

# KDTree（等距圆柱近似：x=lon*cos(lat0), y=lat，单位度，最后 x111.19 km/度）
latm = math.radians((LAT0 + LAT1) / 2)
cosf = math.cos(latm)
XY = np.array([[lo * cosf, la] for la, lo in pts])
tree = cKDTree(XY)

rows = []
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        la, lo = float(row["lat_c"]), float(row["lon_c"])
        d, i = tree.query([lo * cosf, la])
        rows.append((row["gid"], round(float(d) * 111.19, 3), ptypes[i]))

with open(f"{BASE}/data/processed/dist_settlement.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "dist_settlement_km", "nearest_place_type"])
    w.writerows(rows)

arr = [r[1] for r in rows]
print(f"写出 {len(rows)} 行 -> data/processed/dist_settlement.csv")
print(f"距离统计(km): min={min(arr)} max={max(arr)} mean={sum(arr)/len(arr):.2f}")
