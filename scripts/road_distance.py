# -*- coding: utf-8 -*-
"""
距最近道路距离提取（Geofabrik 中国 pbf → 云南网格）
- 用 pyosmium 流式过滤云南省 bbox 内的 highway 路段（motorway/trunk/primary/secondary/tertiary）
- 对每个 0.1° 网格中心点计算到最近路段顶点的近似距离（km）
  （近似口径：到路段节点集的最短球面距离，论文中如实注明）
输出：data/processed/dist_road.csv (gid, dist_road_km)
"""
import csv, math, os, sys
import osmium
from scipy.spatial import cKDTree
import numpy as np

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
PBF = f"{BASE}/data/geo/china-latest.osm.pbf"
BBOX = (21.1, 97.5, 29.3, 106.2)  # lat_min, lon_min, lat_max, lon_max
HWY = {"motorway", "trunk", "primary", "secondary", "tertiary"}

nodes = []

class WayHandler(osmium.SimpleHandler):
    def way(self, w):
        hw = w.tags.get("highway")
        if hw not in HWY:
            return
        for n in w.nodes:
            if n.location.valid():
                lat, lon = n.location.lat, n.location.lon
                if BBOX[0] - 0.2 <= lat <= BBOX[2] + 0.2 and BBOX[1] - 0.2 <= lon <= BBOX[3] + 0.2:
                    nodes.append((lat, lon))

print("解析 pbf（需要数分钟）...", flush=True)
h = WayHandler()
h.apply_file(PBF, locations=True)
print(f"路段节点数: {len(nodes)}", flush=True)

# 等距圆柱近似（云南纬度约24°，1°纬≈110.9km，1°经≈101.4km）
LATK, LONK = 110.9, 101.4
pts = np.array([(la * LATK, lo * LONK) for la, lo in nodes])
tree = cKDTree(pts)

out = []
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        la, lo = float(row["lat_c"]), float(row["lon_c"])
        d, _ = tree.query([(la * LATK, lo * LONK)])
        out.append((row["gid"], round(float(d[0]), 3)))

with open(f"{BASE}/data/processed/dist_road.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "dist_road_km"])
    w.writerows(out)
print(f"完成：{len(out)} 个网格", flush=True)
