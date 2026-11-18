# -*- coding: utf-8 -*-
"""
MOD13A2 NDVI 网格化提取
- HDF4 (pyhdf) 读取 1km 16天 NDVI，有效值 -2000~10000，比例因子 0.0001
- MODIS 正弦投影：景宽 1111950.354m，像元 926.625m，1200x1200
- 对云南省 3430 个 0.1° 网格中心点取 5x5 像元窗口均值
- 合成期起始日期映射到防火期 (fire_year, dekad_seq)
输出：data/processed/ndvi_dekad.csv  (gid, fire_year, dekad_seq, ndvi)
"""
import csv, glob, os, re
import numpy as np
from datetime import datetime
from pyhdf.SD import SD, SDC
from pyproj import Transformer

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
TILE_W = 1111950.354
PIX = 926.625
X0 = -20015109.354
Y0 = 10007554.677

trans = Transformer.from_crs("EPSG:4326", "+proj=sinu +R=6371007.181 +nadgrids=@null", always_xy=True)

def dekad_seq(dt):
    m, d = dt.month, dt.day
    if m >= 11:
        fy, mm = dt.year, m - 11
    elif m <= 5:
        fy, mm = dt.year - 1, m + 1
    elif m == 6 and d <= 15:
        fy, mm = dt.year - 1, 7
        return fy, mm * 3 + (0 if d <= 10 else 1)
    else:
        return None
    return fy, mm * 3 + (0 if d <= 10 else (1 if d <= 20 else 2))

# 网格中心点 → 正弦投影坐标（只算一次）
grids = {}
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        x, y = trans.transform(float(row["lon_c"]), float(row["lat_c"]))
        grids[row["gid"]] = (x, y)
print(f"网格数: {len(grids)}")

out_rows = []
files = sorted(glob.glob(f"{BASE}/data/ndvi/*.hdf"))
print(f"HDF 文件数: {len(files)}")
n_ok = 0
for fp in files:
    fn = os.path.basename(fp)
    m = re.search(r"\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.", fn)
    if not m:
        continue
    yr, doy, tile = int(m.group(1)), int(m.group(2)), m.group(3)
    dt = datetime(yr, 1, 1) + __import__("datetime").timedelta(days=doy - 1)
    ds = dekad_seq(dt)
    if ds is None:
        continue
    fy, seq = ds
    h, v = int(tile[1:3]), int(tile[4:6])
    tx0 = X0 + h * TILE_W
    ty0 = Y0 - v * TILE_W
    try:
        hdf = SD(fp, SDC.READ)
        arr = hdf.select("1 km 16 days NDVI").get()
        hdf.end()
    except Exception as e:
        print(f"[读取失败] {fn}: {e}")
        continue
    n_ok += 1
    for gid, (gx, gy) in grids.items():
        col = (gx - tx0) / PIX
        row = (ty0 - gy) / PIX
        c = int(col); r = int(row)
        if not (2 <= c < 1198 and 2 <= r < 1198):
            continue
        win = arr[r-2:r+3, c-2:c+3].astype("float64")
        valid = win[(win >= -2000) & (win <= 10000)]
        if valid.size == 0:
            continue
        out_rows.append((gid, fy, seq, round(float(valid.mean()) * 0.0001, 4)))

print(f"成功处理 {n_ok} 个文件，输出 {len(out_rows)} 条记录")
with open(f"{BASE}/data/processed/ndvi_dekad.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["gid", "fire_year", "dekad_seq", "ndvi"])
    w.writerows(out_rows)
print("已写出 ndvi_dekad.csv")
