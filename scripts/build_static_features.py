# -*- coding: utf-8 -*-
"""
静态变量网格化提取（第3章数据集第二步）
输入：data/geo/ 下的 WorldCover(10m)、SRTM(90m zip)、WorldPop(1km)
输出：data/processed/static_features.csv
  gid, forest_frac(林地占比), dominant_lc(主导地类), elev_mean, slope_mean, aspect_ns(阴阳坡), pop_density
林地判定：WorldCover 类别 10（乔木覆盖）
"""
import csv, glob, io, os, zipfile
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
GRID = 0.1
LAT0, LON0 = 21.1, 97.5

grids = {}
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        iy, ix = row["gid"].split("_")
        grids[row["gid"]] = (int(iy), int(ix))
print(f"网格数: {len(grids)}")

def grid_window(lat_min, lon_min, transform, inv=True):
    """0.1°网格在栅格中的行列范围（近似，栅格均为WGS84经纬度）。"""
    c0 = int((lon_min - transform.c) / transform.a)
    c1 = int((lon_min + GRID - transform.c) / transform.a)
    r0 = int((transform.f - (lat_min + GRID)) / abs(transform.e))
    r1 = int((transform.f - lat_min) / abs(transform.e))
    return max(r0, 0), max(r1, 0), max(c0, 0), max(c1, 0)

results = {g: {} for g in grids}

# ---------- 1. WorldCover：林地占比与主导地类 ----------
wc_files = sorted(glob.glob(f"{BASE}/data/geo/wc_*.tif"))
print(f"WorldCover 文件数: {len(wc_files)}")
if wc_files:
    for fp in wc_files:
        with rasterio.open(fp) as src:
            tr = src.transform
            data = None
            for gid, (iy, ix) in grids.items():
                lat_min = LAT0 + iy * GRID
                lon_min = LON0 + ix * GRID
                if not (src.bounds.left <= lon_min <= src.bounds.right and
                        src.bounds.bottom <= lat_min + GRID <= src.bounds.top + 0.2):
                    continue
                r0, r1, c0, c1 = grid_window(lat_min, lon_min, tr)
                if r1 <= r0 or c1 <= c0:
                    continue
                if data is None:
                    data = src.read(1)
                sub = data[r0:r1, c0:c1]
                if sub.size == 0:
                    continue
                valid = sub[sub != 0]
                if valid.size == 0:
                    continue
                forest = float((valid == 10).mean())
                vals, cnt = np.unique(valid, return_counts=True)
                dom = int(vals[cnt.argmax()])
                if "forest_frac" in results[gid]:
                    # 跨景取平均（边缘网格可能被两景覆盖）
                    prev = results[gid]["forest_frac"]
                    results[gid]["forest_frac"] = (prev + forest) / 2
                else:
                    results[gid]["forest_frac"] = round(forest, 4)
                    results[gid]["dominant_lc"] = dom
    print("WorldCover 提取完成")

# ---------- 2. SRTM：海拔、坡度、坡向 ----------
srtm_zips = sorted(glob.glob(f"{BASE}/data/geo/srtm_*.zip"))
print(f"SRTM 压缩包数: {len(srtm_zips)}")
if srtm_zips:
    for zp in srtm_zips:
        with zipfile.ZipFile(zp) as z:
            tif_name = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
            raw = z.read(tif_name)
        with rasterio.io.MemoryFile(raw) as memfile:
            with memfile.open() as src:
                tr = src.transform
                data = src.read(1).astype("float32")
                data[data < -1000] = np.nan
                # 坡度坡向（近似，度/度转坡度）
                dx_deg = abs(tr.a); dy_deg = abs(tr.e)
                dx_m = dx_deg * 111320.0
                dy_m = dy_deg * 110540.0
                gy, gx = np.gradient(data, dy_m, dx_m)
                slope = np.degrees(np.arctan(np.hypot(gx, gy)))
                aspect = np.degrees(np.arctan2(-gx, gy)) % 360  # 0=北
                for gid, (iy, ix) in grids.items():
                    lat_min = LAT0 + iy * GRID
                    lon_min = LON0 + ix * GRID
                    if not (src.bounds.left - 0.01 <= lon_min and lon_min + GRID <= src.bounds.right + 0.01):
                        continue
                    r0, r1, c0, c1 = grid_window(lat_min, lon_min, tr)
                    r0 = max(r0, 0); c0 = max(c0, 0)
                    r1 = min(r1, data.shape[0]); c1 = min(c1, data.shape[1])
                    if r1 <= r0 or c1 <= c0:
                        continue
                    sub_e = data[r0:r1, c0:c1]
                    sub_s = slope[r0:r1, c0:c1]
                    sub_a = aspect[r0:r1, c0:c1]
                    if np.isnan(sub_e).all():
                        continue
                    elev = float(np.nanmean(sub_e))
                    slp = float(np.nanmean(sub_s))
                    # 阴阳坡：北坡(0-45,315-360)阴，南坡(135-225)阳 → 南坡占比
                    a = sub_a[~np.isnan(sub_a)]
                    south_frac = float(((a >= 135) & (a <= 225)).mean()) if a.size else np.nan
                    for k, v in (("elev_mean", elev), ("slope_mean", slp), ("south_aspect_frac", south_frac)):
                        if k in results[gid]:
                            results[gid][k] = round((results[gid][k] + v) / 2, 3)
                        else:
                            results[gid][k] = round(v, 3)
    print("SRTM 提取完成")

# ---------- 3. WorldPop：人口密度 ----------
wp = f"{BASE}/data/geo/chn_pd_2020_1km.tif"
if os.path.exists(wp):
    from rasterio.windows import Window
    with rasterio.open(wp) as src:
        tr = src.transform
        H, W = src.height, src.width
        for gid, (iy, ix) in grids.items():
            lat_min = LAT0 + iy * GRID
            lon_min = LON0 + ix * GRID
            r0, r1, c0, c1 = grid_window(lat_min, lon_min, tr)
            r0 = max(r0, 0); c0 = max(c0, 0)
            r1 = min(r1, H); c1 = min(c1, W)
            if r1 <= r0 or c1 <= c0:
                continue
            sub = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype("float32")
            sub[sub < 0] = np.nan
            if np.isnan(sub).all():
                continue
            results[gid]["pop_density"] = round(float(np.nanmean(sub)), 3)
    print("WorldPop 提取完成")

# ---------- 输出 ----------
fields = ["gid", "forest_frac", "dominant_lc", "elev_mean", "slope_mean", "south_aspect_frac", "pop_density"]
with open(f"{BASE}/data/processed/static_features.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(fields)
    for gid in sorted(grids):
        r = results[gid]
        w.writerow([gid] + [r.get(k, "") for k in fields[1:]])

covered = sum(1 for g in grids if results[g])
print(f"\n输出 static_features.csv，覆盖网格 {covered}/{len(grids)}")
