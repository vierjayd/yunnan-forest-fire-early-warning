# -*- coding: utf-8 -*-
"""
ERA5 逐时 ZIP -> 逐网格日值 JSON（v2，hourly 版，2026-10-27 全部 60 段下载完成后执行）
输入: data/era5/era5_*.nc（实为 ZIP，内含 data_stream-oper_stepType-instant.nc
      [t2m/d2m/u10/v10 逐时] 与 data_stream-oper_stepType-accum.nc [tp 逐时累计]）
处理: 1) 解压到临时文件读取；
      2) 日界按北京时间（UTC+8）：UTC 时刻 +8h 的日期为该日；
      3) 跨文件同日的分量按 求和/取最大/计小时数 累积（修正段界日的部分小时问题）；
      4) t2m -> 日均温/日最高气温；d2m -> 日均露点；风速 = hypot(u,v) 后日均；
         tp 逐时值即 1 小时累计量，日降水 = 当日整点值求和；
      5) RH 由日均温、日均露点经 Magnus 公式推算。
输出: data/meteo_era5/{gid}.json（与 assemble_dataset.py 约定字段一致）
"""
import csv, glob, json, os, shutil, zipfile, tempfile
import numpy as np
import xarray as xr

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
SRC = f"{BASE}/data/era5"
OUT = f"{BASE}/data/meteo_era5"
os.makedirs(OUT, exist_ok=True)

# 1. 网格中心（顺序固定，后续用位置索引）
gids, lats, lons = [], [], []
with open(f"{BASE}/data/processed/grids_yunnan.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gids.append(row["gid"]); lats.append(float(row["lat_c"])); lons.append(float(row["lon_c"]))
lats = np.array(lats); lons = np.array(lons)
G = len(gids)
print(f"网格数: {G}", flush=True)

TMPDIR = tempfile.mkdtemp(prefix="era5_")

def read_member(zf, member, tag, kind):
    path = os.path.join(TMPDIR, f"{tag}_{kind}.nc")
    with open(path, "wb") as f:
        f.write(zf.read(member))
    return xr.open_dataset(path)

# 全局按日期累积：date -> {s_t2m, m_t2m, s_d2m, s_wind, s_tp, n}
acc = {}

def rh_from_t_td(t, td):
    return 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))

files = sorted(glob.glob(f"{SRC}/era5_*.nc"))
files = [f for f in files if "smoke" not in f]
print(f"ERA5 文件数: {len(files)}", flush=True)

for fp in files:
    tag = os.path.basename(fp)
    zf = zipfile.ZipFile(fp)
    members = zf.namelist()
    inst_name = next(m for m in members if "instant" in m)
    acc_name = next(m for m in members if "accum" in m)

    inst = read_member(zf, inst_name, tag, "inst")
    tdim = "time" if "time" in inst.dims else "valid_time"
    tt = inst[tdim].values
    tb = tt.astype("datetime64[h]") + np.timedelta64(8, "h")
    dates = tb.astype("datetime64[D]").astype(str)  # 北京日期 YYYY-MM-DD

    elat = inst["latitude"].values; elon = inst["longitude"].values
    ilat = np.argmin(np.abs(elat[None, :] - lats[:, None]), axis=1)
    ilon = np.argmin(np.abs(elon[None, :] - lons[:, None]), axis=1)

    def take(v):
        return v.transpose(tdim, "latitude", "longitude").isel(
            latitude=xr.DataArray(ilat, dims="g"),
            longitude=xr.DataArray(ilon, dims="g")).values  # (T, G)

    t2m = take(inst["t2m"]) - 273.15
    d2m = take(inst["d2m"]) - 273.15
    wind = np.hypot(take(inst["u10"]), take(inst["v10"]))
    inst.close()

    acc_ds = read_member(zf, acc_name, tag, "acc")
    tp = take(acc_ds["tp"]) * 1000.0  # mm
    acc_ds.close()
    zf.close()

    udates = np.unique(dates)
    for d in udates:
        ii = np.where(dates == d)[0]
        a = acc.get(d)
        if a is None:
            a = {"s_t2m": np.zeros(G), "m_t2m": np.full(G, -1e30),
                 "s_d2m": np.zeros(G), "s_wind": np.zeros(G),
                 "s_tp": np.zeros(G), "n": 0}
            acc[d] = a
        a["s_t2m"] += t2m[ii].sum(axis=0)
        np.maximum(a["m_t2m"], t2m[ii].max(axis=0), out=a["m_t2m"])
        a["s_d2m"] += d2m[ii].sum(axis=0)
        a["s_wind"] += wind[ii].sum(axis=0)
        a["s_tp"] += tp[ii].sum(axis=0)
        a["n"] += len(ii)
    print(f"[聚合] {tag} 天数={len(udates)} 总日期数={len(acc)}", flush=True)

# 2. 逐网格写 JSON
udates = sorted(acc.keys())
print(f"唯一北京日期: {len(udates)} ({udates[0]} ~ {udates[-1]})", flush=True)
n_files = 0
for gi, g in enumerate(gids):
    tmax_l, tmean_l, rh_l, wind_l, pr_l = [], [], [], [], []
    for d in udates:
        a = acc[d]
        n = a["n"]
        if n == 0:
            continue
        tm = a["s_t2m"][gi] / n
        td = a["s_d2m"][gi] / n
        tmax_l.append(round(float(a["m_t2m"][gi]), 2))
        tmean_l.append(round(float(tm), 2))
        rh_l.append(round(float(np.clip(rh_from_t_td(tm, td), 0, 100)), 1))
        wind_l.append(round(float(a["s_wind"][gi] / n), 2))
        pr_l.append(round(float(a["s_tp"][gi]), 2))
    payload = {"daily": {"time": udates,
                         "temperature_2m_max": tmax_l,
                         "temperature_2m_mean": tmean_l,
                         "relative_humidity_2m_mean": rh_l,
                         "wind_speed_10m_mean": wind_l,
                         "precipitation_sum": pr_l}}
    with open(f"{OUT}/{g}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    n_files += 1
print(f"写出 {n_files} 个网格 JSON -> {OUT}", flush=True)
shutil.rmtree(TMPDIR, ignore_errors=True)
