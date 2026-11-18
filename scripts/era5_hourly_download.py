# -*- coding: utf-8 -*-
"""
ERA5 逐时数据下载 v2（修正版，CDS 官方 reanalysis-era5-single-levels 通道）
口径: 云南区域 (N29.4, W97.4, S21.0, E106.4)，2009-11-01 至 2024-06-15 全部防火期。
变量: t2m / d2m / u10 / v10 / tp（5 变量一次请求，逐时）。
v0 教训: 6个月段(1-6月) x 5变量 x 24h 触发 cost limit(约2200万值)；队列拥塞源于残留的
         accepted 僵尸任务。v2 修正: 每防火年度拆 4 段(11-12月/1-2月/3-4月/5-6月)，
         每段 2个月 x 5变量 x 24h ≈ 910万值 < 上限(实测可下载16MB)，全年 4x15=60 请求。
         提交前自动清理本账号 accepted 僵尸任务，避免队列自锁。
已存在文件自动跳过（2009/2010/2012/2013/2014 的 11-12 月段沿用 v0 成果）。
输出: data/era5/era5_{fy}1112.nc, era5_{fy+1}0102.nc, era5_{fy+1}0304.nc, era5_{fy+1}0506.nc
"""
import os, sys, time
import cdsapi

# 凭证仅从环境变量读取，不在代码或公开材料中保存任何密钥。
# 用法（本机）：set CDS_PRIVATE_TOKEN=<CDS 个人令牌> 后再运行；cdsapi.Client() 另从 ~/.cdsapirc 读取。
CDS_TOKEN = os.environ.get("CDS_PRIVATE_TOKEN")

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
OUT = f"{BASE}/data/era5"
os.makedirs(OUT, exist_ok=True)

DATASET = "reanalysis-era5-single-levels"
VARS = ["2m_temperature", "2m_dewpoint_temperature",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation"]
AREA = [29.4, 97.4, 21.0, 106.4]
HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

# 每个防火年度 4 段：11-12月、次年1-2月、3-4月、5-6月（5-6月含6.16-30多余日，聚合时忽略）
segments = []
for fy in range(2009, 2024):
    segments.append((f"{fy}1112", str(fy), ["11", "12"]))
    segments.append((f"{fy+1}0102", str(fy + 1), ["01", "02"]))
    segments.append((f"{fy+1}0304", str(fy + 1), ["03", "04"]))
    segments.append((f"{fy+1}0506", str(fy + 1), ["05", "06"]))

# 启动前清理 accepted 僵尸任务，防止队列自锁
def clear_zombies():
    if not CDS_TOKEN:
        print("[清理跳过] 未设置 CDS_PRIVATE_TOKEN 环境变量", flush=True)
        return
    import requests
    try:
        r = requests.get(
            "https://cds.climate.copernicus.eu/api/retrieve/v1/jobs?limit=100",
            headers={"PRIVATE-TOKEN": CDS_TOKEN},
            timeout=30)
        jobs = r.json().get("jobs", [])
        n = 0
        for j in jobs:
            if j.get("status") in ("accepted", "running"):
                requests.delete(
                    f"https://cds.climate.copernicus.eu/api/retrieve/v1/jobs/{j['jobID']}",
                    headers={"PRIVATE-TOKEN": CDS_TOKEN},
                    timeout=30)
                n += 1
        if n:
            print(f"[清理] 移除 {n} 个僵尸排队任务", flush=True)
    except Exception as e:
        print(f"[清理跳过] {type(e).__name__} {str(e)[:120]}", flush=True)

clear_zombies()
c = cdsapi.Client()
done = skip = fail = 0
for tag, year, months in segments:
    fp = f"{OUT}/era5_{tag}.nc"
    if os.path.exists(fp) and os.path.getsize(fp) > 1_000_000:
        print(f"[跳过] {tag}", flush=True)
        skip += 1
        continue
    req = {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": year,
        "month": months,
        "day": DAYS,
        "time": HOURS,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    ok = False
    for attempt in range(3):
        try:
            print(f"[提交] {tag} ...", flush=True)
            c.retrieve(DATASET, req, fp)
            print(f"[完成] {tag} -> {os.path.getsize(fp)/1e6:.1f}MB", flush=True)
            done += 1
            ok = True
            break
        except Exception as e:
            msg = str(e)[:300]
            print(f"[失败重试{attempt+1}] {tag}: {type(e).__name__} {msg}", flush=True)
            low = msg.lower()
            if "licence" in low or "terms of use" in low:
                print("[需接受条款] 请在 CDS 网页接受 reanalysis-era5-single-levels 条款", flush=True)
                sys.exit(2)
            time.sleep(180 * (attempt + 1))
    if not ok:
        print(f"[放弃] {tag}", flush=True)
        fail += 1
    time.sleep(30)

print(f"全部请求处理完毕: 完成{done} 跳过{skip} 失败{fail}", flush=True)
