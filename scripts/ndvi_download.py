# -*- coding: utf-8 -*-
"""
MOD13A2 NDVI 下载流水线（CMR 检索 + LP DAAC 云存储下载）
步骤1（枚举）：对 2009—2024 年 15 个防火期的每个 16 天合成期，用 CMR 按云南 bbox 检索 h26v06/h27v06 两景 URL；
步骤2（下载）： Bearer 令牌下载至 data/ndvi/，跳过已存在文件。
"""
import json, os, sys, time, re
import requests
from datetime import datetime, timedelta

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
OUT = f"{BASE}/data/ndvi"
os.makedirs(OUT, exist_ok=True)
TOKEN = open(f"{BASE}/data/geo/.earthdata_token").read().strip()
HDR = {"Authorization": f"Bearer {TOKEN}"}
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
BBOX = "97.5,21.1,106.2,29.3"

def periods():
    """防火期 16 天合成期起始日期列表。"""
    out = []
    for fy in range(2009, 2024):
        for doy in (305, 321, 337, 353):
            out.append(datetime(fy, 1, 1) + timedelta(days=doy - 1))
        for doy in (1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161):
            out.append(datetime(fy + 1, 1, 1) + timedelta(days=doy - 1))
    return out

plist = periods()
print(f"合成期数: {len(plist)}")

# 步骤1：枚举 URL
url_file = f"{BASE}/data/ndvi/urls.txt"
if "--enumerate" in sys.argv or not os.path.exists(url_file):
    urls = []
    with requests.Session() as s:
        for i, p0 in enumerate(plist):
            p1 = p0 + timedelta(days=15, hours=23, minutes=59)
            q = {"short_name": "MOD13A2", "version": "061",
                 "temporal": f"{p0:%Y-%m-%dT%H:%M:%SZ},{p1:%Y-%m-%dT%H:%M:%SZ}",
                 "bounding_box": BBOX, "page_size": 50}
            for attempt in range(3):
                try:
                    r = s.get(CMR, params=q, timeout=60)
                    j = r.json()
                    break
                except Exception:
                    time.sleep(5)
            else:
                print(f"[枚举失败] {p0:%Y-%m-%d}")
                continue
            for e in j.get("feed", {}).get("entry", []):
                for l in e.get("links", []):
                    href = l.get("href", "")
                    if "lp-prod-protected" in href and href.endswith(".hdf"):
                        urls.append(href)
            if (i + 1) % 25 == 0:
                print(f"枚举进度 {i+1}/{len(plist)}，累计URL {len(urls)}", flush=True)
    urls = sorted(set(urls))
    with open(url_file, "w") as f:
        f.write("\n".join(urls))
    print(f"枚举完成：{len(urls)} 个文件")

# 步骤2：下载
if "--download" in sys.argv:
    urls = [u.strip() for u in open(url_file) if u.strip()]
    print(f"待下载: {len(urls)} 个文件")
    ok = skip = fail = 0
    with requests.Session() as s:
        s.headers.update(HDR)
        for i, u in enumerate(urls):
            fn = u.split("/")[-1]
            fp = f"{OUT}/{fn}"
            if os.path.exists(fp) and os.path.getsize(fp) > 1_000_000:
                skip += 1
                continue
            for attempt in range(4):
                try:
                    with s.get(u, stream=True, timeout=300) as r:
                        if r.status_code == 200:
                            with open(fp, "wb") as f:
                                for chunk in r.iter_content(1 << 20):
                                    f.write(chunk)
                            ok += 1
                            break
                        elif r.status_code in (429, 500, 502, 503):
                            time.sleep(15 * (attempt + 1))
                        else:
                            print(f"[{r.status_code}] {fn}")
                            fail += 1
                            break
                except Exception:
                    time.sleep(10 * (attempt + 1))
            else:
                fail += 1
                print(f"[重试耗尽] {fn}")
            if (ok + skip + fail) % 25 == 0:
                print(f"下载进度 {ok+skip+fail}/{len(urls)}（新{ok} 跳过{skip} 失败{fail}）", flush=True)
    print(f"下载完成：新 {ok}，跳过 {skip}，失败 {fail}")
