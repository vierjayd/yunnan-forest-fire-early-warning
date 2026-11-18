# -*- coding: utf-8 -*-
"""
FIRMS 火点样本量验证脚本（云南省，0.1° 网格 × 旬）
=================================================
用途：在正式动笔前验证论文因变量（正样本）规模是否充足，锁定全文最大数据风险。

用法：
    python firms_sample_check.py <firms_csv路径> [<另一个csv路径> ...]
        [--years 2010-2024] [--min-conf nominal] [--grid 0.1]
        [--report 输出报告路径.txt]

输入：NASA FIRMS 官网下载的 MODIS 或 VIIRS 火点 CSV（firms.modaps.eosdis.nasa.gov，
      免费注册后可下载国家或自定义区域的存档 CSV；MODIS 与 VIIRS 可同时传入，
      本脚本会自动兼容两套列名并分别统计）。

输出：控制台打印 + 写入报告文件，内容包括：
      1. 云南省范围内火点总数（按年、按干季/雨季拆分）；
      2. 0.1° 网格数量与有火网格占比；
      3. “网格×旬”正样本单元总数（核心指标）及按年分布；
      4. 结论判定：正样本规模是否满足论文建模需求（阈值：>=3000 稳妥，
         1000-3000 可行但需合并旬为月，<1000 需回改设计）。

口径说明（与论文第3章数据质量小节对应）：
      - 云南省范围按经纬度包围盒初筛（北纬 21.1-29.3，东经 97.5-106.2），
        正式数据集构建时再用省界矢量精确裁剪；
      - 置信度过滤：MODIS 数值型 confidence（默认保留 >=50），
        VIIRS 类别型 confidence（默认保留 nominal 与 high，剔除 low）；
      - 旬的定义：每月 1-10 日为上旬、11-20 日为中旬、21 日及以后为下旬；
      - 防火期定义为每年 11 月 1 日至次年 6 月 15 日（省法定防火期与迪庆州防火期的并集）。

仅使用 Python 标准库，无需安装任何第三方包。
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

YUNNAN_BBOX = (21.1, 29.3, 97.5, 106.2)  # lat_min, lat_max, lon_min, lon_max
DRY_SEASON_MONTHS = {11, 12, 1, 2, 3, 4, 5}  # 防火期主口径的月份部分；6月1-15日由 in_fire_season 单独判定


def in_fire_season(dt):
    """防火期全集口径：每年11月1日—次年6月15日（省法定防火期与迪庆州防火期的并集）。"""
    if dt.month in DRY_SEASON_MONTHS:
        return True
    if dt.month == 6 and dt.day <= 15:
        return True
    return False
POS_UNIT_THRESHOLD_SAFE = 3000
POS_UNIT_THRESHOLD_OK = 1000


def parse_args():
    p = argparse.ArgumentParser(description="FIRMS 云南省火点样本量验证")
    p.add_argument("csv_files", nargs="+", help="FIRMS 火点 CSV 文件路径（可多个）")
    p.add_argument("--years", default="2010-2024", help="统计年份范围，如 2010-2024")
    p.add_argument("--min-conf", default="nominal",
                   help="VIIRS 置信度下限：nominal（默认）或 low")
    p.add_argument("--modis-conf-min", type=int, default=50,
                   help="MODIS 数值置信度下限（默认 50）")
    p.add_argument("--grid", type=float, default=0.1, help="网格边长（度，默认 0.1）")
    p.add_argument("--report", default="", help="报告输出路径（txt），默认不输出文件")
    return p.parse_args()


def detect_source(header):
    """根据列名判断 MODIS / VIIRS，返回标准化列名映射。"""
    cols = {c.strip().lower(): c for c in header}
    if "latitude" not in cols or "longitude" not in cols or "acq_date" not in cols:
        raise ValueError("无法识别的 FIRMS CSV：缺少 latitude/longitude/acq_date 列，"
                         "请确认文件来自 FIRMS 官网下载。")
    instrument = "MODIS" if "brightness" in cols else ("VIIRS" if "bright_ti4" in cols else "未知")
    return instrument, {
        "lat": cols["latitude"], "lon": cols["longitude"],
        "date": cols["acq_date"], "conf": cols.get("confidence"),
    }


def conf_ok(instrument, raw_conf, min_conf_label, modis_min):
    if raw_conf is None or raw_conf == "":
        return True  # 无置信度列时不过滤
    if instrument == "MODIS":
        try:
            return float(raw_conf) >= modis_min
        except ValueError:
            return True
    # VIIRS：low / nominal / high
    v = raw_conf.strip().lower()
    if min_conf_label == "low":
        return v in ("low", "nominal", "high", "l", "n", "h")
    return v in ("nominal", "high", "n", "h")


def dekad_of(day):
    if day <= 10:
        return 1
    if day <= 20:
        return 2
    return 3


def in_bbox(lat, lon, bbox):
    return bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]


def main():
    args = parse_args()
    y0, y1 = [int(x) for x in args.years.split("-")]
    grid = args.grid

    total_rows = 0
    in_yunnan = 0
    kept = 0
    by_year = defaultdict(int)
    dry_by_year = defaultdict(int)
    wet_by_year = defaultdict(int)
    by_source = defaultdict(int)
    pos_units = set()            # (gy, gx, year, month, dekad)
    pos_units_dry = set()
    grid_cells_with_fire = set()

    for path in args.csv_files:
        if not os.path.exists(path):
            print(f"[警告] 文件不存在，已跳过：{path}")
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            instrument, col = detect_source(header)
            idx = {k: list(header).index(v) for k, v in col.items() if v}
            for row in reader:
                total_rows += 1
                try:
                    lat = float(row[idx["lat"]])
                    lon = float(row[idx["lon"]])
                    dt = datetime.strptime(row[idx["date"]].strip(), "%Y-%m-%d")
                except (ValueError, IndexError):
                    continue
                if not (y0 <= dt.year <= y1):
                    continue
                if not in_bbox(lat, lon, YUNNAN_BBOX):
                    continue
                in_yunnan += 1
                conf_val = row[idx["conf"]] if "conf" in idx else None
                if not conf_ok(instrument, conf_val, args.min_conf, args.modis_conf_min):
                    continue
                kept += 1
                by_source[instrument] += 1
                by_year[dt.year] += 1
                if in_fire_season(dt):
                    dry_by_year[dt.year] += 1
                else:
                    wet_by_year[dt.year] += 1
                gy = int(lat // grid)
                gx = int(lon // grid)
                grid_cells_with_fire.add((gy, gx))
                unit = (gy, gx, dt.year, dt.month, dekad_of(dt.day))
                pos_units.add(unit)
                if in_fire_season(dt):
                    pos_units_dry.add(unit)

    n_grid_total = int((YUNNAN_BBOX[1] - YUNNAN_BBOX[0]) / grid) * \
        int((YUNNAN_BBOX[3] - YUNNAN_BBOX[2]) / grid)
    n_units = len(pos_units)
    n_units_dry = len(pos_units_dry)

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 60)
    out("FIRMS 云南省火点样本量验证报告")
    out(f"统计年份：{y0}-{y1}；网格：{grid}°；统计单元：网格×旬")
    out(f"输入文件：{len(args.csv_files)} 个；原始记录总行数：{total_rows}")
    out("=" * 60)
    out(f"云南省包围盒内火点数：{in_yunnan}")
    out(f"置信度过滤后保留：{kept}（来源："
        + "，".join(f"{k} {v}" for k, v in sorted(by_source.items())) + "）")
    out("")
    out("[按年分布]  年份 | 总火点 | 防火期(11.1-6.15) | 雨季(5-10月)")
    for y in range(y0, y1 + 1):
        out(f"  {y} | {by_year.get(y,0):>7} | {dry_by_year.get(y,0):>7} | {wet_by_year.get(y,0):>7}")
    out("")
    out(f"云南省范围 {grid}° 网格总数（近似）：{n_grid_total}")
    out(f"出现过火点的网格数：{len(grid_cells_with_fire)}"
        f"（占比 {len(grid_cells_with_fire)/n_grid_total:.1%}）")
    out("")
    out(f"★ 核心指标 —— “网格×旬”正样本单元总数：{n_units}")
    out(f"   其中防火期正样本单元：{n_units_dry}")
    out("")

    if n_units_dry >= POS_UNIT_THRESHOLD_SAFE:
        verdict = ("结论：防火期正样本充足（>=3000），按 0.1° 网格×旬 + 近15个防火期的设计"
                   "可直接推进，无需调整框架。")
    elif n_units_dry >= POS_UNIT_THRESHOLD_OK:
        verdict = ("结论：干季正样本处于 1000-3000 区间，可行但建议将统计单元由旬放宽至月，"
                   "或将网格细化评估后确定；建议与我确认后再定稿。")
    else:
        verdict = ("结论：干季正样本不足 1000，原设计需回改——可考虑扩大年份范围、"
                   "改旬为月、或合并 MODIS 与 VIIRS 双源后再评估。")
    out(verdict)
    out("")
    out("[提醒] 本脚本按经纬度包围盒初筛，且未做林地掩膜剔除；")
    out("       正式数据集需叠加 GlobeLand30 林地掩膜剔除非森林火点，")
    out("       并在第3章说明漏报（云/烟雾遮挡）与时间戳对齐口径。")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n报告已写入：{args.report}")


if __name__ == "__main__":
    sys.exit(main())
