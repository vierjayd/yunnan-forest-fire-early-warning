# -*- coding: utf-8 -*-
"""V3 口径统一数据入口（2026-11-08 盲审整改）。

训练集：1:5 下采样装配的建模样本（仅 2010—2018 防火期），用于模型拟合。
校准集/测试集：**未抽样的完整单元**（2019—2020 / 2021—2023），
使保序回归直接拟合业务真实先验，全部评价指标与风险分级均在业务口径下报告。

所有实验脚本统一通过本模块取数，避免各脚本自行划分造成口径漂移。
"""
from pathlib import Path
import pandas as pd

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
FEATS = ["tmax7_mean", "rh7_mean", "wind7_mean", "precip7_sum", "dry_days",
         "ndvi", "forest_frac", "elev_mean", "slope_mean", "south_aspect_frac",
         "pop_density", "dist_road_km", "dist_settlement_km"]
CN = {"tmax7_mean": "前7日平均日最高气温", "rh7_mean": "前7日平均相对湿度",
      "wind7_mean": "前7日平均风速", "precip7_sum": "前7日累计降水量",
      "dry_days": "前7日连续无降水日数", "ndvi": "NDVI", "forest_frac": "林地占比",
      "elev_mean": "平均海拔", "slope_mean": "平均坡度", "south_aspect_frac": "南坡占比",
      "pop_density": "人口密度", "dist_road_km": "距最近道路距离",
      "dist_settlement_km": "距最近居民点距离"}

TRAIN_FILE = "model_dataset.csv"
EVAL_FILE = "eval_dataset.csv"
TRAIN_VIIRS = "model_dataset_viirs.csv"
EVAL_VIIRS = "eval_dataset_viirs.csv"


def load_split(source="modis", train_end=2018, calib_end=2020):
    """返回 (train, calib, test)；train 为下采样建模样本，calib/test 为完整单元。"""
    tf = TRAIN_VIIRS if source == "viirs" else TRAIN_FILE
    ef = EVAL_VIIRS if source == "viirs" else EVAL_FILE
    df = pd.read_csv(f"{BASE}/data/processed/{tf}")
    ev = pd.read_csv(f"{BASE}/data/processed/{ef}")
    train = df[df.fire_year <= train_end].copy()
    calib = ev[(ev.fire_year > train_end) & (ev.fire_year <= calib_end)].copy()
    test = ev[ev.fire_year > calib_end].copy()
    return train, calib, test


def prior_info(train, calib, test):
    """返回业务先验与抽样先验，供论文 2.4.3 与 4.2.2 引用。

    2026-11-09 答辩整改 Q4=A：业务先验 pi 以及依赖它的分级边界、F1 工作点
    一律在**校准期（2019—2020）**确定并冻结，测试期标签不参与任何规则选择。
    故 business_prior 取 calib.y.mean()；test_prior 仅作结果描述保留。
    """
    all_units = 2634 * 14 * 22
    return {
        "train_prior": float(train.y.mean()),
        "calib_prior": float(calib.y.mean()),
        "test_prior": float(test.y.mean()),
        "business_prior": float(calib.y.mean()),
        "all_units": all_units,
    }


def xy(d):
    return d[FEATS].values, d.y.values
