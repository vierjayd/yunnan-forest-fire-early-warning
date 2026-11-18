# -*- coding: utf-8 -*-
"""V3 口径 VIIRS 双源稳健性检验（2026-11-08 按盲审意见第 2 条整改）。

关键修正：双源对照限定在两源**共同完整覆盖的 2012—2023 共 12 个防火年度**，
在同一网格、同一旬、同一划分、同一超参、同一「完整单元」评估集上分别以 MODIS
与 VIIRS 标签训练 XGBoost 并比较，唯一差别是标签来源。原口径中 2009—2011 年
VIIRS 无火点却仍装配为负样本的系统性偏差已消除。

注意：两源正样本规模不同（VIIRS 探测更灵敏），AUC 数值不宜作绝对横比，
本检验只用于说明因子—风险结构对不同探测灵敏度是否稳定。
"""
import os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss)
from xgboost import XGBClassifier

from v3_data import BASE, FEATS, load_split

WINDOW_LO, WINDOW_HI = 2012, 2023


def fit_eval(source):
    df = pd.read_csv(f"{BASE}/data/processed/"
                     f"{'model_dataset_viirs.csv' if source == 'VIIRS' else 'model_dataset.csv'}")
    ev = pd.read_csv(f"{BASE}/data/processed/"
                     f"{'eval_dataset_viirs.csv' if source == 'VIIRS' else 'eval_dataset.csv'}")
    train = df[(df.fire_year >= WINDOW_LO) & (df.fire_year <= 2018)].copy()
    calib = ev[(ev.fire_year >= 2019) & (ev.fire_year <= 2020)].copy()
    test = ev[ev.fire_year >= 2021].copy()
    Xtr, ytr = train[FEATS].values, train.y.values
    Xca, yca = calib[FEATS].values, calib.y.values
    Xte, yte = test[FEATS].values, test.y.values
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                      eval_metric="logloss", random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr)
    p_raw = m.predict_proba(Xte)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(Xca)[:, 1], yca)
    p_cal = iso.transform(p_raw)
    fpr, tpr, _ = roc_curve(yte, p_raw)
    out = {
        "标签来源": source, "共同窗口": f"{WINDOW_LO}—{WINDOW_HI}",
        "训练样本": len(train), "测试单元": len(test), "测试正样本": int(yte.sum()),
        "业务先验": yca.mean(), "AUC": roc_auc_score(yte, p_raw),
        "KS": float(np.max(tpr - fpr)), "Brier_校准后": brier_score_loss(yte, p_cal),
    }
    years = []
    for fy in sorted(test.fire_year.unique()):
        mask = test.fire_year.values == fy
        years.append({"标签来源": source, "fire_year": fy, "n": int(mask.sum()),
                      "pos": int(yte[mask].sum()),
                      "AUC": roc_auc_score(yte[mask], p_raw[mask])})
    return out, years


res, yrs = [], []
for src in ["MODIS", "VIIRS"]:
    r, y = fit_eval(src)
    res.append(r); yrs += y
    print(src, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
pd.DataFrame(res).to_csv(f"{BASE}/results/viirs_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(yrs).to_csv(f"{BASE}/results/viirs_by_year.csv", index=False, encoding="utf-8-sig")

# 双源正样本重叠（限定共同窗口，同为「下一旬」标签口径）
def load_pos(name):
    s = set()
    for row in pd.read_csv(f"{BASE}/data/processed/{name}").itertuples(index=False):
        if WINDOW_LO <= int(row.fire_year) <= WINDOW_HI:
            s.add((row.gid, int(row.fire_year), int(row.dekad_seq)))
    return s

mod = load_pos("fire_labels_next_dekad.csv")
vii = load_pos("fire_labels_viirs_next_dekad.csv")
both = mod & vii
ov = {"共同窗口": f"{WINDOW_LO}—{WINDOW_HI}",
      "MODIS正样本": len(mod), "VIIRS正样本": len(vii), "共同": len(both),
      "MODIS独有": len(mod - vii), "VIIRS独有": len(vii - mod),
      "共同占MODIS": len(both) / len(mod), "共同占VIIRS": len(both) / len(vii)}
print("双源重叠:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ov.items()})
pd.DataFrame([ov]).to_csv(f"{BASE}/results/dual_source_overlap.csv",
                          index=False, encoding="utf-8-sig")
