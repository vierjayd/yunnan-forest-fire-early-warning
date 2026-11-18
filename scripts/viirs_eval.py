# -*- coding: utf-8 -*-
"""VIIRS 双源稳健性检验：与 MODIS 完全同口径装配、训练、校准、评价。"""
import os, json, random
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, brier_score_loss, roc_curve
from xgboost import XGBClassifier

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
random.seed(42)
FEATS = ["tmax7_mean", "rh7_mean", "wind7_mean", "precip7_sum", "dry_days",
         "ndvi", "forest_frac", "elev_mean", "slope_mean", "south_aspect_frac",
         "pop_density", "dist_road_km", "dist_settlement_km"]

def dekad_end_date(fy, seq):
    mm, dk = seq // 3, seq % 3
    if mm <= 1:
        y, m = fy, 11 + mm
    elif mm <= 6:
        y, m = fy + 1, mm - 1
    else:
        y, m = fy + 1, 6
    if m == 6:
        return datetime(y, 6, 10) if dk == 0 else datetime(y, 6, 15)
    d = (10, 20, 28)[dk] if dk < 2 else None
    if dk == 2:
        d = 31 if m == 12 else (datetime(y, m + 1, 1) - timedelta(days=1)).day
    return datetime(y, m, d)

# 网格与静态特征
grids_df = pd.read_csv(f"{BASE}/data/processed/grids_yunnan.csv")
static = pd.read_csv(f"{BASE}/data/processed/static_features.csv").set_index("gid")
road = pd.read_csv(f"{BASE}/data/processed/dist_road.csv").set_index("gid")
stl = pd.read_csv(f"{BASE}/data/processed/dist_settlement.csv").set_index("gid")
ndvi = {}
nd = pd.read_csv(f"{BASE}/data/processed/ndvi_dekad_imputed.csv")
for gid, fy, seq, v in nd.itertuples(index=False):
    ndvi[(gid, fy, seq)] = v

# VIIRS 标签
pos = {}
vl = pd.read_csv(f"{BASE}/data/processed/fire_labels_forest_viirs.csv")
for gid, fy, seq, n in vl.itertuples(index=False):
    if fy <= 2023:
        pos[(gid, fy, seq)] = n
print(f"VIIRS 正样本单元: {len(pos)}")

# 气象 JSON 加载（缓存）
meteo_cache = {}
for gid in grids_df.gid:
    fp = f"{BASE}/data/meteo_era5/{gid}.json"
    if not os.path.exists(fp):
        continue
    d = json.load(open(fp, encoding="utf-8"))["daily"]
    out = {}
    for i, ds in enumerate(d["time"]):
        tmax = d["temperature_2m_max"][i]; rh = d["relative_humidity_2m_mean"][i]
        wnd = d["wind_speed_10m_mean"][i]; pr = d["precipitation_sum"][i]
        if None in (tmax, rh, wnd, pr):
            continue
        out[ds] = (tmax, rh, wnd, pr)
    meteo_cache[gid] = out

def features_for(met, end_dt):
    days = [(end_dt - timedelta(days=k)).strftime("%Y-%m-%d") for k in range(6, -1, -1)]
    vals = [met.get(ds) for ds in days]
    if any(v is None or any(x is None for x in v) for v in vals):
        return None
    tmax = [v[0] for v in vals]; rh = [v[1] for v in vals]
    wind = [v[2] for v in vals]; pr = [v[3] for v in vals]
    dry = 0
    for v in reversed(pr):
        if v < 1.0:
            dry += 1
        else:
            break
    return (np.mean(tmax), np.mean(rh), np.mean(wind), sum(pr), dry)

# 装配（与 MODIS 相同的 1:5 抽样）
rows = []
for gid in grids_df.gid:
    if gid not in meteo_cache:
        continue
    for fy in range(2009, 2024):
        for seq in range(23):
            end = dekad_end_date(fy, seq)
            if end > datetime(2024, 6, 15):
                continue
            rows.append((gid, fy, seq, end, 1 if (gid, fy, seq) in pos else 0))
pos_rows = [r for r in rows if r[4] == 1]
neg_rows = [r for r in rows if r[4] == 0]
neg_sample = random.sample(neg_rows, min(5 * len(pos_rows), len(neg_rows)))
sample = pos_rows + neg_sample
random.shuffle(sample)
print(f"VIIRS 装配样本量: {len(sample)}（正 {len(pos_rows)}，负 {len(neg_sample)}）")

data = []
for gid, fy, seq, end, y in sample:
    feat = features_for(meteo_cache[gid], end)
    if feat is None:
        continue
    st = static.loc[gid]
    row = {"gid": gid, "fire_year": fy, "dekad_seq": seq, "y": y,
           "tmax7_mean": feat[0], "rh7_mean": feat[1], "wind7_mean": feat[2],
           "precip7_sum": feat[3], "dry_days": feat[4],
           "ndvi": ndvi.get((gid, fy, seq), np.nan),
           "forest_frac": st.forest_frac, "elev_mean": st.elev_mean,
           "slope_mean": st.slope_mean, "south_aspect_frac": st.south_aspect_frac,
           "pop_density": st.pop_density,
           "dist_road_km": road.loc[gid].dist_road_km,
           "dist_settlement_km": stl.loc[gid].dist_settlement_km}
    data.append(row)
df = pd.DataFrame(data).dropna()
df.to_csv(f"{BASE}/data/processed/model_dataset_viirs.csv", index=False)
print(f"VIIRS 建模数据集: {len(df)} 行")

train = df[df.fire_year <= 2018]
calib = df[(df.fire_year >= 2019) & (df.fire_year <= 2020)]
test = df[df.fire_year >= 2021]
Xtr, ytr = train[FEATS].values, train.y.values
Xca, yca = calib[FEATS].values, calib.y.values
Xte, yte = test[FEATS].values, test.y.values
model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=(len(ytr)-ytr.sum())/ytr.sum(),
                      eval_metric="logloss", random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)
pca = model.predict_proba(Xca)[:, 1]
pte = model.predict_proba(Xte)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip").fit(pca, yca)
pte_cal = iso.transform(pte)

fpr, tpr, _ = roc_curve(yte, pte)
ks = float(np.max(tpr - fpr))
pred = (pte >= 0.5).astype(int)
res = {"AUC": roc_auc_score(yte, pte), "KS": ks,
       "Recall": recall_score(yte, pred), "Precision": precision_score(yte, pred),
       "F1": f1_score(yte, pred), "Brier": brier_score_loss(yte, pte),
       "Brier_校准后": brier_score_loss(yte, pte_cal)}
print("\nVIIRS 主结果:", {k: round(v, 4) for k, v in res.items()})
pd.DataFrame([res]).to_csv(f"{BASE}/results/viirs_metrics.csv", index=False, encoding="utf-8-sig")

year_rows = []
for fy in sorted(test.fire_year.unique()):
    m = test.fire_year.values == fy
    year_rows.append({"fire_year": fy, "AUC": roc_auc_score(yte[m], pte[m])})
print("VIIRS 分年度 AUC:", year_rows)
pd.DataFrame(year_rows).to_csv(f"{BASE}/results/viirs_by_year.csv", index=False, encoding="utf-8-sig")

# 双源正样本重叠
mod_pos = set()
ml = pd.read_csv(f"{BASE}/data/processed/fire_labels_forest.csv")
for gid, fy, seq, n in ml.itertuples(index=False):
    if fy <= 2023:
        mod_pos.add((gid, fy, seq))
both = mod_pos & set(pos)
print(f"\n双源正样本重叠: {len(both)}；MODIS 独有 {len(mod_pos-set(pos))}；VIIRS 独有 {len(set(pos)-mod_pos)}")
print(f"重叠占 MODIS 正样本比例: {len(both)/len(mod_pos):.2%}；占 VIIRS 正样本比例: {len(both)/len(pos):.2%}")
