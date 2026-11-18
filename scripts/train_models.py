# -*- coding: utf-8 -*-
"""
第4章主实验（V3 口径，2026-11-08 按盲审意见整改）：

  · 训练集：1:5 下采样装配样本（2010—2018 防火期），用于模型拟合与不平衡敏感性。
  · 校准集/测试集：未抽样的完整单元（2019—2020 / 2021—2023），业务真实先验约 2.06%。
    保序回归在完整校准集上拟合，直接输出业务口径概率，从根源消除先验偏移。
  · 补充统计推断：AUC 的 bootstrap 95% 置信区间、模型间配对 bootstrap 差异检验、
    PR-AUC（average precision）、内部模型基线和三个简单预警基线。
  · 补充五折空间留出验证，并保持每折内部的训练—校准—测试时间顺序。
  · Top-k 统一按校准概率降序、原始模型分数降序次级排序，处理保序校准并列值。
  · 主表不再使用任意的 0.5 阈值报召回率与精确率，改报校准概率下 F1 最优工作点。

输出：results/*.csv、results/figure_cache.npz、figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             average_precision_score, f1_score)
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

from v3_data import BASE, FEATS, CN, load_split, prior_info

os.makedirs(f"{BASE}/results", exist_ok=True)
os.makedirs(f"{BASE}/figures", exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df_sample = pd.read_csv(f"{BASE}/data/processed/model_dataset.csv")
train, calib, test = load_split("modis")
pi = prior_info(train, calib, test)
print(f"训练用 1:5 装配样本 {len(df_sample)} 行；完整评估单元 {len(calib)+len(test)} 行")
for name, d in [("训练(下采样)", train), ("校准(完整)", calib), ("测试(完整)", test)]:
    print(f"  {name}集 {len(d)} 行，正样本 {int(d.y.sum())}（{d.y.mean():.2%}）")

Xtr, ytr = train[FEATS].values, train.y.values
Xca, yca = calib[FEATS].values, calib.y.values
Xte, yte = test[FEATS].values, test.y.values
gid_test = test.gid.values.astype(str)
year_test = test.fire_year.values.astype(str)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xca_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xca), scaler.transform(Xte)


def best_f1(y, p):
    """Q4=A：仅在**校准期**业务口径概率上扫描阈值，确定并冻结 F1 工作点。

    测试期标签不参与阈值搜索，避免「选择后评估」造成的乐观偏差。
    """
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 200)))
    best = (-1, 0.5, 0.0, 0.0)
    for t in grid:
        pred = (p >= t).astype(int)
        f = f1_score(y, pred, zero_division=0)
        if f > best[0]:
            best = (f, float(t), float(pred.mean()),
                    float(np.mean(y[pred == 1])) if pred.any() else 0.0)
    return best


def f1_at(y, p, thr):
    """Q4=A：测试期按校准期冻结的阈值评价一次，不再扫描测试标签。"""
    pred = (p >= thr).astype(int)
    return (f1_score(y, pred, zero_division=0), float(pred.mean()),
            float(np.mean(y[pred == 1])) if pred.any() else 0.0)


CAPTURE_FRACS = (0.05, 0.10, 0.20, 0.30)


def capture_at_topk(y, calibrated_score, raw_score, fractions=CAPTURE_FRACS):
    """按校准概率降序、原始分数降序、原行号升序进行确定性 Top-k 排序。

    保序校准会产生并列概率；原始分数用于次级排序，原行号只用于完全并列时
    保证结果可复现。返回各覆盖比例对应的单元数、正样本数和捕获率。
    """
    y = np.asarray(y, dtype=int)
    calibrated_score = np.asarray(calibrated_score, dtype=float)
    raw_score = np.asarray(raw_score, dtype=float)
    if not (len(y) == len(calibrated_score) == len(raw_score)):
        raise ValueError("Top-k 评价的标签与分数长度不一致")
    order = np.lexsort((np.arange(len(y)), -raw_score, -calibrated_score))
    ranked_y = y[order]
    rows = []
    for frac in fractions:
        k = int(len(y) * frac)
        fires = int(ranked_y[:k].sum())
        rows.append({"top_frac": frac, "n": k, "fires": fires,
                     "capture": fires / y.sum()})
    return rows


def capture_columns(y, calibrated_score, raw_score):
    """将 Top-k 结果转换为结果表列。"""
    rows = capture_at_topk(y, calibrated_score, raw_score)
    return {f"Top_{int(r['top_frac'] * 100)}_capture": r["capture"] for r in rows}


def cluster_index(rng, groups):
    """Q3=A：按分组（网格或防火年）聚类 bootstrap，有放回抽组，取整组单元。"""
    uniq = np.unique(groups)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.flatnonzero(np.isin(groups, pick))


def make_models(pos, neg):
    spw = neg / pos
    return {
        "逻辑回归": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=20,
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                                 subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                 eval_metric="logloss", random_state=42, n_jobs=-1),
    }


def bootstrap_auc(y, p, groups, n_boot=500, seed=42):
    """Q3=A：按网格聚类 bootstrap，避免同一网格多旬、相邻网格共享气象格点
    造成的行级独立假设失真。"""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = cluster_index(rng, groups)
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


results, proba_test, proba_calib, proba_cal = {}, {}, {}, {}
frozen_thresholds = {}
for name, model in make_models(ytr.sum(), len(ytr) - ytr.sum()).items():
    X_use = Xtr_s if name == "逻辑回归" else Xtr
    Xc_use = Xca_s if name == "逻辑回归" else Xca
    Xt_use = Xte_s if name == "逻辑回归" else Xte
    model.fit(X_use, ytr)
    pca = model.predict_proba(Xc_use)[:, 1]
    pte = model.predict_proba(Xt_use)[:, 1]
    proba_calib[name], proba_test[name] = pca, pte
    fpr, tpr, _ = roc_curve(yte, pte)
    # 保序回归在完整校准集上拟合 -> 业务口径概率
    iso = IsotonicRegression(out_of_bounds="clip").fit(pca, yca)
    pca_cal = iso.transform(pca)
    pte_cal = iso.transform(pte)
    proba_cal[name] = pte_cal
    # Q3=A：置信区间按网格聚类 bootstrap
    lo, hi = bootstrap_auc(yte, pte, gid_test)
    # Q4=A：阈值只在校准期搜索并冻结，测试期仅按该阈值评价一次
    f1_ca, thr_b, mark_ca, hit_ca = best_f1(yca, pca_cal)
    frozen_thresholds[name] = float(thr_b)
    f1b, mark_frac, hit_rate = f1_at(yte, pte_cal, thr_b)
    results[name] = {
        "AUC": roc_auc_score(yte, pte), "AUC_CI_low": lo, "AUC_CI_high": hi,
        "KS": float(np.max(tpr - fpr)), "PR_AUC": average_precision_score(yte, pte),
        "Brier_校准前": brier_score_loss(yte, pte), "Brier_校准后": brier_score_loss(yte, pte_cal),
        "F1_最优": f1b, "最优阈值": thr_b, "标记比例": mark_frac, "命中率": hit_rate,
        "F1_校准期": f1_ca, "阈值确定集": "校准期2019—2020",
    }
    print(f"{name}: AUC={results[name]['AUC']:.4f} [{lo:.4f},{hi:.4f}] "
          f"KS={results[name]['KS']:.4f} PR-AUC={results[name]['PR_AUC']:.4f} "
          f"Brier={results[name]['Brier_校准前']:.4f}->{results[name]['Brier_校准后']:.4f} "
          f"F1={f1b:.4f}@{thr_b:.4f}", flush=True)

main = pd.DataFrame(results).T
main.to_csv(f"{BASE}/results/metrics_main.csv", encoding="utf-8-sig")
print("\n主结果:\n", main.round(4))

# ---- Q3=A：配对 bootstrap 按网格聚类；另做防火年整块重采样敏感性 ----
rng = np.random.RandomState(7)
pairs, n = {}, len(yte)
for other in ["随机森林", "逻辑回归"]:
    diffs = []
    for _ in range(500):
        idx = cluster_index(rng, gid_test)
        if yte[idx].sum() in (0, len(idx)):
            continue
        diffs.append(roc_auc_score(yte[idx], proba_test["XGBoost"][idx]) -
                     roc_auc_score(yte[idx], proba_test[other][idx]))
    d = np.array(diffs)
    obs = results["XGBoost"]["AUC"] - results[other]["AUC"]
    pairs[f"XGBoost-{other}"] = {
        "观测差": obs, "CI_low": float(np.percentile(d, 2.5)),
        "CI_high": float(np.percentile(d, 97.5)),
        "p_双侧": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
        "重采样单位": "网格聚类（2634 个林区网格）",
    }
unc = pd.DataFrame(pairs).T
unc.to_csv(f"{BASE}/results/uncertainty.csv", encoding="utf-8-sig")
print("\nAUC 差异检验（网格聚类）:\n", unc.round(5))

# 防火年整块 bootstrap 敏感性：检验结论是否依赖某一年份
rng_y = np.random.RandomState(11)
sens = []
for other in ["随机森林", "逻辑回归"]:
    diffs = []
    for _ in range(500):
        idx = cluster_index(rng_y, year_test)
        if yte[idx].sum() in (0, len(idx)):
            continue
        diffs.append(roc_auc_score(yte[idx], proba_test["XGBoost"][idx]) -
                     roc_auc_score(yte[idx], proba_test[other][idx]))
    d = np.array(diffs)
    sens.append({"对照": other, "重采样单位": "防火年整块（2021—2023）",
                 "CI_low": float(np.percentile(d, 2.5)),
                 "CI_high": float(np.percentile(d, 97.5)),
                 "p_双侧": float(2 * min((d <= 0).mean(), (d >= 0).mean()))})
pd.DataFrame(sens).to_csv(f"{BASE}/results/uncertainty_sensitivity.csv",
                          index=False, encoding="utf-8-sig")
print("\n年份块敏感性:\n", pd.DataFrame(sens).round(5).to_string(index=False))

# ---- Q4=A：规则冻结留痕，证明测试标签未参与任何选择 ----
import json
from pathlib import Path
frozen = {
    "规则确定集": "校准期 2019—2020 完整单元",
    "校准单元数": int(len(calib)),
    "业务先验pi": float(pi["business_prior"]),
    "测试期先验_仅作结果描述": float(pi["test_prior"]),
    "主模型": "XGBoost（依据校准期判别、校准与可解释性预先指定）",
    "主不平衡方案": "1:5 下采样 + 类别加权",
    "F1冻结阈值": frozen_thresholds,
    "风险分级边界": "0.5π、π、2π、4π（π 取校准期先验）",
    "测试集用途": "仅报告已冻结规则下的评价结果，不参与模型、阈值、π 或方案选择",
}
Path(f"{BASE}/results_v3").mkdir(exist_ok=True)
Path(f"{BASE}/results_v3/frozen_config.json").write_text(
    json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n冻结配置已写入 results_v3/frozen_config.json")

# ---- 内部模型基线与简单预警基线：相同时间划分、校准和 Top-k 规则 ----
STATIC = ["forest_frac", "elev_mean", "slope_mean", "south_aspect_frac",
          "pop_density", "dist_road_km", "dist_settlement_km"]
METEO = ["tmax7_mean", "rh7_mean", "wind7_mean", "precip7_sum", "dry_days"]
baseline_rows = []
baseline_capture_rows = []


def add_baseline(label, label_en, pca_raw, pte_raw):
    """在2019—2020完整校准集拟合保序回归，再于2021—2023测试集评价。"""
    # 保留主模型 predict_proba 的原始 dtype，避免 float32→float64 转换改变
    # 保序校准的数值并列结构，进而影响 Top-k 次级排序。
    pca_raw = np.asarray(pca_raw)
    pte_raw = np.asarray(pte_raw)
    iso_base = IsotonicRegression(out_of_bounds="clip").fit(pca_raw, yca)
    pte_cal = iso_base.transform(pte_raw)
    row = {"model": label, "model_en": label_en,
           "AUC": roc_auc_score(yte, pte_raw),
           "PR_AUC": average_precision_score(yte, pte_raw),
           "Brier_校准前": brier_score_loss(yte, pte_raw),
           "Brier_校准后": brier_score_loss(yte, pte_cal),
           # 保留旧列名以兼容既有表格脚本；现在所有模型均为校准后口径。
           "Brier": brier_score_loss(yte, pte_cal),
           "n_test": len(yte), "positive_test": int(yte.sum()),
           "unique_raw_scores": int(np.unique(pte_raw).size),
           "unique_calibrated_scores": int(np.unique(pte_cal).size)}
    row.update(capture_columns(yte, pte_cal, pte_raw))
    baseline_rows.append(row)
    for cap_row in capture_at_topk(yte, pte_cal, pte_raw):
        baseline_capture_rows.append({"model": label, "model_en": label_en, **cap_row})


for label, label_en, cols in [
        ("仅静态因子", "Static-only logistic regression", STATIC),
        ("仅气象因子", "Meteorology-only logistic regression", METEO)]:
    sc = StandardScaler().fit(train[cols].values)
    m = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
    m.fit(sc.transform(train[cols].values), ytr)
    add_baseline(label, label_en,
                 m.predict_proba(sc.transform(calib[cols].values))[:, 1],
                 m.predict_proba(sc.transform(test[cols].values))[:, 1])

# 历史网格频率沿用训练期1:5装配样本的网格正样本率，仅作为排序基线；
# 校准集保序回归将其转换为业务概率，抽样先验不直接进入测试概率。
hist = train.groupby("gid").y.mean()
# 网格×旬历史气候频率使用完整火点标签计数，分母为9个训练防火年，避免下采样分母失真。
labels_all = pd.read_csv(f"{BASE}/data/processed/fire_labels_next_dekad.csv")
labels_all["y"] = (labels_all.n_fires > 0).astype(int)
train_labels = labels_all[(labels_all.fire_year >= 2010) & (labels_all.fire_year <= 2018)]
seasonal_train = train_labels.groupby(["gid", "dekad_seq"]).y.sum() / 9
positive_keys = set(zip(labels_all.gid, labels_all.fire_year, labels_all.dekad_seq))


def simple_baseline_scores(d):
    overall = d.gid.map(hist).fillna(train.y.mean()).to_numpy()
    seasonal_keys = pd.MultiIndex.from_arrays([d.gid, d.dekad_seq])
    seasonal = seasonal_train.reindex(seasonal_keys).fillna(0).to_numpy()
    # 当前特征旬 d 预测 d+1；最近可观测的上一旬火点标签存放在 d-1。
    previous = np.array([
        int(dekad > 0 and (gid, year, dekad - 1) in positive_keys)
        for gid, year, dekad in zip(d.gid, d.fire_year, d.dekad_seq)
    ])
    return {"历史网格火点频率": overall,
            "网格×旬历史气候频率": seasonal,
            "上一旬火点状态": previous}


scores_calib_simple = simple_baseline_scores(calib)
scores_test_simple = simple_baseline_scores(test)
for label, label_en in [
        ("历史网格火点频率", "Historical grid fire frequency"),
        ("网格×旬历史气候频率", "Grid-dekad fire climatology"),
        ("上一旬火点状态", "Previous-dekad fire state")]:
    add_baseline(label, label_en, scores_calib_simple[label], scores_test_simple[label])
add_baseline("XGBoost全变量", "XGBoost (all predictors)",
             proba_calib["XGBoost"], proba_test["XGBoost"])

baseline_table = pd.DataFrame(baseline_rows)
baseline_table.to_csv(f"{BASE}/results/baselines.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(baseline_capture_rows).to_csv(f"{BASE}/results/baseline_capture.csv",
                                           index=False, encoding="utf-8-sig")
print("\n模型与简单预警基线:\n",
      baseline_table[["model", "AUC", "PR_AUC", "Brier_校准后",
                      "Top_20_capture"]].round(4).to_string(index=False))

# ---- 敏感性：1:1 配比 & SMOTE（只看 AUC/KS 的排序稳定性） ----
sens = {}
rng2 = np.random.RandomState(42)
pos_idx = np.where(ytr == 1)[0]
neg_idx = np.where(ytr == 0)[0]
neg_11 = rng2.choice(neg_idx, size=len(pos_idx), replace=False)
idx_11 = np.concatenate([pos_idx, neg_11])
for name, model in make_models(ytr[idx_11].sum(), len(idx_11) - ytr[idx_11].sum()).items():
    X_use = Xtr_s[idx_11] if name == "逻辑回归" else Xtr[idx_11]
    Xt_use = Xte_s if name == "逻辑回归" else Xte
    model.fit(X_use, ytr[idx_11])
    p = model.predict_proba(Xt_use)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    sens[f"{name}_1比1"] = {"AUC": roc_auc_score(yte, p), "KS": float(np.max(tpr - fpr))}

X_sm, y_sm = SMOTE(random_state=42).fit_resample(Xtr, ytr)
print("SMOTE 后训练集:", X_sm.shape, "正样本占比:", y_sm.mean(), flush=True)
for name, model in make_models(y_sm.sum(), len(y_sm) - y_sm.sum()).items():
    scaler_sm = StandardScaler().fit(X_sm)
    X_use = scaler_sm.transform(X_sm) if name == "逻辑回归" else X_sm
    Xt_use = Xte_s if name == "逻辑回归" else Xte
    model.fit(X_use, y_sm)
    p = model.predict_proba(Xt_use)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    sens[f"{name}_SMOTE"] = {"AUC": roc_auc_score(yte, p), "KS": float(np.max(tpr - fpr))}
pd.DataFrame(sens).T.to_csv(f"{BASE}/results/metrics_sensitivity.csv", encoding="utf-8-sig")
print("\n敏感性结果:\n", pd.DataFrame(sens).T.round(4))

# ---- 分年度 AUC ----
year_rows = []
for fy in sorted(test.fire_year.unique()):
    m = test.fire_year.values == fy
    year_rows.append({"fire_year": fy, "n": int(m.sum()), "pos": int(yte[m].sum()),
                      "AUC_XGBoost": roc_auc_score(yte[m], proba_test["XGBoost"][m]),
                      "AUC_RF": roc_auc_score(yte[m], proba_test["随机森林"][m]),
                      "AUC_LR": roc_auc_score(yte[m], proba_test["逻辑回归"][m])})
pd.DataFrame(year_rows).to_csv(f"{BASE}/results/metrics_by_year.csv", index=False, encoding="utf-8-sig")
print("\n分年度 AUC:\n", pd.DataFrame(year_rows).round(4))

# ---- 五折空间留出验证：纯地理分折 + 折内严格时间隔离 ----
# 1°块由经纬度直接生成；KMeans 只使用块质心坐标和网格数权重，不使用任何火点标签。
# 每折的林区网格从训练、校准和测试中同步留出，保持2010—2018/2019—2020/2021—2023顺序。
grids = pd.read_csv(f"{BASE}/data/processed/grids_yunnan.csv")
grids = grids[grids.gid.isin(test.gid.unique())].copy()
grids["block"] = (np.floor(grids.lat_min).astype(int).astype(str) + "_" +
                  np.floor(grids.lon_min).astype(int).astype(str))
blocks = grids.groupby("block").agg(lat=("lat_c", "mean"), lon=("lon_c", "mean"),
                                   grids=("gid", "size")).reset_index()
coords = blocks[["lon", "lat"]].to_numpy(dtype=float)
coords[:, 0] *= np.cos(np.deg2rad(blocks.lat.mean()))
blocks["spatial_fold"] = KMeans(n_clusters=5, random_state=42, n_init=20).fit_predict(
    coords, sample_weight=blocks.grids) + 1
grid_folds = grids.merge(blocks[["block", "spatial_fold"]], on="block", how="left")
grid_folds[["gid", "lat_c", "lon_c", "block", "spatial_fold"]].to_csv(
    f"{BASE}/results/spatial_fold_assignments.csv", index=False, encoding="utf-8-sig")
fold_by_gid = grid_folds.set_index("gid").spatial_fold
fold_test = test.gid.map(fold_by_gid).to_numpy(dtype=int)
if pd.isna(fold_test).any():
    raise RuntimeError("部分测试网格缺少空间折编号")

spatial_rows, spatial_capture_rows = [], []
spatial_oof_raw = {name: np.full(len(test), np.nan) for name in results}
spatial_oof_cal = {name: np.full(len(test), np.nan) for name in results}
for fold in sorted(np.unique(fold_test)):
    held_gids = set(grid_folds.loc[grid_folds.spatial_fold == fold, "gid"])
    tr_s = train[~train.gid.isin(held_gids)]
    ca_s = calib[~calib.gid.isin(held_gids)]
    mask_s = fold_test == fold
    te_s = test[mask_s]
    ytr_s, yca_s, yte_s = tr_s.y.to_numpy(), ca_s.y.to_numpy(), te_s.y.to_numpy()
    if min(ytr_s.sum(), yca_s.sum(), yte_s.sum()) == 0:
        raise RuntimeError(f"空间折{fold}存在无正样本的数据子集")
    scaler_s = StandardScaler().fit(tr_s[FEATS].values)
    for name, model_s in make_models(ytr_s.sum(), len(ytr_s) - ytr_s.sum()).items():
        Xtr_s = scaler_s.transform(tr_s[FEATS].values) if name == "逻辑回归" else tr_s[FEATS].values
        Xca_s = scaler_s.transform(ca_s[FEATS].values) if name == "逻辑回归" else ca_s[FEATS].values
        Xte_s = scaler_s.transform(te_s[FEATS].values) if name == "逻辑回归" else te_s[FEATS].values
        model_s.fit(Xtr_s, ytr_s)
        pca_s = model_s.predict_proba(Xca_s)[:, 1]
        pte_s = model_s.predict_proba(Xte_s)[:, 1]
        iso_s = IsotonicRegression(out_of_bounds="clip").fit(pca_s, yca_s)
        pte_cal_s = iso_s.transform(pte_s)
        spatial_oof_raw[name][mask_s] = pte_s
        spatial_oof_cal[name][mask_s] = pte_cal_s
        row = {"spatial_fold": int(fold), "model": name,
               "n_grids_test": len(held_gids), "n_train": len(tr_s),
               "n_calib": len(ca_s), "n_test": len(te_s),
               "positive_train": int(ytr_s.sum()), "positive_calib": int(yca_s.sum()),
               "positive_test": int(yte_s.sum()), "test_prior": yte_s.mean(),
               "AUC": roc_auc_score(yte_s, pte_s),
               "PR_AUC": average_precision_score(yte_s, pte_s),
               "Brier_校准前": brier_score_loss(yte_s, pte_s),
               "Brier_校准后": brier_score_loss(yte_s, pte_cal_s)}
        row.update(capture_columns(yte_s, pte_cal_s, pte_s))
        spatial_rows.append(row)
        for cap_row in capture_at_topk(yte_s, pte_cal_s, pte_s):
            spatial_capture_rows.append({"spatial_fold": int(fold), "model": name, **cap_row})
        if name == "XGBoost":
            print(f"空间折{fold}: n_test={len(te_s)}, positives={int(yte_s.sum())}, "
                  f"AUC={row['AUC']:.4f}, Top20={row['Top_20_capture']:.4f}", flush=True)

spatial_metrics = pd.DataFrame(spatial_rows)
spatial_metrics.to_csv(f"{BASE}/results/spatial_cv_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(spatial_capture_rows).to_csv(f"{BASE}/results/spatial_cv_capture.csv",
                                          index=False, encoding="utf-8-sig")
spatial_summary_rows = []
for name in results:
    sm = spatial_metrics[spatial_metrics.model == name]
    row = {"model": name, "AUC_pooled": roc_auc_score(yte, spatial_oof_raw[name]),
           "AUC_macro_mean": sm.AUC.mean(), "AUC_fold_min": sm.AUC.min(),
           "AUC_fold_max": sm.AUC.max(),
           "PR_AUC_pooled": average_precision_score(yte, spatial_oof_raw[name]),
           "PR_AUC_macro_mean": sm.PR_AUC.mean(),
           "Brier_pooled_校准前": brier_score_loss(yte, spatial_oof_raw[name]),
           "Brier_pooled_校准后": brier_score_loss(yte, spatial_oof_cal[name])}
    row.update({f"{k}_pooled": v for k, v in
                capture_columns(yte, spatial_oof_cal[name], spatial_oof_raw[name]).items()})
    row["Top_20_capture_macro_mean"] = sm.Top_20_capture.mean()
    spatial_summary_rows.append(row)
spatial_summary = pd.DataFrame(spatial_summary_rows)
spatial_summary.to_csv(f"{BASE}/results/spatial_cv_summary.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(f"{BASE}/results/spatial_cv_predictions.npz",
                    yte=yte, spatial_fold=fold_test,
                    p_lr=spatial_oof_raw["逻辑回归"], p_lr_cal=spatial_oof_cal["逻辑回归"],
                    p_rf=spatial_oof_raw["随机森林"], p_rf_cal=spatial_oof_cal["随机森林"],
                    p_xgb=spatial_oof_raw["XGBoost"], p_xgb_cal=spatial_oof_cal["XGBoost"])
print("\n空间留出汇总:\n",
      spatial_summary[["model", "AUC_pooled", "AUC_macro_mean", "AUC_fold_min",
                       "AUC_fold_max", "Top_20_capture_pooled"]].round(4).to_string(index=False))

# ---- SHAP 与图件缓存 ----
xgb_final = make_models(ytr.sum(), len(ytr) - ytr.sum())["XGBoost"]
xgb_final.fit(Xtr, ytr)
sample_idx = np.random.RandomState(0).choice(len(Xte), size=min(5000, len(Xte)), replace=False)
explainer = shap.TreeExplainer(xgb_final)
sv = explainer.shap_values(Xte[sample_idx])
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)
shap.summary_plot(sv, Xte[sample_idx], feature_names=[CN[f] for f in FEATS], show=False)
plt.xlabel("SHAP 值（对模型 log-odds 输出的贡献）")
try:
    plt.gcf().axes[-1].set_ylabel("特征值高低", rotation=270, labelpad=14)
except Exception:
    pass
plt.tight_layout(); plt.savefig(f"{BASE}/figures/fig4_3_shap_summary.png", dpi=200); plt.close()
shap.summary_plot(sv, Xte[sample_idx], feature_names=[CN[f] for f in FEATS], plot_type="bar", show=False)
plt.xlabel("平均绝对 SHAP 值（log-odds 量纲）")
plt.tight_layout(); plt.savefig(f"{BASE}/figures/fig4_4_shap_bar.png", dpi=200); plt.close()

imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(axis=0)}) \
        .sort_values("mean_abs_shap", ascending=False)
imp.to_csv(f"{BASE}/results/shap_importance.csv", index=False, encoding="utf-8-sig")
print("\nSHAP 重要性:\n", imp)

np.savez_compressed(f"{BASE}/results/figure_cache.npz",
                    p_lr=proba_test["逻辑回归"], p_rf=proba_test["随机森林"],
                    p_xgb=proba_test["XGBoost"], p_xgb_cal=proba_cal["XGBoost"],
                    Xsample=Xte[sample_idx], sv=sv, yte=yte,
                    gid_test=test.gid.values.astype(str))
print("\n全部实验完成；图件缓存已更新")
