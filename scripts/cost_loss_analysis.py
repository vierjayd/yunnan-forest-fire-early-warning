# -*- coding: utf-8 -*-
"""成本—损失（cost-loss）决策价值分析：补强稿件第2.9 / 3.8 节。

只读冻结结果与正式数据划分，不修改任何既有结果文件。
输出目录：analysis_materials/cost_loss_results/
图件输出：E:/workbuddyoutput/2026-10-16-11-07-31/figures/fig_article_cost_loss_value.{png,pdf}

方法：标准二行动 cost-loss 模型。令 C 为对一个"网格×旬"单元采取重点核查/前置的
边际成本，L 为未覆盖单元发生火点造成的额外损失，alpha = C/L。校准概率 p > alpha
时采取防护行动。以 L 为单位，总期望费用：
    E_forecast = [ alpha*(hits+false_alarms) + misses ] / n
    E_clim     = min(alpha, pi)                （气候学最优：全覆盖 或 全不覆盖）
    E_perfect  = alpha*pi                      （完美预报：只对真实事件防护）
相对经济价值 REV = (E_clim - E_forecast) / (E_clim - E_perfect)。
REV>0 表示基于该概率的逐单元决策优于任何无预报的固定策略。
"""
import io
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
PROJ = "E:/workbuddyoutput/2026-11-10-10-24-23"
OUT = PROJ + "/analysis_materials/cost_loss_results"
FIG = BASE + "/figures"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BASE + "/scripts")
from v3_data import load_split, FEATS  # noqa: E402

STATIC = ["forest_frac", "elev_mean", "slope_mean", "south_aspect_frac",
          "pop_density", "dist_road_km", "dist_settlement_km"]
METEO = ["tmax7_mean", "rh7_mean", "wind7_mean", "precip7_sum", "dry_days"]

train, calib, test = load_split("modis")
yca = calib.y.values
ytr = train.y.values
z = np.load(BASE + "/results/figure_cache.npz")
y = z["yte"]
gid = z["gid_test"]
assert len(y) == len(test) and np.array_equal(y, test.y.values)

# ---------- 重建各方法的校准概率（与主模型同一校准期、同一保序规则） ----------
p_xgb_cal = z["p_xgb_cal"].astype(np.float64)

hist = train.groupby("gid").y.mean()
labels_all = pd.read_csv(BASE + "/data/processed/fire_labels_next_dekad.csv")
labels_all["y"] = (labels_all.n_fires > 0).astype(int)
tr_lab = labels_all[(labels_all.fire_year >= 2010) & (labels_all.fire_year <= 2018)]
seasonal_train = tr_lab.groupby(["gid", "dekad_seq"]).y.sum() / 9
positive_keys = set(zip(labels_all.gid, labels_all.fire_year, labels_all.dekad_seq))


def simple_scores(d):
    overall = d.gid.map(hist).fillna(train.y.mean()).to_numpy()
    sk = pd.MultiIndex.from_arrays([d.gid, d.dekad_seq])
    seasonal = seasonal_train.reindex(sk).fillna(0).to_numpy()
    previous = np.array([int(dk > 0 and (g, fy, dk - 1) in positive_keys)
                         for g, fy, dk in zip(d.gid, d.fire_year, d.dekad_seq)])
    return overall, seasonal, previous


def iso_apply(pca, pte):
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(pca), yca)
    return iso.transform(np.asarray(pte))


def ablation(cols):
    sc = StandardScaler().fit(train[cols].values)
    m = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
    m.fit(sc.transform(train[cols].values), ytr)
    return (m.predict_proba(sc.transform(calib[cols].values))[:, 1],
            m.predict_proba(sc.transform(test[cols].values))[:, 1])


ca_h, ca_s, ca_pv = simple_scores(calib)
te_h, te_s, te_pv = simple_scores(test)
ca_st, te_st = ablation(STATIC)
ca_me, te_me = ablation(METEO)

# 主模型校准概率直接取冻结缓存，避免重复训练造成精度差异
methods = {
    "XGBoost全变量": p_xgb_cal,
    "历史网格火点频率": iso_apply(ca_h, te_h),
    "网格×旬历史气候频率": iso_apply(ca_s, te_s),
    "上一旬火点状态": iso_apply(ca_pv, te_pv),
    "仅静态因子逻辑回归": iso_apply(ca_st, te_st),
    "仅气象因子逻辑回归": iso_apply(ca_me, te_me),
}
# 重建的消融模型须与冻结表4一致，否则不得纳入决策分析
_auc_st = roc_auc_score(y, te_st)
_auc_me = roc_auc_score(y, te_me)
assert abs(_auc_st - 0.7028630754204204) < 1e-6, "静态消融重建偏差: %.6f" % _auc_st
assert abs(_auc_me - 0.6392844510418083) < 1e-6, "气象消融重建偏差: %.6f" % _auc_me
print("重建消融模型与冻结表4一致: static=%.6f meteo=%.6f" % (_auc_st, _auc_me))

n = len(y)
pos = int(y.sum())
pi = pos / n
print("n=%d pos=%d pi=%.8f" % (n, pos, pi))
for name, p in methods.items():
    print("  %-24s pmax=%.4f unique=%d" % (name, p.max(), np.unique(p).size))

# ---------- REV 核心 ----------
def decision(yy, pp, alpha):
    flag = pp > alpha
    hit = int((yy[flag] == 1).sum())
    fa = int(flag.sum() - hit)
    miss = int(yy.sum() - hit)
    cr = int(len(yy) - hit - fa - miss)
    return hit, fa, miss, cr


def rev_at(yy, pp, alpha):
    nn = len(yy)
    pii = yy.sum() / nn
    hit, fa, miss, _ = decision(yy, pp, alpha)
    e_f = (alpha * (hit + fa) + miss) / nn
    e_clim = min(alpha, pii)
    e_perf = alpha * pii
    denom = e_clim - e_perf
    if denom <= 1e-15:
        return np.nan
    return (e_clim - e_f) / denom


alphas = np.round(np.linspace(0.001, 0.30, 1200), 6)
rows = []
for name, p in methods.items():
    for a in alphas:
        v = rev_at(y, p, a)
        hit, fa, miss, cr = decision(y, p, a)
        rows.append({
            "method": name, "alpha": a, "REV": v,
            "hits": hit, "false_alarms": fa, "misses": miss, "correct_rejections": cr,
            "flag_count": hit + fa, "flag_frac": (hit + fa) / n,
            "recall": hit / pos, "precision": hit / (hit + fa) if hit + fa else 0.0,
            "expense_per_unit": (a * (hit + fa) + miss) / n,
        })
curve = pd.DataFrame(rows)
curve.to_csv(OUT + "/rev_curve.csv", index=False, encoding="utf-8-sig")

# ---------- 汇总表 ----------
summary = []
for name, g in curve.groupby("method", sort=False):
    g = g.dropna(subset=["REV"])
    imax = g.REV.idxmax()
    r = g.loc[imax]
    posv = g[g.REV > 0]
    lo = posv.alpha.min() if len(posv) else np.nan
    hi = posv.alpha.max() if len(posv) else np.nan
    summary.append({
        "方法": name, "最大REV": r.REV, "最优成本比alpha": r.alpha,
        "最优alpha下预警单元比例": r.flag_frac, "最优alpha下捕获率": r.recall,
        "最优alpha下精确率": r.precision,
        "REV为正的alpha下限": lo, "REV为正的alpha上限": hi,
    })
summ = pd.DataFrame(summary).sort_values("最大REV", ascending=False)
summ.to_csv(OUT + "/rev_summary.csv", index=False, encoding="utf-8-sig")
print("\n=== REV 汇总 ===")
print(summ.to_string(index=False, float_format=lambda x: "%.4f" % x))

# ---------- 典型成本比下的 2x2 决策表 ----------
anchor = [0.005, 0.01, float(round(pi, 4)), 0.04, 0.06, 0.08, 0.12]
anchor = sorted(set(anchor))
drows = []
for a in anchor:
    for name, p in methods.items():
        hit, fa, miss, cr = decision(y, p, a)
        v = rev_at(y, p, a)
        e_f = (a * (hit + fa) + miss) / n
        e_clim = min(a, pi)
        drows.append({
            "成本比alpha": a, "方法": name, "REV": v,
            "命中": hit, "虚警": fa, "漏报": miss, "正确拒绝": cr,
            "预警单元数": hit + fa, "预警比例": (hit + fa) / n,
            "捕获率": hit / pos, "精确率": hit / (hit + fa) if hit + fa else 0,
            "总费用(以L为单位)": a * (hit + fa) + miss,
            "气候学总费用(以L为单位)": e_clim * n,
            "相对气候学节省(以L为单位)": e_clim * n - (a * (hit + fa) + miss),
        })
dtab = pd.DataFrame(drows)
dtab.to_csv(OUT + "/decision_table.csv", index=False, encoding="utf-8-sig")

# ---------- 增量经济价值：配对 bootstrap（网格聚类 / 防火年整块） ----------
p_x = methods["XGBoost全变量"]
p_b = methods["历史网格火点频率"]
a_star = float(summ.loc[summ["方法"] == "XGBoost全变量", "最优成本比alpha"].iloc[0])


def cluster_index(rng, groups, uniq=None):
    if uniq is None:
        uniq = np.unique(groups)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.flatnonzero(np.isin(groups, pick))


def paired(groups, seed=7, n_boot=500):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    diffs, diffs_vmax = [], []
    for _ in range(n_boot):
        idx = cluster_index(rng, groups, uniq)
        yy = y[idx]
        if yy.sum() in (0, len(idx)):
            continue
        diffs.append(rev_at(yy, p_x[idx], a_star) - rev_at(yy, p_b[idx], a_star))
        diffs_vmax.append(rev_at(yy, p_x[idx], a_star))
    v = np.asarray(diffs)
    return {
        "alpha": a_star,
        "观测dREV": rev_at(y, p_x, a_star) - rev_at(y, p_b, a_star),
        "dREV_CI_low": float(np.percentile(v, 2.5)),
        "dREV_CI_high": float(np.percentile(v, 97.5)),
        "p_双侧": float(2 * min((v <= 0).mean(), (v >= 0).mean())),
        "重采样单位": str(np.unique(groups).size) + " 组",
        "n_boot": len(v),
    }


boot_rows = [paired(gid), paired(test.fire_year.values)]
boot = pd.DataFrame(boot_rows)
boot.index = ["网格聚类", "防火年度整块"]
boot.to_csv(OUT + "/incremental_value_bootstrap.csv", encoding="utf-8-sig")
print("\n=== 增量 REV 配对 bootstrap（alpha*=%.4f）===" % a_star)
print(boot.to_string(float_format=lambda x: "%.4f" % x))

# ---------- 空间留出下的经济价值 ----------
zs = np.load(BASE + "/results/spatial_cv_predictions.npz")
ys, folds, ps = zs["yte"], zs["spatial_fold"], zs["p_xgb_cal"].astype(np.float64)
sp_rows = []
for f in np.unique(folds):
    m = folds == f
    yy, pp = ys[m], ps[m]
    g = [(a, rev_at(yy, pp, a)) for a in alphas]
    g = [(a, v) for a, v in g if np.isfinite(v)]
    amax, vmax = max(g, key=lambda t: t[1])
    sp_rows.append({"空间折": int(f), "折内先验": yy.mean(), "正样本": int(yy.sum()),
                    "折内最大REV": vmax, "折内最优alpha": amax,
                    "在alpha*下的REV": rev_at(yy, pp, a_star)})
vpool_all = [(a, rev_at(ys, ps, a)) for a in alphas]
vpool_all = [(a, v) for a, v in vpool_all if np.isfinite(v)]
apool, vpool = max(vpool_all, key=lambda t: t[1])
sp_rows.append({"空间折": "汇总pooled", "折内先验": ys.mean(), "正样本": int(ys.sum()),
                "折内最大REV": vpool, "折内最优alpha": apool,
                "在alpha*下的REV": rev_at(ys, ps, a_star)})
sp = pd.DataFrame(sp_rows)
sp.to_csv(OUT + "/spatial_rev.csv", index=False, encoding="utf-8-sig")
print("\n=== 空间留出 REV ===")
print(sp.to_string(index=False, float_format=lambda x: "%.4f" % x))
print("时间外推在 alpha* 下 REV = %.4f" % rev_at(y, p_x, a_star))

# ---------- 正文候选数字 ----------
vx = float(summ.loc[summ["方法"] == "XGBoost全变量", "最大REV"].iloc[0])
vb = float(summ.loc[summ["方法"] == "历史网格火点频率", "最大REV"].iloc[0])
r_at = dtab[(dtab["方法"] == "XGBoost全变量") & (np.isclose(dtab["成本比alpha"], round(pi, 4)))]
key = {
    "test_n": n, "test_positives": pos, "test_prior": pi,
    "alpha_star_XGB": a_star, "REV_max_XGB": vx,
    "alpha_star_baseline": float(summ.loc[summ["方法"] == "历史网格火点频率", "最优成本比alpha"].iloc[0]),
    "REV_max_baseline": vb, "REV_max_gain": vx - vb,
    "alpha_positive_range_XGB": [float(summ.loc[summ["方法"] == "XGBoost全变量", "REV为正的alpha下限"].iloc[0]),
                                 float(summ.loc[summ["方法"] == "XGBoost全变量", "REV为正的alpha上限"].iloc[0])],
    "calibrated_p_max": float(p_x.max()), "calibrated_unique": int(np.unique(p_x).size),
    "REV_at_alpha_star_XGB": rev_at(y, p_x, a_star),
    "REV_at_alpha_star_baseline": rev_at(y, p_b, a_star),
    "spatial_pooled_REV_at_alpha_star": rev_at(ys, ps, a_star),
    "spatial_pooled_REV_max": vpool,
    "spatial_fold_REV_mean_at_alpha_star": float(sp[sp["空间折"] != "汇总pooled"]["在alpha*下的REV"].mean()),
}
with io.open(OUT + "/key_numbers.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print("\n=== key_numbers.json ===")
print(json.dumps(key, ensure_ascii=False, indent=2))

# ---------- 图件 ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
COL = {"XGBoost全变量": "#c0392b", "历史网格火点频率": "#2c3e50",
       "网格×旬历史气候频率": "#7f8c8d", "上一旬火点状态": "#95a5a6"}
LBL = {"XGBoost全变量": "XGBoost (all predictors)",
       "历史网格火点频率": "Historical grid fire frequency",
       "网格×旬历史气候频率": "Grid-dekad fire climatology",
       "上一旬火点状态": "Previous-dekad fire state"}

ax = axes[0]
for name in ["XGBoost全变量", "历史网格火点频率", "网格×旬历史气候频率", "上一旬火点状态"]:
    g = curve[curve["method"] == name].dropna(subset=["REV"])
    ax.plot(g.alpha, g.REV, lw=2, color=COL[name], label=LBL[name])
ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
ax.axhline(1, color="#27ae60", lw=1.2, ls=":", label="Perfect forecast (REV = 1)")
ax.axvline(pi, color="#2980b9", lw=1, ls="-.", alpha=0.7,
           label="Base rate π = %.3f" % pi)
ax.set_xlim(0, 0.16)
ax.set_ylim(-0.05, 1.02)
ax.set_xlabel("Cost-loss ratio  $\\alpha = C/L$")
ax.set_ylabel("Relative economic value  (REV)")
ax.set_title("(a) Temporal hold-out (2021-2023)")
ax.legend(fontsize=7.6, loc="upper right", framealpha=0.9)

ax2 = axes[1]
g = curve[curve["method"] == "XGBoost全变量"].dropna(subset=["REV"])
ax2.plot(g.alpha, g.REV, lw=2, color="#c0392b",
         label="Temporal hold-out (XGBoost)")
sp_curve = []
for a in alphas:
    v = rev_at(ys, ps, a)
    if np.isfinite(v):
        sp_curve.append((a, v))
sp_curve = np.array(sp_curve)
ax2.plot(sp_curve[:, 0], sp_curve[:, 1], lw=2, color="#8e44ad",
         label="Spatial hold-out, pooled (XGBoost)")
fold_curves, fold_alphas = [], None
for f in np.unique(folds):
    m = folds == f
    fc = [(a, rev_at(ys[m], ps[m], a)) for a in alphas]
    fc = [(a, v) for a, v in fc if np.isfinite(v)]
    if fold_alphas is None:
        fold_alphas = np.array([a for a, v in fc])
    fold_curves.append([v for a, v in fc])
fold_curves = np.array(fold_curves)
fold_at_star = [rev_at(ys[folds == f], ps[folds == f], a_star)
                for f in np.unique(folds)]
n_negative = sum(v < 0 for v in fold_at_star)
ax2.axhline(0, color="k", lw=1, ls="--", alpha=0.6)
ax2.axvline(pi, color="#2980b9", lw=1, ls="-.", alpha=0.7,
            label="Base rate π")
ax2.set_xlim(0, 0.16)
ax2.set_ylim(-0.08, 0.55)
ax2.set_xlabel("Cost-loss ratio  $\\alpha = C/L$")
ax2.set_ylabel("Relative economic value  (REV)")
ax2.set_title("(b) Transferability boundary")
ax2.legend(fontsize=8, loc="upper right", framealpha=0.95)
# 折内负值的细节交由图注与表9承载，避免图内文字遮挡曲线
print("图注用: pooled spatial REV=%.3f vs temporal %.3f; %d/5 folds negative at alpha* (min %.2f)"
      % (vpool, rev_at(y, p_x, a_star), n_negative, min(fold_at_star)))

fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(FIG + "/fig_article_cost_loss_value." + ext, dpi=200,
                bbox_inches="tight")
print("\n图件已写出: " + FIG + "/fig_article_cost_loss_value.{png,pdf}")
print("结果目录: " + OUT)
