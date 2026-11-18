# -*- coding: utf-8 -*-
"""第4章应用层分析（V3 口径）：业务先验下的风险分级、捕获率、风险空间图、SHAP 依赖图。

风险等级以测试集业务先验 pi 为基准按「相对风险倍数」划分，避免把下采样样本口径的
概率阈值直接移植到业务系统（盲审意见第 3 条）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

from v3_data import BASE, FEATS, load_split, prior_info

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

train, calib, test = load_split("modis")
pi = prior_info(train, calib, test)["business_prior"]
Xtr, ytr = train[FEATS].values, train.y.values
Xca = calib[FEATS].values
Xte, yte = test[FEATS].values, test.y.values
print(f"业务先验 pi={pi:.4%}；训练 {len(train)}（下采样）/ 校准 {len(calib)} / 测试 {len(test)}（完整单元）")

model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                      eval_metric="logloss", random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)  # 仅用于下方 SHAP 依赖图
# 所有风险与 Top-k 结果直接读取主训练脚本保存的同一组预测，避免重复训练的数值漂移。
with np.load(f"{BASE}/results/figure_cache.npz") as cache:
    if not np.array_equal(cache["yte"], yte):
        raise RuntimeError("主模型缓存的测试标签与当前测试集不一致")
    test["p_raw"] = cache["p_xgb"]
    test["p_cal"] = cache["p_xgb_cal"]

# 1. 风险分级：以业务先验 pi 为基准的相对风险倍数
edges = [0, 0.5 * pi, pi, 2 * pi, 4 * pi, 1.0]
labels = [f"低(<0.5π, <{0.5*pi:.3f})", f"较低(0.5π—π, {0.5*pi:.3f}—{pi:.3f})",
          f"中(π—2π, {pi:.3f}—{2*pi:.3f})", f"较高(2π—4π, {2*pi:.3f}—{4*pi:.3f})",
          f"高(≥4π, ≥{4*pi:.3f})"]
test["level"] = pd.cut(test.p_cal, bins=edges, labels=labels, include_lowest=True)
grp = test.groupby("level", observed=False).agg(n=("y", "size"), fires=("y", "sum"),
                                                p_mean=("p_cal", "mean"))
grp["fire_rate"] = grp.fires / grp.n
grp["fire_share"] = grp.fires / test.y.sum()
print("\n风险分级（业务先验口径）:\n", grp.round(5))
grp.to_csv(f"{BASE}/results/risk_levels.csv", encoding="utf-8-sig")
pd.DataFrame({"下界": edges[:-1], "上界": edges[1:], "等级": labels,
              "相对业务先验倍数": [0, 0.5, 1, 2, 4]}) \
  .to_csv(f"{BASE}/results/risk_level_edges.csv", index=False, encoding="utf-8-sig")

# 2. 高风险单元捕获率：校准概率降序，原始 XGBoost 分数降序次级排序。
# 保序校准会产生大量并列值；固定次级排序避免不同排序方式造成结果漂移。
order = np.lexsort((np.arange(len(test)), -test.p_raw.to_numpy(), -test.p_cal.to_numpy()))
srt = test.iloc[order]
rows = []
for frac in [0.05, 0.10, 0.20, 0.30]:
    k = int(len(srt) * frac)
    rows.append({"top_frac": frac, "n": k, "fires": int(srt.y.iloc[:k].sum()),
                 "capture": srt.y.iloc[:k].sum() / test.y.sum()})
cap = pd.DataFrame(rows)
print("\n最高风险单元捕获率（校准概率＋原始分数次级排序）:\n", cap.round(5))
cap.to_csv(f"{BASE}/results/risk_capture.csv", index=False, encoding="utf-8-sig")

# 3. 风险空间分布
grids = pd.read_csv(f"{BASE}/data/processed/grids_yunnan.csv")
gm = test.groupby("gid").p_cal.mean().rename("p_mean").reset_index()
gm = grids.merge(gm, on="gid", how="left")
vmax = float(np.percentile(gm.p_mean.dropna(), 99))
plt.figure(figsize=(8, 7))
sc = plt.scatter(gm.lon_c, gm.lat_c, c=gm.p_mean, s=14, cmap="YlOrRd", vmin=0, vmax=vmax)
plt.colorbar(sc, label="平均校准起火概率（业务口径）")
plt.xlabel("经度"); plt.ylabel("纬度")
plt.tight_layout(); plt.savefig(f"{BASE}/figures/fig4_5_risk_map.png", dpi=200); plt.close()
top = gm.dropna().sort_values("p_mean", ascending=False).head(10)
top.to_csv(f"{BASE}/results/top_risk_grids.csv", index=False, encoding="utf-8-sig")
print(f"\n空间图色标上限(P99)={vmax:.4f}；最高风险网格:\n", top.round(4))

# 4. SHAP 依赖图
explainer = shap.TreeExplainer(model)
idx = np.random.RandomState(0).choice(len(Xte), size=min(5000, len(Xte)), replace=False)
sv = explainer.shap_values(Xte[idx])
if isinstance(sv, list):
    sv = sv[1]
for feat, cn in [("forest_frac", "林地占比"), ("rh7_mean", "前7日平均相对湿度")]:
    j = FEATS.index(feat)
    plt.figure(figsize=(7, 5))
    plt.scatter(Xte[idx, j], sv[:, j], s=5, alpha=0.3)
    plt.xlabel(cn); plt.ylabel("SHAP 值（log-odds 量纲）")
    plt.tight_layout()
    plt.savefig(f"{BASE}/figures/fig4_6_depend_{feat}.png", dpi=200)
    plt.close()
# 5. 小论文图件：主模型与简单预警基线的 Top-k 捕获曲线
baseline_cap = pd.read_csv(f"{BASE}/results/baseline_capture.csv")
article_models = {
    "XGBoost全变量": ("Calibrated XGBoost", "#C44E52", 2.2),
    "历史网格火点频率": ("Historical grid frequency", "#4C72B0", 1.6),
    "网格×旬历史气候频率": ("Grid-dekad climatology", "#55A868", 1.6),
    "上一旬火点状态": ("Previous-dekad fire state", "#8172B3", 1.6),
}
fig, ax = plt.subplots(figsize=(7.2, 5.6))
for name, (label, color, width) in article_models.items():
    d = baseline_cap[baseline_cap.model == name].sort_values("top_frac")
    ax.plot(d.top_frac * 100, d.capture * 100, marker="o", linewidth=width,
            color=color, label=label)
ax.plot([0, 30], [0, 30], color="#888888", linestyle="--", linewidth=1.1,
        label="Random ranking")
ax.set_xlabel("Units flagged as high risk (%)")
ax.set_ylabel("Fire-positive units captured (%)")
ax.set_xlim(0, 30); ax.set_ylim(0, 80)
ax.set_xticks([0, 5, 10, 20, 30])
ax.grid(alpha=0.25, linestyle=":")
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{BASE}/figures/fig_article_topk_capture.png", dpi=300)
fig.savefig(f"{BASE}/figures/fig_article_topk_capture.pdf")
plt.close(fig)

# 6. 小论文补充图：五折空间留出的判别能力与 Top-20% 捕获率
spatial_file = f"{BASE}/results/spatial_cv_metrics.csv"
spatial_summary_file = f"{BASE}/results/spatial_cv_summary.csv"
if os.path.exists(spatial_file) and os.path.exists(spatial_summary_file):
    sm = pd.read_csv(spatial_file)
    ss = pd.read_csv(spatial_summary_file)
    sx = sm[sm.model == "XGBoost"].sort_values("spatial_fold")
    ssx = ss[ss.model == "XGBoost"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    axes[0].plot(sx.spatial_fold, sx.AUC, color="#C44E52", marker="o", linewidth=1.8)
    axes[0].axhline(ssx.AUC_pooled, color="#4C72B0", linestyle="--", linewidth=1.3,
                    label=f"Pooled AUC = {ssx.AUC_pooled:.3f}")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.5, 0.9)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(sx.spatial_fold, sx.Top_20_capture * 100, color="#C44E52",
                 marker="o", linewidth=1.8)
    axes[1].axhline(ssx.Top_20_capture_pooled * 100, color="#4C72B0", linestyle="--",
                    linewidth=1.3,
                    label=f"Pooled capture = {ssx.Top_20_capture_pooled * 100:.1f}%")
    axes[1].set_ylabel("Top-20% capture (%)")
    axes[1].set_ylim(0, 80)
    axes[1].legend(frameon=False, fontsize=9)
    for ax, panel in zip(axes, ["(a)", "(b)"]):
        ax.set_xlabel("Spatial hold-out fold")
        ax.set_xticks(sorted(sm.spatial_fold.unique()))
        ax.grid(alpha=0.25, linestyle=":")
        ax.text(0.02, 0.96, panel, transform=ax.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{BASE}/figures/fig_article_spatial_validation.png", dpi=300)
    fig.savefig(f"{BASE}/figures/fig_article_spatial_validation.pdf")
    plt.close(fig)

# 7. 摘要与结果段候选数字：全部从本次冻结结果自动提取，避免手工抄录漂移。
import json
main_metrics = pd.read_csv(f"{BASE}/results/metrics_main.csv", index_col=0)
key_results = {
    "ranking_rule": "calibrated probability descending; raw XGBoost score descending as tie-breaker",
    "test_n": int(len(test)), "test_positives": int(test.y.sum()),
    "test_prior": float(test.y.mean()),
    "XGBoost_AUC_temporal_test": float(main_metrics.loc["XGBoost", "AUC"]),
    "XGBoost_PR_AUC_temporal_test": float(main_metrics.loc["XGBoost", "PR_AUC"]),
    "XGBoost_Brier_before_calibration": float(main_metrics.loc["XGBoost", "Brier_校准前"]),
    "XGBoost_Brier_after_calibration": float(main_metrics.loc["XGBoost", "Brier_校准后"]),
    "Top_k_capture_temporal_test": {str(r.top_frac): float(r.capture) for r in cap.itertuples()},
}
if os.path.exists(spatial_summary_file):
    ss = pd.read_csv(spatial_summary_file)
    ssx = ss[ss.model == "XGBoost"].iloc[0]
    key_results["spatial_holdout_XGBoost"] = {
        "AUC_pooled": float(ssx.AUC_pooled), "AUC_macro_mean": float(ssx.AUC_macro_mean),
        "AUC_fold_min": float(ssx.AUC_fold_min), "AUC_fold_max": float(ssx.AUC_fold_max),
        "Top_20_capture_pooled": float(ssx.Top_20_capture_pooled),
    }
with open(f"{BASE}/results/article_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key_results, f, ensure_ascii=False, indent=2)
print("\n应用层分析、Top-k 图件与摘要候选数字已更新")
