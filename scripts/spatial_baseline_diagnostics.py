# -*- coding: utf-8 -*-
"""A5 补充诊断：五折空间留出下的简单基线对照与折内标准化。

性质：**补充诊断**，不修改任何冻结结果文件、不重训主模型。
所有输入来自正式数据划分 load_split()、冻结缓存 figure_cache.npz /
spatial_cv_predictions.npz、空间折映射 spatial_fold_assignments.csv。

构造的基线：对每一折 f，只用「不属于 f 的网格」的训练期（2010—2018）
完整火点标签，按旬序计算历史发生频率；该基线不含任何留出区域的空间信息，
因此可作为「零空间迁移成本」的对照。校准器同样只用非留出折的校准期
（2019—2020）单元拟合，严格遵守「留出折网格不参与任何建模或校准」。

输出目录：analysis_materials/spatial_baseline/
"""
import io
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
OUT = "E:/workbuddyoutput/2026-11-10-10-24-23/analysis_materials/spatial_baseline"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BASE + "/scripts")
from v3_data import load_split  # noqa: E402

buf = []


def p(*a):
    s = " ".join(str(x) for x in a)
    buf.append(s)
    print(s, flush=True)


def cap20(y, pcal, praw):
    """与主稿一致的确定性 Top-20% 捕获率：校准概率→原始分数→行号。"""
    order = np.lexsort((np.arange(len(y)), -praw, -pcal))
    k = int(len(y) * 0.2)
    return y[order[:k]].sum() / y.sum()


def auc_ci(y, score, groups, n_boot=300, seed=11):
    """折内按网格聚类的 AUC bootstrap 95%CI（整数分组索引，避免字符串 isin）。"""
    rng = np.random.RandomState(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.flatnonzero(inv == j) for j in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[j] for j in pick])
        yy = y[idx]
        if yy.sum() in (0, len(idx)):
            continue
        vals.append(roc_auc_score(yy, score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, len(vals)


train, calib, test = load_split("modis")
for d in (train, calib, test):
    d["gid"] = d["gid"].astype(str)

z = np.load(BASE + "/results/figure_cache.npz")
sp = np.load(BASE + "/results/spatial_cv_predictions.npz")
yte = sp["yte"]
folds = sp["spatial_fold"]
gid = z["gid_test"].astype(str)
p_xgb = sp["p_xgb"].astype(np.float64)
p_xgb_cal = sp["p_xgb_cal"].astype(np.float64)

assign = pd.read_csv(BASE + "/results/spatial_fold_assignments.csv", encoding="utf-8-sig")
assign["gid"] = assign["gid"].astype(str)
g2f = dict(zip(assign["gid"], assign["spatial_fold"]))

p("=" * 78)
p("A5 补充诊断：空间留出的简单基线对照与折内标准化")
p("（补充诊断，仅新增结果，不修改任何冻结主结果）")
p("=" * 78)
assert len(test) == len(yte) == len(folds)
assert np.array_equal(test.y.values, yte), "测试标签与空间缓存不一致"
assert np.array_equal(test.y.values, z["yte"]), "测试标签与主缓存不一致"
assert np.array_equal(test.gid.values.astype(str), gid), "网格ID与缓存不一致"
assert np.array_equal(np.array([g2f[g] for g in gid]), folds), "折映射与缓存不一致"
p("一致性检查 OK：len=%d，标签/网格ID/折映射与主缓存完全一致" % len(test))

lab = pd.read_csv(BASE + "/data/processed/fire_labels_next_dekad.csv")
lab["gid"] = lab["gid"].astype(str)
lab["y"] = (lab["n_fires"] > 0).astype(int)
tr_lab = lab[(lab.fire_year >= 2010) & (lab.fire_year <= 2018)].copy()
all_gids = np.unique(gid)
p("训练期完整火点标签行数=%d；林区网格数=%d" % (len(tr_lab), len(all_gids)))

rows = []
chunks = []
for f in sorted(np.unique(folds)):
    fold_gids = set(assign.loc[assign.spatial_fold == f, "gid"])
    nonf_gids = [g for g in all_gids if g not in fold_gids]
    # 旬气候频率：9 个训练防火年度、非留出网格的平均发生频率
    cnt = tr_lab[~tr_lab.gid.isin(fold_gids)].groupby("dekad_seq").size()
    freq = cnt / (len(nonf_gids) * 9.0)
    gmean = freq.mean()

    def score(d):
        return d.dekad_seq.map(freq).fillna(gmean).to_numpy(dtype=float)

    ca_f = calib[~calib.gid.isin(fold_gids)]
    te_f = test[folds == f]
    s_ca, s_te = score(ca_f), score(te_f)
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_ca, ca_f.y.values)
    p_base_cal = iso.transform(s_te)

    y = te_f.y.values
    g = te_f.gid.values
    i = np.flatnonzero(folds == f)
    y_x, px, pxc = yte[i], p_xgb[i], p_xgb_cal[i]

    auc_x = roc_auc_score(y_x, px)
    auc_b = roc_auc_score(y, s_te)
    cap_x = cap20(y_x, pxc, px)
    cap_b = cap20(y, p_base_cal, s_te)
    lo_x, hi_x, nb_x = auc_ci(y_x, px, g)
    lo_b, hi_b, nb_b = auc_ci(y, s_te, g)

    rows.append(dict(
        spatial_fold=f, n_grids_test=len(np.unique(g)), n_test=len(y),
        positive_test=int(y.sum()), test_prior=float(y.mean()),
        AUC_XGB=auc_x, AUC_XGB_CI_lo=lo_x, AUC_XGB_CI_hi=hi_x,
        AUC_baseline=auc_b, AUC_baseline_CI_lo=lo_b, AUC_baseline_CI_hi=hi_b,
        Top20_fold_XGB=cap_x, Top20_fold_baseline=cap_b,
        PR_AUC_XGB=average_precision_score(y_x, px),
        PR_AUC_baseline=average_precision_score(y, s_te),
        n_boot_valid=min(nb_x, nb_b),
    ))
    chunks.append((y, p_base_cal, s_te, y_x, pxc, px))
    p("折%d 网格%4d 单元%6d 正样本%5d 先验%.4f | XGB AUC %.4f [%.4f,%.4f] Top20 %.4f | 基线 AUC %.4f [%.4f,%.4f] Top20 %.4f"
      % (f, len(np.unique(g)), len(y), y.sum(), y.mean(),
         auc_x, lo_x, hi_x, cap_x, auc_b, lo_b, hi_b, cap_b))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "spatial_baseline_metrics.csv"),
                              index=False, encoding="utf-8-sig")
    with io.open(os.path.join(OUT, "spatial_baseline_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(buf) + "\n")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "spatial_baseline_metrics.csv"), index=False, encoding="utf-8-sig")

# ---- pooled：把五折预测合并后统一排序（与冻结汇总口径一致）----
y_all = np.concatenate([c[0] for c in chunks])
pb_cal = np.concatenate([c[1] for c in chunks])
pb_raw = np.concatenate([c[2] for c in chunks])
yx_all = np.concatenate([c[3] for c in chunks])
pxc_all = np.concatenate([c[4] for c in chunks])
px_all = np.concatenate([c[5] for c in chunks])
assert np.array_equal(y_all, yx_all)

cap_x_pool = cap20(yx_all, pxc_all, px_all)
cap_b_pool = cap20(y_all, pb_cal, pb_raw)
auc_x_pool = roc_auc_score(yx_all, px_all)
auc_b_pool = roc_auc_score(y_all, pb_raw)

frozen_pool = 0.3000277546489037
frozen_macro = 0.3961843709773795
p("")
p("=" * 78)
p("汇总与口径核对")
p("=" * 78)
p("XGB pooled Top20 复算 = %.6f （冻结值 %.6f，差 %.2e）"
  % (cap_x_pool, frozen_pool, abs(cap_x_pool - frozen_pool)))
assert abs(cap_x_pool - frozen_pool) < 1e-9, "排序实现与冻结结果不一致"
p("XGB 折内 Top20 均值 = %.6f （冻结值 %.6f，差 %.2e）"
  % (df.Top20_fold_XGB.mean(), frozen_macro, abs(df.Top20_fold_XGB.mean() - frozen_macro)))
assert abs(df.Top20_fold_XGB.mean() - frozen_macro) < 1e-9
p("→ 本脚本的排序口径与冻结结果完全一致，新增基线数字可直接对照")
p("")
p("XGB    pooled AUC=%.4f  pooled Top20=%.4f  折内Top20均值=%.4f  宏平均AUC=%.4f"
  % (auc_x_pool, cap_x_pool, df.Top20_fold_XGB.mean(), df.AUC_XGB.mean()))
p("基线  pooled AUC=%.4f  pooled Top20=%.4f  折内Top20均值=%.4f  宏平均AUC=%.4f"
  % (auc_b_pool, cap_b_pool, df.Top20_fold_baseline.mean(), df.AUC_baseline.mean()))
p("pooled 增量：ΔAUC=%+.4f  ΔTop20=%+.2f 个百分点"
  % (auc_x_pool - auc_b_pool, 100 * (cap_x_pool - cap_b_pool)))
p("随机排序 Top20 期望=20.00%%；XGB pooled 净超随机=%.2f 个百分点；基线净超随机=%.2f 个百分点"
  % (100 * (cap_x_pool - 0.2), 100 * (cap_b_pool - 0.2)))
p("")
p("折间异质性：先验 %.4f—%.4f（%.1f 倍）；正样本 %d—%d（%.1f 倍）"
  % (df.test_prior.min(), df.test_prior.max(),
     df.test_prior.max() / df.test_prior.min(),
     df.positive_test.min(), df.positive_test.max(),
     df.positive_test.max() / df.positive_test.min()))
p("pooled 与折内均值差异：XGB %.2f 个百分点；基线 %.2f 个百分点"
  % (100 * (df.Top20_fold_XGB.mean() - cap_x_pool),
     100 * (df.Top20_fold_baseline.mean() - cap_b_pool)))
p("→ 混折汇总把不同先验的折直接合并，Top20 受构成差异影响；应同时报告折内标准化结果")

summary = pd.DataFrame([dict(
    method="XGBoost", AUC_pooled=auc_x_pool, AUC_macro_mean=df.AUC_XGB.mean(),
    Top20_pooled=cap_x_pool, Top20_fold_mean=df.Top20_fold_XGB.mean(),
    PR_AUC_pooled=average_precision_score(yx_all, px_all)),
    dict(method="Seasonal climatology baseline", AUC_pooled=auc_b_pool,
         AUC_macro_mean=df.AUC_baseline.mean(), Top20_pooled=cap_b_pool,
         Top20_fold_mean=df.Top20_fold_baseline.mean(),
         PR_AUC_pooled=average_precision_score(y_all, pb_raw))])
summary.to_csv(os.path.join(OUT, "spatial_baseline_summary.csv"), index=False,
               encoding="utf-8-sig")

p("")
p("说明：该基线仅使用非留出网格的旬序历史频率，不含任何留出区域的空间信息，")
p("      用于回答「空间留出下的排序能力是否优于一条无需迁移的简单规则」；")
p("      各折单独拟合保序校准器，严格保持留出折网格不参与建模与校准。")
p("结果目录：" + OUT)

with io.open(os.path.join(OUT, "spatial_baseline_diagnostics.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(buf) + "\n")
