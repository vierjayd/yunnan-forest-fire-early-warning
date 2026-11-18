import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE = "E:/workbuddyoutput/2026-10-16-11-07-31"
sys.path.insert(0, f"{BASE}/scripts")
from v3_data import load_split

train, calib, test = load_split("modis")
labels = pd.read_csv(f"{BASE}/data/processed/fire_labels_next_dekad.csv")
labels["y"] = (labels["n_fires"] > 0).astype(int)
train_labels = labels[(labels.fire_year >= 2010) & (labels.fire_year <= 2018)]

# Existing baseline: grid-specific positive proportion in the downsampled training data.
hist_train = train.groupby("gid").y.mean()

# Seasonal climatology: all observed positives for the same grid and dekad in 2010-2018,
# divided by nine training fire years. This uses counts, not the downsampled denominator.
seasonal_train = train_labels.groupby(["gid", "dekad_seq"]).y.sum() / 9
positive_keys = set(zip(labels.gid, labels.fire_year, labels.dekad_seq))


def make_scores(d):
    overall = d.gid.map(hist_train).fillna(train.y.mean()).to_numpy()
    seasonal_keys = pd.MultiIndex.from_arrays([d.gid, d.dekad_seq])
    seasonal = seasonal_train.reindex(seasonal_keys).fillna(0).to_numpy()
    # fire_labels_next_dekad at index d stores the observed label for dekad d+1.
    # For a row predicting d+1, the latest observable previous-dekad label is stored at d-1.
    previous = np.array([
        int(dekad > 0 and (gid, year, dekad - 1) in positive_keys)
        for gid, year, dekad in zip(d.gid, d.fire_year, d.dekad_seq)
    ])
    return {"overall_grid_history": overall,
            "grid_dekad_climatology": seasonal,
            "previous_dekad_fire": previous}


scores_calib = make_scores(calib)
scores_test = make_scores(test)
rows = []
for name in scores_test:
    pc = scores_calib[name]
    pt = scores_test[name]
    iso = IsotonicRegression(out_of_bounds="clip").fit(pc, calib.y.to_numpy())
    pt_cal = iso.transform(pt)
    rows.append({
        "baseline": name,
        "AUC": roc_auc_score(test.y, pt),
        "PR_AUC": average_precision_score(test.y, pt),
        "Brier_raw": brier_score_loss(test.y, pt),
        "Brier_calibrated": brier_score_loss(test.y, pt_cal),
        "test_mean_raw": pt.mean(),
        "test_positive_rate": test.y.mean(),
    })

cache = np.load(f"{BASE}/results/figure_cache.npz")
rows.append({
    "baseline": "XGBoost_all_features",
    "AUC": roc_auc_score(test.y, cache["p_xgb"]),
    "PR_AUC": average_precision_score(test.y, cache["p_xgb"]),
    "Brier_raw": brier_score_loss(test.y, cache["p_xgb"]),
    "Brier_calibrated": brier_score_loss(test.y, cache["p_xgb_cal"]),
    "test_mean_raw": cache["p_xgb"].mean(),
    "test_positive_rate": test.y.mean(),
})

out = pd.DataFrame(rows)
print(out.round(6).to_string(index=False))


def tie_aware_capture(y, score, frac):
    """Expected capture when observations tied at the cutoff are selected at random."""
    k = int(len(y) * frac)
    captured = 0.0
    filled = 0
    for value in np.sort(np.unique(score))[::-1]:
        idx = np.flatnonzero(score == value)
        if filled + len(idx) <= k:
            captured += y[idx].sum()
            filled += len(idx)
        else:
            slots = k - filled
            captured += slots * y[idx].mean()
            filled = k
            break
    return captured / y.sum()


capture_rows = []
for name in scores_test:
    row = {"baseline": name}
    for frac in [0.05, 0.10, 0.20, 0.30]:
        row[f"top_{int(frac * 100)}_capture"] = tie_aware_capture(
            test.y.to_numpy(), scores_test[name], frac)
    capture_rows.append(row)
row = {"baseline": "XGBoost_all_features"}
for frac in [0.05, 0.10, 0.20, 0.30]:
    row[f"top_{int(frac * 100)}_capture"] = tie_aware_capture(
        test.y.to_numpy(), cache["p_xgb_cal"], frac)
capture_rows.append(row)
print("\nTie-aware expected capture rates:\n")
print(pd.DataFrame(capture_rows).round(6).to_string(index=False))
