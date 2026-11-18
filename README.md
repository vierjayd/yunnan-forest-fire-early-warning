# A Calibrated Machine-Learning Framework for Forest Fire Early Warning: A Case Study of Yunnan Province, China

Reproducibility package for the manuscript *基于校准机器学习的森林火灾预警研究：以云南省为例*.

This repository contains the processed grid–dekad modelling data, analysis scripts, frozen results, and figure sources needed to reproduce the reported numbers. The study performs **retrospective risk ranking** of satellite-detectable forest fire hotspots; it is **not** a real-time operational forecasting or resource-dispatch system.

## 1. Contents

| Directory | Description |
|---|---|
| `data/processed/` | Processed grid–dekad datasets (training / calibration / test, MODIS and VIIRS labels, static features, distance variables, NDVI). Third-party raw products are **not** redistributed. |
| `results/` | Frozen result tables and diagnostics (metrics, baselines, calibration bins, risk levels, spatial cross-validation, sensitivity analyses). |
| `analysis/cost_loss_results/` | Cost–loss decision curves, decision tables, prior sensitivity, bootstrap intervals, spatial REV. |
| `analysis/spatial_baseline/` | Five-fold spatial holdout diagnostics and seasonal-baseline comparison. |
| `scripts/` | Analysis pipeline: data assembly, model fitting, calibration, baselines, spatial holdout, cost–loss analysis, and figure generation. |
| `figures/` | Eight 300-dpi PNG figures used in the manuscript. |
| `docs/supplementary/` | Supplementary Materials S1–S3 (Markdown). |
| `docs/字段字典.csv` | Field dictionary for the processed CSV files. |

## 2. Data sources and licensing

- **Code**: MIT License (see `LICENSE`).
- **Self-generated data** (processed datasets, results, figures): CC BY 4.0.
- **OSM-derived distance variables**: ODbL 1.0; attribution to OpenStreetMap contributors.
- **Third-party raw products are not redistributed**. Obtain them from their official providers:

| Product | Provider / entry point |
|---|---|
| MODIS C6 active fire (FIRMS) | https://firms.modaps.eosdis.nasa.gov/download/ |
| VIIRS S-NPP 375 m active fire | https://firms.modaps.eosdis.nasa.gov/download/ |
| ERA5 hourly single levels | https://cds.climate.copernicus.eu/ (dataset `reanalysis-era5-single-levels`) |
| MOD13A2 V061 NDVI | https://lpdaac.usgs.gov/products/mod13a2v061/ |
| ESA WorldCover 2021 v200 | https://doi.org/10.5281/zenodo.7254221 |
| SRTM | https://srtm.csi.cgiar.org/ |
| WorldPop 2020 | https://www.worldpop.org/ |
| OpenStreetMap (Geofabrik China snapshot, 2026-10-17) | https://download.geofabrik.de/asia/china.html |

No API keys, tokens, or personal credentials are included. `scripts/era5_hourly_download.py` reads the CDS token from the environment variable `CDS_PRIVATE_TOKEN`; `cdsapi.Client()` otherwise uses the local `~/.cdsapirc`.

## 3. Reproduction

Environment used for the frozen results: Python 3.13.14; NumPy 2.5.2; pandas 2.3.3; scikit-learn 1.9.0; XGBoost 3.4.1; matplotlib 3.11.2.

All scripts were developed with hard-coded base directories reflecting the original workstation. Before running, replace the base paths in the script headers with your local paths:

- Upstream data scripts (`data assembly` group): replace `E:/workbuddyoutput/2026-10-16-11-07-31` with the directory containing this repository.
- Project analysis scripts (`analysis` group): replace both `E:/workbuddyoutput/2026-10-16-11-07-31` and `E:/workbuddyoutput/2026-11-10-10-24-23` accordingly.

Suggested execution order:

```bash
# 1) data assembly (requires third-party raw products)
python scripts/build_labels.py
python scripts/build_static_features.py
python scripts/road_distance.py
python scripts/settlement_distance.py
python scripts/ndvi_download.py
python scripts/ndvi_extract.py
python scripts/ndvi_impute.py
python scripts/era5_hourly_download.py
python scripts/era5_to_grid_daily.py
python scripts/assemble_dataset.py

# 2) modelling, calibration, and evaluation
python scripts/train_models.py
python scripts/application_analysis.py
python scripts/check_baselines.py
python scripts/spatial_baseline_diagnostics.py

# 3) cost-loss decision analysis
python scripts/cost_loss_analysis.py
python scripts/cost_loss_prior_sensitivity.py

# 4) figures
python scripts/draw_article_figures_zh.py
```

With the processed datasets supplied here, step 1 can be skipped and step 2 onward can be run directly. The frozen numbers in `results/` and `analysis/` are the reference values; small differences may arise from library version or platform.

## 4. Key design points

- Prediction unit: `0.1° forest grid × characteristic dekad` → presence of a satellite-detected forest hotspot in the **next** dekad.
- Temporal split by fire year: train 2010–2018, calibration 2019–2020, test 2021–2023.
- XGBoost was pre-specified as the primary model; random forest and logistic regression are comparators.
- Isotonic calibration was fitted on the independent calibration period and applied unchanged to the test period.
- Five-fold spatial holdout uses 1°×1° block K-means with strict within-fold temporal ordering.
- The cost–loss ratio `alpha = C/L` is an **environmental decision parameter**, not an estimated model threshold. Reported `V_max` values are ex-post upper bounds; `alpha` was not calibrated to real financial costs.
- Top-k capture is a pooled ranking over the entire test period, not a per-dekad fixed-capacity policy.

## 5. Citation

Please cite the accompanying manuscript and this archived release. Citation metadata are provided in `CITATION.cff` (Zenodo reads this file when archiving a GitHub release). The Zenodo DOI will be inserted after the release is archived; until then, cite the repository URL and version tag `v1.0.0`.

## 6. Contact

Corresponding author: Zhi Zhang (24091852@qq.com). Authors: Yujie Fu, Feng Xu. Affiliation: Nanning Fire and Rescue Mobile Detachment.
