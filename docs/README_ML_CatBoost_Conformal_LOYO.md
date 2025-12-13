# ML Prediction Pipeline: CatBoost + Conformal Uncertainty + LOYO CV

## Table of contents
1. How to run
2. Purpose and scope
3. Data inputs
4. Modeling approach
5. Outputs
6. Post-processing plots
7. Observed annual load uncertainty
8. Performance and runtime notes
9. Recommended manuscript language
10. Troubleshooting

---

## 1. How to run

### 1.1 Train and generate ML outputs

Activate the ML environment:

```bash
conda activate wq_ml
```

Run the LOYO training pipeline (from repo root or from `code/`):

```bash
python code/ml_catboost_conformal_loyo.py
```

Common optional flags:

```bash
python code/ml_catboost_conformal_loyo.py --alpha 0.05 --draws 2000
```

Show CatBoost training progress every N iterations:

```bash
python code/ml_catboost_conformal_loyo.py --cb_verbose_every 200
```

Fast debug run (reduced iterations/draws):

```bash
python code/ml_catboost_conformal_loyo.py --fast
```

Interrupting safely:
- Press **Ctrl + C** in Anaconda Prompt.
- Checkpointed results from completed folds remain in `out/ml_catboost_conformal_loyo/`.

### 1.2 Create publication-style plots (post-processing)

After the model run completes, generate the core figures from the saved CSV outputs:

```bash
python code/ml_postprocess_plots.py
```

Optional arguments:
- Convert annual loads to different units (default is grams):

```bash
python code/ml_postprocess_plots.py --units g
python code/ml_postprocess_plots.py --units mg
python code/ml_postprocess_plots.py --units kg
```

- Plot only a subset of analytes:

```bash
python code/ml_postprocess_plots.py --analytes TP,TSS,NO3
```

- Control observed-load bootstrap uncertainty (default B=2000, 95% interval):

```bash
python code/ml_postprocess_plots.py --obs_boot 5000 --obs_alpha 0.05
```

---

## 2. Purpose and scope

This pipeline provides a **prediction-focused ML benchmark** parallel to the Bayesian workflow in this repository.

Goals:
- benchmark predictive skill against Bayesian models,
- identify nonlinear or high-dimensional interactions,
- generate uncertainty-aware annual loads for comparison.

This ML workflow is **not causal** and does not replace the Bayesian do-calculus analysis.

---

## 3. Data inputs

### Required input file
- `out/wq_with_stir_by_season.csv`

### Required columns
- `Year` or `Date`
- `Treatment` or `System` (CT/MT/ST)
- `Analyte`
- `Result_mg_L` (outflow concentration)
- `Volume` (runoff volume, L)

Rows with `NoRunoff == TRUE` are excluded by default (if present).

### Optional columns used automatically (if present)
- STIR: `CumAll_STIR_toDate`, `Season_STIR_toDate`
- Replication: `Irrigation`, `Rep`
- Inflow context: `Inflow_Result_mg_L`
- Methods metadata: `FlumeMethod`, `MeasureMethod`, `IrrMethod`, `TSSMethod`, `Lab`
- QA/QC flags: `Flag`, `Inflow_Flag`
- Crop context: `Crop`, `SeasonYear`, `PlantDate`, `HarvestDate`

---

## 4. Modeling approach

Two models are trained separately:

1) Concentration model: `log1p(Result_mg_L)`
2) Volume model: `log1p(Volume)`

Estimator:
- CatBoost regression (categorical handling, strong tabular performance)

Uncertainty:
- Split conformal prediction intervals (MAPIE v1)

Validation:
- Leave-One-Year-Out cross-validation across the full monitoring record

Annual load propagation:
- Compute sample load as `C × V`
- Sample within prediction intervals in log space
- Back-transform and aggregate to annual loads by Year × Treatment × Analyte

---

## 5. Outputs

### 5.1 Data outputs
Written to: `out/ml_catboost_conformal_loyo/`

| File | Description |
|---|---|
| `cv_metrics_by_year.csv` | Fold-level MAE, RMSE, R² for `logC` and `logV` |
| `cv_predictions_samplelevel.csv` | Sample-level predictions with PI bounds (original units) |
| `annual_load_draws.csv` | Monte Carlo draws of annual loads (mg) |
| `annual_load_summary.csv` | Summary statistics for annual loads (mean, median, PI) |

### 5.2 Figure outputs during training
Written to: `figs/ml_catboost_conformal_loyo/`

| Figure | Description |
|---|---|
| `cv_rmse_by_year.png` | RMSE by held-out year (logC and logV) |

---

## 6. Post-processing plots

Created by `python code/ml_postprocess_plots.py` and written to `figs/ml_catboost_conformal_loyo/`:

| Figure | Description |
|---|---|
| `cv_rmse_by_year.jpg` | CV RMSE plot saved as JPG |
| `annual_load_<analyte>_obs_vs_ml.jpg` | Observed vs ML annual load by treatment with uncertainty intervals |
| `parity_logC.jpg` | 1:1 parity plot for concentration (log scale) |
| `parity_logV.jpg` | 1:1 parity plot for volume (log scale) |
| `pi_coverage_overall.jpg` | Empirical PI coverage by target |
| `pi_coverage_by_year.jpg` | Empirical PI coverage by year |

---

## 7. Observed annual load uncertainty

Observed annual loads are computed from measured event pairs:

- Event load (mg) = `Result_mg_L × Volume_L`
- Annual load (mg) = sum of event loads within Year × Treatment × Analyte

To reflect within-year variability and heterogeneity, the post-processing script adds **nonparametric bootstrap intervals**:

- Resample events with replacement within each Year × Treatment × Analyte
- Recompute annual load for each bootstrap replicate
- Report the central (1 − alpha) interval (default: 95%)

These bootstrap intervals are plotted as error bars on the observed annual load points.

---

## 8. Performance and runtime notes

- CatBoost uses multi-threading internally by default.
- LOYO folds are run sequentially to avoid thread oversubscription.
- Runtime scales with the number of years, boosting iterations, and Monte Carlo draws.
- Use `--fast` for quick validation of the full workflow.

---

## 9. Recommended manuscript language

Suggested phrasing:

> “Machine learning models were used as a prediction-focused benchmark. Concentration and runoff volume were modeled using CatBoost regression with split conformal prediction intervals, and uncertainty was propagated to annual loads via Monte Carlo sampling. Temporal generalization was evaluated using leave-one-year-out cross-validation. Observed annual loads were summarized with nonparametric bootstrap intervals.”

When comparing to Bayesian results, explicitly distinguish:
- Bayesian posterior credible intervals (posterior predictive uncertainty)
- ML conformal prediction intervals (predictive coverage intervals)

---

## 10. Troubleshooting

Repo root detection:
```bash
python code/ml_postprocess_plots.py --repo "C:/path/to/kerbel-long-term-impacts"
python code/ml_catboost_conformal_loyo.py --repo "C:/path/to/kerbel-long-term-impacts"
```

Data file not found:
```bash
python code/ml_catboost_conformal_loyo.py --data out/wq_with_stir_by_season.csv
```

Slow run:
```bash
python code/ml_catboost_conformal_loyo.py --fast
```

Need training diagnostics:
```bash
python code/ml_catboost_conformal_loyo.py --cb_verbose_every 200
```
