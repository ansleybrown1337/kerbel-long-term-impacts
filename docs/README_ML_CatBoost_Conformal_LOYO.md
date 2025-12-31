
# ML Prediction Pipeline: CatBoost + Conformal Uncertainty + LOYO CV

## Table of contents
1. How to run
2. Purpose and scope
3. Data inputs
4. Modeling approach
5. Outputs
6. Imputation and impute draws (NEW, DEFAULT)
7. Post-processing plots
8. Observed annual load uncertainty
9. Performance and runtime notes
10. Recommended manuscript language
11. Troubleshooting

---

## 1. How to run

### 1.1 Default behavior (recommended)

With **no flags**, the pipeline now performs the full workflow:

- Leave-One-Year-Out (LOYO) cross-validation  
- Refit on all observed data  
- **Impute missing concentration and volume values**  
- **Generate 2000 imputation draws per missing row**  
- **Save fitted models and conformal metadata for reuse**  

```bash
conda activate wq_ml
cd C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts
python code/ml_catboost_conformal_loyo.py
```

This is the recommended and documented default.

---

### 1.2 Optional overrides

Disable imputation entirely (LOYO only):

```bash
python code/ml_catboost_conformal_loyo.py --no-impute-missing
```

Run LOYO and impute, but disable imputation draws:

```bash
python code/ml_catboost_conformal_loyo.py --impute-draws 0
```

Change miscoverage rate or annual-load Monte Carlo draws:

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
- Completed LOYO folds are written incrementally to `out/ml_catboost_conformal_loyo/`.

---

### 1.3 Impute only (no recalibration, no LOYO rerun)

If LOYO has already been run and fitted models were saved, you may **skip LOYO entirely** and only impute missing rows:

```bash
python code/ml_catboost_conformal_loyo.py --impute_only
```

This mode:
- loads the saved CatBoost models,
- loads the saved conformal calibration values,
- imputes missing rows with prediction intervals,
- optionally generates imputation draws,
- **does not** rerun LOYO cross-validation.

Optional Monte Carlo draws for imputed rows:

```bash
python code/ml_catboost_conformal_loyo.py --impute_only --impute_draws 2000
```

---


### 1.4 Regenerate annual summaries from saved models (NO LOYO, NO recalibration)

Use this when you have already run the full pipeline once (so the trained models and conformal metadata exist in
`out/ml_catboost_conformal_loyo/models/`) and you want to **regenerate predictions and overwrite the annual summaries**
without rerunning LOYO CV or recalibrating conformal intervals.

This mode:
- loads the saved CatBoost models (`*.cbm`),
- loads the saved conformal quantiles (`*_meta.json`),
- regenerates row-level predictions on the modeling dataset(s),
- propagates uncertainty to annual totals via Monte Carlo sampling,
- **overwrites** the existing summary files with the **Bayes-aligned column schema** (including volume, concentration, and load):

Outputs overwritten (no new “extra” CSVs are created):
- `out/ml_catboost_conformal_loyo/annual_load_summary.csv`
- `out/ml_catboost_conformal_loyo/annual_load_summary_imputed.csv`

Command (Anaconda Prompt / Windows cmd.exe):

```bash
conda activate wq_ml
cd C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts
python code\ml_regenerate_from_saved_models.py --repo . --overwrite --alpha 0.05 --draws 2000 --seed 123
```

Notes:
- This is a **deployment/regeneration** step only. It does **not** run LOYO CV and does **not** compute new conformal quantiles.
- `--alpha 0.05` corresponds to `interval_prob = 0.95` in the annual summaries.


## 2. Purpose and scope

This pipeline provides a **prediction-focused ML benchmark** parallel to the Bayesian workflow in this repository.

Goals:
- benchmark predictive skill against Bayesian models,
- identify nonlinear or high-dimensional interactions,
- generate uncertainty-aware annual loads for comparison.

This ML workflow is **not causal** and does not replace Bayesian generative or causal inference.

---

## 3. Data inputs

### Required input file
- `out/wq_with_stir_by_season.csv`

### Required columns
- `Year` or `Date`
- `Treatment` (CT / MT / ST)
- `Analyte`
- `Result_mg_L` (outflow concentration)
- `Volume` (runoff volume, L)

Rows with `NoRunoff == TRUE` are excluded by default.

### Optional columns used automatically (if present)
- STIR metrics
- Replication identifiers
- Inflow concentration and volume
- Measurement and laboratory metadata
- QA/QC flags
- Crop and season context

---

## 4. Modeling approach

Two models are trained independently:

1. Concentration model: `log1p(Result_mg_L)`
2. Volume model: `log1p(Volume)`

Estimator:
- CatBoost regression with native categorical handling

Uncertainty:
- Split conformal prediction intervals

Validation:
- Leave-One-Year-Out cross-validation across the monitoring record

Annual load propagation:
- Event load = `C × V`
- Sampling within prediction intervals (log scale)
- Back-transform and aggregate to annual loads by Year × Treatment × Analyte

---

## 5. Outputs

Written to: `out/ml_catboost_conformal_loyo/`

### Core evaluation outputs

| File | Description |
|---|---|
| `cv_metrics_by_year.csv` | Fold-level MAE, RMSE, R² |
| `cv_predictions_samplelevel.csv` | Sample-level LOYO predictions with PI bounds |
| `annual_load_draws.csv` | Monte Carlo annual load draws |
| `annual_load_summary.csv` | Annual summary table (Bayes-aligned schema; includes volume_*, conc_*, load_*) |

### Saved model artifacts

| File | Description |
|---|---|
| `models/model_logC.cbm` | CatBoost model for log-concentration |
| `models/model_logV.cbm` | CatBoost model for log-volume |
| `models/model_logC_meta.json` | Feature list, categorical cols, conformal q |
| `models/model_logV_meta.json` | Feature list, categorical cols, conformal q |

---

## 6. Imputation and impute draws (DEFAULT)

### 6.1 Imputation (`--impute_missing`, default: TRUE)

By default, after LOYO CV the pipeline:

- refits the CatBoost models on **all observed data**,
- applies the **same conformal calibration** used during LOYO,
- imputes missing `Result_mg_L` and/or `Volume` values,
- writes:

```text
wq_cleaned_ml_imputed.csv
```

Key point:
**No recalibration or retuning occurs.**  
This is a deployment step using the already-validated ML model.

### 6.2 Imputation draws (`--impute_draws`, default: 2000)

Imputation draws control **how ML uncertainty is propagated**, not calibration.

- Imputation always produces point predictions + prediction intervals
- Imputation draws generate **multiple realizations within those intervals**

These draws:
- are sampled uniformly within the conformal interval (log space),
- are back-transformed to original units,
- are written to:

```text
imputed_row_draws.csv
```

Use cases:
- propagate imputation uncertainty into annual load aggregation,
- mirror Bayesian posterior predictive draws,
- compare variability (not just interval width) between Bayes and ML.

Setting `--impute-draws 0` disables draw generation while retaining intervals.

---

## 7. Post-processing plots

Generated by:

```bash
python code/ml_postprocess_plots.py
```

If you regenerated summaries using `ml_regenerate_from_saved_models.py`, you do **not** need to rerun LOYO to update the summary tables; you may rerun this plotting script at any time.

Figures written to `figs/ml_catboost_conformal_loyo/` include:
- CV RMSE by year
- Observed vs ML annual loads
- Parity plots
- Prediction interval coverage diagnostics

---

## 8. Observed annual load uncertainty

Observed annual loads use nonparametric bootstrap intervals:

- resample events within each Year × Treatment × Analyte,
- recompute annual loads,
- report central (1 − alpha) interval.

These intervals represent **sampling variability**, not model uncertainty.

---

## 9. Performance and runtime notes

- CatBoost uses internal multithreading.
- LOYO folds run sequentially to avoid oversubscription.
- Runtime scales with years, iterations, and draw count.
- `--fast` is recommended for workflow testing only.

---

## 10. Recommended manuscript language

Suggested phrasing:

> “Machine learning models were used as a prediction-focused benchmark. Concentration and runoff volume were modeled using CatBoost regression with split conformal prediction intervals. Temporal generalization was assessed using leave-one-year-out cross-validation. Missing observations were imputed using the fitted models without recalibration, and uncertainty was propagated to annual loads via Monte Carlo sampling.”

Explicitly distinguish:
- Bayesian posterior predictive uncertainty
- ML conformal prediction intervals

---

## 11. Troubleshooting

Specify repo root manually:

```bash
python code/ml_catboost_conformal_loyo.py --repo "C:/path/to/kerbel-long-term-impacts"
```

Slow run:

```bash
python code/ml_catboost_conformal_loyo.py --fast
```

Need diagnostics:

```bash
python code/ml_catboost_conformal_loyo.py --cb_verbose_every 200
```
