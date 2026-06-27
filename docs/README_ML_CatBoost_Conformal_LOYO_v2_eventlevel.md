# Kerbel CatBoost conformal LOYO v2: event-level volume model

## Purpose

This workflow is a site-specific, prediction-oriented conditional reconstruction of the 2011-2025 Kerbel monitoring record. It is intended to complete and summarize the empirical record where causal prediction is weaker, particularly for soluble analytes and runoff volume. It is not a transferable forecasting model and does not replace the Bayesian causal and mechanistic analysis.

The central v2 correction is the physical unit used for runoff volume:

- Concentration remains at the analyte-row level because each concentration belongs to a specific analyte.
- Runoff volume is fitted once per physical measurement event, rather than once for every analyte row that repeats the same volume.

## Default input and compatibility contract

The exact default input is:

```text
out/wq_cleaned.csv
```

The new scripts retain the original scripts and overwrite the existing ML output locations by default:

```text
out/ml_catboost_conformal_loyo/
figs/ml_catboost_conformal_loyo/
```

This is intentional. It preserves the paths and standard filenames used by `code/annual_load_bayes_vs_ml.py`. Use `--output_dir` and `--fig_dir` when a separate run is needed.

## Models and predictors

### Concentration (`logC`)

The target is `log1p(Result_mg_L)`. Rows with `NoRunoff == TRUE` are not used to train or evaluate concentration because they do not represent water leaving the field. In reconstruction mode, measured same-event `Volume` is an allowed predictor. The model never uses `Result_mg_L`, `Result_mg_L_cens`, `cout_z`, `Result_is_nd`, or another transformation/status field derived from the concentration target as a predictor.

The available monitoring, management, timing, inflow, analytical-method, crop, STIR, residue, and detection-limit fields are selected defensively. `previous_crop` is categorical and complete in the current input. It represents crop-history information that measured residue cover alone may not capture, including residue architecture and decomposition, soil structure, infiltration opportunity, nutrient carryover, and sediment availability.

`residue_prop` is the primary residue-cover feature. `Residue_PercentCover` is not included at the same time because it is a duplicate encoding. `Residue_DryMass_kg_m2` is retained as a secondary feature when its missing fraction does not exceed `--dry_mass_missing_threshold` (default 0.60). `residue_obs` and `Residue_n` are audit/quality fields rather than primary predictors.

### Volume (`logV`)

The target is `log1p(Volume)` and the training table contains one row per event. The preferred event key is:

```text
Date + Year + Irrigation + Rep + Treatment + SampleID + MeasureMethod
```

Grouping retains missing key values, including missing `SampleID`. If a future input lacks one of these columns, the script warns and records the degraded key. Conflicting nonmissing volumes are audited before a median resolution is used.

The volume model uses event-level management, timing, inflow, crop, STIR, residue, measurement, and irrigation features. It does not use analyte identity, lab identity, `Volume`, `volume_z`, load, or any volume-derived predictor.

In reconstruction mode, observed same-event concentrations are summarized into event-level features. Duplicate measurements of an analyte within an event are reduced with the median `log1p(Result_mg_L)`. Features include the event-wide mean, maximum, observed count, and analyte-specific fields such as `event_logC_TSS`, `event_logC_TP`, and `event_logC_NO3` when those analytes are present. Missing analytes remain missing for CatBoost to handle.

Zero-volume `NoRunoff == TRUE` events remain valid volume targets in the single `log1p(Volume)` model. Their reconstructed load is explicitly set to zero regardless of concentration.

### Irrigation infrastructure

The current `wq_cleaned.csv` has complete `IrrMethod` values distinguishing `Siphon` from `Gated Pipe`; `IrrMethod` is the primary model predictor. A cleaned audit field is derived from it. If `IrrMethod` is absent in a future input, the fallback management-era proxy is:

- year <= 2022: `Siphon`
- year >= 2023: `Gated Pipe`

The audit records which source was used.

## Cross-validation and uncertainty

Leave-one-year-out cross-validation is performed independently at the correct unit for each target:

- held-out analyte rows for `logC`;
- held-out unique physical events for `logV`.

For annual held-out loads, event-level volume predictions are mapped back to analyte rows through `EventVolumeID`. Concentration and volume are drawn in log space from their conformal intervals, back-transformed, multiplied, and summed by `Year x Treatment x Analyte`. A single volume-draw vector is shared by analyte rows from the same event.

The default `--calibration random` retains the original random split-conformal method. `--calibration grouped-year` estimates calibration residuals by internally holding out whole training years, then fits the fold model on all available training years. Grouped-year calibration is more aligned with LOYO evaluation but is substantially more expensive. Coverage by held-out year is always written and plotted.

Metrics from reconstruction mode describe conditional reconstruction performance. They should not be described as forward-forecasting performance because same-event co-outcomes are available.

## Reconstruction and strict-prediction modes

The default mode is:

```text
--mode reconstruction
```

It permits observed outflow volume in the concentration model and observed event concentration summaries in the volume model.

The sensitivity mode:

```text
--mode strict_prediction
```

removes both sets of current-event response predictors. Strict mode requires an explicit `--output_dir` or `--out_subdir`, preventing accidental replacement of the primary reconstruction results. Its default figure folder is also mode-specific unless `--fig_dir` is supplied.

## Run commands

Activate the project environment and run:

```bash
conda activate wq_ml
python code/ml_catboost_conformal_loyo_v2_eventlevel.py
python code/ml_postprocess_plots_v2_eventlevel.py
```

A small workflow test can use:

```bash
python code/ml_catboost_conformal_loyo_v2_eventlevel.py --fast --impute_draws 50
```

Year-blocked conformal calibration:

```bash
python code/ml_catboost_conformal_loyo_v2_eventlevel.py --calibration grouped-year
```

Strict-prediction sensitivity run without overwriting the primary results:

```bash
python code/ml_catboost_conformal_loyo_v2_eventlevel.py \
  --mode strict_prediction \
  --output_dir out/ml_catboost_conformal_loyo_strict_prediction \
  --fig_dir figs/ml_catboost_conformal_loyo_strict_prediction
```

## Outputs

Core compatible outputs:

- `cv_metrics_by_year.csv`
- `cv_predictions_samplelevel.csv`
- `annual_load_draws.csv`
- `annual_load_summary.csv`
- `annual_load_summary_imputed.csv`
- `feature_importance_logC.csv`
- `feature_importance_logV.csv`
- `wq_cleaned_ml_imputed.csv`
- `imputed_row_draws.csv`
- `predictions_from_saved_models.csv`
- `models/model_logC.cbm` and `model_logC_meta.json`
- `models/model_logV.cbm` and `model_logV_meta.json`

New audit/diagnostic outputs:

- `event_volume_training_table.csv`
- `event_volume_audit_summary.csv`
- `event_volume_audit_conflicts.csv`
- `event_volume_predictions.csv`
- `feature_audit_summary.csv`
- `cv_interval_coverage_by_year.csv`

The model metadata records the unit of analysis, feature list, categorical fields, mode, calibration method, conformal quantile, and event key. The v2 script's `--impute_only` option can regenerate compatible row and imputation outputs from these v2 saved models. The older `ml_regenerate_from_saved_models.py` remains available for older row-level model artifacts and should not be used with the v2 event-level `logV` model.

Post-processing preserves the established figure names where applicable and adds `cv_r2_by_year`, `event_volume_unit_audit`, and `event_volume_rows_per_event` figures. Annual plots retain separate LOYO-supported and `_imputed` versions.

## Pre-Chapter 3 review checklist

Before using a run in Chapter 3, verify:

1. `event_volume_audit_conflicts.csv` is empty or every conflict is scientifically resolved.
2. `event_volume_audit_summary.csv` shows the expected analyte-row to event-row collapse and retains missing-`SampleID` groups.
3. `feature_audit_summary.csv` confirms `residue_prop`, `previous_crop`, and `IrrMethod` were selected, with no target-derived leakage fields.
4. `cv_metrics_by_year.csv` and the RMSE/R-squared figures do not show isolated failure years.
5. Concentration and event-volume interval coverage are reviewed by held-out year, not only overall.
6. `event_volume_unit_audit` and the event training table confirm `logV` was fitted on unique events.
7. Imputed annual load intervals are plausible for soluble analytes and years with substantial missingness.
8. `annual_load_summary_imputed.csv` and `annual_load_draws.csv` remain readable by `code/annual_load_bayes_vs_ml.py` before regenerating Chapter 3 comparison figures.
