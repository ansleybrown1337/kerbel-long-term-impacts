# Machine-learning workflow

## Scientific purpose and unit of analysis

The accepted event-level CatBoost workflow is a conditional reconstruction of the historical record, with LOYO diagnostics and conformal intervals. Concentration is modeled at the analyte-row level. Volume is modeled once per accepted `EventVolumeID`, whose key includes `SampleID` and `MeasureMethod`.

## Inputs

- `out/wq_cleaned.csv`
- predictor definitions in `code/ml_catboost_conformal_loyo_v2_eventlevel.py`

## Entry point and command

Full-model entry points:

```powershell
python code/ml_catboost_conformal_loyo_v2_eventlevel.py
python code/ml_postprocess_plots_v2_eventlevel.py
```

These commands were not run during the current task. They fit/recalibrate models and regenerate Monte Carlo outputs. Runtime depends on CatBoost hardware and calibration mode and remains to be recorded before release.

## Saved-model regeneration

The accepted v2 workflow supports `--impute_only` with compatible saved v2 models. Do not use the older `ml_regenerate_from_saved_models.py` for the v2 event-level volume model.

## Outputs and version

Principal final artifacts are under `out/ml_catboost_conformal_loyo/`: saved `.cbm` models and metadata, `annual_load_summary_imputed.csv`, `wq_cleaned_ml_imputed.csv`, and deposited imputation draws. LOYO predictions, coverage tables, feature importance, and audit tables are diagnostic or interpretive. `predictions_from_saved_models.csv` is byte-identical to `wq_cleaned_ml_imputed.csv` in the audited snapshot and is recommended as an excluded provenance alias; retain `wq_cleaned_ml_imputed.csv` as the canonical release artifact.

## Manuscript support

The workflow supports ML methods, LOYO error/calibration tables, feature importance, annual reconstructions, and the ML inputs to comparison tables/figures. Feature importance is descriptive and noncausal.

See `docs/README_ML_CatBoost_Conformal_LOYO_v2_eventlevel.md` and `docs/manuscript_crosswalk.md`.
