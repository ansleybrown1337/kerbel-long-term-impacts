# Bayesian-versus-ML comparison workflow

## Scientific purpose and unit of analysis

This workflow compares existing saved annual/posterior/Monte Carlo outputs. It computes annual agreement metrics, temporal Spearman agreement, 2011–2025 cumulative loads, draw-level MT/ST differences relative to CT, Bayesian nonnegative-load sensitivities, calibration/feature-importance summaries, and publication tables. It never fits or recalibrates either model.

The modeled analytical unit is the accepted upstream annual `Year × Analyte × Treatment × Draw` total. The validation report separately flags the unresolved physical-event interpretation inherited from upstream event keys.

## Required inputs

- `out/annual_load_draws_bayes_v2p1.csv`
- `out/annual_load_summary_bayes_v2p1.csv`
- `out/annual_load_summary_bayes_plus_observed_v2p1.csv`
- `out/ml_catboost_conformal_loyo/wq_cleaned_ml_imputed.csv`
- `out/ml_catboost_conformal_loyo/imputed_row_draws.csv`
- saved ML LOYO, calibration, and feature-importance tables
- existing `out/bayes_vs_ml_metrics_v2p1/` metrics and Spearman products

Exact paths, sizes, modification times, SHA-256 checksums, schemas, units, aliases, keys, missingness, and negative counts are recorded in `out/bayes_vs_ml_postprocessing_v2p1/input_audit.csv`.

## Entry point and command

```powershell
python code/bayes_ml_postprocessing_v2p1.py --repo . --rebuild-current-run
python -m unittest discover -s tests -p "test_bayes_ml_postprocessing_v2p1.py" -v
```

The validated run took about 12 seconds on a desktop with Python 3.12.4. It streams the 439 MB deposited ML imputation-draw file and does not generate random values.

## Outputs and version

Version: `bayes_vs_ml_postprocessing_v2p1`.

- Raw scientific tables: final calculations, with provisional QC fields.
- Publication tables: manuscript-review formatting derived from the raw tables.
- Validation report/checks/input audit/data dictionary: required release provenance.
- Figures: comparison-only visual products in the versioned comparison figure directory.

Both Bayesian sensitivity variants are retained. No preferred interpretation is selected in code or filenames.

## Manuscript support and status

See `docs/manuscript_crosswalk.md`. The tables are structurally ready for manuscript review but numerically provisional pending resolution of the upstream physical-event unit. The post-processing does not draft or imply a manuscript Results/Discussion interpretation.
