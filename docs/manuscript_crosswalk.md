# Manuscript table and figure crosswalk

This crosswalk identifies validated technical products; it does not draft Results or Discussion text.

| Manuscript use | Current validated product | Workflow | Status |
|---|---|---|---|
| Study-period cumulative loads, Bayes Variant A | `out/bayes_vs_ml_postprocessing_v2p1/master_cumulative_loads_raw_bound_floor_pub.csv` | Comparison | Structurally ready; provisional unit-of-analysis flag |
| Study-period cumulative loads, Bayes Variant B | `out/bayes_vs_ml_postprocessing_v2p1/master_cumulative_loads_annual_truncation_pub.csv` | Comparison | Structurally ready; provisional unit-of-analysis flag |
| Bayesian nonnegative sensitivity | `out/bayes_vs_ml_postprocessing_v2p1/bayes_nonnegative_sensitivity_pub.csv` | Comparison | Ready for author interpretation; neither variant preferred in code |
| CT-relative modeled differences | `out/bayes_vs_ml_postprocessing_v2p1/treatment_differences_vs_ct_raw.csv` | Comparison | Ready with denominator QC; provisional |
| Observed subtotals | `out/bayes_vs_ml_postprocessing_v2p1/observed_study_period_subtotals_raw.csv` | Shared/comparison | Descriptive only; incomplete and unequal coverage |
| Temporal rank agreement | `out/bayes_vs_ml_postprocessing_v2p1/spearman_by_analyte_pub.csv` | Comparison | Ready; exploratory unadjusted tests |
| Annual concentration/volume error | `out/bayes_vs_ml_postprocessing_v2p1/performance_comparison_pub.csv` | Comparison | Annual central-estimate diagnostic; not common sample-level validation |
| ML interval calibration | `out/bayes_vs_ml_postprocessing_v2p1/loyo_interval_calibration_pub.csv` | ML/comparison | Ready from existing LOYO intervals; no recalibration |
| ML feature importance | `out/bayes_vs_ml_postprocessing_v2p1/catboost_feature_importance_pub.csv` | ML/comparison | Ready as descriptive/noncausal interpretation |
| Cumulative-load figure | `figs/bayes_vs_ml_postprocessing_v2p1/study_period_cumulative_loads.png` | Comparison | Provisional |
| CT-relative figure | `figs/bayes_vs_ml_postprocessing_v2p1/treatment_differences_vs_ct.png` | Comparison | Provisional |
| Concentration RMSE figure | `figs/bayes_vs_ml_postprocessing_v2p1/concentration_rmse_original_units.png` | Comparison | Diagnostic |
| Mean-normalized RMSE figure, including Volume | `figs/bayes_vs_ml_postprocessing_v2p1/nrmse_mean_comparison.png` | Comparison | Dimensionless cross-response diagnostic |
| LOYO coverage figure | `figs/bayes_vs_ml_postprocessing_v2p1/loyo_interval_coverage_by_year.png` | ML/comparison | Ready |
| Concentration feature importance | `figs/bayes_vs_ml_postprocessing_v2p1/feature_importance_concentration.png` | ML/comparison | Descriptive/noncausal |
| Volume feature importance | `figs/bayes_vs_ml_postprocessing_v2p1/feature_importance_volume.png` | ML/comparison | Descriptive/noncausal |

Definitions, units, interval notation, formulae, observed-subtotal warnings, and sensitivity rules are in `comparison_table_footnotes.md` and `postprocessing_data_dictionary.csv` beside the tables.
