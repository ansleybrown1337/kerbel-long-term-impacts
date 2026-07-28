# Manuscript table and figure crosswalk

This crosswalk describes the corrected v3p1 products that will be created only
after the Bayesian, ML, and comparison workflows are run successfully. Legacy
v2p1 products are not valid substitutes for these physical-event outputs.

| Manuscript use | Corrected v3p1 product | Workflow | Interpretation |
|---|---|---|---|
| Annual modeled loads | `results/comparison/v3p1_physical_event/annual_load_summary_publication.csv` | Comparison | Mean per treatment plot for primary TSS, TP, and TN; all-analyte raw export retained |
| Annual loads with complete observed references | `results/comparison/v3p1_physical_event/annual_load_complete_observed_publication.csv` | Comparison | Mean of complete replicate plot totals, or the single available complete plot; no incomplete subtotal enters the point |
| Observed annual-load completeness audit | `results/comparison/v3p1_physical_event/observed_annual_load_completeness_publication.csv` | Comparison | Plot-level expected/observed counts, missing Irrigation labels, completeness, totals, SD/SE, and descriptive ranges |
| Annual runoff volume with complete observed references | `results/comparison/v3p1_physical_event/annual_runoff_volume_complete_observed_publication.csv` | Comparison | Mean per treatment plot with the same replicate-completeness rule |
| Observed annual-volume completeness audit | `results/comparison/v3p1_physical_event/observed_annual_volume_completeness_publication.csv` | Comparison | Plot-level volume completeness and descriptive replicate summaries |
| Study-period cumulative loads | `results/comparison/v3p1_physical_event/cumulative_load_2011_2025_publication.csv` | Comparison | Cumulative mean-per-plot Bayes model-only and ML full-record scenarios |
| Bayesian nonnegative sensitivity | `results/comparison/v3p1_physical_event/bayes_negative_draw_sensitivity_publication.csv` | Comparison | Raw annual draws with display-only lower-bound floor versus annual-draw truncation at zero |
| CT-relative modeled differences | `results/comparison/v3p1_physical_event/ct_relative_summary_raw.csv` | Comparison | Computed within draw; positive means lower load than CT |
| Observed subtotals | `results/comparison/v3p1_physical_event/observed_subtotals_raw.csv` | Comparison | Descriptive audit only; incomplete replicate-plot subtotals are not plotted or reported as observed annual loads |
| Temporal rank agreement | `results/comparison/v3p1_physical_event/temporal_spearman_publication.csv` | Comparison | Includes paired-year n and significance marker; raw table retains rho and p |
| Concentration and volume error | `results/comparison/v3p1_physical_event/performance_and_calibration_publication.csv` | Comparison | Concentration by analyte/treatment; volume overall and by treatment |
| ML LOYO interval calibration | `results/comparison/v3p1_physical_event/loyo_interval_coverage_by_year_target_publication.csv` | ML/comparison | Missing observed-volume years remain explicit as n = 0 and NA coverage |
| ML feature importance | `results/comparison/v3p1_physical_event/feature_importance_descriptive_noncausal_publication.csv` | ML/comparison | Descriptive CatBoost importance; not causal evidence |
| Cross-model review ledger | `results/comparison/v3p1_physical_event/cross_model_observation_disagreement.csv` | Comparison | Review aid only; disagreement is not an automatic error label |
| Annual-load figures | `figures/comparison/v3p1_physical_event/annual_complete_observed/annual_load_*_complete_observed_v3p1.png` | Comparison | Replicate ranges for n=2; x marker and no interval for n=1; no point for n=0 |
| Annual-volume figure | `figures/comparison/v3p1_physical_event/annual_complete_observed/annual_runoff_volume_complete_observed_v3p1.png` | Comparison | Same replicate-completeness and marker convention for runoff volume |
| Cumulative-load figure | `figures/comparison/v3p1_physical_event/cumulative_loads_2011_2025.png` | Comparison | Corrected physical-event cumulative summaries |

Definitions and unit rules are in `data_unit_dictionary_v3p1.md` and
`physical_event_methods_change_v3p1.md`.
