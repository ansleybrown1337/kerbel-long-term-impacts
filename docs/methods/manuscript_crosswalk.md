# Manuscript table and figure crosswalk

This crosswalk describes the corrected v3p0 products that will be created only
after the Bayesian, ML, and comparison workflows are run successfully. Legacy
v2p1 products are not valid substitutes for these physical-event outputs.

| Manuscript use | Corrected v3p0 product | Workflow | Interpretation |
|---|---|---|---|
| Annual modeled loads | `results/comparison/v3p0_physical_event/annual_load_summary_publication.csv` | Comparison | One physical-event/analyte/draw ledger; complete 2011-2025 coverage required |
| Annual loads with complete observed references | `results/comparison/v3p0_physical_event/annual_load_complete_observed_publication.csv` | Comparison | Observed annual values are shown only when every expected physical event has an observed event load; incomplete-year subtotals remain NA |
| Observed annual-load completeness audit | `results/comparison/v3p0_physical_event/observed_annual_load_completeness_publication.csv` | Comparison | Documents expected and observed event counts by year, analyte, and treatment |
| Annual runoff volume with complete observed references | `results/comparison/v3p0_physical_event/annual_runoff_volume_complete_observed_publication.csv` | Comparison | Observed annual volume is shown only when every expected physical event has an observed runoff-volume observation |
| Observed annual-volume completeness audit | `results/comparison/v3p0_physical_event/observed_annual_volume_completeness_publication.csv` | Comparison | Documents expected and observed physical-event volume counts by year and treatment |
| Study-period cumulative loads | `results/comparison/v3p0_physical_event/cumulative_load_2011_2025_publication.csv` | Comparison | Bayes model-only and ML full-record scenarios are primary |
| Bayesian nonnegative sensitivity | `results/comparison/v3p0_physical_event/bayes_negative_draw_sensitivity_publication.csv` | Comparison | Raw annual draws with display-only lower-bound floor versus annual-draw truncation at zero |
| CT-relative modeled differences | `results/comparison/v3p0_physical_event/ct_relative_summary_raw.csv` | Comparison | Computed within draw; positive means lower load than CT |
| Observed subtotals | `results/comparison/v3p0_physical_event/observed_subtotals_raw.csv` | Comparison | Descriptive audit only; incomplete annual subtotals are not plotted or reported as observed annual loads |
| Temporal rank agreement | `results/comparison/v3p0_physical_event/temporal_spearman_publication.csv` | Comparison | Includes paired-year n and significance marker; raw table retains rho and p |
| Concentration and volume error | `results/comparison/v3p0_physical_event/performance_and_calibration_publication.csv` | Comparison | Concentration by analyte/treatment; volume overall and by treatment |
| ML LOYO interval calibration | `results/comparison/v3p0_physical_event/loyo_interval_coverage_by_year_target_publication.csv` | ML/comparison | Missing observed-volume years remain explicit as n = 0 and NA coverage |
| ML feature importance | `results/comparison/v3p0_physical_event/feature_importance_descriptive_noncausal_publication.csv` | ML/comparison | Descriptive CatBoost importance; not causal evidence |
| Cross-model review ledger | `results/comparison/v3p0_physical_event/cross_model_observation_disagreement.csv` | Comparison | Review aid only; disagreement is not an automatic error label |
| Annual-load figures | `figures/comparison/v3p0_physical_event/annual_complete_observed/annual_load_*_complete_observed_v3p0.png` | Comparison | Corrected physical-event model summaries with complete observed annual markers only |
| Annual-volume figure | `figures/comparison/v3p0_physical_event/annual_complete_observed/annual_runoff_volume_complete_observed_v3p0.png` | Comparison | Corrected physical-event model summaries with complete observed annual-volume markers only |
| Cumulative-load figure | `figures/comparison/v3p0_physical_event/cumulative_loads_2011_2025.png` | Comparison | Corrected physical-event cumulative summaries |

Definitions and unit rules are in `data_unit_dictionary_v3p0.md` and
`physical_event_methods_change_v3p0.md`.
