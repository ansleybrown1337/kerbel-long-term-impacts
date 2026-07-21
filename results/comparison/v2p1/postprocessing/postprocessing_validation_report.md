# Post-processing validation report

## Outcome

The saved Bayesian and ML artifacts were post-processed without refitting, recompiling, recalibrating, or generating new model predictions. The requested cumulative-load, CT-relative, sensitivity, performance, calibration, feature-importance, publication-table, figure, audit, and dictionary outputs are in `C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\out\bayes_vs_ml_postprocessing_v2p1` and `C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\figs\bayes_vs_ml_postprocessing_v2p1`.

All cumulative and CT-relative results are **provisional** because the limited read-only audit could not establish an unambiguous physical runoff-event unit across first-flush, outflow, and duplicate SampleIDs. No upstream code or saved model output was changed.

## Key QC findings

- Bayesian saved annual draws below zero: 19,572 of 180,000 (10.87%).
- ML reconstructed annual draws below zero: 0; no parallel ML transformation was imposed.
- Variant A preserves raw cumulative draws and floors only a negative publication lower bound. Variant B truncates each annual draw at zero before summation.
- Invalid/unstable CT-denominator groups: 2; exact draw counts are in `treatment_differences_vs_ct_raw.csv`.
- Saved ML LOYO annual summaries reconcile to the saved LOYO annual draws to floating-point precision. Bayesian summaries do not reconcile exactly because the saved annual draw file contains only 400 posterior draws while the saved summary used the fuller posterior calculation; the audit records the discrepancy.
- The saved full-record ML annual summary and the deposited ML row draws use different upstream random-number streams (`seed+80001` and `seed+70001`). Their independent Monte Carlo reconciliation is quantified rather than presented as an exact identity.
- The saved ML LOYO annual draw file is intentionally incomplete for full-period use. Full-period ML cumulative products were reconstructed deterministically from the saved 2,000-column `imputed_row_draws.csv` and `wq_cleaned_ml_imputed.csv`; missing years were never treated as zero.
- Unit-of-analysis audit: {'saved_ml_imputed_rows': 15366, 'saved_ml_event_volume_ids': 1018, 'year_treatment_analyte_event_groups_with_multiple_rows': 1134, 'rows_in_year_treatment_analyte_event_groups_with_multiple_rows': 2408, 'base_plot_event_groups': 535, 'base_plot_event_groups_with_multiple_event_ids': 229, 'base_analyte_groups_with_multiple_event_ids': 3186, 'rows_flagged_duplicate': 3248}.

## Validation status

| check                                   | status   | detail                                                                                                                                                                      |
|:----------------------------------------|:---------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Expected cumulative groups              | PASS     | Expected 90; found 90.                                                                                                                                                      |
| Expected treatment-difference groups    | PASS     | Expected 60; found 60.                                                                                                                                                      |
| Expected Spearman groups                | PASS     | Expected 30; found 30.                                                                                                                                                      |
| Spearman paired years                   | PASS     | Minimum paired years=15.                                                                                                                                                    |
| ML imputed row-draw key uniqueness      | PASS     | Duplicate keys=0.                                                                                                                                                           |
| ML annual draws nonnegative             | PASS     | No new ML transformation applied.                                                                                                                                           |
| Variant A display floor only            | PASS     | Raw lower bound retained separately.                                                                                                                                        |
| Variant B nonnegative cumulative        | PASS     | Annual draws truncated before summation.                                                                                                                                    |
| ML saved annual summary reconciliation  | PASS     | Saved LOYO summary recalculated from saved LOYO annual draws.                                                                                                               |
| ML full-record summary reconciliation   | WARN     | Saved row draws and saved imputed annual summary use different documented upstream RNG streams; relative discrepancies are quantified in annual_summary_reconciliation.csv. |
| Bayes saved summary reconciliation      | WARN     | Saved annual draw file is a 400-draw posterior subsample; saved summary used fuller posterior calculation and upstream clamping.                                            |
| Physical event aggregation demonstrated | WARN     | Not demonstrated: 1134 Year × Treatment × Analyte × event groups contain multiple rows; outputs marked provisional.                                                         |
| CT denominator QC                       | WARN     | Total invalid percent-difference draws across groups=486; retained in denominator-QC fields.                                                                                |

## Readiness

- The two master cumulative-load tables, Bayesian sensitivity table, Spearman table, calibration tables, feature-importance tables, and performance tables are structurally ready for manuscript review.
- Their numerical interpretation remains provisional pending a domain decision on whether first-flush/outflow/duplicate sample records are separate load-bearing events or repeated measurements of one physical plot runoff event.
- No glaring target leakage, held-out-year leakage, unit-conversion error, analyte mismatch, or wrong model-version use was identified in this limited read-only audit. The unresolved unit-of-analysis evidence is the upstream issue requiring attention before definitive release claims.

## Reproduction

Run from the repository root:

```powershell
python code/bayes_ml_postprocessing_v2p1.py --repo . --rebuild-current-run
python -m unittest discover -s tests -p "test_bayes_ml_postprocessing_v2p1.py" -v
```

Post-processing runtime for this run: 11.3 seconds. Dependency versions are saved in `postprocessing_run_metadata.json`.
