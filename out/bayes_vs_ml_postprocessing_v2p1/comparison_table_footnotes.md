# Comparison table footnotes

1. Study period: 2011–2025. Loads are kilograms. Bayes annual draws were saved in grams and divided by 1,000; ML loads were reconstructed from saved concentration (mg/L) and volume (L) draws and divided by 1,000,000.
2. Modeled entries are `mean [2.5th percentile, 97.5th percentile]`. Raw machine-readable values are unrounded.
3. Observed values are **observed subtotals**, not complete annual truth. Parenthetical values give the number of contributing annual subtotals. Missing concentration-volume pairs and unequal treatment coverage preclude interpreting them as complete treatment comparisons.
4. CT-relative percent difference is `100 × (CT − T) / CT`. Positive values mean T is lower than CT; negative values mean T is higher than CT. Percent calculations exclude missing, nonfinite, zero, negative, and CT values at or below 1e-12 kg. Absolute differences retain all finite aligned draws.
5. Bayes Variant A (`raw_draws_bound_floor`) sums unmodified saved annual draws. Only the publication display lower bound is floored at zero when the raw 2.5th percentile is negative. Means, medians, upper bounds, and raw draws are unchanged.
6. Bayes Variant B (`annual_draw_truncation`) applies `max(annual_load, 0)` to every saved annual draw before summing. This is annual-draw-level truncation, not event-level truncation.
7. ML uses the accepted saved full-record imputation artifacts. The 2,000 draw columns in `imputed_row_draws.csv` are aligned by saved draw index; no new random values were generated. Reconstruction QC: {"duplicate_row_draw_keys": 0, "n_draws": 2000, "n_filtered_rows": 10984, "n_fixed_rows": 3716, "n_imputed_concentration_rows": 5613, "n_imputed_volume_rows": 3999, "n_no_runoff_rows": 35}.
8. Spearman rho is calculated across paired annual central estimates within Analyte × Treatment. `*` denotes an exact unadjusted p < 0.05; parentheses contain the number of paired years. These are exploratory multiple tests, not multiplicity-adjusted confirmatory inference.
9. NRMSE_mean = RMSE / mean(abs(observed)); publication percentages multiply this ratio by 100.
10. CatBoost feature importance is descriptive and noncausal. It reflects model use of a predictor, not a management effect or environmental mechanism.
11. All cumulative and CT-relative modeled results are **provisional**. Both accepted workflows define event units using SampleID and MeasureMethod, and the saved ML imputed routine sums repeated analyte-event rows. The current audit cannot establish that all such rows are independent physical runoff-load units.
