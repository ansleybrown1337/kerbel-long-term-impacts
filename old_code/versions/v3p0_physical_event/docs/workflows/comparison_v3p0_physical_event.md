# Bayesian-versus-ML comparison v3p0

Active entry point: `code/comparison/bayes_ml_comparison_v3p0_physical_event.py`.

The comparison consumes only completed Bayesian and ML
`v3p0_physical_event` manifests, event-analyte point ledgers, and
event-analyte-draw ledgers. It stops for missing files, legacy versions,
incomplete 2011-2025 coverage within any publication analyte/treatment, or
duplicate `PhysicalEventID x Analyte` point rows or
`PhysicalEventID x Analyte x Draw` uncertainty rows. It never substitutes a
legacy path or inserts missing years as zero.

The ten publication analytes use the shared abbreviated key. Additional ML-only
analytes are recorded in the comparison manifest and excluded from
cross-framework tables. LOYO load ledgers remain validation diagnostics when
held-out years lack observed volume targets; only complete-year full-record
ledgers contribute to study-period comparison totals.

Primary annual, cumulative, treatment-relative, and Spearman comparisons use
the Bayesian posterior median and the ML deterministic physical-event point
total as their respective central estimates. The Bayesian band is a 95%
posterior credible interval. The ML band is the 95% Monte Carlo empirical
calibration-residual prediction interval: signed log-scale residuals are
resampled from the physical-event-grouped split-conformal calibration set, with
performance evaluated by outer LOYO. Observed event-bootstrap markers remain a
separate reference and are never substituted into either model-only line.

Prepared raw and publication tables include original-unit performance and
calibration, descriptive/noncausal CatBoost importance, annual and cumulative
loads, Spearman agreement with paired-year n/rho/p, and within-draw CT-relative
and absolute differences with invalid-denominator counts. Bayesian raw-draw
display-floor and annual-draw-truncation sensitivities remain separate. The
workflow also exports observed subtotals, explicitly named ML model-only and
observed-plus-imputed products, and cross-model row disagreement without
automatic exclusion labels.

Run only after both model manifests and checkpoint reviews pass:

```powershell
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\comparison\bayes_ml_comparison_v3p0_physical_event.py --repo .
```

Inspect `results/comparison/v3p0_physical_event/run_manifest_comparison_v3p0_physical_event.json`,
raw tables before publication tables, and `figures/comparison/v3p0_physical_event/`.
