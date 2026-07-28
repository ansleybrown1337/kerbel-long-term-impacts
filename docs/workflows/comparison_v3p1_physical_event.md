# Bayesian-versus-ML comparison v3p1

Active entry point: `code/comparison/bayes_ml_comparison_v3p1_physical_event.py`.

The comparison consumes only completed Bayesian and ML
`v3p1_physical_event` manifests, event-analyte point ledgers, and
event-analyte-draw ledgers. It stops for missing files, legacy versions,
incomplete 2011-2025 coverage within any publication analyte/treatment, or
duplicate `PhysicalEventID x Analyte` point rows or
`PhysicalEventID x Analyte x Draw` uncertainty rows. It never substitutes a
legacy path or inserts missing years as zero.

The ten shared analytes remain in technical raw exports. Primary annual
manuscript-supporting tables are restricted to TSS, TP, and TN; runoff volume
is reported in its dedicated tables. Additional ML-only analytes are recorded
in the comparison manifest and excluded from cross-framework tables. LOYO load ledgers remain validation diagnostics when
held-out years lack observed volume targets; only complete-year full-record
ledgers contribute to study-period comparison totals.

Primary annual, cumulative, treatment-relative, and Spearman comparisons use
the Bayesian posterior median and the ML deterministic mean of replicate annual
plot totals as their respective central estimates. Every model draw first sums
events within Rep and then averages replicate totals. CT-relative percentages
are calculated within draw from those treatment means, and cumulative contrasts
use cumulative mean-per-plot draws. The Bayesian band is a 95%
posterior credible interval. The ML band is the 95% Monte Carlo empirical
calibration-residual prediction interval: signed log-scale residuals are
resampled from the physical-event-grouped split-conformal calibration set, with
performance evaluated by outer LOYO. Annual ML runoff-volume draws also sum
within Rep and then average replicate plot totals; their prediction interval is
retained in the comparison output.

Observed annual points do not use event bootstrap resampling. Completeness is
evaluated independently for each `Year × Treatment × Rep × Analyte` against
that year's actual Irrigation labels. Two complete plots provide their mean and
descriptive minimum-to-maximum range. A single complete plot is shown with an x
marker and no interval. No complete plot yields no primary point. Incomplete
subtotals remain in separate audit exports, and the replicate range is never
described as a confidence interval.

Prepared raw and publication tables include original-unit performance and
calibration, descriptive/noncausal CatBoost importance, annual and cumulative
loads, Spearman agreement with paired-year n/rho/p, and within-draw CT-relative
and absolute differences with invalid-denominator counts. Bayesian raw-draw
display-floor and annual-draw-truncation sensitivities remain separate. The
workflow also exports observed subtotals and cross-model row disagreement
without automatic exclusion labels.

Primary Bayes and ML comparisons use model predictions at every modeled row
and physical event. Bayes models the 10 prespecified study analytes; ML models
all 14 analytes, while shared Bayes-versus-ML tables use the 10 analytes common
to both models. Observed outcomes appear only as reference markers and
evaluation targets; the comparison never substitutes them into either modeled
series and does not require the optional ML observed-plus-imputed sensitivity.

Generate tables first, inspect them, then generate figures:

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\comparison\bayes_ml_comparison_v3p1_physical_event.py --repo . --skip-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\comparison\bayes_ml_comparison_v3p1_physical_event.py --repo . --figures-only
```

Inspect `results/comparison/v3p1_physical_event/run_manifest_comparison_v3p1_physical_event.json`,
raw tables before publication tables, and `figures/comparison/v3p1_physical_event/`.
The manifest records the `mean_per_treatment_plot` unit and current storm-handling limitation.
