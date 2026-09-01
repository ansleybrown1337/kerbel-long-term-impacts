# Bayesian-versus-ML comparison v3p4

Active entry point: `code/comparison/bayes_ml_comparison_v3p4_physical_event.py`.

The comparison pairs completed Bayesian `v3p3_physical_event` outputs with
completed ML `v3p4_physical_event` outputs. Both runoff-volume models include
the same reviewed event-level `FurrowTireCompaction` exposure. Bayesian v3p3
does not add a direct concentration term, while ML v3p4 includes the exposure
as a concentration-prediction feature. It consumes
their manifests, event-analyte point ledgers, and event-analyte-draw ledgers.
It stops for missing files, unexpected versions,
incomplete 2011-2025 coverage within any publication analyte/treatment, or
duplicate `PhysicalEventID x Analyte` point rows or
`PhysicalEventID x Analyte x Draw` uncertainty rows. It never substitutes a
fallback path or inserts missing years as zero.

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
descriptive minimum-to-maximum range. A single complete plot is shown with a
white-filled square and no interval. No complete plot yields no primary point. Incomplete
subtotals remain in separate audit exports, and the replicate range is never
described as a confidence interval.

Prepared raw and publication tables include original-unit performance and
calibration, descriptive/noncausal CatBoost importance, annual and cumulative
loads, Spearman agreement with paired-year n/rho/p, and within-draw CT-relative
and absolute differences with invalid-denominator counts. Bayesian raw-draw
display-floor and annual-draw-truncation sensitivities remain separate. The
workflow also exports observed subtotals and cross-model row disagreement
without automatic exclusion labels.

The Chapter 4 feature-importance figures show the top 20 inputs ranked by mean
importance across LOYO folds, or all inputs when fewer than 20 are available.
They use a taller layout and larger text; complete rankings remain in the raw
and publication CSV tables.

Primary Bayes and ML comparisons use model predictions at every modeled row
and physical event. Bayes models the 10 prespecified study analytes; ML models
all 14 analytes, while shared Bayes-versus-ML tables use the 10 analytes common
to both models. Observed outcomes appear only as reference markers and
evaluation targets; the comparison never substitutes them into either modeled
series and does not require the optional ML observed-plus-imputed sensitivity.

Primary annual-load and annual-runoff-volume comparison figures show both the
Bayesian 95% credible ribbon and the ML 95% calibration-residual prediction
ribbon. Their shared y-axis is set from Bayesian upper credible bounds while
also retaining observed references and ML central estimates. ML prediction-
interval bounds are deliberately excluded from axis scaling, so an extreme ML
ribbon can be clipped at the panel boundary instead of flattening the useful
Bayesian-scale comparison. Figure titles and legends disclose that clipping.
The supplemental uncertainty panels retain the complete, unclipped intervals
on method-specific scales.

Generate tables first, inspect them, then generate figures:

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\comparison\bayes_ml_comparison_v3p4_physical_event.py --repo . --skip-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\comparison\bayes_ml_comparison_v3p4_physical_event.py --repo . --figures-only
```

Inspect `results/comparison/v3p4_physical_event/run_manifest_comparison_v3p4_physical_event.json`,
raw tables before publication tables, and `figures/comparison/v3p4_physical_event/`.
The manifest records the `mean_per_treatment_plot` unit and current storm-handling limitation.
