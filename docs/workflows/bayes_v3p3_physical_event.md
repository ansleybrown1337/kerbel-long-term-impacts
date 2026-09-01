# Bayesian v3p3 workflow

## Scope

Bayes v3p3 is the inferential framework for the public release. It models the
hierarchical residue, runoff-volume, and analyte-concentration processes over
the shared 528-event record. Directional causal assumptions support inference
about tillage-system mechanisms and management effects.

The physical-event identity is `Year + Irrigation + Rep + Treatment`. Event
date is metadata. S1 and S2 storms remain distinct irrigation labels and are
mapped to codes 11 and 12 for the model's existing linear irrigation term.
The documented `FurrowTireCompaction` indicator enters the runoff-volume process only.

## Entry points

- Stan model: `code/bayes/m_stir_mogp_v3p3_physical_event.stan`
- Batch workflow: `code/bayes/stir-bayes-load_v3p3_physical_event.R`
- R Markdown entry point: `code/bayes/stir-bayes-load_v3p3_physical_event.Rmd`
- Configuration: `config/physical_event_v3p3.json`
- Data-contract preflight: `results/preflight/`
- Prior specification: `docs/methods/bayesian_priors.md`

Run from the repository root after producing `out/wq_cleaned.csv` and verifying
that the preflight reports no blocking rows:

```powershell
Rscript code/bayes/stir-bayes-load_v3p3_physical_event.R
```

No pipeline or Python data transformation is performed during model fitting.
The R workflow consumes the cleaned table and documented configuration.

## Outputs and reporting

Compact results are written to `results/bayes/v3p3_physical_event/`; figures
are written to `figures/bayes/v3p3_physical_event/`. Posterior concentration
and volume predictions are resolved at event level before load is calculated.
Annual treatment estimates sum within replicate plots, average the plot totals,
and then summarize posterior draws.

Observed annual values are reference markers only. A white-filled square marks
one complete plot without an interval; two complete plots may show a descriptive
replicate range. The volume display uses the volume-specific complete-plot count.

Large chain CSVs and draw ledgers are generated locally and ignored by Git.
The compact fitted object, summaries, diagnostics, audits, and publication
tables are retained.

## Diagnostic qualification

The released saved fit has 28 divergent transitions and 20 finite parameters
with R-hat above 1.04 (maximum 1.068). It supports the archived analysis but is
not fully converged; report that qualification with Bayesian estimates.
