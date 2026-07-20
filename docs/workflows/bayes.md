# Bayesian workflow

## Scientific purpose and unit of analysis

The accepted v2p1 hierarchical Bayesian workflow estimates analyte-specific concentration and runoff-load structure with shared temporal components and propagated uncertainty. Concentration remains at the analyte-row level. The outflow-volume unit uses the accepted event key `Date + Year + Irrigation + Rep + Treatment + SampleID + MeasureMethod`; inflow volume uses the physical plot-event key.

## Inputs

- `out/wq_cleaned.csv`
- the accepted Stan model `code/m_stir_mogp_v2p1.stan`
- the R Markdown driver `code/stir-bayes-load2p1_nonneg.Rmd`

## Entry point and command

Full-model entry point:

```powershell
Rscript -e "rmarkdown::render('code/stir-bayes-load2p1_nonneg.Rmd')"
```

This command is documentation only for this task: the Bayesian model was not recompiled or rerun. Full runtime and hardware requirements depend on CmdStan configuration and remain to be recorded before release.

## Outputs and version

Accepted version: **v2p1**. Principal saved outputs include `out/annual_load_draws_bayes_v2p1.csv`, `out/annual_load_summary_bayes_v2p1.csv`, `out/annual_load_summary_bayes_plus_observed_v2p1.csv`, volume tables, model diagnostics, and unit-of-analysis audits. v2p1 tables and posterior draws are final model artifacts; convergence and unit audits are diagnostic; comparison-derived products belong to the comparison workflow.

## Manuscript support

The v2p1 outputs support Bayesian-only methods, diagnostics, annual estimates, and the Bayesian inputs to comparison tables/figures. See `docs/README_bayes_methods_v2p1_notes.md`, `docs/bayes-model_versions.md`, and `docs/manuscript_crosswalk.md`.

## Saved-output inspection

Readers who only need article tables should use the deposited v2p1 draws and the comparison post-processing command instead of rerunning Stan.
