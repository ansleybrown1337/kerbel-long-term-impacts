# Bayesian v3p1 physical-event workflow

Active files:

- `code/bayes/stir-bayes-load_v3p1_physical_event.Rmd`
- `code/bayes/stir-bayes-load_v3p1_physical_event.R` (primary all-at-once entry point)
- `code/bayes/m_stir_mogp_v3p1_physical_event.stan`
- `code/bayes/stir-bayes-backend.R`

## Structural correction

`E_n` is the number of `PhysicalEventID` values and `E[N]` maps each concentration row to its event. Stan contains one `V_true_event[e]`. The observed-volume interface is `J_VOL`, `VOL_obs[J_VOL]`, and `VOL_event_id[J_VOL]`; each genuine observation uses the existing common `sigma_VOL_obs` likelihood. An event with missing volume has no observation row. A confirmed zero is a finite observation.

The key is `Year + Irrigation + Rep + Treatment`; `Date` is not identity. `EventDate` comes from the unique genuine-volume observation date where one exists, otherwise the earliest valid contributing date. All observation dates remain in the audit. Preflight predictor conflicts block the run.

No measurement-method coefficient, method bias, method-specific error scale, or new scientific parameter was added. Inflow volume, residue, crop, concentration, censoring, random effects, GP structure, priors, and causal adjustment structure remain as in the accepted predecessor.

The STIR input definition is corrected in v3p1. The Stan predictor named
`STIR` is standardized `Season_STIR_toDate`, calculated over
the period after the preceding crop's named harvest operation through the
observation date. Same-day follow-up operations, fall operations, and pre-plant
operations carry into the next crop. The same seasonal predictor enters the
residue, runoff-volume, and concentration processes. Although the pipeline also
exports and standardizes `CumAll_STIR_toDate`, the current Bayesian Stan model
does not include that all-years predictor.

## Full-record prediction and final load

All eligible cleaned-data concentration rows for the 10 prespecified study
analytes (`OP`, `TP`, `NO3`, `TN`, `TSS`, `TKN`, `NH4`, `Se`, `NO2`, and
`TDS`) receive model predictions, whether their concentration is observed or
missing. The four additional cleaned-data analytes (`ICP`, `TSP`, `NPOC`, and
`NOx`) are intentionally reserved for the exploratory ML model and are not
Bayesian targets. All 528 physical events receive modeled runoff volumes,
whether volume is observed, confirmed zero, or missing. Observed outcomes enter
the likelihood and appear as diagnostic/reference points; observed values are
never substituted into the modeled prediction products. Within each draw, row
predictions resolve to one
`PhysicalEventID × Analyte` concentration (median default). Mean and
method-priority modes are implemented; the priority hierarchy is empty until
explicitly reviewed. One latent physical-event volume is paired to that
concentration. The exported ledger asserts uniqueness by
`PhysicalEventID × Analyte × Draw`.

Annual products sum event loads within `Year × Treatment × Rep × Analyte × Draw`, average replicate-specific plot totals within treatment and draw, and then summarize those treatment-mean draws. Annual runoff volume and 2011–2025 cumulative loads use the same hierarchy. CT-relative reductions are calculated within draw from treatment-mean loads. The raw Gaussian load draws remain unchanged; presentation-only lower-bound flooring and annual-draw truncation are separate labeled sensitivities.

Observed annual values use replicate-plot completeness, not event bootstrap resampling. Two complete plots show their mean and minimum-to-maximum range; one complete plot shows its value with no interval and a distinct marker; zero complete plots show no point. The range is descriptive, not a confidence interval.

S1/S2 are retained as distinct Irrigation levels and mapped to codes 11/12 for the accepted linear irrigation-effect term. ML treats them categorically; this difference is a documented limitation.

## Batch execution

Run only after the preflight reports ready. The batch script performs the full
fit, post-processing, diagnostics, and exports without interactive R Markdown
cells. It can be launched from any working directory because it resolves the
repository from its own file path.

```powershell
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p1_physical_event.R
```

The batch version omits two transient pre-dashboard diagnostic chunks that
only printed or displayed unsaved results. It also runs `cmdstan_dashboard()`
once directly to the persistent PNG instead of first repeating the same costly
dashboard interactively. The final overall diagnostic summary is saved as
`results/bayes/v3p1_physical_event/overall_diagnostics_bayes_v3p1_physical_event.csv`.

The batch runner reuses a valid compiled Stan executable by default. The first
run still compiles when no current executable exists. To require a fresh build:

```powershell
$env:BAYES_FORCE_RECOMPILE = "true"
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p1_physical_event.R
Remove-Item Env:BAYES_FORCE_RECOMPILE
```

The Rmd remains available for exploratory, cell-by-cell work but is no longer
the recommended production entry point.

Inspect `results/bayes/v3p1_physical_event/run_manifest_bayes_v3p1_physical_event.json`, sampler diagnostics, row/event prediction diagnostics, event-date audits, the event-analyte ledger, replicate-aware annual draws, observed plot-completeness exports, and `figures/bayes/v3p1_physical_event/` before proceeding.
