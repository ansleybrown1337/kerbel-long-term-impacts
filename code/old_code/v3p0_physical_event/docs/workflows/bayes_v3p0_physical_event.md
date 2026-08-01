# Bayesian v3p0 physical-event workflow

Active files:

- `code/bayes/stir-bayes-load_v3p0_physical_event.Rmd`
- `code/bayes/stir-bayes-load_v3p0_physical_event.R` (primary all-at-once entry point)
- `code/bayes/m_stir_mogp_v3p0_physical_event.stan`
- `code/bayes/stir-bayes-backend.R`

## Structural correction

`E_n` is the number of `PhysicalEventID` values and `E[N]` maps each concentration row to its event. Stan contains one `V_true_event[e]`. The observed-volume interface is `J_VOL`, `VOL_obs[J_VOL]`, and `VOL_event_id[J_VOL]`; each genuine observation uses the existing common `sigma_VOL_obs` likelihood. An event with missing volume has no observation row. A confirmed zero is a finite observation.

No measurement-method coefficient, method bias, method-specific error scale, or new scientific parameter was added. STIR, inflow volume, residue, crop, concentration, censoring, random effects, GP structure, priors, and causal adjustment structure remain as in the accepted predecessor except for mechanical physical-event indexing.

## Final load

All concentration rows receive predictions. Within each draw, row predictions resolve to one `PhysicalEventID × Analyte` concentration (median default). Mean and method-priority modes are implemented; the priority hierarchy is empty until explicitly reviewed. One latent physical-event volume is paired to that concentration. The exported ledger asserts uniqueness by `PhysicalEventID × Analyte × Draw` before annual and cumulative summaries. It retains the accepted Gaussian model's raw load draws; the comparison workflow keeps the presentation-only lower-bound floor and annual-draw truncation alternatives separate and labeled.

## Batch execution

Run only after the preflight reports ready. The batch script performs the full
fit, post-processing, diagnostics, and exports without interactive R Markdown
cells. It can be launched from any working directory because it resolves the
repository from its own file path.

```powershell
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p0_physical_event.R
```

The batch version omits two transient pre-dashboard diagnostic chunks that
only printed or displayed unsaved results. It also runs `cmdstan_dashboard()`
once directly to the persistent PNG instead of first repeating the same costly
dashboard interactively. The final overall diagnostic summary is saved as
`results/bayes/v3p0_physical_event/overall_diagnostics_bayes_v3p0_physical_event.csv`.

The batch runner reuses a valid compiled Stan executable by default. The first
run still compiles when no current executable exists. To require a fresh build:

```powershell
$env:BAYES_FORCE_RECOMPILE = "true"
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p0_physical_event.R
Remove-Item Env:BAYES_FORCE_RECOMPILE
```

The Rmd remains available for exploratory, cell-by-cell work but is no longer
the recommended production entry point.

Inspect `results/bayes/v3p0_physical_event/run_manifest_bayes_v3p0_physical_event.json`, sampler diagnostics, row/event prediction diagnostics, the event-analyte ledger, annual draws, and `figures/bayes/v3p0_physical_event/` before proceeding.
