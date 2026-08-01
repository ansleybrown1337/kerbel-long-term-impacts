# Bayesian v3p3 physical-event workflow

Active files:

- `code/bayes/stir-bayes-load_v3p3_physical_event.Rmd`
- `code/bayes/stir-bayes-load_v3p3_physical_event.R` (primary all-at-once entry point)
- `code/bayes/m_stir_mogp_v3p3_physical_event.stan`
- `code/bayes/stir-bayes-backend.R`

## v3p3 model change

v3p3 preserves the complete v3p2 model, including its global standardized
runoff-volume effect on concentration and accepted prior conventions. It adds
one documented event-level Boolean exposure, `FurrowTireCompaction`, and one
global coefficient, `beta_tire_comp_V`, to the runoff-volume process only.

The approved exposure scope is ST in 2021 and MT plus ST in 2022–2025, both
replicates. All affected tractor passes occurred after residue measurement and
before the first runoff event, so every physical event in an affected
treatment-year is marked exposed. CT and all pre-2021 events remain unexposed.
The source is `data/furrow_tire_compaction_records.csv`; absent Year × Treatment
rows default to zero. The source is deliberately separate from the tillage
records because the follow-on tractor pass is not a STIR operation and cannot
be inferred from the logged clean-furrow or pipe-ditch label alone.

The new coefficient uses `normal(0, 1)` without a directional constraint. No
other v3p2 prior changes. The complete audit is in
`docs/methods/bayesian_prior_audit_v3p3.md`.

This is a Bayesian-only release. ML remains v3p1 and completed comparison
outputs remain v3p2. Because the cleaned input and 528-event definition did not change, the
v3p3 runner deliberately reuses the accepted
`validation/preflight/physical_event_v3p1` data contract while writing every
new Bayesian fit, table, diagnostic, and figure under v3p3 paths.

## Furrow tire-compaction pathway

The runoff-volume mean is:

`a_V + b_V * STIR + beta_vin * VIN + beta_res_V * residue +
beta_tire_comp_V * FurrowTireCompaction + crop effect`.

Compaction does not enter the residue process because the documented passes
occurred after residue measurement. It does not enter concentration directly;
any load consequence propagates through the fitted latent runoff volume. v3p3
does not add a residue interaction, treatment-specific compaction slopes, or
carryover into later years.

The runner writes a 528-row exposure audit and a Year × Treatment summary
before compilation. The current support is 120 exposed physical events, of
which 70 have genuine runoff-volume observations. The 2022 exposed events have
no direct genuine volume observations and therefore do not directly identify
the coefficient from the runoff-volume likelihood.

## Structural correction

`E_n` is the number of `PhysicalEventID` values and `E[N]` maps each concentration row to its event. Stan contains one `V_true_event[e]`. The observed-volume interface is `J_VOL`, `VOL_obs[J_VOL]`, and `VOL_event_id[J_VOL]`; each genuine observation uses the existing common `sigma_VOL_obs` likelihood. An event with missing volume has no observation row. A confirmed zero is a finite observation.

The key is `Year + Irrigation + Rep + Treatment`; `Date` is not identity. `EventDate` comes from the unique genuine-volume observation date where one exists, otherwise the earliest valid contributing date. All observation dates remain in the audit. Preflight predictor conflicts block the run.

Sampler, flume, and duplicate effects are observation-layer effects: they
describe how a recorded observation may differ from the one physical truth for
its event and analyte. They do not create separate physical truths. No
laboratory analytical-method coefficient was added.

The event-analyte concentration innovations use a multivariate normal
correlation matrix across the 10 prespecified analytes. Consequently, observed
information for analytes such as TSS can inform an unobserved TP concentration
for the same event, with the strength and direction estimated from the data.
The runoff-volume effect on concentration is global. Irrigation and the other
approved analyte-specific effects retain their existing pooling structures.

The STIR input definition remains the corrected definition introduced in
v3p1. The Stan predictor named
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
never substituted into the modeled prediction products. The reconstruction is
direct, not conditional: each posterior draw supplies one latent concentration
for every `PhysicalEventID × Analyte` combination and one latent volume for
every physical event. Concentration and volume are clamped to zero within each
draw before their product is used to calculate event load. The exported ledger
asserts uniqueness by `PhysicalEventID × Analyte × Draw`.
Bayes observed-versus-modeled figure generation is likewise restricted to
these 10 analytes; observed-only figures are not created for the four
exploratory ML-only analytes.

The shared configuration keeps median row resolution for the ML workflow. Its
separate `bayesian_prediction_resolution` section labels Bayes correctly as
`latent_event_analyte_truth` and `latent_physical_event_truth`.

Annual products sum event loads within `Year × Treatment × Rep × Analyte ×
Draw`, average replicate-specific plot totals within treatment and draw, and
then summarize those treatment-mean draws. Annual runoff volume and 2011–2025
cumulative loads use the same hierarchy. CT-relative reductions are calculated
within draw from treatment-mean loads. Nonnegative clamping is applied at the
event-draw concentration and volume level before load multiplication; observed
values are still reference data and are not substituted into predictions.

Observed annual values use replicate-plot completeness, not event bootstrap resampling. Two complete plots show their mean and minimum-to-maximum range with a white-filled circle; one complete plot shows its value with no interval as a white-filled square; zero complete plots show no point. Both observed marker types appear in the estimate-type legend. Annual runoff-volume figures use the volume-specific complete-plot count rather than analyte-load completeness. The range is descriptive, not a confidence interval.

All analyte-specific annual-load images use the same 12-by-5.5-inch
publication canvas. TSS alone is displayed in megagrams (Mg) so its large
values do not compress the plotting panel with seven-digit gram labels. This
display conversion does not alter the gram-based posterior calculations or
saved annual tables.

S1/S2 are retained as distinct Irrigation levels and mapped to codes 11/12 for the accepted linear irrigation-effect term. ML treats them categorically; this difference is a documented limitation.

## Batch execution

The existing v3p1 data preflight must report ready. No pipeline or Python data
preparation rerun is required for this model-only change. The batch script performs the full
fit, post-processing, diagnostics, and exports without interactive R Markdown
cells. It can be launched from any working directory because it resolves the
repository from its own file path.

From RStudio, open and source
`code/bayes/stir-bayes-load_v3p3_physical_event.R`. From Anaconda Prompt or
PowerShell at the repository root, run:

```powershell
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p3_physical_event.R
```

The batch version omits two transient pre-dashboard diagnostic chunks that
only printed or displayed unsaved results. It also runs `cmdstan_dashboard()`
once directly to the persistent PNG instead of first repeating the same costly
dashboard interactively. The final overall diagnostic summary is saved as
`results/bayes/v3p3_physical_event/overall_diagnostics_bayes_v3p3_physical_event.csv`.

The batch runner reuses a valid compiled Stan executable by default. The first
run still compiles when no current executable exists. To require a fresh build:

```powershell
$env:BAYES_FORCE_RECOMPILE = "true"
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p3_physical_event.R
Remove-Item Env:BAYES_FORCE_RECOMPILE
```

The Rmd is a thin compatibility wrapper that sources the same R script, so the
production workflow has one implementation.

## Completion email

When the existing SMTP environment variables are configured and the `blastula`
package is installed, the runner sends a compact diagnostic email immediately
after sampling. It attaches a small four-panel PNG and a CSV with maximum
R-hat, total parameter count, counts and percentage at or below the 1.04 R-hat
review threshold, minimum bulk/tail effective sample sizes, divergence count,
maximum treedepth count, minimum E-BFMI, and R-hats/ESS for prespecified key
effects. The subject says either `QUICK CHECKS OK` or `REVIEW DIAGNOSTICS`;
this is informational and never stops figures or post-processing. The full
dashboard is still generated later and remains on disk.

To regenerate diagnostics and finish post-processing from an existing
serialized fit without compiling or sampling, run:

```powershell
$env:BAYES_REUSE_FIT = "true"
$env:BAYES_FORCE_RECOMPILE = "false"
$env:BAYES_REUSE_QUICK_DIAGNOSTICS = "true"
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p3_physical_event.R
Remove-Item Env:BAYES_REUSE_FIT
Remove-Item Env:BAYES_FORCE_RECOMPILE
Remove-Item Env:BAYES_REUSE_QUICK_DIAGNOSTICS
```

Set `BAYES_REUSE_QUICK_DIAGNOSTICS` only when the compact CSV and PNG already
match the saved fit; it skips the expensive all-parameter summary while
retaining those saved metrics for the final overall diagnostic export. This
saved-fit resume does not resend the sampling-finished email. For a targeted
residue convergence audit only, run
`Rscript code/bayes/diagnose_saved_fit_v3p3.R`; it writes parameter, chain,
divergence, and posterior-correlation CSVs plus a residue-volume chain
density/trace figure without changing the fit. CSV writes use short bounded
retries because OneDrive can briefly lock an existing output while syncing.

To regenerate only the all-years, 2011-2020 pre-tire-compaction-era, and
2021-2025 tire-compaction-era total-load tables from the saved replicate-aware
annual posterior draws, run:

```powershell
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\export_period_total_load_tables_v3p3_physical_event.R
```

This targeted runner reads
`annual_load_draws_bayes_v3p3_physical_event.csv` only; it does not open the
serialized Stan fit, compile, sample, or rebuild event-level predictions. Each
period receives a numeric audit table and a PUB-ready table. PUB-ready cells
contain the estimate and its parenthesized 95% interval on separate lines for
direct pasting into Word.

To run only the structural preflight without compiling or sampling:

```powershell
$env:BAYES_PREFLIGHT_ONLY = "true"
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p3_physical_event.R
Remove-Item Env:BAYES_PREFLIGHT_ONLY
```

Inspect `results/bayes/v3p3_physical_event/run_manifest_bayes_v3p3_physical_event.json`, sampler diagnostics, row/event prediction diagnostics, event-date audits, the event-analyte ledger, replicate-aware annual draws, observed plot-completeness exports, and `figures/bayes/v3p3_physical_event/` before proceeding.
