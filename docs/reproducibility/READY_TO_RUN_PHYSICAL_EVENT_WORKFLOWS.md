# Ready-to-run physical-event workflows: Bayes, ML, and comparison v3p3

## Status

The v3p1 shared schema and corrected preflight remain the active data contract. They report exactly 528 physical events (510 numeric-irrigation events and 18 S1/S2 storm events), zero unresolved predictor conflicts, zero blocking rows, and `ready_for_model_execution: true`. Bayes and ML v3p3 each hard-check the same runoff-volume-only compaction roster: 120 exposed physical events and 70 exposed events with genuine volume observations. Earlier Bayes v3p2, ML v3p1, and comparison v3p2 artifacts remain preserved as baselines.

## Scientific contract

- `PhysicalEventID = Year + Irrigation + Rep + Treatment`.
- `Date` is observation metadata. `EventDate` is the unique genuine-volume observation date when one exists; otherwise it is the earliest valid contributing date.
- Legitimate concentration and genuine runoff-volume observations are retained across dates. Unresolved event-predictor conflicts are blocking.
- `Season_STIR_toDate` starts after the preceding crop's named harvest operation; same-day follow-up, post-harvest, and pre-plant operations apply to the following crop rather than resetting on January 1.
- S1/S2 remain distinct Irrigation levels. Bayes maps them to 11/12 for the existing linear irrigation term; ML treats them categorically. This framework difference is a limitation, not a v3p1 redesign.
- Model annual and cumulative results are means per treatment plot: sum within Rep, average replicate plot totals within each draw, and then summarize treatment-mean draws.
- CT-relative percentages are calculated within draw from treatment-mean loads.
- Event-level calibration metrics remain in their existing event-level units.
- Observed annual completeness is assessed independently for each replicate plot from the year's actual Irrigation roster. Replicate ranges are descriptive, not confidence intervals.

## Corrected preflight findings

- Cleaned rows: 15,366.
- Corrected physical events: 528.
- Numeric-irrigation plot events: 510.
- Recorded storm plot events: 18.
- Multi-date corrected events: 7.
- Genuine runoff-volume observations: 372.
- Copied volume rows excluded from the observation count: 9,395.
- Events with at least one volume observation: 333.
- Events without a volume observation: 195.
- Confirmed-zero volume observations: 12.
- Unresolved event-level predictor conflicts: 0.
- Blocking rows: 0.
- Harvest-anchored seasonal STIR conflicts within corrected physical events: 0.
- First-season left-censored rows: 504 (2011; preceding 2010 harvest date unavailable).
- Known-boundary physical events differing from the legacy calendar reset: 460 of 492 (436 previously undercounted, 24 previously overcounted).
- Physical events receiving same-harvest-date stalk-shredding/baling carryover: 36; carryover STIR is 30.7125 for each affected 2015 plot event.

The seven multi-date events are 2020 Irrigation 3 Rep 2 for CT/MT/ST, 2020 Irrigation 7 Rep 1 for CT, and 2022 Irrigation 7 Rep 2 for CT/MT/ST. CT and ST in the 2020 Irrigation 3 group use 2020-07-13 because each has a unique genuine volume-observation date. The other five use their earliest valid contributing date (2020-07-13, 2020-08-25, or 2022-08-23 as applicable). All contributing dates and concentration/volume observation IDs are retained in `multi_date_event_audit.csv`; the audit found no blocking predictor conflicts.

## Ordered PowerShell production runbook

Run each block from the repository root. Stop at a failed command or checkpoint.

### 1. Preprocessing

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\pipeline\run_pipeline.py --debug
```

Expected success: exit code 0, the merge command visibly includes `--season postharvest`, the merge reports the harvest-operation/same-day carry-forward definition, the seasonal-definition audit reports `460/492` known-boundary physical runoff events differ from the legacy calendar reset, and `[OK] Wrote wq_cleaned.csv -> out\wq_cleaned.csv`. Review `out/pipeline_csvs/seasonal_stir_definition_audit.csv` and do not continue if the cleaned output is missing.

### 2. Corrected preflight

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\shared\audit_physical_events.py --input out\wq_cleaned.csv --output-dir validation\preflight\physical_event_v3p1
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' -c "import json; from pathlib import Path; p=Path('validation/preflight/physical_event_v3p1/preflight_metadata.json'); m=json.loads(p.read_text()); assert m['workflow_version']=='v3p1_physical_event'; assert m['physical_events']==528; assert m['numeric_irrigation_events']==510; assert m['storm_events']==18; assert m['observed_storm_labels']==['S1','S2']; assert m['event_level_predictor_conflicts']==0; assert m['blocking_rows']==0; assert m['ready_for_model_execution'] is True; print('v3p1 preflight gate: PASS')"
```

Expected success: the audit prints `ready_for_model_execution: True`, and the gate prints `v3p1 preflight gate: PASS`. Check:

- `preflight_metadata.json`
- `BLOCKING_REVIEW.csv` (header only)
- `event_date_audit.csv`
- `multi_date_event_audit.csv`
- `event_level_predictor_conflicts.csv` (header only)
- `yearly_irrigation_roster.csv`

### 3. Bayesian fit and post-processing

```powershell
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p3_physical_event.R
```

Expected success: final `[OK] Wrote:` messages, `Saved overall diagnostics to:`, and exit code 0. Check:

- `results/bayes/v3p3_physical_event/run_manifest_bayes_v3p3_physical_event.json`
- `results/bayes/v3p3_physical_event/overall_diagnostics_bayes_v3p3_physical_event.csv`
- `results/bayes/v3p3_physical_event/event_analyte_draw_ledger_bayes_v3p3_physical_event.csv`
- `results/bayes/v3p3_physical_event/annual_load_draws_bayes_v3p3_physical_event.csv`
- `results/bayes/v3p3_physical_event/tire_compaction_event_audit_v3p3_physical_event.csv`
- `results/bayes/v3p3_physical_event/tire_compaction_volume_effect_v3p3_physical_event.csv`
- sampler diagnostics, posterior predictive checks, event/row diagnostics, and `figures/bayes/v3p3_physical_event/`

The Bayes prediction table must cover all 10,984 cleaned-data rows for the 10
prespecified study analytes (`OP`, `TP`, `NO3`, `TN`, `TSS`, `TKN`, `NH4`,
`Se`, `NO2`, and `TDS`) and all 528 physical events. `ICP`, `TSP`, `NPOC`,
and `NOx` are intentionally ML-only targets. Observed outcomes are retained as
likelihood inputs and reference points but must not replace modeled predictions.

Do not proceed on sampling failures, unacceptable diagnostics, duplicate ledger keys, or missing 2011–2025 coverage.

### 4. ML training and calibration

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_catboost_conformal_loyo_v3p3_physical_event.py --repo . --preflight-only --no-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_catboost_conformal_loyo_v3p3_physical_event.py --repo . --no-impute_missing --no-figures
```

Expected success: the preflight reports 528 physical events, 120 compacted events, 70 compacted events with genuine volume, compaction in `logV` only, and no fitting. The full LOYO command then writes `results/ml/v3p3_physical_event/run_manifest_ml_v3p3_physical_event.json`. Check LOYO exclusions, physical-event calibration splits, event-balanced weights, coverage, residuals, and event-date audits. The manifest must report:

- `workflow_version = v3p3_physical_event`
- `calibration_split_unit = PhysicalEventID`
- `annual_reporting_unit = mean_per_treatment_plot`
- `primary_ml_central_estimate = mean_of_replicate_annual_plot_totals`
- `full_record_models_saved = false`
- `final_model_fit_deferred = true`

### 5. Final full-record model fit and calibration

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_catboost_conformal_loyo_v3p3_physical_event.py --repo . --fit_final_models_only --no-figures
```

Expected success: `[DONE] Final full-record CatBoost models and calibration residuals saved; reconstruction not started.`, exit code 0, two `.cbm` files and matching metadata under `results/ml/v3p3_physical_event/models/`, plus `calibration_residual_distribution_logC.csv` and `calibration_residual_distribution_logV.csv`. This command does not repeat LOYO and does not generate reconstructed values.

### 6. Full-record prediction from saved models

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_regenerate_from_saved_models_v3p3.py --repo . --no-figures
```

Expected success: `[DONE] Full-record predictions regenerated from saved event-level models.`, exit code 0, and full-record point/draw ledgers. Check:

- `event_analyte_point_ledger_full_record_model_only.csv`
- `event_analyte_draw_ledger_full_record_model_only.csv`
- `annual_load_summary_full_record_model_only.csv`
- point-load resolution audits

The primary ML row table must contain model predictions for all 15,366 eligible
concentration rows across all 14 analytes, and the volume table must cover all
528 physical events.
Observed outcomes are reference/evaluation values and must not replace the
primary predictions. All point ledgers must be unique by
`PhysicalEventID × Analyte`; draw ledgers must be unique by
`PhysicalEventID × Analyte × Draw`.

### 7. Comparison tables

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\comparison\bayes_ml_comparison_v3p3_physical_event.py --repo . --skip-figures
```

Expected success: `[DONE] Corrected comparison outputs written to ...`, exit code 0, and `results/comparison/v3p3_physical_event/run_manifest_comparison_v3p3_physical_event.json`. Inspect raw tables before publication tables. Confirm the intentional Bayes v3p3 plus ML v3p3 pairing, mean-per-plot units, all 2011–2025 years, CT-relative within-draw calculations, no zero insertion for missing years, primary TSS/TP/TN plus runoff-volume products, and separate incomplete-observed subtotals.

### 8. Figures

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_postprocess_plots_v3p3_physical_event.py --repo .
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\comparison\bayes_ml_comparison_v3p3_physical_event.py --repo . --figures-only
```

Expected success: `[DONE] v3p3 physical-event post-processing figures created.`, `[DONE] Corrected comparison figures regenerated from saved tables in ...`, both commands exit 0, and refreshed `figures/ml/v3p3_physical_event/` and `figures/comparison/v3p3_physical_event/`. Visually inspect every primary figure. In Bayes and comparison v3p3 figures, observed n=2 points show a white circle and the replicate minimum-to-maximum range; n=1 uses a white square with no interval; n=0 shows no point.

## Post-run checks

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' -m py_compile code\pipeline\merge_wq_stir_by_season.py code\pipeline\run_pipeline.py code\shared\physical_event.py code\shared\audit_physical_events.py code\ml\ml_catboost_conformal_loyo_v3p3_physical_event.py code\ml\ml_regenerate_from_saved_models_v3p3.py code\ml\ml_postprocess_plots_v3p3_physical_event.py code\comparison\bayes_ml_comparison_v3p3_physical_event.py
& 'C:\Users\ansle\anaconda3\python.exe' -m pytest -q
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' -e "invisible(parse(file='code/bayes/stir-bayes-load_v3p3_physical_event.R')); f <- tempfile(fileext='.R'); knitr::purl('code/bayes/stir-bayes-load_v3p3_physical_event.Rmd', output=f, documentation=0, quiet=TRUE); invisible(parse(file=f)); unlink(f); cat('R and Rmd parse: PASS\n')"
```

Expected success: Python compilation exits 0, pytest reports all focused tests passed, and R prints `R and Rmd parse: PASS`. The `wq_ml` environment currently supplies the scientific runtime; the base Anaconda interpreter is used for pytest because pytest is not installed in `wq_ml`.

## Output namespaces

Active outputs:

- `results/bayes/v3p3_physical_event/`
- `results/ml/v3p3_physical_event/`
- `results/comparison/v3p3_physical_event/`
- matching versioned figure folders
- `validation/preflight/physical_event_v3p1/`

Accepted v3p0 artifacts remain archived under:

- `old_code/versions/v3p0_physical_event/`
- `results/{bayes,ml,comparison}/old_versions/v3p0_physical_event/`
- `figures/{bayes,ml,comparison}/old_versions/v3p0_physical_event/`
- `validation/preflight/old_versions/physical_event_v3p0/`

Superseded Bayes v3p1 artifacts are archived under:

- `old_code/versions/v3p1_physical_event/`
- `results/bayes/old_versions/v3p1_physical_event/`
- `figures/bayes/old_versions/v3p1_physical_event/`

No files are staged or committed by this migration.
