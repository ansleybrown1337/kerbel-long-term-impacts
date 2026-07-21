# Ready-to-run physical-event workflows: v3p0 handoff

## Status

The code, shared schema, audit utility, output contracts, documentation, and
synthetic tests are prepared. The corrected end-to-end pipeline and physical-
event audit completed successfully. The current preflight has zero blocking
rows and `ready_for_model_execution: true`.

No Bayesian model was compiled or sampled. No R Markdown document was rendered. No CatBoost model was fitted, calibrated, or used for Monte Carlo reconstruction. No full Bayesian-versus-ML comparison was run during this refactor.

## Scientific contract

`PhysicalEventID = Date + Year + Irrigation + Rep + Treatment`. Concentration observations remain row-level. Genuine volume observations remain observation-level and map to one physical-event volume. Copied volume repetitions do not become measurements. Missing volume creates no Bayesian observation-likelihood row; confirmed zero remains observed. Final load is unique by `PhysicalEventID × Analyte × Draw` after prediction resolution.

Bayesian and ML corrected versions are both `v3p0_physical_event`. Default prediction resolution is median; mean is available. Method priority is available only when the currently empty hierarchy is explicitly configured.

## Preflight findings on current `out/wq_cleaned.csv`

- 15,366 cleaned/concentration rows and 535 physical-event groups.
- 9,767 nonmissing copied volume candidate rows before deduplication.
- 372 genuine `VolumeObservationID` rows after deterministic deduplication; 9,395 copied rows removed from the observation count.
- 333 events with at least one volume observation and 202 events with no volume observation.
- 12 genuine confirmed-zero volume observations.
- Zero blocking rows and zero blocking physical events.
- Source storm-event labels `S1`/`S2` are preserved through final cleaning rather than being erased by numeric coercion.
- Two confirmed lab-duplicate transcription errors were corrected at source and logged in `data/source_corrections_v3p0.csv`: MT duplicate volume 1389.06 to 1388.06 L and ST duplicate volume 7674.99 to 7673.99 L.

`validation/preflight/physical_event_v3p0/BLOCKING_REVIEW.csv` is currently empty except for its header. Any future source-data change must rerun the pipeline and audit; do not bypass the gate or resolve conflicts through silent averaging.

## Verification completed

- Python `compileall`: passed.
- 27 focused synthetic/static tests: passed, including the requested
  invariance/split/refusal cases, deterministic point-ledger uniqueness,
  point-versus-draw center separation, pooled analyte-plus-volume NRMSE,
  empirical calibration-residual propagation enforcement, shared final-input
  enforcement, storm-label preservation, batch-runner single-pass enforcement,
  version-folder isolation, date-ID stability, LOYO-ledger separation, explicit
  absent-year coverage, Bayesian annual-draw sensitivity, and observed-plot
  ledger checks.
- Extracted R code from the Bayesian R Markdown: parse passed without rendering.
- Static Stan interface: `J_VOL`, `VOL_obs`, `VOL_event_id`, mapped likelihood, and absence of method-specific volume parameters passed.
- The end-to-end data pipeline completed successfully and wrote the repository-root `out/` products. It reported one nonfatal warning that the optional STIR crop-window cumulative helper expected a `Date` column in the crop table; the required season merge completed with zero unmatched rows.
- The audit utility ran successfully, wrote the preflight directory, and marked the current dataset ready for model execution.

No scientific runtime validation is claimed until the manual model sequence below is completed.

## Exact manual sequence

Run from the repository root in PowerShell.

### 1. Preflight audit

```powershell
C:\Users\ansle\anaconda3\python.exe code\shared\audit_physical_events.py --input out\wq_cleaned.csv --output-dir validation\preflight\physical_event_v3p0
```

Checkpoint: open `AUDIT_README.md`, `preflight_summary.csv`, `BLOCKING_REVIEW.csv`, `physical_events_by_year_treatment.csv`, `concentration_observation_counts.csv`, `copied_volume_values.csv`, `events_with_multiple_volume_methods.csv`, `ambiguous_volume_observations.csv`, `zero_missing_volume_status.csv`, and `event_multiplicity_by_year.csv`. Proceed only when the metadata says ready and the blocking table is empty. Expected runtime is seconds.

### 2. Bayesian v3p0

```powershell
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' code\bayes\stir-bayes-load_v3p0_physical_event.R
```

Checkpoint: inspect `results/bayes/v3p0_physical_event/run_manifest_bayes_v3p0_physical_event.json`, `results/bayes/v3p0_physical_event/overall_diagnostics_bayes_v3p0_physical_event.csv`, the saved CmdStan dashboard, posterior predictive checks, row/volume diagnostics, event provenance, the unique event-analyte-draw ledger, annual summaries, and `figures/bayes/v3p0_physical_event/`. This may take hours; record actual hardware/runtime. The Rmd remains an optional exploratory interface rather than the production runner.

### 3. Bayesian post-run validation

```powershell
C:\Users\ansle\anaconda3\python.exe -c "import json,pandas as p; from pathlib import Path; d=Path('results/bayes/v3p0_physical_event'); m=json.loads((d/'run_manifest_bayes_v3p0_physical_event.json').read_text()); x=p.read_csv(d/'event_analyte_draw_ledger_bayes_v3p0_physical_event.csv'); assert m['workflow_version']=='v3p0_physical_event'; assert m['event_unit']=='PhysicalEventID'; assert sorted(x.Year.unique())==list(range(2011,2026)); assert not x.duplicated(['PhysicalEventID','Analyte','Draw']).any(); print('Bayesian v3p0 ledger validation: PASS')"
```

Checkpoint: do not proceed if the command or scientific diagnostics fail.

### 4. ML v3p0

```powershell
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\ml\ml_catboost_conformal_loyo_v3p0_physical_event.py --repo .
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\ml\ml_postprocess_plots_v3p0_physical_event.py --repo .
```

Checkpoint: inspect `results/ml/v3p0_physical_event/run_manifest_ml_v3p0_physical_event.json`,
volume training observations, event-balanced weights, LOYO metrics/coverage,
row residuals/standardized discrepancies, feature importance, the point-load
resolution audits, and the separately named point and uncertainty-draw ledgers.
The ML annual line must equal the saved physical-event point total, not the
median of the propagated draws. The uncertainty draws must report weighted
resampling of signed log-scale calibration residuals; the superseded uniform
sampling between conformal endpoints must be false in the manifest. Record
actual runtime; CatBoost plus reconstruction may take hours.

### 5. ML post-run validation

```powershell
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe -c "import json,pandas as p; from pathlib import Path; d=Path('results/ml/v3p0_physical_event'); m=json.loads((d/'run_manifest_ml_v3p0_physical_event.json').read_text()); assert m['workflow_version']=='v3p0_physical_event'; assert m['calibration_split_unit']=='PhysicalEventID'; assert m['primary_ml_central_estimate']=='sum_of_physical_event_point_loads'; assert m['legacy_uniform_between_conformal_bounds_used'] is False; [(_ for _ in ()).throw(AssertionError(f'duplicate draw ledger: {f}')) if p.read_csv(f).duplicated(['PhysicalEventID','Analyte','Draw']).any() else None for f in [d/'event_analyte_draw_ledger_ml_v3p0.csv',d/'event_analyte_draw_ledger_full_record_model_only.csv',d/'event_analyte_draw_ledger_observed_plus_imputed_sensitivity.csv']]; [(_ for _ in ()).throw(AssertionError(f'duplicate point ledger: {f}')) if p.read_csv(f).duplicated(['PhysicalEventID','Analyte']).any() else None for f in [d/'event_analyte_point_ledger_full_record_model_only.csv',d/'event_analyte_point_ledger_observed_plus_imputed_sensitivity.csv']]; print('ML v3p0 point/draw ledger validation: PASS')"
```

### 6. Comparison v3p0

```powershell
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\comparison\bayes_ml_comparison_v3p0_physical_event.py --repo .
```

Checkpoint: inspect the comparison manifest and every raw table before publication tables/figures. Confirm no missing years, no zero substitution, coherent units, Spearman sample sizes, within-draw CT-relative calculations, sensitivity labels, and disagreement rows.

### 7. Final static and synthetic checks

```powershell
C:\Users\ansle\anaconda3\python.exe -m compileall -q code tests
C:\Users\ansle\anaconda3\python.exe -m pytest -q tests\test_physical_event_v3p0.py
& 'C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe' -e "f <- tempfile(fileext='.R'); knitr::purl('code/bayes/stir-bayes-load_v3p0_physical_event.Rmd', output=f, documentation=0, quiet=TRUE); parse(file=f); unlink(f); cat('R parse: PASS\n')"
```

## Organization and path changes

The user confirmed that the complete pre-correction repository baseline was
pushed before this refactor. The active predecessors and all path changes are
listed in `refactor_path_inventory_v3p0.md`; no Git operation was performed in
this task.

New or modified handoff files are:

- shared contract and audit: `config/physical_event_v3p0.json`,
  `code/shared/__init__.py`, `code/shared/physical_event.py`, and
  `code/shared/audit_physical_events.py`;
- moved/updated pipeline: every file under `code/pipeline/`;
- moved/updated Bayes: `code/bayes/stir-bayes-load_v3p0_physical_event.R`,
  `code/bayes/stir-bayes-load_v3p0_physical_event.Rmd`,
  `code/bayes/m_stir_mogp_v3p0_physical_event.stan`,
  `code/bayes/stir-bayes-backend.R`, and `code/bayes/stir_bayes_backend.py`;
- moved/updated ML: all three files under `code/ml/`;
- moved/rebuilt comparison:
  `code/comparison/bayes_ml_comparison_v3p0_physical_event.py`;
- tests: `tests/test_physical_event_v3p0.py`;
- audit artifacts: every CSV, JSON, and Markdown file under
  `validation/preflight/physical_event_v3p0/`;
- workflow/method/reproducibility documentation: every Markdown/R Markdown
  file under `docs/workflows/`, `docs/methods/`, and `docs/reproducibility/`;
- release/runtime metadata: `README.md`, `.gitignore`, `CITATION.cff`, and
  `environment/README.md`;
- dedicated corrected output roots under versioned subfolders in
  `results/{bayes,ml,comparison}/` and `figures/{bayes,ml,comparison}/`.

Active pipeline, Bayesian, ML, comparison, and shared code are separated under `code/`. Corrected outputs target `results/` and `figures/`. Redundant Zenodo scaffolding and superseded active source copies were removed; the complete move/create/remove inventory is in `refactor_path_inventory_v3p0.md`.

Legacy artifacts were moved without deletion into workflow/version folders; see
`legacy_artifact_migration_v3p0.md`. Follow `repository_cleanup_plan.md` and
`github_release_checklist.md` after corrected runs are validated.
