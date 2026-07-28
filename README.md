# Kerbel long-term impacts on edge-of-field water quality

This repository supports analysis of the 2011–2025 Kerbel agricultural monitoring record in Colorado. The active scientific workflows are versioned `v3p1_physical_event` and share one hydrologic unit: `Year + Irrigation + Rep + Treatment`. `Date` is observation metadata; a deterministic `EventDate` is selected for event-level predictors.

## Workflow map

| Stage | Active entry point | Reads | Writes |
|---|---|---|---|
| Data pipeline | `code/pipeline/run_pipeline.py` | `data/` | `out/wq_cleaned.csv` and pipeline intermediates |
| Physical-event preflight | `code/shared/audit_physical_events.py` | `out/wq_cleaned.csv` | `validation/preflight/physical_event_v3p1/` |
| Bayesian v3p1 | `code/bayes/stir-bayes-load_v3p1_physical_event.R` and `code/bayes/m_stir_mogp_v3p1_physical_event.stan` | cleaned data plus passing preflight | `results/bayes/v3p1_physical_event/`, `figures/bayes/v3p1_physical_event/` |
| ML v3p1 | `code/ml/ml_catboost_conformal_loyo_v3p1_physical_event.py` | cleaned data plus passing preflight | `results/ml/v3p1_physical_event/`, `figures/ml/v3p1_physical_event/` |
| Comparison v3p1 | `code/comparison/bayes_ml_comparison_v3p1_physical_event.py` | completed v3p1 manifests and ledgers only | `results/comparison/v3p1_physical_event/`, `figures/comparison/v3p1_physical_event/` |

Legacy artifacts formerly stored in `figs/`, model-output directories under
`out/`, and `code/out_cmdstanr/` have been migrated into versioned subfolders
under `figures/` and `results/`. The active `out/` directory is now reserved for
pipeline/model-input data products.

The comparison stage does not fit either model and has no fallback to legacy results. It rejects wrong versions, missing artifacts, incomplete years, and duplicated `PhysicalEventID × Analyte × Draw` ledger rows.

## Scientific data units

- `PhysicalEventID` identifies one plot runoff event and therefore one latent/event-resolved runoff volume.
- `VolumeObservationID` identifies one genuine recorded volume measurement. Exact copied values across analyte, sample, and duplicate rows do not become new measurements.
- `ConcentrationObservationID` identifies one analyte result row. Legitimate rows remain separate through model fitting and prediction.
- Final physical load has one row per `PhysicalEventID × Analyte × Draw`, after within-draw prediction resolution (median by default; mean and an explicitly configured method priority are sensitivity options).
- Annual and cumulative load and runoff-volume products are means per treatment plot: event values are summed within `Year × Treatment × Rep` and the replicate-specific annual plot totals are then averaged.

See the [data-unit dictionary](docs/methods/data_unit_dictionary_v3p1.md) for the complete contract.

## Current status

The code refactor and physical-event preflight are complete. The corrected
pipeline preserves the source storm-event labels (`S1`/`S2`), and two confirmed
duplicate-sample volume transcription errors are documented in
`data/source_corrections_v3p1.csv`. The current audit has zero blocking rows and
is ready for model execution. No v3p1 Bayesian model, CatBoost model, full-record
reconstruction, or comparison has been run.

Start with [READY_TO_RUN_PHYSICAL_EVENT_WORKFLOWS.md](docs/reproducibility/READY_TO_RUN_PHYSICAL_EVENT_WORKFLOWS.md). Do not proceed to model execution until `preflight_metadata.json` reports `"ready_for_model_execution": true`.

## Documentation

- [Pipeline workflow](docs/workflows/pipeline_v3p1.md)
- [Bayesian v3p1 workflow](docs/workflows/bayes_v3p1_physical_event.md)
- [ML v3p1 workflow](docs/workflows/ml_v3p1_physical_event.md)
- [Comparison v3p1 workflow](docs/workflows/comparison_v3p1_physical_event.md)
- [Methods change note](docs/methods/physical_event_methods_change_v3p1.md)
- [Repository cleanup plan](docs/reproducibility/repository_cleanup_plan.md)
- [GitHub release checklist](docs/reproducibility/github_release_checklist.md)
- [Environment notes](environment/README.md)
- [Citation metadata](CITATION.cff)

The accepted v3p0 code, results, figures, and preflight are retained under `old_code/versions/v3p0_physical_event/` and the workflow-specific `old_versions/` output folders. The intended release archive is the tagged GitHub snapshot through the Zenodo–GitHub integration.

## Authorship and license

Principal investigator: AJ Brown, Colorado State University Agricultural Water Quality Program.

The code is licensed under the [GNU General Public License version 2](LICENSE). Confirm the final author list, ORCIDs, funding, repository URL, article DOI, and release date before creating the GitHub release.
