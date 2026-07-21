# v3p0 refactor path inventory

## Moved and renamed active source

- `code/main.py` → `code/pipeline/main.py`
- `code/merge_residue.py` → `code/pipeline/merge_residue.py`
- `code/merge_wq_stir_by_season.py` → `code/pipeline/merge_wq_stir_by_season.py`
- `code/run_pipeline.py` → `code/pipeline/run_pipeline.py`
- `code/stir_pipeline.py` → `code/pipeline/stir_pipeline.py`
- `code/wq_longify.py` → `code/pipeline/wq_longify.py`
- `code/stir-bayes-load2p1_nonneg.Rmd` → `code/bayes/stir-bayes-load_v3p0_physical_event.Rmd`
- `code/m_stir_mogp_v2p1.stan` → `code/bayes/m_stir_mogp_v3p0_physical_event.stan`
- `code/stir_bayes_backend.py` → `code/bayes/stir_bayes_backend.py`
- `code/stir-bayes-backend.R` → `code/bayes/stir-bayes-backend.R`
- `code/ml_catboost_conformal_loyo_v2_eventlevel.py` → `code/ml/ml_catboost_conformal_loyo_v3p0_physical_event.py`
- `code/ml_postprocess_plots_v2_eventlevel.py` → `code/ml/ml_postprocess_plots_v3p0_physical_event.py`
- `code/ml_regenerate_from_saved_models.py` → `code/ml/ml_regenerate_from_saved_models_v3p0.py`
- `code/bayes_ml_postprocessing_v2p1.py` → `code/comparison/bayes_ml_comparison_v3p0_physical_event.py`

## Moved documentation

- `README_data_pipeline.md` → `docs/workflows/pipeline_reference.md`
- `STIR calculations.md` → `docs/methods/STIR_calculations.md`
- `manuscript_crosswalk.md` → `docs/methods/manuscript_crosswalk.md`
- the predecessor Bayesian model note → `docs/methods/bayesian_model_history_pre_v3p0.md`
- `code/daggity.Rmd` → `docs/methods/daggity.Rmd`
- the initial status snapshot → `docs/reproducibility/initial_worktree_status_2026-07-19.md`

## Created

- `code/bayes/stir-bayes-load_v3p0_physical_event.R` all-at-once batch runner
- `code/shared/__init__.py`
- `code/shared/physical_event.py`
- `code/shared/audit_physical_events.py`
- `config/physical_event_v3p0.json`
- `tests/test_physical_event_v3p0.py`
- `validation/preflight/physical_event_v3p0/` audit outputs
- `docs/workflows/*_v3p0*.md`
- `docs/methods/data_unit_dictionary_v3p0.md`
- `docs/methods/physical_event_methods_change_v3p0.md`
- `docs/reproducibility/READY_TO_RUN_PHYSICAL_EVENT_WORKFLOWS.md`
- `docs/reproducibility/github_release_checklist.md`
- `docs/reproducibility/repository_cleanup_plan.md`
- this inventory
- placeholder `.gitkeep` files under the v3p0 versioned results/figure roots
- `docs/reproducibility/legacy_artifact_migration_v3p0.md`

## Removed as superseded source or redundant release scaffolding

- `.zenodo.json`
- `code/build_zenodo_inventory.py`
- `docs/zenodo_release_manifest.csv`
- `docs/output_manifest.csv` and `docs/output_manifest.md`
- `code/annual_load_bayes_vs_ml.py`
- superseded root-level Bayesian/ML source copies, compiled Stan executables, and `code/Old Bayes/`
- v2p1 comparison test and obsolete Bayes/ML/comparison README copies

## Modified in place

- `README.md`
- `CITATION.cff`
- the moved active workflows and pipeline path references listed above

Existing model result directories were intentionally untouched.
