# Superseded Bayes v3p1 physical-event archive

This folder preserves the Bayesian v3p1 source and workflow documentation
superseded by Bayes v3p2. It is historical code and is not an active workflow.
The archive was created after the accepted v3p2 fit completed; no model was
rerun to create it.

## Archived Bayesian source

- `code/bayes/m_stir_mogp_v3p1_physical_event.stan`
- `code/bayes/m_stir_mogp_v3p1_physical_event.exe`
- `code/bayes/stir-bayes-load_v3p1_physical_event.R`
- `code/bayes/stir-bayes-load_v3p1_physical_event.Rmd`
- `code/bayes/diagnose_saved_fit_v3p1.R`
- `docs/workflows/bayes_v3p1_physical_event.md`

## Archived Bayesian outputs

- `results/bayes/old_versions/v3p1_physical_event/`
- `figures/bayes/old_versions/v3p1_physical_event/`

The saved fit, CmdStan chain outputs, diagnostics, tables, ledgers, and figures
were moved without regeneration. Existing ignore rules continue to exclude
large runtime artifacts such as CmdStan chain CSVs.

## v3p1 assets intentionally retained as active

Bayes v3p2 changes only the Bayesian parameterization and priors. The corrected
528-event data contract, pipeline, ML model, comparison workflow, preflight,
configuration, shared code, tests, and related documentation remain v3p1 and
must stay in their active locations. In particular:

- `config/physical_event_v3p1.json`
- `validation/preflight/physical_event_v3p1/`
- `code/ml/*v3p1*`
- `code/comparison/*v3p1*`
- `results/ml/v3p1_physical_event/`
- `figures/ml/v3p1_physical_event/`
