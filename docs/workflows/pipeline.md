# Data-pipeline workflow

## Scientific purpose and unit of analysis

The pipeline standardizes the Kerbel monitoring, management, STIR, crop, and residue records into analyte-row and event-linked tables used by both modeling frameworks. Raw observations remain unchanged; derived fields and joins are written to processed outputs.

## Inputs

- CSV files under `data/`
- field definitions and cleaning rules documented in `docs/README_data_pipeline.md`

## Entry point and command

Principal entry point: `code/run_pipeline.py`.

```powershell
python code/run_pipeline.py
```

Runtime and computational requirements: modest desktop Python workload; exact release runtime remains to be recorded.

## Outputs and version

Primary processed tables are under `out/pipeline_csvs/` and `out/wq_cleaned.csv`. They support all Bayes v2p1 and event-level CatBoost inputs. Treat raw-source tables as final inputs, processed tables as reproducible intermediates, and unmatched/QC tables as diagnostics.

## Manuscript support

The pipeline supports the analytical dataset definition and all downstream tables/figures. See `docs/manuscript_crosswalk.md` and the detailed pipeline README for cleaning and join rules.

## Release note

The release manifest proposes `data/raw/` and `data/processed/` paths. No active files were moved during the current dirty-worktree task.
