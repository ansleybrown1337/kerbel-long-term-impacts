# Repository cleanup plan

## Current structure

Active source is separated under `code/pipeline`, `code/bayes`, `code/ml`,
`code/comparison`, and `code/shared`. Active v3p1 outputs use dedicated version
folders:

- `results/bayes/v3p1_physical_event/`
- `results/ml/v3p1_physical_event/`
- `results/comparison/v3p1_physical_event/`
- matching paths under `figures/`

The `out/` directory is reserved for pipeline/model-input products, including
`out/wq_cleaned.csv` and `out/pipeline_csvs/`.

## Completed cleanup and migration

- Copied the accepted v3p0 Bayesian, ML, comparison, shared, configuration,
  test, and workflow files to `old_code/versions/v3p0_physical_event/`.
- Renamed active Bayesian, ML, comparison, configuration, tests, and workflow
  documents to v3p1.
- Moved all v3p0 results and figures into each framework's
  `old_versions/v3p0_physical_event/` folder.
- Moved the v3p0 preflight to
  `validation/preflight/old_versions/physical_event_v3p0/`.
- Moved the compiled v3p0 Stan executable out of active source and into the
  v3p0 code archive.
- Removed `.zenodo.json`, the Zenodo include manifest, output/checksum catalogs,
  and their inventory builder.
- Replaced version-obsolete workflow documents with v3p1 documents.
- Migrated legacy figures formerly under `figs/` into workflow/version folders
  under `figures/{bayes,ml,comparison}/`.
- Migrated legacy model/comparison tables formerly mixed into `out/` into
  workflow/version folders under `results/{bayes,ml,comparison}/`.
- Migrated legacy fits, chain CSVs, and dashboard images from
  `code/out_cmdstanr/` into their matching Bayesian version folders.
- Retained `README.md`, `LICENSE`, and `CITATION.cff` as the release metadata
  surface.

See `old_code/versions/v3p0_physical_event/ARCHIVE_MANIFEST.md` for the exact
archive map.

## Remaining release review

After successful v3p1 model and comparison runs, retain the historical
`old_versions/` folders unless a separately reviewed release policy explicitly
changes that decision.

Local session/cache candidates remain separate from scientific outputs:
`catboost_info/`, bytecode caches, `.RDataTmp`, `.Rhistory`, and `.Rproj.user/`.

Do not delete raw or processed source data. Before removing any legacy result,
verify that the corrected result/figure, run manifest, audit, and manuscript
crosswalk replacement are complete.

## Release structure

The tagged GitHub repository snapshot is the release unit used by the
Zenodo-GitHub integration. Do not construct a separate deposit tree, include
manifest, or duplicate bundle.
