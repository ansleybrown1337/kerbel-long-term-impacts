# Repository cleanup plan

## Current structure

Active source is separated under `code/pipeline`, `code/bayes`, `code/ml`,
`code/comparison`, and `code/shared`. Active v3p0 outputs use dedicated version
folders:

- `results/bayes/v3p0_physical_event/`
- `results/ml/v3p0_physical_event/`
- `results/comparison/v3p0_physical_event/`
- matching paths under `figures/`

The `out/` directory is reserved for pipeline/model-input products, including
`out/wq_cleaned.csv` and `out/pipeline_csvs/`.

## Completed cleanup and migration

- Removed superseded active Bayesian, ML, and comparison source copies and the
  `code/Old Bayes/` source tree. Git history remains the source-code archive.
- Removed compiled legacy Stan executables from active source locations.
- Removed `.zenodo.json`, the Zenodo include manifest, output/checksum catalogs,
  and their inventory builder.
- Replaced version-obsolete workflow documents with v3p0 documents.
- Migrated legacy figures formerly under `figs/` into workflow/version folders
  under `figures/{bayes,ml,comparison}/`.
- Migrated legacy model/comparison tables formerly mixed into `out/` into
  workflow/version folders under `results/{bayes,ml,comparison}/`.
- Migrated legacy fits, chain CSVs, and dashboard images from
  `code/out_cmdstanr/` into their matching Bayesian version folders.
- Retained `README.md`, `LICENSE`, and `CITATION.cff` as the release metadata
  surface.

See `legacy_artifact_migration_v3p0.md` for the routing rules and validation.

## Remaining release review

After successful v3p0 model and comparison runs, review whether historical
versions should remain in the release branch. Do not remove an older version
until its provenance and any manuscript dependencies are confirmed.

Local session/cache candidates remain separate from scientific outputs:
`catboost_info/`, bytecode caches, `.RDataTmp`, `.Rhistory`, and `.Rproj.user/`.

Do not delete raw or processed source data. Before removing any legacy result,
verify that the corrected result/figure, run manifest, audit, and manuscript
crosswalk replacement are complete.

## Release structure

The tagged GitHub repository snapshot is the release unit used by the
Zenodo-GitHub integration. Do not construct a separate deposit tree, include
manifest, or duplicate bundle.
