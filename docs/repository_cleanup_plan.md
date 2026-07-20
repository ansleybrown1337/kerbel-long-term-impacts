# Repository cleanup plan for a Zenodo release

## Current decision

Do not mass-move active files in the current working tree. The task began with pre-existing modifications to the comparison script, comparison documentation, v2p1 metrics, a figure, Spearman outputs, and validation directories. Moving active Bayes/ML files now would obscure those changes and risk hard-coded path breakage.

The scientific post-processing already writes new products into clearly separated comparison-only directories. Use `docs/output_manifest.csv` as the exact migration map for a later clean `zenodo-release` branch or a separate release repository.

## Proposed release structure

```text
README.md
CITATION.cff
.zenodo.json
LICENSE
CHANGELOG.md
environment/
data/raw/
data/processed/
code/pipeline/
code/bayes/
code/ml/
code/comparison/
results/bayes/v2p1/
results/ml/catboost_loyo/
results/comparison/v2p1/
figures/bayes/v2p1/
figures/ml/catboost_loyo/
figures/comparison/v2p1/
docs/methods/
docs/reproducibility/
```

## Safe migration sequence

1. Start from a clean branch created specifically for the release; do not archive active files inside the tracked release tree.
2. Freeze the include set in `docs/zenodo_release_manifest.csv` and confirm every included source/result checksum.
3. Move files with version control according to `proposed_release_path`; do not copy multi-gigabyte posterior/CmdStan artifacts merely to match the tree.
4. Update hard-coded paths listed in the manifest, then run only the allowed comparison post-processing and tests.
5. Verify root links, metadata placeholders, licenses, data rights, and manuscript crosswalk.
6. Create the Zenodo deposit from the clean include manifest, not from the development branch.

## Exclude while preserving development history

- `code/Old Bayes/` and superseded Bayesian outputs/figures;
- superseded v1 comparison directories;
- `code/out_cmdstanr/`, local fit/chain objects, and compiled Stan executables;
- `catboost_info/`, `__pycache__/`, bytecode, `.RDataTmp`, and scratch/session files;
- `codex_volume_check` validation directories;
- obsolete meeting presentations and result PDFs;
- redundant PNG/JPG copies;
- exact duplicate prediction datasets.

The two ML files `wq_cleaned_ml_imputed.csv` and `predictions_from_saved_models.csv` have identical length and SHA-256 (`14D1391029F29FADE4CE0D1314D6F1F4C1A50A2DD8E6324817222EEB4906D48D`). Retain `wq_cleaned_ml_imputed.csv` as the canonical deposited table because active comparison code reads it; exclude `predictions_from_saved_models.csv` as a byte-identical provenance alias, while documenting how it was generated.

## Deferred structural moves

All active-file moves are deferred. The manifest proposes paths but the current scripts continue to use their established locations. This keeps the accepted Bayes/ML workflows unchanged and makes the comparison reproducible immediately from saved artifacts.

## Release blockers to resolve

- Decide the physical interpretation of first-flush, outflow, and duplicate SampleIDs before removing provisional labels from cumulative results.
- Confirm the full author list, ORCIDs, funding, article DOI, repository URL, and release version/date.
- Confirm data-sharing rights and whether the large deposited imputation-draw file should be compressed or attached separately.
- Record full Bayes/ML runtime and hardware requirements.
