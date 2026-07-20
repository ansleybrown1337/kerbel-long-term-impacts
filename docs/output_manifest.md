# Output manifest guide

The machine-readable inventory is `docs/output_manifest.csv`. It includes every tracked file plus untracked scientifically relevant files found in the working tree, with current and proposed release paths, workflow classification, status, generating script, manuscript use, include/exclude recommendation, dependencies, and hard-coded-path risk.

`docs/zenodo_release_manifest.csv` is the filtered article-facing recommendation. It is a planning artifact, not evidence that files have been moved, committed, pushed, or deposited.

Use the CSVs rather than an `archive/` folder to define the release. A tracked archive remains part of a Zenodo snapshot unless explicitly excluded from the release branch/repository.
