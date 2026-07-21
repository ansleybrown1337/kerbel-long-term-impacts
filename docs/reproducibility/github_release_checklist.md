# GitHub release checklist

Do not create the release until every item below is complete.

- [ ] Resolve every record in `validation/preflight/physical_event_v3p0/BLOCKING_REVIEW.csv`.
- [ ] Rerun preflight and confirm `ready_for_model_execution: true`.
- [ ] Run Bayesian v3p0; inspect sampler diagnostics, posterior predictive diagnostics, event provenance, and ledger uniqueness.
- [ ] Run ML v3p0; inspect LOYO held-out exclusion, event splits, weights, calibration, residuals, feature importance, and all reconstruction/sensitivity ledgers.
- [ ] Run comparison v3p0; inspect raw tables before publication tables and confirm all 2011–2025 years.
- [ ] Rerun synthetic tests and static parse checks.
- [ ] Update manuscript methods/results and `docs/manuscript_crosswalk.md` to the validated v3p0 outputs.
- [ ] Decide which untouched legacy result directories should be excluded from the release branch.
- [ ] Confirm data-sharing rights and remove local-only/scratch artifacts.
- [ ] Confirm the full author list, ORCIDs, funding, article DOI, repository URL, release date, and final semantic version in `CITATION.cff`.
- [ ] Confirm `README.md`, `LICENSE`, and `CITATION.cff` render/validate in the GitHub repository.
- [ ] User reviews local changes, commits, and pushes them.
- [ ] User creates and validates the tagged GitHub release.
- [ ] Confirm the Zenodo–GitHub integration archived that tagged repository snapshot and record the DOI.

No separate Zenodo include manifest, checksum catalog, `.zenodo.json`, deposit directory, or duplicate release bundle is needed.
