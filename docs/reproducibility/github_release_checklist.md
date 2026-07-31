# GitHub release checklist

Do not create the release until every item below is complete.

- [ ] Resolve every record in `validation/preflight/physical_event_v3p1/BLOCKING_REVIEW.csv`.
- [ ] Rerun preflight and confirm `ready_for_model_execution: true`.
- [ ] Confirm the corrected roster is exactly 528 events: 510 numeric and 18 S1/S2 storm events.
- [ ] Review all seven multi-date events and confirm `event_level_predictor_conflicts.csv` has no data rows.
- [ ] Run Bayesian v3p1; inspect sampler diagnostics, posterior predictive diagnostics, event provenance, and ledger uniqueness.
- [ ] Run ML v3p1; inspect LOYO held-out exclusion, event splits, weights, calibration, residuals, feature importance, and all reconstruction/sensitivity ledgers.
- [ ] Run comparison v3p2 (Bayes v3p2 plus ML v3p1); inspect raw tables before publication tables, confirm all 2011–2025 years, and confirm the mean-per-treatment-plot hierarchy.
- [ ] Confirm observed n=2 points use descriptive replicate ranges, n=1 uses a distinct marker without an interval, and n=0 is absent.
- [ ] Confirm primary annual exports include TSS, TP, TN, and runoff volume, while technical exports retain all analytes.
- [ ] Rerun synthetic tests and static parse checks.
- [ ] Update manuscript methods/results and `docs/methods/manuscript_crosswalk.md` to the validated v3p1 outputs.
- [ ] Verify the accepted v3p0 code snapshot and every v3p0 result/figure/preflight folder remain in their documented archive locations.
- [ ] Confirm data-sharing rights and remove local-only/scratch artifacts.
- [ ] Confirm the full author list, ORCIDs, funding, article DOI, repository URL, release date, and final semantic version in `CITATION.cff`.
- [ ] Confirm `README.md`, `LICENSE`, and `CITATION.cff` render/validate in the GitHub repository.
- [ ] User reviews local changes, commits, and pushes them.
- [ ] User creates and validates the tagged GitHub release.
- [ ] Confirm the Zenodo–GitHub integration archived that tagged repository snapshot and record the DOI.

No separate Zenodo include manifest, checksum catalog, `.zenodo.json`, deposit directory, or duplicate release bundle is needed.
