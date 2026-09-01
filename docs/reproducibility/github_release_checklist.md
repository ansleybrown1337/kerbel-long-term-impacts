# GitHub release checklist

- [ ] Run `python -m pytest -q` and resolve all failures.
- [ ] Regenerate `results/preflight/` and confirm 528 events, 510 numeric
  irrigation events, 18 storm events, zero blocking rows, and readiness `true`.
- [ ] Confirm the Bayes manifest reports v3p3 and the shared v3p4 data contract.
- [ ] Confirm the ML and complementary-synthesis manifests report v3p4.
- [ ] Review Bayesian divergences, R-hat, effective sample sizes, and posterior
  predictive diagnostics; retain the convergence qualification in public text.
- [ ] Review outer-LOYO ML performance, interval coverage, feature importance,
  and saved-model metadata.
- [ ] Confirm publication tables use the mean-per-treatment-plot aggregation
  hierarchy and do not treat replicate ranges as confidence intervals.
- [ ] Confirm `tools/`, `tmp/`, compiled binaries, chain CSVs, and draw-level
  ledgers are ignored and absent from the prospective release.
- [ ] Verify `CITATION.cff`, author order, affiliations, release version, DOI,
  and release date.
- [ ] Regenerate `CHECKSUMS.sha256` after all reviewed changes.
- [ ] Create tag `v3p4` only after repository review and final validation.
