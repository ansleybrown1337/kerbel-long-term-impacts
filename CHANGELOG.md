# Changelog

## 3.1.0 — physical-event and mean-per-plot correction

- Changed `PhysicalEventID` to `Year + Irrigation + Rep + Treatment`; `Date` is retained as observation metadata and deterministic `EventDate` is used for event predictors.
- Added multi-date provenance, dynamic yearly irrigation rosters, predictor-conflict blocking, and the corrected 528-event roster check.
- Corrected `Season_STIR_toDate` to reset after the preceding crop's named harvest operation, carrying same-day follow-up, post-harvest, and pre-plant operations into the following crop instead of resetting on January 1.
- Added explicit `PreviousHarvestDate`, seasonal-window start, and first-season left-censoring metadata plus focused harvest-boundary regression tests.
- Retained S1/S2 storm labels with the existing framework-specific Bayes and ML representations.
- Changed annual and cumulative Bayes/ML reporting to sum within replicate plot and then average replicate totals.
- Replaced pooled event-bootstrap observed references with replicate-plot completeness, descriptive replicate ranges, and distinct single-plot markers.
- Restricted primary manuscript-supporting annual tables to TSS, TP, TN, and runoff volume while preserving all-analyte technical exports.
- Archived accepted v3p0 source snapshots and moved all v3p0 model results, figures, and preflight outputs into `old_versions/` namespaces.
- Added focused v3p1 regression tests and a staged production runbook. No Bayes or ML model was run.

## Earlier unreleased Zenodo preparation

- Added saved-output Bayesian-versus-ML post-processing for 2011–2025 cumulative loads, CT-relative treatment differences, and two Bayesian nonnegative-load sensitivity variants.
- Added raw and publication-ready tables, data dictionaries, validation checks, and comparison figures in new versioned directories.
- Added explicit input checksums, annual-summary reconciliations, LOYO calibration summaries, feature-importance tables, and Spearman temporal-agreement tables.
- Documented the unresolved physical-event unit issue and marked affected cumulative products provisional.
- Added release metadata placeholders, workflow documentation, output/manuscript manifests, and a non-destructive repository migration plan.

No Bayesian or CatBoost model was refitted, recompiled, recalibrated, or rerun while preparing v3p1.
