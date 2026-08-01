# Changelog

## 3.4.0 — ML predictor-contract and feature-importance revision

- Preserved completed Bayesian, ML, and comparison v3p3 artifacts; added an
  isolated ML v3p4 workflow and a Bayes-v3p3-versus-ML-v3p4 comparison.
- Removed derived `Result_lod_mg_L` from the ML concentration predictors while
  retaining the directly interpretable `MDL_mg_L` and `RL_mg_L` fields.
- Added the reviewed event-level `FurrowTireCompaction` exposure to ML
  concentration prediction as well as runoff-volume prediction; Bayesian v3p3
  remains unchanged with a runoff-volume-only compaction pathway.
- Retained `DaysSincePlant` and removed `DaysUntilHarvest` from the ML
  runoff-volume feature set; the concentration model continues to receive both
  crop-calendar fields.
- Kept Chapter 4 feature-importance figures at a maximum of 20 inputs while
  increasing figure height and label sizes and stating the displayed/available
  feature counts in the comparison titles.
- Restored ML prediction-interval ribbons to the primary annual load and
  runoff-volume comparison figures. Their y-axes use Bayesian credible bounds,
  observed references, and ML centers but exclude ML interval bounds; clipped
  ML ribbons are explicitly disclosed, while supplemental panels retain the
  complete method-specific intervals.
- Added hard preflight checks for the revised feature contract. No CatBoost
  calibration, fitting, reconstruction, or comparison regeneration was run
  while preparing the v3p4 code.

## 3.3.0 — furrow tire-compaction runoff-volume pathway

- Added a documented Boolean `FurrowTireCompaction` exposure for 2021 ST and
  2022–2025 MT/ST, both replicates, applying to runoff events after the
  post-residue, pre-irrigation tractor pass.
- Kept the unlogged tractor pass separate from STIR and added one global
  `beta_tire_comp_V ~ normal(0, 1)` coefficient to runoff volume only.
- Added hard-checked event and Year × Treatment exposure audits, confirming
  120 exposed physical events and 70 exposed events with genuine volume
  observations under the unchanged 528-event roster.
- Added compaction-specific R-hat/ESS reporting, posterior correlations,
  original-scale kL-per-event summaries, annual counterfactual contrasts, and
  density/trace diagnostics.
- Preserved all v3p2 source and outputs. Completed the Bayes v3p3 sampling run
  (4,000 post-warmup draws), regenerated the remaining Bayes outputs from the
  serialized fit without recompiling or resampling, and saved the expanded
  convergence and residue/compaction diagnostics.
- Added and completed ML v3p3 with `FurrowTireCompaction` restricted to the
  runoff-volume feature set; retained the 3,000-iteration CatBoost settings,
  outer leave-one-year-out validation, full-record reconstruction, and
  all-point prediction ledgers.
- Completed the Bayes-versus-ML v3p3 comparison tables and publication figures
  using only v3p3 manifests and ledgers, with no legacy fallback or missing-year
  zero fill.

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
