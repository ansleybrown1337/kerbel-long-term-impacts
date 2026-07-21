# Data-unit dictionary: v3p0 physical-event workflows

## PhysicalEventID

One hydrologic plot runoff event, keyed by `Date + Year + Irrigation + Rep + Treatment`. It contributes one latent/resolved runoff volume and no more than one final load for each analyte and draw. `SampleID`, analyte, duplicate status, and measurement method are not part of this identity.

## VolumeObservationID

One genuine runoff-volume measurement mapped to a `PhysicalEventID`. The current cleaned data have no unique source flow-record identifier, so the conservative candidate identity is physical event plus available measurement provenance (`MeasureMethod`, `FlumeMethod`) plus recorded numeric value.

Exact copies across analyte, sample, and duplicate rows collapse to one observation. Different methods remain separate even when values match. Different values within the same available event/method/flume provenance are retained in the audit but block execution rather than being silently averaged. A confirmed zero is an observation. An unconfirmed missing value creates no volume-observation row.

## ConcentrationObservationID

One analyte result row, derived from the pipeline’s unique `_wq_idx`. Legitimate samples, first flushes, and duplicates remain distinct observations through fitting and prediction. `SampleID` is provenance, not a physical runoff-volume multiplier.

## Event-analyte-draw load ledger

The sole load-bearing unit is `PhysicalEventID × Analyte × Draw`.

1. Predict every eligible concentration observation row.
2. Resolve row predictions within event, analyte, and draw (median default; mean optional; method priority optional only with an explicit nonempty hierarchy).
3. Resolve volume predictions within physical event and draw where a workflow produces several observation-level predictions. Bayesian output already has one latent physical-event volume.
4. Calculate `Load_kg = Concentration_mg_L × Volume_L / 1,000,000`.
5. Assert ledger-key uniqueness, then sum to annual and cumulative load.

No pre-fit average or median is used to replace concentration rows or genuine volume observations.

Bayesian Gaussian load draws remain unmodified in the raw ledger. A display-only
floor may change a negative reported lower interval bound to zero without
changing any draw or central estimate. The separate annual-truncation
sensitivity applies `max(annual_load_draw, 0)` before study-period summation;
it is not event-level truncation.
