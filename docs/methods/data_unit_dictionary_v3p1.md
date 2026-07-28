# Data-unit dictionary: v3p1 physical-event workflows

## PhysicalEventID

One hydrologic plot runoff event, keyed by `Year + Irrigation + Rep + Treatment`. It contributes one latent/resolved runoff volume and no more than one final load for each analyte and draw. `Date`, `SampleID`, analyte, duplicate status, and measurement method are not part of this identity. Replicate plots remain distinct.

## EventDate

`Date` remains observation metadata. When one physical event contains several dates, `EventDate` is the date of a unique genuine runoff-volume observation; otherwise it is the earliest valid contributing date. All contributing dates and observation IDs are retained in `event_date_audit.csv` and `multi_date_event_audit.csv`. Unresolved event-level predictor conflicts are blocking.

## Seasonal STIR exposure

`Season_STIR_toDate` is the treatment-specific sum of operation-level STIR
after the preceding crop's named harvest operation through the observation
date. The harvest operation closes the preceding crop; subsequent fall, winter,
spring pre-plant, and in-season operations are assigned to the following crop.
Other operations recorded on the same date after the named harvest operation
also carry forward. `Season_STIR_BoundaryDayCarryover` records that amount. The
exposure therefore does not reset at January 1 or at planting.

The 2011 season is left-censored because the preceding 2010 harvest date is not
available. `Season_STIR_LeftCensored` identifies those rows, which retain all
available STIR from the start of the operation record through the observation.
`CumAll_STIR_toDate` remains the independent all-years running total.

## VolumeObservationID

One genuine runoff-volume measurement mapped to a `PhysicalEventID`. The current cleaned data have no unique source flow-record identifier, so the conservative candidate identity is physical event plus available measurement provenance (`MeasureMethod`, `FlumeMethod`) plus recorded numeric value.

Exact copies across analyte, sample, duplicate rows, and the same observation date collapse to one observation. Records on distinct dates are not discarded merely because their values match. Different methods remain separate even when values match. Different values within the same available event/date/method/flume provenance are retained in the audit but block execution rather than being silently averaged. A confirmed zero is an observation. An unconfirmed missing value creates no volume-observation row.

## ConcentrationObservationID

One analyte result row, derived from the pipeline’s unique `_wq_idx`. Legitimate samples, first flushes, and duplicates remain distinct observations through fitting and prediction. `SampleID` is provenance, not a physical runoff-volume multiplier.

## Event-analyte-draw load ledger

The sole load-bearing unit is `PhysicalEventID × Analyte × Draw`.

1. Predict every eligible concentration observation row.
2. Resolve row predictions within event, analyte, and draw (median default; mean optional; method priority optional only with an explicit nonempty hierarchy).
3. Resolve volume predictions within physical event and draw where a workflow produces several observation-level predictions. Bayesian output already has one latent physical-event volume.
4. Calculate `Load_kg = Concentration_mg_L × Volume_L / 1,000,000`.
5. Assert ledger-key uniqueness.
6. Sum event loads within `Year × Treatment × Rep × Analyte × Draw`.
7. Average replicate-specific plot totals within `Year × Treatment × Analyte × Draw`, then summarize treatment-mean draws and their cumulative contrasts.

No pre-fit average or median is used to replace concentration rows or genuine volume observations.

Bayesian Gaussian load draws remain unmodified in the raw ledger. A display-only
floor may change a negative reported lower interval bound to zero without
changing any draw or central estimate. The separate annual-truncation
sensitivity applies `max(annual_load_draw, 0)` before study-period summation;
it is not event-level truncation.

## Observed annual plot unit

Completeness is determined independently within `Year × Treatment × Rep × Analyte` from that year's actual Irrigation labels. A plot total is complete only when every expected event has observed concentration and genuine runoff volume, or confirmed zero runoff. Two complete plots yield their arithmetic mean and descriptive minimum-to-maximum range. One complete plot yields its value with no interval. No complete plots yield `NA`. The replicate range is not a confidence interval.
