# Physical-event v3p1 preflight audit

Input: `C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\out\wq_cleaned.csv`

- Physical events: 528
- Numeric-irrigation events: 510
- Recorded S1/S2 storm events: 18
- Multi-date physical events: 7
- Event-level predictor conflicts: 0
- Concentration rows: 15,366
- Genuine volume observations: 372
- Copied volume rows removed: 9,395
- Events without a volume observation: 195
- Blocking review rows: 0
- Ready for model execution: `true`

`PhysicalEventID` uses `Year + Irrigation + Rep + Treatment`; Date is observation
metadata and does not change the identity. `EventDate` uses the date of a unique
genuine volume observation when available and otherwise the earliest valid
contributing date. `VolumeObservationID` uses the physical-event identity,
observation Date, available measurement provenance, and exact recorded value.
`SampleID` is not used to manufacture volume observations.

Review `BLOCKING_REVIEW.csv` before model execution. Detailed source rows for
copied values are in `copied_volume_values.csv`; they remain concentration rows
but contribute only one genuine volume observation per deterministic identity.
The full multi-date merge provenance is in `multi_date_event_audit.csv`.

`Season_STIR_toDate` uses
`(previous named harvest operation, Date]`: post-harvest and
pre-plant operations are assigned to the following crop. The first observed
season is explicitly left-censored because its preceding harvest date is not
available.
