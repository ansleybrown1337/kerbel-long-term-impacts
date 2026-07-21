# Physical-event v3p0 preflight audit

Input: `C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\out\wq_cleaned.csv`

- Physical events: 535
- Concentration rows: 15,366
- Genuine volume observations: 372
- Copied volume rows removed: 9,395
- Events without a volume observation: 202
- Blocking review rows: 0
- Ready for model execution: `true`

`PhysicalEventID` uses `Date + Year + Irrigation + Rep + Treatment`. `VolumeObservationID`
uses that physical-event key plus available measurement provenance and the exact
recorded value. `SampleID` is not used to manufacture volume observations.

Review `BLOCKING_REVIEW.csv` before model execution. Detailed source rows for
copied values are in `copied_volume_values.csv`; they remain concentration rows
but contribute only one genuine volume observation per deterministic identity.
