# Data pipeline workflow

The v3p1 pipeline includes two scientific corrections: the physical-event
identity change and a harvest-anchored seasonal STIR exposure. Its active entry
point and helpers live together under `code/pipeline/`.

From the repository root:

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\pipeline\run_pipeline.py --debug
```

Primary cleaned output: `out/wq_cleaned.csv`.

`Season_STIR_toDate` is the treatment-specific STIR sum after the preceding
crop's named harvest operation through the runoff observation date. Thus,
operations after one crop's harvest, including fall and pre-plant operations,
are carried into the following crop's runoff observations. It does not reset on
January 1 or at planting. If another operation is recorded on the same date
after the named harvest operation, it also carries forward; this handles the
December 2, 2014 stalk-shredding and baling records. The first observed 2011
season is marked `Season_STIR_LeftCensored = true` because a 2010 harvest date
is unavailable; it includes all STIR operations available from the start of the
record through each 2011 observation.

The production runner passes `--season postharvest` explicitly. The legacy
`plant` and `calendar` modes remain available only for diagnostic comparisons.
Their event-level differences are exported to
`out/pipeline_csvs/seasonal_stir_definition_audit.csv`; neither legacy
alternative enters model input.

After any pipeline change, rerun the physical-event preflight. It must report exactly 528 physical events (510 numeric and 18 storm), seven currently known multi-date events, no unresolved event-level predictor conflict, no blocking rows, and `ready_for_model_execution: true`. The model scripts bind identity to the audited `_wq_idx` rows and refuse stale or blocked preflight metadata. Focused tests also verify that calendar-year and planting-date resets cannot replace the harvest-anchored definition.

Pipeline intermediates remain in `out/` for compatibility. They are source/processed data, not v3p1 model results.
