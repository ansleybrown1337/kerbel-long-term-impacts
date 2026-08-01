# ML v3p4 physical-event workflow

Active entry point: `code/ml/ml_catboost_conformal_loyo_v3p4_physical_event.py`.

The event key is `Year + Irrigation + Rep + Treatment`; `Date` is observation metadata. A deterministic `EventDate` supplies event-level date predictors, and unresolved event-predictor conflicts block execution. S1/S2 remain distinct categorical Irrigation labels in ML.

Both ML feature sets include `Season_STIR_toDate` and
`CumAll_STIR_toDate`. The seasonal feature is the treatment-specific STIR sum
after the preceding crop's named harvest operation through the observation
date, so same-day follow-up, post-harvest, and pre-plant operations are assigned
to the following crop. The all-years feature remains the independent running
total from the start of the STIR record.

ML v3p4 includes the reviewed binary `FurrowTireCompaction` predictor in both
the concentration (`logC`) and runoff-volume (`logV`) feature sets. Bayesian
v3p3 remains unchanged and retains compaction in its runoff-volume process
only. The exposure covers 120 physical events: 2021 ST and
2022-2025 MT/ST, both replicates. Seventy of those events have at least one
genuine runoff-volume observation. The runner validates that exact roster,
pre-runoff timing, and feature scope before fitting. Genuine parallel volume
measurements retain separate observation rows and event-balanced weights, but
the exposure audit counts unique `PhysicalEventID` values.

The concentration feature set retains the directly interpretable
`MDL_mg_L` and `RL_mg_L` fields and excludes `Result_lod_mg_L`, which is a
derived RL-first censoring limit and exactly duplicates `RL_mg_L` wherever it
is populated in the current training data. The derived field remains in the
cleaned data for Bayesian censoring; it is excluded only from ML prediction.
The runoff-volume feature set retains `DaysSincePlant` and excludes
`DaysUntilHarvest` to avoid carrying both closely collinear crop-calendar
features and to preserve a predictor known at prediction time.

The concentration model retains all eligible `ConcentrationObservationID` rows. Its default weights sum to one within `PhysicalEventID × Analyte`. The volume model trains on genuine `VolumeObservationID` rows; default weights sum to one within `PhysicalEventID`. Exact copied volume repetitions are absent from the training table, while genuine parallel measurements remain. `--no-event-balanced-weights` is an explicit sensitivity switch; the manifest records its use.

`SampleMethod` remains a concentration predictor and a post-prediction concentration-resolution label. It is excluded from the volume model because the current audit does not establish a one-to-one relationship between sampler rows and genuine volume observations. Legitimate volume provenance (`MeasureMethod` and `FlumeMethod`) remains available to the volume model.

Feature-importance figures display at most the top 20 original input columns,
ranked by mean CatBoost importance across LOYO folds. Figure titles state the
number shown and the number available; full rankings remain in the saved CSV
tables. Importance figures use an expanded vertical layout and larger labels
for Word-document readability.

Within each LOYO fold, the held-out year is excluded before the proper-training/calibration split. That split is by whole `PhysicalEventID`, so an event cannot cross it. Predictions and diagnostics remain row/observation-level. Concentration and volume predictions resolve only afterward for physical load.

Outputs distinguish:

- LOYO model-only evaluation;
- primary full-record model prediction at every eligible row and event;
- an explicitly opt-in observed-plus-imputed sensitivity that is disabled by
  default and is never used by the primary comparison workflow.

The primary concentration output predicts every eligible cleaned-data row,
including rows with measured concentrations and rows with missing
concentrations. The primary runoff-volume output predicts all 528 physical
events, including events with genuine measured volumes, confirmed zeros, and
missing volumes. Measured outcomes remain training/evaluation data and
reference markers; they are never substituted into the primary modeled
products.

Each ML load product now has two deliberately separate components. The central
estimate is the deterministic mean of replicate-specific annual plot totals
derived from one resolved point load per `PhysicalEventID x Analyte`; it is not
the median of the uncertainty draws.
The 95% band comes from Monte Carlo resampling of signed log-scale residuals
from the physical-event-grouped split-conformal calibration sets. This replaces
the legacy uniform sampling between interval endpoints, which treated an
interval as though it were a uniform predictive distribution. Interval
performance is evaluated in the outer LOYO folds. Point-load
ledgers stop on duplicate `PhysicalEventID x Analyte` keys, draw ledgers stop on
duplicate `PhysicalEventID x Analyte x Draw` keys, and the saved resolution
audits report the sampler/method rows contributing to every point load.

Every Monte Carlo draw follows the same reporting hierarchy: sum physical-event values within `Year × Treatment × Rep × Analyte`, then average the replicate plot totals. Runoff volume, cumulative totals, CT-relative contrasts, and annual rank comparisons use the treatment-mean product. Event-level RMSE/NRMSE remains unchanged.

For a separated production sequence, first run LOYO without reconstruction.
Then fit and save the two final full-record models and their grouped calibration
residual distributions without generating any reconstructed values. Finally,
regenerate the full-record products from those saved models:

```powershell
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_catboost_conformal_loyo_v3p4_physical_event.py --repo . --preflight-only --no-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_catboost_conformal_loyo_v3p4_physical_event.py --repo . --no-impute_missing --no-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_catboost_conformal_loyo_v3p4_physical_event.py --repo . --fit_final_models_only --no-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_regenerate_from_saved_models_v3p4.py --repo . --no-figures
& 'C:\Users\ansle\anaconda3\envs\wq_ml\python.exe' code\ml\ml_postprocess_plots_v3p4_physical_event.py --repo .
```

`--no-impute_missing` deliberately completes only the LOYO evaluation; it does
not fit or save the two final full-record models. The
`--fit_final_models_only` stage fits and saves those models but stops before
Monte Carlo reconstruction. The flags above defer ML figures so they can be
generated after table review.
Without `--no-figures`, the training and regeneration entry points retain their
convenient automatic plotting behavior.

Inspect `results/ml/v3p4_physical_event/run_manifest_ml_v3p4_physical_event.json`,
LOYO metrics and coverage, event weights/training audit, residual diagnostics,
the primary full-record model-only point and draw ledgers, and their resolution
audits. Observed annual figures use replicate-plot completeness: ranges only
for two complete plots, a white-filled square without an interval for one
complete plot, and no point for zero complete plots. In figures,
"95% calibration-residual PI" is a prediction interval for
reconstructed outcomes, not a confidence interval for a fitted parameter.
