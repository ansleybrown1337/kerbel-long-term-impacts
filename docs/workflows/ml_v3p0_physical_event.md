# ML v3p0 physical-event workflow

Active entry point: `code/ml/ml_catboost_conformal_loyo_v3p0_physical_event.py`.

The concentration model retains all eligible `ConcentrationObservationID` rows. Its default weights sum to one within `PhysicalEventID × Analyte`. The volume model trains on genuine `VolumeObservationID` rows; default weights sum to one within `PhysicalEventID`. Exact copied volume repetitions are absent from the training table, while genuine parallel measurements remain. `--no-event-balanced-weights` is an explicit sensitivity switch; the manifest records its use.

`SampleMethod` remains a concentration predictor and a post-prediction concentration-resolution label. It is excluded from the volume model because the current audit does not establish a one-to-one relationship between sampler rows and genuine volume observations. Legitimate volume provenance (`MeasureMethod` and `FlumeMethod`) remains available to the volume model.

Within each LOYO fold, the held-out year is excluded before the proper-training/calibration split. That split is by whole `PhysicalEventID`, so an event cannot cross it. Predictions and diagnostics remain row/observation-level. Concentration and volume predictions resolve only afterward for physical load.

Outputs distinguish:

- LOYO model-only evaluation;
- full-record model-only reconstruction;
- observed-plus-imputed sensitivity.

Each ML load product now has two deliberately separate components. The central
estimate is the deterministic sum of one resolved point load per
`PhysicalEventID x Analyte`; it is not the median of the uncertainty draws.
The 95% band comes from Monte Carlo resampling of signed log-scale residuals
from the physical-event-grouped split-conformal calibration sets. This replaces
the legacy uniform sampling between interval endpoints, which treated an
interval as though it were a uniform predictive distribution. Interval
performance is evaluated in the outer LOYO folds. Point-load
ledgers stop on duplicate `PhysicalEventID x Analyte` keys, draw ledgers stop on
duplicate `PhysicalEventID x Analyte x Draw` keys, and the saved resolution
audits report the sampler/method rows contributing to every point load.

Run after Bayesian validation (or independently after a passing preflight):

```powershell
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\ml\ml_catboost_conformal_loyo_v3p0_physical_event.py --repo .
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\ml\ml_postprocess_plots_v3p0_physical_event.py --repo .
```

Inspect `results/ml/v3p0_physical_event/run_manifest_ml_v3p0_physical_event.json`,
LOYO metrics and coverage, event weights/training audit, residual diagnostics,
the model-only and observed-plus-imputed point ledgers, their resolution audits,
and the separately named uncertainty-draw ledgers. In figures, “95%
calibration-residual PI” is a prediction interval for reconstructed outcomes,
not a confidence interval for a fitted parameter.
