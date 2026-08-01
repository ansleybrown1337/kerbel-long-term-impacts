# Ready-to-run ML v3p4 and comparison workflow

ML v3p4 retains the corrected v3p1 528-event data contract and pairs with the
completed Bayesian v3p3 outputs. It makes four deliberate ML-only changes:

- retain `MDL_mg_L` and `RL_mg_L` but exclude duplicate `Result_lod_mg_L`
  from concentration prediction;
- include `FurrowTireCompaction` in both concentration and runoff-volume
  prediction;
- use `DaysSincePlant` and exclude `DaysUntilHarvest` in runoff-volume
  prediction; and
- show at most 20 inputs in taller, larger-text feature-importance figures.

Run these commands from an Anaconda Prompt opened at the repository root.
They are separated so each major output can be checked before the next
expensive stage.

```bat
cd /d "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts"
```

## 1. Inexpensive feature-contract preflight

```bat
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\ml\ml_catboost_conformal_loyo_v3p4_physical_event.py --repo . --preflight-only --no-figures
```

Expected: 528 physical events, 120 compacted events, 70 compacted events with
genuine volume, compaction present in both `logC` and `logV`, and no model fit.
Inspect `results/ml/v3p4_physical_event/event_volume_audit_summary.csv` and
`feature_audit_summary.csv` before proceeding.

## 2. Outer leave-one-year-out evaluation

```bat
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\ml\ml_catboost_conformal_loyo_v3p4_physical_event.py --repo . --no-impute_missing --no-figures
```

This is the first expensive stage. Inspect LOYO performance, coverage,
feature-importance tables, row predictions, and the run manifest before fitting
the final full-record models.

## 3. Fit and save final full-record models

```bat
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\ml\ml_catboost_conformal_loyo_v3p4_physical_event.py --repo . --fit_final_models_only --no-figures
```

## 4. Regenerate full-record predictions from the saved models

```bat
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\ml\ml_regenerate_from_saved_models_v3p4.py --repo . --no-figures
```

## 5. Generate ML figures

```bat
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\ml\ml_postprocess_plots_v3p4_physical_event.py --repo .
```

## 6. Regenerate comparison tables and figures

```bat
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\comparison\bayes_ml_comparison_v3p4_physical_event.py --repo . --skip-figures
"C:\Users\ansle\anaconda3\envs\wq_ml\python.exe" code\comparison\bayes_ml_comparison_v3p4_physical_event.py --repo . --figures-only
```

The comparison must report Bayesian `v3p3_physical_event`, ML
`v3p4_physical_event`, and comparison `v3p4_physical_event`. Do not replace or
move the completed v3p3 outputs; v3p4 writes to new versioned result and figure
directories.
