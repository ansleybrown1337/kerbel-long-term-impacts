# Data pipeline workflow

The pipeline remains scientifically unchanged by the physical-event refactor. Its active entry point and helpers now live together under `code/pipeline/`.

From the repository root:

```powershell
C:\Users\ansle\anaconda3\python.exe code\pipeline\run_pipeline.py --debug
```

Primary cleaned output: `out/wq_cleaned.csv`.

After any pipeline change, rerun the physical-event preflight. The model scripts bind identity to the audited `_wq_idx` rows and refuse stale or blocked preflight metadata.

Pipeline intermediates remain in `out/` for compatibility. They are source/processed data, not v3p0 model results.
