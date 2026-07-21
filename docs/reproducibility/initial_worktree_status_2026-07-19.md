# Initial worktree status (2026-07-19)

Recorded before task edits with bundled Git at commit `ab87b17a71fee58b2c7f81bd463bc72a28e1267c` on `main`, tracking `origin/main` (`+0/-0`).

```text
 M code/annual_load_bayes_vs_ml.py
 M docs/README_Bayes_vs_ML_WQ.md
 M figs/annual_bayes_vs_ml_faceted_jpg_v2p1/gof_nrmse_mean_by_analyte.jpg
 M out/bayes_vs_ml_metrics_v2p1/metrics_by_analyte_overall.csv
 M out/bayes_vs_ml_metrics_v2p1/metrics_by_analyte_treatment.csv
?? code/__pycache__/
?? figs/annual_bayes_vs_ml_faceted_jpg_codex_volume_check/
?? out/bayes_vs_ml_metrics_codex_volume_check/
?? out/bayes_vs_ml_metrics_v2p1/spearman_by_analyte_pub.csv
?? out/bayes_vs_ml_metrics_v2p1/spearman_by_analyte_treatment.csv
```

This is a historical snapshot from the earlier post-processing-only phase. The
subsequent physical-event refactor was authorized after the baseline was pushed;
its corrected products now target `results/` and `figures/` under separate
Bayes, ML, and comparison roots. The unrelated pre-existing result changes
listed above were not treated as corrected v3p0 inputs or overwritten.
