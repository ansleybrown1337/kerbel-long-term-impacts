# Release manifest

This manifest describes the intended contents of release 3.4.0 (`v3p4`).

| Path | Release role |
| --- | --- |
| `data/` | Source water-quality, tillage, crop, residue, method, and documented correction tables |
| `code/pipeline/` | Deterministic construction of the cleaned analysis table |
| `code/shared/` | Shared physical-event identities, audits, and aggregation utilities |
| `code/bayes/` | Bayes v3p3 Stan model, R workflow, diagnostics, and saved-draw post-processing |
| `code/ml/` | ML v3p4 CatBoost, LOYO, conformal, plotting, and saved-model workflows |
| `code/comparison/` | v3p4 complementary Bayes/ML synthesis workflow |
| `config/` | Current Bayes v3p3 and integrated v3p4 analysis contracts |
| `out/wq_cleaned.csv` | Cleaned analysis table produced by the pipeline |
| `results/preflight/` | Audited 528-event roster and blocking checks |
| `results/bayes/v3p3_physical_event/` | Compact Bayes results and fitted object |
| `results/ml/v3p4_physical_event/` | Compact ML results and saved CatBoost models |
| `results/comparison/v3p4_physical_event/` | Publication synthesis tables |
| `figures/` | Current Bayes, ML, and synthesis figures |
| `docs/` | Public methods, workflow, and reproducibility documentation |
| `tests/` | Non-fitting contract and workflow tests |
| `environment/` and `requirements.txt` | Reproducible software environments |

Local manuscript utilities (`tools/`, `tmp/`, and `docs/drafts/`), compiled
binaries, chain CSVs, and draw-level ledgers are intentionally excluded by
`.gitignore`.
