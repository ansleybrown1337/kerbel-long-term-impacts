# Environment and reproducibility

## Python environments verified during the v3p0 refactor

The ML workflow requires the repository's `wq_ml` Conda environment documented
by `environment_wq_ml.yml`. The installed environment inspected on 2026-07-19
contains:

- Python 3.10.19
- pandas 2.3.3
- NumPy 2.2.5
- SciPy 1.15.3
- Matplotlib 3.10.7
- scikit-learn 1.7.2
- CatBoost 1.2.8

Run ML with:

```powershell
C:\Users\ansle\anaconda3\envs\wq_ml\python.exe code\ml\ml_catboost_conformal_loyo_v3p0_physical_event.py --repo .
```

The base Anaconda Python inspected during the refactor is Python 3.12.7 with
pandas 2.2.2, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2, and scikit-learn
1.5.1. It does not contain CatBoost. It is suitable for the preflight, focused
tests, and comparison workflow, but not the ML fit.

## R and Bayesian workflow

R 4.4.2 was used for the parse-only validation. Full Bayesian execution also
requires CmdStanR, CmdStan, and the packages loaded by
`code/bayes/stir-bayes-load_v3p0_physical_event.Rmd`.

## Reproducibility boundary

`requirements_mac.txt` is a platform-specific historical snapshot and should
not replace the Windows environments above. Exact model runtimes and minimum
hardware requirements have not yet been measured for v3p0; record them after
the manual runs. Do not deposit platform-specific compiled Stan executables,
local CmdStan output directories, Python caches, or `catboost_info/`.
