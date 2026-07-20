# Environment and reproducibility

## Saved-output comparison

The comparison post-processing requires Python with pandas, NumPy, SciPy, and Matplotlib. The validated run used:

- Python 3.12.4
- pandas 2.2.2
- NumPy 1.26.4
- SciPy 1.13.1
- Matplotlib 3.9.2

Run from the repository root:

```powershell
python code/bayes_ml_postprocessing_v2p1.py --repo . --rebuild-current-run
python -m unittest discover -s tests -p "test_bayes_ml_postprocessing_v2p1.py" -v
```

No randomness is introduced by the post-processing. It reads deposited draws directly. A completed versioned output directory is protected from accidental overwrite; reproduce in a clean checkout or use a new versioned output name for a new release.

## Full workflows

- `environment_wq_ml.yml` documents the established ML environment.
- `requirements_mac.txt` is a platform-specific historical dependency snapshot and should be reviewed before release.
- The Bayesian workflow additionally requires R, CmdStanR, CmdStan, and the R packages named by the R Markdown driver.

Exact full-model runtimes and minimum hardware requirements have not been confirmed in this task. Record them here before the final archive release. Platform-specific compiled Stan executables and local CmdStan output directories should not be deposited.
