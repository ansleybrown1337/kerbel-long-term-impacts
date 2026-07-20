# Kerbel long-term impacts on edge-of-field water quality

This repository supports analysis of the 2011–2025 Kerbel agricultural monitoring record in Colorado. It contains a shared data pipeline, a hierarchical Bayesian workflow, a CatBoost/LOYO/conformal workflow, and a post-processing workflow that compares their saved outputs.

The four workflows are deliberately separate. Comparison code reads deposited Bayesian draws and ML predictions; it does not refit either model.

## Reader routes

1. **Prepare the data** — see [the pipeline workflow](docs/workflows/pipeline.md) and [data documentation](data/README.md).
2. **Reproduce or inspect the Bayesian analysis** — see [the Bayesian workflow](docs/workflows/bayes.md). The accepted model/output version is v2p1.
3. **Reproduce or inspect the ML analysis** — see [the ML workflow](docs/workflows/ml.md). The accepted output is the event-level CatBoost reconstruction with LOYO and conformal diagnostics.
4. **Run the Bayesian-versus-ML post-processing** — see [the comparison workflow](docs/workflows/comparison.md). This is the short, deterministic route from saved model outputs to cumulative-load, CT-relative, sensitivity, performance, calibration, and publication tables.

## Current validated comparison products

The current post-processing products are in:

- `out/bayes_vs_ml_postprocessing_v2p1/`
- `figs/bayes_vs_ml_postprocessing_v2p1/`

Start with:

- `postprocessing_validation_report.md`
- `master_cumulative_loads_raw_bound_floor_pub.csv`
- `master_cumulative_loads_annual_truncation_pub.csv`
- `bayes_nonnegative_sensitivity_pub.csv`
- `postprocessing_data_dictionary.csv`

These cumulative and CT-relative products are marked **provisional**. A read-only audit found that the accepted upstream event keys include `SampleID` and `MeasureMethod`, while first-flush, outflow, and duplicate sample IDs can occur for the same date/irrigation/replicate/treatment. The present repository does not yet demonstrate whether every such record is an independent physical load-bearing runoff event. No upstream model was changed in response.

## Reproduction levels

Two reproduction levels should not be conflated:

- **Saved-output reproduction** reruns tables and figures from deposited Bayesian draws and ML predictions. It is fast and does not require Stan compilation or CatBoost fitting.
- **Full model reproduction** reruns the data pipeline and then the accepted Bayesian or ML model. It is computationally much more demanding and may depend on platform-specific CmdStan/CatBoost environments.

To reproduce the validated comparison from saved outputs in a clean checkout:

```powershell
python code/bayes_ml_postprocessing_v2p1.py --repo . --rebuild-current-run
python -m unittest discover -s tests -p "test_bayes_ml_postprocessing_v2p1.py" -v
```

The recorded run took approximately 12 seconds with Python 3.12.4, pandas 2.2.2, NumPy 1.26.4, SciPy 1.13.1, and Matplotlib 3.9.2. Exact versions are in `out/bayes_vs_ml_postprocessing_v2p1/postprocessing_run_metadata.json`.

## Repository and release documentation

- [Repository cleanup plan](docs/repository_cleanup_plan.md)
- [Output manifest](docs/output_manifest.csv)
- [Zenodo release manifest](docs/zenodo_release_manifest.csv)
- [Manuscript crosswalk](docs/manuscript_crosswalk.md)
- [Environment notes](environment/README.md)
- [Citation metadata](CITATION.cff)
- [Change log](CHANGELOG.md)

Active files have not been mass-moved in the current working tree. The manifests provide an exact proposed release path for a later clean `zenodo-release` branch or separate release repository.

## Authorship, citation, and license

Principal investigator: AJ Brown, Colorado State University Agricultural Water Quality Program.

Release DOI, funding, ORCID, and final repository URL remain explicit metadata placeholders pending author confirmation. The code is licensed under the [GNU General Public License version 2](LICENSE).
