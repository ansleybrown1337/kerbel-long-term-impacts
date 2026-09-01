# Environment and reproducibility

## Python

Create the analysis environment with:

```powershell
conda env create -f environment/environment_wq_ml.yml
conda activate wq_ml
```

The environment provides Python, pandas, NumPy, SciPy, Matplotlib,
scikit-learn, CatBoost, tqdm, pytest, and spreadsheet readers used by the data,
ML, and synthesis workflows. A smaller package list is also provided in the
repository-root `requirements.txt` for users who manage Python outside Conda.

## R and CmdStan

The Bayesian workflow requires R, CmdStanR, CmdStan, and the packages loaded by
`code/bayes/stir-bayes-load_v3p3_physical_event.R`. The checked-in Stan source
is compiled locally; platform-specific executables and chain CSVs are excluded
from Git.

## Reproducibility boundary

Compact fitted objects, model summaries, tables, and figures are retained.
Large posterior/prediction draw ledgers, local caches, compiled binaries, and
manuscript-building utilities are local runtime artifacts and are intentionally
ignored. See the root README and workflow documents for run order.
