
# Bayesian vs Machine Learning Approaches for Long-Term Edge-of-Field Water Quality Analysis

This document describes how Bayesian hierarchical models and machine-learning models are compared in this project, and **how to reproduce the comparison figures and metrics without overwriting previous results**.

The comparison is intentionally designed so that multiple Bayesian model versions (for example v1p6 and v1p7) can be evaluated side by side against the same ML workflow.

---

## Repository prerequisites

These instructions assume:

- You are working from the **repository root**
- You are using **Anaconda Prompt (Windows)** or a comparable conda-enabled shell
- The conda environment used for ML evaluation is already installed

If needed:

```bash
conda activate wq_ml
```

---

## Core comparison script

The comparison between Bayesian and ML annual loads is performed with:

```text
code/annual_load_bayes_vs_ml.py
```

This script:

- reads annual Bayesian load summaries and posterior (or draw-based) uncertainty,
- reads ML-derived annual load estimates,
- computes evaluation metrics (RMSE, NRMSE, coverage, CRPS), including volume
  in the main NRMSE tables and normalized-RMSE comparison chart when volume
  inputs are available,
- generates faceted annual-load comparison figures.

### Important: versioned outputs

The script now supports a **version tag** so that outputs from different Bayesian model versions are written to **separate folders**.

This prevents overwriting figures or metrics when comparing models such as v1p6 and v1p7.

---

## Command-line arguments (summary)

| Argument | Description |
|--------|-------------|
| `--bayes` | Path to Bayesian annual load summary CSV |
| `--bayes_draws` | Path to Bayesian annual load draws CSV |
| `--tag` | Optional label appended to output folders (for example `v1p7`) |

The `--tag` argument is strongly recommended.

---

## How to run (copy/paste friendly)

All commands below are designed to be run **from the repository root** in Anaconda Prompt.

### Run using the default Bayesian outputs (for example v1p6)

If your default files are already named and located as expected by the script:

```bash
python code/annual_load_bayes_vs_ml.py --tag v1p6
```

This writes outputs to:

```text
figs/annual_bayes_vs_ml_faceted_jpg_v1p6/
out/bayes_vs_ml_metrics_v1p6/
```

Your original untagged outputs remain untouched.

---

### Run using Bayesian v1p7 outputs

After exporting v1p7 Bayesian results (example filenames shown below):

```text
out/annual_load_summary_bayes_plus_observed_v1p7.csv
out/annual_load_draws_bayes_v1p7.csv
```

Run:

```bash
python code/annual_load_bayes_vs_ml.py ^
  --bayes out/annual_load_summary_bayes_plus_observed_v1p7.csv ^
  --bayes_draws out/annual_load_draws_bayes_v1p7.csv ^
  --tag v1p7
```

Outputs are written to:

```text
figs/annual_bayes_vs_ml_faceted_jpg_v1p7/
out/bayes_vs_ml_metrics_v1p7/
```

This allows direct visual and quantitative comparison with v1p6.

---

## Output structure

For each `--tag`, the script produces:

### Figures

```text
figs/
  annual_bayes_vs_ml_faceted_jpg_<tag>/
    annual_load_<analyte>_bayes_vs_ml_faceted_<tag>.jpg
```

### Metrics

```text
out/
  bayes_vs_ml_metrics_<tag>/
    bayes_vs_ml_metrics_summary.csv
    bayes_vs_ml_metrics_by_analyte.csv
```

Each tagged run is fully self-contained.

---

## Recommended workflow for Chapter 3

1. Run Bayesian model v1p6
2. Export annual summaries and draws
3. Run comparison script with `--tag v1p6`
4. Run Bayesian model v1p7
5. Export annual summaries and draws
6. Run comparison script with `--tag v1p7`
7. Compare figures and metrics side by side

This workflow preserves reproducibility while allowing methodological iteration.

---

## Interpretation reminder

Differences between v1p6 and v1p7 reflect **model structure choices**, not just numerical noise.  
Comparing tagged outputs is intended to support scientific judgment about:

- uncertainty realism,
- temporal behavior under missing years,
- sensitivity to latent-state modeling assumptions.

No single version is assumed to be “correct” a priori.
