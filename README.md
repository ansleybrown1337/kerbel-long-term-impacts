# Kerbel Long‑Term Impacts on Edge‑of‑Field Water Quality

![Banner](./figs/banner.png)

**Principal Investigator**  
AJ Brown, Agricultural Data Scientist  
Colorado State University – Agricultural Water Quality Program (AWQP)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Organization](#2-repository-organization)
3. [End-to-End Data Pipeline](#3-end-to-end-data-pipeline)
4. [Bayesian Modeling Framework (Overview)](#4-bayesian-modeling-framework-overview)
5. [Machine Learning Modeling Framework (CatBoost + Conformal)](#5-machine-learning-modeling-framework-catboost--conformal)
6. [Key Results at a Glance](#6-key-results-at-a-glance)
7. [How to Reproduce Results](#7-how-to-reproduce-results)
8. [Output Data Products](#8-output-data-products)
9. [Related Documentation](#9-related-documentation)
10. [License and Citation](#10-license-and-citation)


---

## 1. Project Overview

This repository supports a **long‑term (2011–2025) edge‑of‑field (EoF) water‑quality analysis** at the Kerbel agricultural research site in Colorado. The project integrates:

- Event‑scale runoff **concentration**, **volume**, and **load** data
- Crop and tillage management records
- Seasonal and cumulative **Soil Tillage Intensity Rating (STIR)** metrics

Two complementary modeling approaches are implemented:

1. **Hierarchical Bayesian models** to quantify causal STIR effects and propagate uncertainty
2. **Machine learning models (CatBoost)** with conformal prediction to benchmark pattern‑learning performance

The repository is intentionally structured so both approaches operate on the **same processed datasets**, enabling transparent comparison of strengths and limitations.

---

## 2. Repository Organization

```
kerbel-long-term-impacts/
├── code/        # Python + R modeling and preprocessing scripts
├── data/        # Raw and intermediate datasets
├── docs/        # Detailed method and workflow documentation
├── figs/        # Publication‑ready figures
├── out/         # Final model outputs
├── README.md    # This file
```

Key subdirectories:
- `code/` contains both Bayesian (Stan/R) and ML (Python) workflows
- `docs/` contains extended READMEs linked throughout this file
- `out/` contains finalized CSV outputs used for analysis and figures

---

## 3. End‑to‑End Data Pipeline

The preprocessing pipeline is shared by both modeling approaches.

1. **Water‑quality longification**  
   `code/wq_longify.py`

2. **STIR calculation**  
   `code/stir_pipeline.py`  
   Details: [`docs/STIR calculations.md`](./docs/STIR%20calculations.md)

3. **Seasonal merge**  
   `code/merge_wq_stir_by_season.py`

4. **Pipeline runner**  
   ```bash
   python code/main.py --debug
   ```

Outputs are written to `out/` and serve as direct inputs to both Bayes and ML models.

---

## 4. Bayesian Modeling Framework (Overview)

The Bayesian analysis is the primary inferential framework for this project. It focuses on:

- Causal estimation of STIR effects
- Full uncertainty propagation
- Multi‑analyte hierarchical structure
- Temporal persistence using Gaussian Processes

📘 **Full Bayesian methods documentation:**  
➡️ [`docs/README_bayes_methods.md`](./docs/README_bayes_methods.md)

This separation keeps the main README concise while preserving methodological depth.

---

## 5. Machine Learning Framework (CatBoost + Conformal)

A parallel machine‑learning analysis is implemented using **CatBoost regression** with **Leave‑One‑Year‑Out (LOYO)** cross‑validation and **conformal prediction intervals**.

### Purpose of ML analysis
- Benchmark predictive performance against Bayesian models
- Evaluate strengths and weaknesses for long‑term environmental datasets
- Examine interpretability tradeoffs

### Key characteristics
- Separate models for log‑concentration and log‑volume
- Saved calibrated models reused for regeneration
- Prediction intervals derived from stored conformal quantiles
- Annual summaries regenerated without refitting

📘 **Detailed ML documentation:**  
➡️ [`docs/README_ML_CatBoost_Conformal_LOYO.md`](./docs/README_ML_CatBoost_Conformal_LOYO.md)

---

## 6. Key Results at a Glance

### Bayesian findings
- STIR exhibits strong, analyte-specific associations, with the largest and most consistent effects observed for particulate constituents (e.g., TP, TSS, OP, TKN).
- Dissolved species generally show weaker or more variable responses to tillage intensity.
- Temporal persistence is substantial for particulate analytes and is explicitly captured via a multi-output Gaussian process.
- Bayesian imputation propagates uncertainty forward in time, producing widening annual load intervals in years with sparse or missing data.

### Machine-learning findings
- CatBoost models primarily learn analyte identity, inflow conditions, cumulative system state, and monitoring infrastructure effects.
- STIR contributes to prediction skill but is not isolated as a dominant or interpretable driver, particularly for concentration.
- Predictive accuracy is reasonable for central tendencies but degrades for extreme events and high-load years.
- Conformal prediction intervals exhibit under-coverage, especially for runoff volume, and do not adapt to temporal nonstationarity.
- ML-derived annual loads behave as smoothed reconstructions rather than uncertainty-aware estimates.

### Synthesis
- Bayesian models provide causal interpretability, temporal coherence, and principled uncertainty propagation.
- Machine-learning models offer pattern recognition and interpolation capability but struggle with temporal generalization and uncertainty calibration.
- The contrast between these approaches highlights when ML may be useful for descriptive benchmarking and when Bayesian inference is required for environmental decision-making.


---

## 7. How to Reproduce Results

### Bayesian modeling
See the dedicated methods README:
➡️ [`docs/README_bayes_Methods.md`](./docs/README_bayes_Methods.md)

### Machine‑learning modeling
Train and calibrate (rare):
```bash
python code/ml_catboost_conformal_loyo.py --repo . --seed 123
```

Regenerate predictions and summaries (no recalibration):
```bash
python code/ml_regenerate_from_saved_models.py --repo . --overwrite --alpha 0.05 --draws 2000
```

---

## 8. Output Data Products

Key outputs include:

- `out/annual_load_summary_bayes_plus_observed_v1p6.csv`
- `out/ml_catboost_conformal_loyo/annual_load_summary.csv`
- `out/ml_catboost_conformal_loyo/annual_load_summary_imputed.csv`

All annual summaries share the same schema:

- volume (L)
- concentration (mg/L)
- load (g)

📘 **Output field definitions:**  
➡️ [`docs/README_pipeline_final_outputs.md`](./docs/README_pipeline_final_outputs.md)

---

## 9. Related Documentation

- Bayes vs ML comparison notes:  
  [`docs/README_Bayes_vs_ML_WQ.md`](./docs/README_Bayes_vs_ML_WQ.md)
- Pipeline execution notes:  
  [`docs/run data pipeline.md`](./docs/run%20data%20pipeline.md)
- STIR methodology:  
  [`docs/STIR calculations.md`](./docs/STIR%20calculations.md)

---

## 10. License and Citation

Licensed under **GPL‑2**.

Please cite the **Colorado State University Agricultural Water Quality Program (AWQP)** when using this repository.

