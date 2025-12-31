# Model Versioning Summary

**Kerbel Long-Term Tillage Impacts Project**  
**AJ Brown**

This file tracks the structure and development stage of all Bayesian model families used in the STIR–WQ analysis workflow. Each model family has its own Rmd file and evolves independently. This table format is designed for fast updates, minimal text, and clear tracking of model features over time.

---

## Load Models (Vol + Concentration) – ACTIVE DEVELOPMENT

**Primary file:** `code/stir-bayes-load1p6.Rmd`  
**Stan model (current):** `code/m_stir_mogp_v1p6.stan`  
**Current version:** **1.6**

| Version | File    | Description of Model Features | Missing Features / Planned Additions |
| ------- | ------- | ----------------------------- | ------------------------------------ |
| 1.0 | load1p0 | Posterior load computation using posterior C and posterior V, z‑score back‑transformation, per‑analyte load distributions, HPDI summaries | Visualization functions, alternative unit scaling, integration of concentration censoring, inclusion of treatment or year pooling in load summaries |
| 1.1 | load1p1 | Integrates outflow volume model into concentration model for more accurate load estimates. Imputes missing inflow concentrations. | Non‑centered version, consider Gaussian process by incorporating year and/or irrigation number |
| 1.2 | load1p2 | Adds MVN priors and non‑centered parameterization. | Consider Gaussian process by incorporating year and/or irrigation number, add annual load curve visualization |
| 1.3 | load1p3 | Adds single Gaussian process over year and associated plots. | Upgrade to multi‑output GP to share information across analytes |
| 1.4 | load1p4 | Uses multi‑output GP via Stan code. | Clarify/standardize outputs, finalize annual aggregation and plotting conventions |
| 1.5 | load1p5 | Removes rethinking code, standardizes Stan workflow, adds annual load estimates and missing‑year imputation. | Explicitly track “observed vs modeled” annual summaries and harmonize schema across workflows |
| 1.6 | load1p6 | Production version. Multi‑output GP across analytes × years, standardized annual load summaries, and an “observed + modeled” annual output aligned to the ML annual-summary schema. | **Planned extension:** explicit measurement-error submodels for inflow concentration and runoff volume (latent true states); consider event-level censoring; consider additional temporal structure beyond year GP (if warranted) |

**Important clarification (v1.6):** inflow concentration and runoff volume are treated as observed covariates (no latent “true” C/V measurement-error submodels yet). Posterior uncertainty is propagated from the statistical model and missing-data structure, not from explicit covariate measurement-error likelihoods.

---

## Concentration Models – COMPLETED (superseded by Load model family)

**File(s):** `code/stir-bayes-conc1p0.Rmd`, `code/stir-bayes-conc1p1.Rmd`  
**Current status:** **Historical / reference only**

| Version | File    | Description of Model Features | Missing Features / Planned Additions |
| ------- | ------- | ----------------------------- | ------------------------------------ |
| 1.0 | conc1p0 | Multi-analyte concentration model, standardized outflow concentration, analyte-specific intercepts and slopes, posterior slope extraction, HPDI/PI summaries and plots | Superseded by integrated load model family |
| 1.1 | conc1p1 | Adds DAG-motivated covariates to isolate direct STIR effect on concentration | Superseded by integrated load model family |

---

## Volume Models – COMPLETED (superseded by Load model family)

**File:** `code/stir-bayes-vol1p0.Rmd` *(historical; volume is embedded in load models after 1.1)*  
**Current status:** **Historical / reference only**

| Version | File     | Description of Model Features | Missing Features / Planned Additions |
| ------- | -------- | ----------------------------- | ------------------------------------ |
| 1.0 | vol1p0 | Simple regression of standardized volume on seasonal STIR, missing-data imputation enabled, posterior slope density visualization | Superseded by integrated load model family |

---

## Outputs (Bayesian)

Primary Bayesian annual summaries in `out/`:

- `out/annual_load_summary_bayes_v1p6.csv` (modeled-only summary)
- `out/annual_load_summary_bayes_plus_observed_v1p6.csv` (observed + modeled summary; schema harmonized with ML)

Output field definitions and conventions:
- `docs/README_pipeline_final_outputs.md`

---

# TODOs (high-level)

## Load model family
- [ ] If pursued, implement explicit measurement-error submodels for inflow concentration and runoff volume (latent true states with priors on measurement error).
- [ ] If needed, incorporate event-level censoring / lab reporting limits in concentration likelihood.
- [ ] Continue refining “observed + modeled” annual summaries as the canonical comparison product for Chapter 3.

## Documentation
- [ ] Keep `docs/README_bayes_methods.md` synchronized with what is *actually implemented* in the Stan code (avoid describing planned extensions as current features).

---

# Notes for Future Updates

- Add a new row under each model family every time you create a new `.Rmd` version.
- Keep descriptions short (1–2 lines).
- Avoid equations; focus on what the model *does* and what remains out of scope.
- Link new model files directly in the table when they are added to `code/`.

