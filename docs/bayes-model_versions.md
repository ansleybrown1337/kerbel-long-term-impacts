# Model Versioning Summary

**Kerbel Long-Term Tillage Impacts Project**  
**AJ Brown**

This file tracks the structure and development stage of all Bayesian model families used in the STIR–WQ analysis workflow. Each model family has its own Rmd file and evolves independently. This table format is designed for fast updates, minimal text, and clear tracking of model features over time.

---

## Load Models (Vol + Concentration) – ACTIVE DEVELOPMENT

Yes, this is a clean place to formalize v1.7 and to be explicit about what changed and why. Below is a **drop-in replacement** for your README section, keeping your existing structure and adding v1.7 in a way that is defensible and honest given the convergence discussion.

I have also added a short **Important clarification (v1.7)** block that mirrors your v1.6 note, so future readers understand exactly what is new and what is still aspirational.

---

### Version history and model evolution

**Primary file:** `code/stir-bayes-load1p7.Rmd`
**Stan model (current):** `code/m_stir_mogp_v1p7.stan`
**Current version:** **1.7**

| Version   | File          | Description of Model Features                                                                                                                            | Convergence Status   | Notes / Differences                                                                                                                                                                            |
| --------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0       | load1p0       | Posterior load computation using posterior C and V, z-score back-transformation, per-analyte load distributions, HPDI summaries                          | Converged            | Initial integrated load workflow                                                                                                                                                               |
| 1.1       | load1p1       | Integrates volume model into concentration model for joint load estimation; imputes missing inflow concentrations                                        | Converged            | First unified Vol + C structure                                                                                                                                                                |
| 1.2       | load1p2       | Adds MVN priors and non-centered parameterization                                                                                                        | Converged            | Improved geometry and sampling stability                                                                                                                                                       |
| 1.3       | load1p3       | Adds single Gaussian process over year                                                                                                                   | Converged            | Temporal smoothing across years                                                                                                                                                                |
| 1.4       | load1p4       | Multi-output Gaussian process (shared temporal structure across analytes)                                                                                | Converged            | Cross-analyte information sharing                                                                                                                                                              |
| 1.5       | load1p5       | Removes rethinking dependency; standardizes full Stan workflow; adds annual load summaries                                                               | Converged            | Reproducible CmdStan pipeline                                                                                                                                                                  |
| 1.6       | load1p6       | Production version. Multi-output GP across analytes × years, standardized annual summaries, harmonized “observed + modeled” outputs                      | Converged            | Baseline for Chapter 2 inference                                                                                                                                                               |
| **1p7**   | **load1p7**   | Introduced latent “true” states for inflow concentration and runoff volume; added crop and residue effects to both volume and concentration models       | **Did NOT converge** | Severe divergences, treedepth saturation, and poor E-BFMI. Latent-state parameterization and hierarchical structure were not computationally stable. Model results are not used for inference. |
| **1p7p1** | **load1p7p1** | Retains crop and residue covariates in both process models; removes latent true states; simplifies parameterization; preserves multi-output GP structure | **Convergence TBD**  | Stabilized version of 1p7. Measurement error propagation removed; crop and residue retained; generative load functions updated to use residue as fixed or observed-average proportion.         |


---

### Important clarification (v1.6)

In v1.6, inflow concentration and runoff volume are treated as observed covariates. Posterior uncertainty reflects model structure, missing-data imputation, and GP uncertainty, but **does not include explicit likelihood-based measurement-error propagation for C or V**.

### Important clarification (v1.7)

In v1.7, inflow concentration and runoff volume are modeled with **latent “true” states** and explicit measurement-error likelihoods. This enables separation of process variability from observation error and allows direct assessment of how measurement uncertainty propagates into annual load estimates and treatment effects.

If convergence criteria are not satisfied, a reduced variant retaining crop and residue effects but excluding latent true states may be used for inference.

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

