# Model Versioning Summary

**Kerbel Long-Term Tillage Impacts Project**
**AJ Brown**

This file tracks the structure and development stage of all Bayesian model families used in the STIR–WQ analysis workflow. Each model family has its own Rmd file and evolves independently. This table format is designed for fast updates, minimal text, and clear tracking of model features over time.

---
## Load Models (Vol + Concentration) - IN PROGRESS

**File:** `code/stir-bayes-load1p0.Rmd`
**Current Version:** **1.4**

| Version | File    | Description of Model Features                                                                                                             | Missing Features / Planned Additions                                                                                                                |
| ------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | load1p0 | Posterior load computation using posterior C and posterior V, z-score back-transformation, per-analyte load distributions, HPDI summaries | Visualization functions, alternative unit scaling, integration of concentration censoring, inclusion of treatment or year pooling in load summaries |
| 1.1     | load1p1 | Same as 1.0 but now includes outflow volume model integrated into concentration model for more accurate load estimates. Imputes missing inflow concentrations. | Create non-centered version. Consider Gaussian processes by incorporating year and irrigation number |
| 1.2     | load1p2 | Same as 1.1 but uses MVN priors and non-centered parameterization| Consider Gaussian processes by incorporating year and/or irrigation number; plot relationship over each year for 1 analyte for load curves |
| 1.3     | load1p3 | Same as 1.2 but uses single Gaussin process for year and now graphs accordingly| we need a multi-output GP next for a per-analyte year relationship I think. |
| 1.4     | load1p4 | Uses multi-output GP via stan code| consider treating C as a latent variable? |

---

## Concentration Models - DONE

**File:** `code/stir-bayes-conc1p0.Rmd`
**Current Version:** **FINAL 1.1**

| Version | File    | Description of Model Features                                                                                                                                     | Missing Features / Planned Additions                                                                         |
| ------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1.0     | conc1p0 | Multianalyte concentration model, standardized OUT concentration, analyte-specific intercepts and slopes, posterior slope extraction, HPDI/PI summaries and plots | |
| 1.1     | conc1p1 | Same as 1.0 but now includes all DAG dependencies to isolate direct effect of STIR on concentration. | Consider making nested irrigation variable with temporal dependency; integrate MVN priors; non centered version of model |

---

## Volume Models - DONE

**File:** `code/stir-bayes-vold1p0.Rmd`
**Current Version:** **FINAL 1.0**

| Version | File     | Description of Model Features                                                                                                            | Missing Features / Planned Additions                                                                                                                    |
| ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | embedded | Simple linear regression of standardized volume on seasonal STIR, missing-data imputation enabled, posterior slope density visualization | Treatment-level or year-level pooling, residual correlation modeling, alternative predictors (e.g., cumulative STIR), integration into full joint model |

---


# TODOs

* Concentration Model
  - [ ] Add DAG conditional dependencies for STIR direct effects on concentration
  - [ ] Create nested year-irrigation var for partial pooling via gaussian kernel process

* Volume Model
  - **Done:** DAG implies no conditional dependencies to characterize the direct relationship between STIR and volume

* Load Model
  - [ ] ensure outflow volume model is properly integrated into conc. model
  - [ ] Finalize concentration model updates before adding new features to load model
---

# Notes for Future Updates

* Add a new row under each model family every time you create a new `.Rmd` version.
* Keep descriptions short (1–2 lines).
* Avoid equations; focus on what the model *does* and what remains out of scope.
* Link new model files directly in the table.

---

If you want, I can also generate:

* A **CHANGELOG.md** template
* A **Makefile** or R script that auto-renders all models and stores outputs consistently
* A **README badge system** showing which models are “complete”, “draft”, or “in development”
