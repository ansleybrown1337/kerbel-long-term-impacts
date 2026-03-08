# Model Versioning Summary

**Kerbel Long-Term Tillage Impacts Project**  
**AJ Brown**

This document tracks the structure and convergence status of all Bayesian load-model versions used in the STIR–water-quality analysis workflow. Each version corresponds to a specific `.Rmd` driver file and Stan implementation.

The tables below document what was implemented and whether the model converged under full diagnostic criteria (R̂ ≈ 1, no divergences, acceptable E-BFMI, no persistent treedepth saturation).

---

## Load Models (Vol + Concentration) – CURRENT

**Current selected model:** `load1p8`  
**Stan file:** `code/m_stir_mogp_v1p8.stan`  
**Driver file:** `code/stir-bayes-load1p8.Rmd`  
**Status:** **CURRENT – Converged and selected for inference**

| Version    | File            | Key Structural Features                                                                 | Convergence Status | Notes |
|------------|----------------|------------------------------------------------------------------------------------------|--------------------|-------|
| 1.0        | load1p0        | Posterior load computation from separate C and V models                                 | Converged          | Initial integrated load workflow |
| 1.1        | load1p1        | Joint concentration–volume structure; inflow imputation                                 | Converged          | First unified Vol + C model |
| 1.2        | load1p2        | MVN priors; non-centered parameterization                                               | Converged          | Improved sampling geometry |
| 1.3        | load1p3        | Single-output Gaussian process over year                                                | Converged          | Temporal smoothing introduced |
| 1.4        | load1p4        | Multi-output GP across analytes                                                         | Converged          | Cross-analyte temporal structure |
| 1.5        | load1p5        | CmdStan standardization; annual load summaries                                          | Converged          | Production-ready pipeline |
| 1.6        | load1p6        | Stable multi-output GP baseline                                                         | Converged          | Baseline reference model |
| 1p7        | load1p7        | Introduced latent true states for volume and inflow; expanded hierarchy                 | **Did NOT converge** | Severe divergences and treedepth saturation |
| 1p7p1      | load1p7p1      | Simplified latent structure; retained crop and residue                                  | **Did NOT converge** | Geometry instability persisted |
| 1p7p2      | load1p7p2      | Additional reparameterization attempts                                                  | **Did NOT converge** | Poor mixing; E-BFMI failures |
| 1p7p3      | load1p7p3      | Refined covariance and hierarchical structure                                           | **Did NOT converge** | Persistent divergences |
| 1p7p4      | load1p7p4      | Pre-final restructuring before full stabilization                                       | **Did NOT converge** | Parameterization still unstable |
| 1p7p5      | load1p7p5      | Finalized latent-state formulation; stabilized non-centered parameterization; full MOGP; includes analyte-specific crop, residue, and irrigation effects | Converged          | Previous selected model |
| **1p8**    | **load1p8**    | Adds explicit concentration censoring via left-censored likelihood, inflow-volume (`VIN`) imputation, updated residue submodel using previous crop, and retained stabilized latent-state + MOGP structure | **Converged**      | Current selected model |

---

### Clarification on 1p7 Series

Versions **1p7 through 1p7p4** introduced expanded latent-state structures and additional hierarchy but failed to satisfy convergence diagnostics. These models are retained for record-keeping but are **not used for inference or reporting**.

Version **1p7p5** resolved prior geometric pathologies through:

- Fully non-centered latent truth parameterization  
- Stabilized multi-output Gaussian process structure  
- Corrected covariance factorization  
- Improved missing-data integration  
- Removal of unstable parameter couplings  

Version **1p8** extends 1p7p5 by preserving the stabilized geometry while adding:

- Explicit concentration censoring via row-level reporting-limit thresholds  
- Missing inflow-volume (`VIN`) imputation  
- Current residue submodel with previous-crop effects on logit residue proportion  
- Posterior predictive quantities for concentration, volume, and residue  

This version satisfies convergence diagnostics and is the official analysis model.

---

## Concentration Models – Historical (Superseded)

These were early standalone concentration models before integration into the joint load framework.

| Version | File    | Description | Status |
|---------|--------|-------------|--------|
| 1.0     | conc1p0 | Multi-analyte concentration regression | Historical |
| 1.1     | conc1p1 | Added DAG-motivated structure | Historical |

---

## Volume Models – Historical (Superseded)

| Version | File   | Description | Status |
|---------|--------|-------------|--------|
| 1.0     | vol1p0 | Standardized volume regression on STIR | Historical |

---

## Notes for Future Updates

- Add a new row for each version created.  
- Record convergence status explicitly.  
- Do not describe planned features as implemented features.  
- The selected model for inference must always be clearly labeled.  
- If a newer version changes the data interface (e.g., censoring inputs, imputation indices), note that explicitly in the table.
