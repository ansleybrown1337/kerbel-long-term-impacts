# Bayesian Modeling Framework for Kerbel Long‑Term Impacts on Edge‑of‑Field Water Quality

This document provides full technical documentation for the **hierarchical Bayesian modeling framework** used to analyze long‑term (2011–2025) edge‑of‑field (EoF) water‑quality responses to tillage disturbance (STIR) at the Kerbel agricultural research site. It is intended to serve as a **methods appendix–level reference** for manuscripts and dissertation chapters.

The Bayesian framework is the **primary inferential approach** in this project and is explicitly designed for causal interpretation, uncertainty propagation, and temporal inference.

---

## 1. Modeling Objectives

The Bayesian models are constructed to:

1. Quantify **analyte‑specific causal effects** of tillage intensity (STIR) on runoff concentration, volume, and load
2. Propagate **measurement uncertainty** in concentration and volume through to annual loads
3. Separate **management signals** from sampler, laboratory, and infrastructure artifacts
4. Share information across analytes using **hierarchical partial pooling**
5. Capture **multi‑year temporal persistence** and correlation across analytes
6. Enable principled **imputation of missing years** with uncertainty that grows appropriately

---

## 2. Causal Structure and DAG

The modeling framework is grounded in an explicit causal directed acyclic graph (DAG), developed using agronomic knowledge and monitoring design constraints.

### Key causal assumptions

- Tillage intensity (STIR) influences **soil disturbance**, which affects runoff generation and particulate transport
- Runoff volume mediates a substantial portion of load variability
- Inflow concentration affects outflow concentration but is not affected by STIR at the field edge
- Measurement method, sampler, flume, and laboratory introduce systematic but non‑causal variation

### Conceptual DAG

Nodes include:

- STIR (seasonal and cumulative)
- Runoff volume
- Inflow concentration
- Outflow concentration
- Analyte identity
- Year (latent environmental state)
- Sampler, flume, laboratory, and replication

A graphical representation of the DAG is shown below.

![Causal DAG](../figs/dagitty-model.jpeg)

---

## 3. Observation Model

For analyte \(a\), event \(i\), the observed outflow concentration is modeled as:

\[
C_{i,a} \sim \text{Normal}(\mu_{i,a},\; \sigma_a)
\]

where \(\sigma_a\) is an analyte‑specific residual standard deviation.

The linear predictor is:

\[
\mu_{i,a} = \alpha_a
+ \beta_{\text{STIR},a} \cdot \text{STIR}_{i}
+ \beta_{\text{inflow},a} \cdot \log(\tilde{C}^{\text{in}}_{i})
+ \beta_{\text{vol},a} \cdot \log(\tilde{V}_{i})
+ \mathbf{Z}_{i} \boldsymbol{\gamma}_a
+ f_{a}(\text{Year}_i)
\]

where:

- \(\tilde{C}^{\text{in}}_{i}\) is the latent inflow concentration
- \(\tilde{V}_{i}\) is the latent runoff volume
- \(\mathbf{Z}_{i}\) is a design matrix of categorical effects (sampler, flume, lab, replication)
- \(f_a(\cdot)\) is a year‑specific latent deviation modeled via a Gaussian process

---

## 4. Measurement Error Submodels

Observed inflow concentration and runoff volume are treated as noisy measurements of latent true values:

\[
\log(C^{\text{in,obs}}_{i}) \sim \text{Normal}(\log(\tilde{C}^{\text{in}}_{i}),\; \tau_{C})
\]

\[
\log(V^{\text{obs}}_{i}) \sim \text{Normal}(\log(\tilde{V}_{i}),\; \tau_{V})
\]

This structure ensures uncertainty in measurements propagates forward into concentration and load estimates rather than being implicitly ignored.

---

## 5. Hierarchical Analyte Structure

Regression coefficients are modeled hierarchically across analytes:

\[
\boldsymbol{\beta}_a \sim \text{MVN}(\boldsymbol{\mu}_{\beta},\; \Sigma_{\beta})
\]

where:

- \(\boldsymbol{\beta}_a\) includes STIR, inflow, and volume effects for analyte \(a\)
- \(\Sigma_{\beta}\) captures cross‑analyte covariance

This allows strong analytes (e.g., TSS, TP) to inform weaker ones without forcing identical behavior.

---

## 6. Random Effects

Random intercepts are included for multiple non‑causal sources of variation:

\[
\alpha_{a,j} = \alpha_a + u_{a,j}
\]

with

\[
\mathbf{u}_{a} \sim \text{MVN}(\mathbf{0},\; \Sigma_{u})
\]

Random effects include:

- Replication (block)
- Sampler type
- Flume type
- Laboratory

Each is estimated with analyte‑level covariance.

---

## 7. Temporal Structure: Multi‑Output Gaussian Process

Year‑to‑year latent deviations are modeled using a separable multi‑output Gaussian process:

\[
\mathbf{f}(y) \sim \text{GP}(\mathbf{0},\; \Sigma_A \otimes K_{\text{year}})
\]

where:

- \(\Sigma_A\) captures covariance across analytes
- \(K_{\text{year}}\) is a squared‑exponential kernel over years

This allows analytes to share temporal information and enables principled interpolation across missing years.

---

## 8. Prior Distributions

Key priors include:

- Regression coefficients:
\[
\boldsymbol{\mu}_{\beta} \sim \text{Normal}(0,\; 1)
\]

- Covariance matrices:
\[
\Sigma \sim \text{LKJcorr}(2) \cdot \text{diag}(\sigma)
\]

- Standard deviations:
\[
\sigma \sim \text{Half‑Normal}(0,\; 1)
\]

These priors are weakly informative and scaled to the log‑transformed data.

---

## 9. Posterior Load Generation

Event‑scale loads are computed as:

\[
L_{i,a} = C_{i,a} \times V_{i}
\]

Annual loads are obtained by summing posterior predictive draws within year × treatment × analyte groups, preserving uncertainty.

This enables direct comparison between modeled and observed annual loads.

---

## 10. Computation and Inference

- Models are implemented in **Stan** and compiled via `cmdstanr`
- Posterior sampling is performed using **Hamiltonian Monte Carlo (HMC)** with the No‑U‑Turn Sampler (NUTS)
- Typical fits involve ~27,000 parameters and ~15,000 observations
- Convergence is assessed via \(\hat{R}\), effective sample size, and trace diagnostics

Model versions are tracked in `docs/bayes-model_versions.md`.

---

## 11. Relationship to Machine Learning Analysis

Bayesian models provide:

- Explicit causal interpretation
- Time‑evolving uncertainty
- Principled handling of missing data

Machine‑learning models are used strictly as a **benchmark for pattern learning**, not causal inference.

---

## 12. Graphical Model Results

Representative Bayesian results are shown below.

### STIR effects on runoff volume
![STIR volume effects](../figs/1p6_post_STIR_effect_on_volume.jpeg)

### Annual load curves conditioned on STIR
![STIR load curves](../figs/load1p6_STIR_load_curves.jpeg)

### Year‑to‑year latent deviations
![Latent deviations](../figs/yearly_latent_deviations_v1p6.jpeg)

### Year covariance structure
![Year covariance](../figs/year_covariance_gp_v1p6.jpeg)

---

## 13. References

Harmel, R.D. et al. (2006). Cumulative uncertainty in measured streamflow and water‑quality data for small watersheds. *Transactions of the ASABE*, 49(3), 689–701.

USDA‑NRCS (2023). Revised Universal Soil Loss Equation (RUSLE2).

