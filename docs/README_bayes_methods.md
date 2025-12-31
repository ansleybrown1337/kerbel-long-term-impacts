# Bayesian Modeling Framework for Kerbel Long-Term Impacts on Edge-of-Field Water Quality

This document provides full technical documentation for the **hierarchical Bayesian modeling framework** used to analyze long-term (2011–2025) edge-of-field (EoF) water-quality responses to tillage disturbance (STIR) at the Kerbel agricultural research site. It is intended to serve as a methods appendix–level reference for manuscripts and dissertation chapters.

The Bayesian framework is the primary inferential approach in this project and is explicitly designed for causal interpretation, uncertainty propagation, and temporal inference.

---

## 1. Modeling objectives

The Bayesian models are constructed to:

1. Quantify analyte-specific causal effects of tillage intensity (STIR) on runoff concentration, volume, and load  
2. Propagate uncertainty from the statistical model through to annual loads  
3. Separate management signals from sampler, laboratory, and infrastructure artifacts  
4. Share information across analytes using hierarchical partial pooling  
5. Capture multi-year temporal persistence and correlation across analytes  
6. Enable principled imputation of missing years with uncertainty that grows appropriately  

---

## 2. Causal structure and DAG

The modeling framework is grounded in an explicit causal directed acyclic graph (DAG), developed using agronomic knowledge and monitoring design constraints.

### Key causal assumptions

- Tillage intensity (STIR) influences soil disturbance and surface condition, which affects runoff generation and particulate transport.  
- Runoff volume mediates a substantial portion of load variability.  
- Inflow concentration affects outflow concentration but is not affected by STIR at the field edge (conditional on design and timing).  
- Measurement method, sampler, flume, and laboratory introduce systematic but non-causal variation.  

### Conceptual DAG

![Causal DAG](../figs/dagitty-model.jpeg)

---

## 3. Observation model (outflow concentration)

For analyte $a$ and event $i$, the observed outflow concentration is modeled as:

$$
C_{i,a} \sim \mathrm{Normal}(\mu_{i,a},\, \sigma_a)
$$

where $\sigma_a$ is an analyte-specific residual standard deviation.

### Linear predictor

The linear predictor is specified as:

```math
\mu_{i,a}
=
\alpha_a
+ \beta_{\mathrm{STIR},a} \, \mathrm{STIR}_i
+ \beta_{\mathrm{inflow},a} \, \log\left(C^{\mathrm{in}}_{i}\right)
+ \beta_{\mathrm{vol},a} \, \log\left(V_{i}\right)
+ \mathbf{Z}_{i} \, \boldsymbol{\gamma}_a
+ f_a\left(\mathrm{Year}_i\right)
```

where:

- $C^{\mathrm{in}}_{i}$ is the observed inflow concentration for event $i$  
- $V_{i}$ is the observed runoff volume for event $i$  
- $\mathbf{Z}_{i}$ is a design matrix for categorical factors (sampler, flume, lab, replication)  
- $\boldsymbol{\gamma}_a$ are analyte-specific coefficients for those factors  
- $f_a(\cdot)$ is a year-specific latent deviation for analyte $a$  

---

## 4. Treatment of runoff volume and inflow concentration

Runoff volume and inflow concentration are treated as **observed covariates** in the current model implementation (v1p6). Measured values of runoff volume and inflow concentration enter the linear predictors directly after transformation (log or log1p, as appropriate).

No explicit measurement-error submodels are specified for these covariates in v1p6. Consequently, uncertainty in volume and inflow concentration measurements is not propagated forward into concentration or load uncertainty. This design choice reflects data availability and a desire to maintain comparability with the machine-learning models evaluated in Chapter 3.

The inclusion of latent “true” runoff volume and inflow concentration states, along with corresponding measurement-error models, is a planned extension of this framework but is not implemented in the current version.

---

## 5. Hierarchical analyte structure (partial pooling)

Regression coefficients are modeled hierarchically across analytes:

$$
\boldsymbol{\beta}_a \sim \mathrm{MVN}\left(\boldsymbol{\mu}_{\beta},\, \sigma_{\beta}\right)
$$

where:

- **$\beta_a$** includes the STIR, inflow, and volume effects for analyte *a*
- **$\sigma_{\beta}$** captures cross-analyte covariance across analytes



---

## 6. Random effects for monitoring artifacts

Random intercepts are included for multiple non-causal sources of variation. For a factor level $j$ (for example laboratory, sampler, flume, or replication), we write:

$$
\alpha_{a,j} = \alpha_a + u_{a,j}
$$

with:

$$
\mathbf{u}_a \sim \mathrm{MVN}\left(\mathbf{0},\, \sigma_u\right)
$$

This formulation supports analyte-specific sensitivity to sampling and infrastructure differences and allows correlated deviations across analytes.

---

## 7. Temporal structure: multi-output Gaussian process

Year-to-year latent deviations are modeled using a separable multi-output Gaussian process:

$$
\mathbf{f}(y) \sim \mathrm{GP}\left(\mathbf{0},\, \Sigma_A \otimes K_{\mathrm{year}}\right)
$$

A common kernel choice is the squared-exponential:

$$
K_{\mathrm{year}}(y, y') = \eta^2 \, \exp\left(-\frac{(y-y')^2}{2\ell^2}\right)
$$

This structure allows analytes to share temporal information and enables principled interpolation across missing years.

---

## 8. Prior distributions

Priors are weakly informative and scaled to the log-transformed data.

### Regression coefficients

$$
\boldsymbol{\mu}_{\beta} \sim \mathrm{Normal}(0,\, 1)
$$

### Standard deviations

$$
\sigma \sim \mathrm{Half\text{-}Normal}(0,\, 1)
$$

### Correlation and covariance structure

$$
\Sigma = \mathrm{diag}(\boldsymbol{\sigma}) \, \Omega \, \mathrm{diag}(\boldsymbol{\sigma}),
\qquad
\Omega \sim \mathrm{LKJcorr}(2)
$$

---

## 9. Posterior load generation

Event-scale loads are computed as:

$$
L_{i,a} = C_{i,a} \, V_i
$$

Annual loads are obtained by summing posterior predictive draws within year × treatment × analyte groups:

$$
L_{y,t,a} = \sum_{i \in (y,t,a)} L_{i,a}
$$

---

## 10. Computation and inference

- Models are implemented in **Stan** and compiled via `cmdstanr`.  
- Posterior sampling is performed using **Hamiltonian Monte Carlo (HMC)** with the **No-U-Turn Sampler (NUTS)**.  
- Typical fits involve tens of thousands of parameters and thousands to tens of thousands of observations.  
- Convergence is assessed using $\hat{R}$, effective sample size, trace diagnostics, and posterior predictive checks.  

Model versions are tracked in `docs/bayes-model_versions.md`.

---

## 11. Relationship to the machine-learning analysis

Bayesian models provide explicit causal interpretability, time-evolving uncertainty, and principled handling of missing data. Machine-learning models are used strictly as a benchmark for pattern learning rather than causal inference.

---

## 12. Graphical results (representative)

### STIR effects on runoff volume
![STIR volume effects](../figs/1p6_post_STIR_effect_on_volume.jpeg)

### Annual load curves conditioned on STIR
![STIR load curves](../figs/load1p6_STIR_load_curves.jpeg)

### Year-to-year latent deviations
![Latent deviations](../figs/yearly_latent_deviations_v1p6.jpeg)

### Year covariance structure
![Year covariance](../figs/year_covariance_gp_v1p6.jpeg)

---

## 13. References

Harmel, R.D., Cooper, R.J., Slade, R.M., Haney, R.L., & Arnold, J.G. (2006). Cumulative uncertainty in measured streamflow and water-quality data for small watersheds. *Transactions of the ASABE*, 49(3), 689–701.

USDA-NRCS. (2023). Revised Universal Soil Loss Equation (RUSLE2) documentation and methodology.
