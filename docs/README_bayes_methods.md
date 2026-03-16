# Bayesian STIR–Water Quality Model  
**Version 1p8 (Current Analysis Model)**  
Kerbel Long-Term Tillage Impacts Study, CSU ARDEC (Fort Collins, Colorado)  
A.J. Brown  

---

## Overview

This document provides technical documentation for the **hierarchical Bayesian modeling framework** used to analyze long-term (2011–2025) edge-of-field (EoF) water-quality responses to tillage disturbance (STIR) at the Kerbel agricultural research site, a long-term tillage systems experiment located at Colorado State University’s Agricultural Research, Development, and Education Center (ARDEC) in Fort Collins, Colorado.

The model supports research objectives focused on quantifying the effect of tillage intensity on runoff concentration, runoff volume, and annual nutrient and sediment loads, while explicitly accounting for temporal persistence, cross-analyte dependence, structural field design variables, missing-data mechanisms, residue effects, inflow predictors, and censoring of concentration observations below reporting limits.

Inference is conducted in **Stan** using **Hamiltonian Monte Carlo (HMC)** with the **No-U-Turn Sampler (NUTS)**.

**Stan model:** `code/m_stir_mogp_v1p8.stan`  
**Driver analysis:** `code/stir-bayes-load1p8_nonneg.Rmd`  
**Primary dataset:** `out/wq_cleaned.csv`  

---

## Model summary

| Component | Description |
| --- | --- |
| Concentration model | Multi-analyte hierarchical regression for latent true concentration with analyte-specific coefficients and measurement error |
| Volume model | Hierarchical regression for latent true outflow volume with measurement error |
| Residue model | Logit-normal submodel for residue proportion, including missing-data imputation and previous-crop effects |
| Missing predictors | In-model imputation for missing inflow concentration (`CIN`), inflow volume (`VIN`), and residue (`RES`) |
| Censoring | Left-censored normal likelihood for concentration observations below reporting limits |
| Temporal structure | Multi-output Gaussian process over year with shared cross-analyte covariance |
| Latent variables | `C_true` and `V_true` separate underlying event states from observation error; imputed predictors propagate covariate uncertainty |
| Outputs | Posterior distributions for concentration, volume, and annual loads, plus posterior predictive replicates |
| Inference | Hamiltonian Monte Carlo in Stan using NUTS |

---

# Data and scaling (as implemented in Stan)

Both primary outcomes are modeled on the **z-standardized scale** as supplied to Stan:

- `C[i]`: outflow concentration, z scale  
- `VOL[i]`: outflow volume, z scale  

The model introduces row-level latent “true” values, `C_true` and `V_true`, and applies observation likelihoods only to rows with observed outcomes. Missing rows are inferred through the latent process. Concentration rows flagged as censored are handled through a left-censored likelihood using reporting-limit values transformed to the same z scale as `C`.

Residue enters the model on the **proportion scale** in `(0,1)` and is modeled through a separate logit-normal regression with in-model imputation for missing residue rows.

---

# Model specification (matches `m_stir_mogp_v1p8.stan`)

## 1) Concentration model (multi-analyte, latent true concentration)

### Latent process

For row `i` with analyte `a = A[i]` and year index `y = Y[i]`:

```math
\mu_{C,i} =
\alpha_{a}
+ \beta_{\mathrm{stir},a}\,\mathrm{STIR}_i
+ \beta_{\mathrm{cin},a}\,\mathrm{CIN}^{\ast}_i
+ \beta_{\mathrm{vol}}\,V_{\mathrm{true},i}
+ \beta_{\mathrm{irr},a}\,\mathrm{IRR}_z[i]
+ \beta_{\mathrm{dup},a}\,\mathrm{DUP}_i
+ \beta_{\mathrm{res},a}\,\mathrm{RES}^{\ast}_i
+ \gamma_{a,\mathrm{Cr}[i]}
+ \gamma_{a,\mathrm{B}[i]}
+ \gamma_{a,\mathrm{S}[i]}
+ \gamma_{a,\mathrm{Fu}[i]}
+ f_{y,a}
```

Row-level latent truth is parameterized non-centrally:

```math
C_{\mathrm{true},i} = \mu_{C,i} + \sigma_{\mathrm{analyte}}\,z_{C,i},
\qquad z_{C,i} \sim \mathrm{Normal}(0,1)
```

### Observation model

For observed, uncensored concentration rows:

```math
C_i \sim \mathrm{Normal}\!\left(C_{\mathrm{true},i},\ \sigma_{\mathrm{C,obs},a}\right)
```

For observed, left-censored concentration rows, for example nondetects or values below reporting limit:

```math
C_i < L_i,
\qquad
\Pr(C_i < L_i) = \Phi\!\left(\frac{L_i - C_{\mathrm{true},i}}{\sigma_{\mathrm{C,obs},a}}\right)
```

which is implemented in Stan as:

```math
\log \Pr(C_i < L_i)
=
\mathrm{normal\_lcdf}\!\left(L_i \mid C_{\mathrm{true},i}, \sigma_{\mathrm{C,obs},a}\right)
```

where `L_i` is the row-specific censoring threshold, `C_cens_limit[i]`, on the same z scale as `C`.

### Parameter meanings

- $\alpha_a$: analyte-specific intercept, hierarchical multivariate prior  
- $\beta_{\mathrm{stir},a}$: analyte-specific slope on $\mathrm{STIR}_i$  
- $\beta_{\mathrm{cin},a}$: analyte-specific slope on imputed or observed inflow concentration $\mathrm{CIN}^{\ast}_i$  
- $\beta_{\mathrm{vol}}$: global coupling from latent true volume $V_{\mathrm{true},i}$ to concentration  
- $\beta_{\mathrm{irr},a}$: analyte-specific slope on standardized irrigation-order covariate $\mathrm{IRR}_z$  
- $\beta_{\mathrm{dup},a}$: analyte-specific duplicate indicator effect  
- $\beta_{\mathrm{res},a}$: analyte-specific slope on residue proportion $\mathrm{RES}^{\ast}$  
- $\gamma_{a,\mathrm{Cr}}$: analyte-specific crop-type random effects  
- $\gamma_{a,\mathrm{B}}$: analyte-specific block random effects  
- $\gamma_{a,\mathrm{S}}$: analyte-specific sampler random effects  
- $\gamma_{a,\mathrm{Fu}}$: analyte-specific flume random effects  
- $f_{y,a}$: year-by-analyte Gaussian-process effect  

**Design note:** treatment is not included as a separate causal predictor in v1p8 by design. Tillage disturbance enters through $\mathrm{STIR}_i$.

---

## 2) Volume model (latent true outflow volume)

### Latent process

```math
\mu_{V,i} =
a_V
+ b_V\,\mathrm{STIR}_i
+ \beta_{\mathrm{vin}}\,\mathrm{VIN}^{\ast}_i
+ \beta_{\mathrm{res},V}\,\mathrm{RES}^{\ast}_i
+ \gamma^{(V)}_{\mathrm{Cr}[i]}
```

with non-centered row-level latent truth:

```math
V_{\mathrm{true},i} = \mu_{V,i} + \sigma_V\,z_{V,i},
\qquad z_{V,i} \sim \mathrm{Normal}(0,1)
```

### Observation model

For observed volume rows:

```math
\mathrm{VOL}_i \sim \mathrm{Normal}\!\left(V_{\mathrm{true},i},\ \sigma_{\mathrm{VOL,obs}}\right)
```

### Parameter meanings

- $a_V$: volume intercept, z scale  
- $b_V$: slope on $\mathrm{STIR}_i$ for outflow volume  
- $\beta_{\mathrm{vin}}$: slope of inflow volume predictor $\mathrm{VIN}^{\ast}_i$ on outflow volume  
- $\beta_{\mathrm{res},V}$: slope of residue proportion on outflow volume  
- $\gamma^{(V)}_{\mathrm{Cr}}$: crop-type effect on volume  
- $\sigma_V$: latent process standard deviation for volume  
- $\sigma_{\mathrm{VOL,obs}}$: volume observation standard deviation  

**Scaling note:** volume is modeled on the **z scale** in Stan, not log scale, consistent with the upstream pipeline.

---

## 3) Residue model (logit-normal regression with missing-data imputation)

Residue is treated as a proportion in `(0,1)` and modeled on the logit scale.

### Linear predictor

```math
\mu_{\mathrm{res},i} =
\mathrm{logit}(\mathrm{res\_base})
+ b_{\mathrm{res,stir}}\,\mathrm{STIR}_i
+ \gamma_{\mathrm{PrevCr}[i]}^{(\mathrm{res})}
```

where `PrevCr` is the previous-year crop indicator used as a parent of residue in the model.

### Observation and imputation

Observed residue rows satisfy:

```math
\mathrm{logit}(\mathrm{RES}_i)
\sim
\mathrm{Normal}\!\left(\mu_{\mathrm{res},i},\ \sigma_{\mathrm{res,obs}}\right)
```

Missing residue rows are imputed as parameters $\mathrm{RES}_{\mathrm{miss},j} \in (10^{-4}, 1 - 10^{-4})$ with:

```math
\mathrm{logit}(\mathrm{RES}_{\mathrm{miss},j})
\sim
\mathrm{Normal}\!\left(\mu_{\mathrm{res},\,\mathrm{idx}(j)},\ \sigma_{\mathrm{res,obs}}\right)
```

A merged residue vector is then used as a predictor in the concentration and volume submodels:

```math
\mathrm{RES}^{\ast}_i =
\begin{cases}
\mathrm{RES}_i, & \text{if observed} \\
\mathrm{RES}_{\mathrm{miss},j}, & \text{if missing, mapped by index}
\end{cases}
```

### Parameter meanings

- $\mathrm{res\_base}$: baseline residue proportion  
- $b_{\mathrm{res,stir}}$: STIR effect on residue proportion  
- $\gamma^{(\mathrm{res})}_{\mathrm{PrevCr}}$: previous-crop effect on residue  
- $\sigma_{\mathrm{res,obs}}$: logit-scale residue observation or process standard deviation  

---

## 4) Inflow predictor models for missing-data integration

### Inflow concentration (`CIN`)

Missing inflow concentration values are imputed in-model via:

```math
\mathrm{CIN}_{\mathrm{impute}} \sim \mathrm{Normal}(0,1)
```

A merged inflow concentration vector is then used in the concentration model:

```math
\mathrm{CIN}^{\ast}_i =
\begin{cases}
\mathrm{CIN}_i, & \text{if observed} \\
\mathrm{CIN}_{\mathrm{impute},j}, & \text{if missing, mapped by index}
\end{cases}
```

### Inflow volume (`VIN`)

Missing inflow volume values are likewise imputed in-model:

```math
\mathrm{VIN}_{\mathrm{impute}} \sim \mathrm{Normal}(0,1)
```

with merged predictor:

```math
\mathrm{VIN}^{\ast}_i =
\begin{cases}
\mathrm{VIN}_i, & \text{if observed} \\
\mathrm{VIN}_{\mathrm{impute},j}, & \text{if missing, mapped by index}
\end{cases}
```

---

# Temporal structure: multi-output Gaussian process (as implemented)

Year-by-analyte Gaussian-process effects $f_{y,a}$ are constructed from a separable covariance:

```math
\mathrm{vec}(F_{\mathrm{year}}) \sim \mathrm{Normal}\!\left(0,\ \Sigma_A \otimes K_{\mathrm{year}}\right)
```

Stan uses `cov_GPL2(D, etasq_year, rhosq_year, delta)` to build $K_{\mathrm{year}}$ from the year-distance matrix `D`:

```math
K_{\mathrm{year},ij} = \eta^2 \exp\!\left(-\rho^2 d_{ij}^2\right),
\qquad K_{\mathrm{year},ii} \leftarrow K_{\mathrm{year},ii} + \delta
```

with $\delta = 0.01$ in the Stan implementation.

Cross-analyte covariance is represented via a Cholesky factor:

```math
\Sigma_A = L_{Agp} L_{Agp}^{\top},
\qquad L_{Agp} = \mathrm{diag}(\sigma_{Agp})\,L_{\mathrm{corr},Agp}
```

and the Gaussian-process realization is built non-centrally:

```math
F_{\mathrm{year}} = L_t Z_{gp} L_{Agp}^{\top}
```

where `L_t` is the Cholesky factor of $K_{\mathrm{year}}$ and `Z_gp` is standard normal.

This structure allows analytes to share temporal information while retaining analyte-specific deviations in year effects.

---

# Load propagation (post-processing)

Event loads are computed from posterior draws after back-transformation of concentration and volume from their modeled scales to original units in post-processing, as implemented in `stir-bayes-load1p8_nonneg.Rmd`. At the event level for analyte `a`:

```math
L_{i,a} = C_{i,a}\,V_i
```

Annual loads are then computed as:

```math
L_{y,a} = \sum_{i \in y} L_{i,a}
```

Uncertainty is propagated by applying these calculations to each posterior draw.

---

# Missingness and censoring summary (as implemented)

- Missing `C` rows: excluded from the direct observation likelihood and inferred via $C_{\mathrm{true}}$  
- Censored `C` rows: included via a left-censored normal likelihood using `C_cens_limit`  
- Missing `VOL` rows: excluded from the direct observation likelihood and inferred via $V_{\mathrm{true}}$  
- Missing `RES` rows: imputed through the residue submodel into $\mathrm{RES}^{\ast}$  
- Missing `CIN` rows: imputed via $\mathrm{CIN}_{\mathrm{impute}}$ into $\mathrm{CIN}^{\ast}$  
- Missing `VIN` rows: imputed via $\mathrm{VIN}_{\mathrm{impute}}$ into $\mathrm{VIN}^{\ast}$  

---

# Generated quantities

The Stan model returns posterior predictive replicates for:

- `VOL_rep[i]`: replicated outflow volume  
- `C_rep[i]`: replicated concentration  
- `RES_rep01[i]`: replicated residue proportion  

These quantities support posterior predictive checking of measurement-scale fit.

---

**Version 1p8** is the current Bayesian model implementation for Kerbel STIR × EoF water-quality analyses. Relative to 1p7p5, it adds explicit concentration censoring, inflow-volume imputation, and the current residue submodel structure while preserving the stabilized latent-state formulation and multi-output Gaussian process framework.
