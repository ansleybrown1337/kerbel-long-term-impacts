# Bayesian STIR–Water Quality Model  
**Version 1p7p5 (Final Analysis Model)**  
Kerbel Long-Term Tillage Impacts Study, CSU ARDEC (Fort Collins, Colorado)  
A.J. Brown  

---

## Overview

This document provides full technical documentation for the **hierarchical Bayesian modeling framework** used to analyze long-term (2011–2025) edge-of-field (EoF) water-quality responses to tillage disturbance (STIR) at the Kerbel agricultural research site, a long-term tillage systems experiment located at Colorado State University’s Agricultural Research, Development, and Education Center (ARDEC) in Fort Collins, Colorado.

The model supports research objectives focused on quantifying the effect of tillage intensity on runoff concentration, runoff volume, and annual nutrient and sediment loads, while explicitly accounting for temporal persistence, cross-analyte dependence, structural field design variables, and missing-data mechanisms.

**Stan model:** `code/m_stir_mogp_v1p7p5.stan`  
**Driver analysis:** `code/stir-bayes-load1p7p5.Rmd`  
**Primary dataset:** `out/wq_with_stir_by_season.csv`  

---

# Data and scaling (as implemented in Stan)

Both primary outcomes are modeled on the **z-standardized scale** (mean 0, SD 1) as supplied to Stan:

- `C[i]`: outflow concentration (z scale)  
- `VOL[i]`: outflow volume (z scale)  

The model introduces row-level latent “true” values (`C_true`, `V_true`) and applies measurement-error likelihoods only to observed rows (missing rows are inferred through the latent process).

---

# Model specification (matches `m_stir_mogp_v1p7p5.stan`)

## 1) Concentration model (multi-analyte, latent true concentration)

### Latent process

For row $i$ with analyte $a=A[i]$ and year index $y=Y[i]$:

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

Non-centered row-level latent truth:

```math
C_{\mathrm{true},i} = \mu_{C,i} + \sigma_{\mathrm{analyte}}\,z_{C,i},
\qquad z_{C,i}\sim\mathrm{Normal}(0,1)
```

### Observation (measurement) model

For observed concentration rows:

```math
C_i \sim \mathrm{Normal}\!\left(C_{\mathrm{true},i},\ \sigma_{\mathrm{C,obs},a}\right)
```

### Where (parameter meanings)

- $\alpha_a$: analyte-specific intercept (hierarchical MVN)  
- $\beta_{\mathrm{stir},a}$: analyte-specific slope on $\mathrm{STIR}_i$  
- $\beta_{\mathrm{cin},a}$: analyte-specific slope on inflow concentration $\mathrm{CIN}^{\ast}_i$ (z scale)  
- $\beta_{\mathrm{vol}}$: global coupling from latent true volume $V_{\mathrm{true}}$ to concentration  
- $\beta_{\mathrm{irr},a}$: analyte-specific slope on standardized irrigation covariate $\mathrm{IRR}_z$  
- $\beta_{\mathrm{dup},a}$: analyte-specific duplicate indicator effect  
- $\beta_{\mathrm{res},a}$: analyte-specific slope on residue proportion $\mathrm{RES}^{\ast}$  
- $\gamma_{a,\mathrm{Cr}}$: analyte-specific crop-type random effects  
- $\gamma_{a,\mathrm{B}}$: analyte-specific block random effects  
- $\gamma_{a,\mathrm{S}}$: analyte-specific sampler random effects  
- $\gamma_{a,\mathrm{Fu}}$: analyte-specific flume random effects  
- $f_{y,a}$: year-by-analyte Gaussian-process effect (multi-output GP)

**Design note:** Treatment is not included as a separate covariate in 1p7p5 by design. Tillage disturbance enters through $\mathrm{STIR}_i$.

---

## 2) Volume model (latent true volume)

### Latent process

```math
\mu_{V,i} =
a_V
+ b_V\,\mathrm{STIR}_i
+ \beta_{\mathrm{res},V}\,\mathrm{RES}^{\ast}_i
+ \gamma^{(V)}_{\mathrm{Cr}[i]}
```

Non-centered row-level latent truth:

```math
V_{\mathrm{true},i} = \mu_{V,i} + \sigma_V\,z_{V,i},
\qquad z_{V,i}\sim\mathrm{Normal}(0,1)
```

### Observation (measurement) model

For observed volume rows:

```math
\mathrm{VOL}_i \sim \mathrm{Normal}\!\left(V_{\mathrm{true},i},\ \sigma_{\mathrm{VOL,obs}}\right)
```

### Where (parameter meanings)

- $a_V$: volume intercept (z scale)  
- $b_V$: slope on $\mathrm{STIR}_i$ for volume (z scale)  
- $\beta_{\mathrm{res},V}$: slope of residue proportion on volume  
- $\gamma^{(V)}_{\mathrm{Cr}}$: crop-type effect on volume  
- $\sigma_V$: latent process SD for volume  
- $\sigma_{\mathrm{VOL,obs}}$: volume measurement SD (estimated)

**Scaling note:** Volume is modeled on the **z scale** in Stan (not log-scale) per `VOL` and `V_true` implementation.

---

## 3) Residue model (logit-normal regression with missing-data imputation)

Residue is treated as a proportion in $(0,1)$ and modeled on the logit scale.

### Linear predictor (logit scale)

```math
\mu_{\mathrm{res},i} =
\mathrm{logit}(\mathrm{res\_base})
+ b_{\mathrm{res,stir}}\,\mathrm{STIR}_i
+ \gamma^{(\mathrm{res})}_{\mathrm{Cr}[i]}
```

### Observation and imputation

Observed residue rows:

```math
\mathrm{logit}(\mathrm{RES}_i)\sim \mathrm{Normal}\!\left(\mu_{\mathrm{res},i},\ \sigma_{\mathrm{res,obs}}\right)
```

Missing residue rows are imputed as parameters $\mathrm{RES}_{\mathrm{miss},j}\in(10^{-4}, 1-10^{-4})$:

```math
\mathrm{logit}(\mathrm{RES}_{\mathrm{miss},j})\sim \mathrm{Normal}\!\left(\mu_{\mathrm{res},\,\mathrm{idx}(j)},\ \sigma_{\mathrm{res,obs}}\right)
```

A merged residue vector is then used as a predictor in the volume and concentration models:

```math
\mathrm{RES}^{\ast}_i =
\begin{cases}
\mathrm{RES}_i, & \text{if observed}\\
\mathrm{RES}_{\mathrm{miss},j}, & \text{if missing (mapped by index)}
\end{cases}
```

---

## 4) Inflow concentration model for missing data (CIN imputation)

Inflow concentration covariate values (z scale) are imputed in-model for rows where CIN is missing.

Missing CIN values are parameterized as `CIN_impute` with a standard normal prior:

```math
\mathrm{CIN}_{\mathrm{impute}} \sim \mathrm{Normal}(0,1)
```

A merged inflow vector is constructed deterministically and used in the concentration model:

```math
\mathrm{CIN}^{\ast}_i =
\begin{cases}
\mathrm{CIN}_i, & \text{if observed}\\
\mathrm{CIN}_{\mathrm{impute},j}, & \text{if missing (mapped by index)}
\end{cases}
```

---

# Temporal structure: multi-output Gaussian process (as implemented)

Year-by-analyte GP term $f_{y,a}$ is constructed from a separable covariance:

```math
\mathrm{vec}(F_{\mathrm{year}})\sim \mathrm{Normal}\!\left(0,\ \Sigma_A \otimes K_{\mathrm{year}}\right)
```

Stan uses `cov_GPL2(D, etasq_year, rhosq_year, delta)` to build $K_{\mathrm{year}}$ from the year-distance matrix $D$:

```math
K_{\mathrm{year},ij} = \eta^2 \exp\!\left(-\rho^2 d_{ij}^2\right),
\qquad K_{\mathrm{year},ii} \leftarrow K_{\mathrm{year},ii} + \delta
```

with $\delta = 0.01$ in the Stan call.

Cross-analyte covariance is represented via a Cholesky factor:

```math
\Sigma_A = L_{Agp} L_{Agp}^\top,
\qquad L_{Agp} = \mathrm{diag}(\sigma_{Agp})\,L_{\mathrm{corr},Agp}
```

and the GP realization is built non-centrally:

```math
F_{\mathrm{year}} = L_t Z_{gp} L_{Agp}^\top
```

where $L_t$ is the Cholesky factor of $K_{\mathrm{year}}$ and $Z_{gp}$ is standard normal.

---

# Load propagation (post-processing)

Event loads are computed from posterior draws after back-transforming concentration and volume from their modeled scales to original units in post-processing (see `stir-bayes-load1p7p5.Rmd`). At the event level for analyte $a$:

```math
L_{i,a} = C_{i,a}\,V_i
```

Annual loads:

```math
L_{y,a} = \sum_{i\in y} L_{i,a}
```

Uncertainty is propagated by applying these calculations to each posterior draw.

---

# Missingness handling summary (as implemented)

- Missing `C` rows: excluded from observation likelihood; inferred via $C_{\mathrm{true}}$  
- Missing `VOL` rows: excluded from observation likelihood; inferred via $V_{\mathrm{true}}$  
- Missing `RES` rows: imputed via residue submodel into $\mathrm{RES}^{\ast}$  
- Missing `CIN` rows: imputed via $\mathrm{CIN}_{\mathrm{impute}}$ into $\mathrm{CIN}^{\ast}$  

---

**Version 1p7p5** is the finalized Bayesian model implementation for Kerbel STIR × EoF water-quality analyses.
