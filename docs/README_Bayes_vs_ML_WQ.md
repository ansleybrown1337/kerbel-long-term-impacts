# Bayesian vs Machine Learning Approaches for Long-Term Edge-of-Field Water Quality Analysis
Bayesian and machine learning approaches address complementary objectives in long-term environmental monitoring. Bayesian hierarchical models enable causal interpretation and principled uncertainty propagation under sparse data, while machine learning models provide accurate, scalable predictions with empirically calibrated uncertainty. Comparing these approaches clarifies the tradeoffs between interpretability, uncertainty realism, and predictive performance.

This document compares the two analytical frameworks used in this project for long-term edge-of-field (EoF) runoff water-quality analysis:

* **Bayesian hierarchical modeling**
* **Machine learning modeling using CatBoost with conformal uncertainty**

The purpose is not to rank the methods, but to clarify **what scientific questions each approach can and cannot answer**, how uncertainty is treated, and what researchers must understand before choosing one over the other.

---

## 1. Framing the comparison correctly

Both the Bayesian and machine learning approaches used here are capable of:

* producing predictions,
* generating distributions or intervals around predictions,
* filling missing observations.

The distinction is **not** that one is predictive and the other is not, nor that only one is “generative.”

The real distinction lies in:

1. **How variables are selected and justified**,
2. **What assumptions are encoded explicitly**,
3. **What the resulting uncertainty represents**, and
4. **How outputs should be interpreted and used**.

Failing to distinguish these points leads to incorrect conclusions, particularly when comparing uncertainty under missing years.

---

## 2. Variable selection: causal structure vs predictive convenience

### Bayesian workflow, DAG-driven selection

In the Bayesian workflow, variables are chosen using **explicit causal assumptions**, formalized through directed acyclic graphs (DAGs) and do-calculus logic.

Key characteristics:

* Covariates are included or excluded based on their role in the causal graph.
* Adjustment sets are chosen to identify specific causal estimands (e.g., tillage effects).
* Variables that would introduce bias (e.g., colliders or post-treatment variables) are deliberately excluded.
* Model structure reflects scientific judgment about processes, not just data availability.

As a result, parameter estimates have **interpretable meaning conditional on stated assumptions**.

### ML workflow, feature-rich predictive inclusion

In the ML workflow:

* A broad set of available columns is included by default.
* Feature inclusion is driven by predictive performance, not causal identifiability.
* The model may condition on variables that are downstream of management actions or correlated through unobserved mechanisms.
* Feature importance reflects predictive contribution within the fitted model, not causal influence.

This is appropriate for prediction, but incompatible with causal claims.

### Summary: variable selection philosophy

| Aspect                   | Bayesian workflow        | ML workflow               |
| ------------------------ | ------------------------ | ------------------------- |
| Variable inclusion       | DAG-based, theory-driven | Feature-rich, data-driven |
| Colliders avoided        | Yes                      | No guarantee              |
| Post-treatment variables | Excluded intentionally   | Often included            |
| Causal interpretability  | Yes (conditional on DAG) | No                        |
| Predictive flexibility   | Moderate                 | High                      |

---

## 3. How parameters are chosen and constrained

### Bayesian workflow

* Parameters are **random variables** with prior distributions.
* Regularization occurs through priors and hierarchical pooling.
* Temporal behavior is encoded explicitly via Gaussian processes.
* Multi-output structure allows analytes to share information.

Parameter uncertainty is inferred, not tuned.

### ML workflow

* Parameters are algorithmic weights optimized to minimize loss.
* Regularization occurs via tree depth, learning rate, subsampling, and early stopping.
* Temporal information is learned only through supplied features.
* Models are trained independently for concentration and volume.

Parameters are optimized, not interpreted.

### Parameter comparison

| Aspect                 | Bayesian        | ML              |
| ---------------------- | --------------- | --------------- |
| Parameter meaning      | Scientific      | Algorithmic     |
| Regularization         | Priors, pooling | Hyperparameters |
| Temporal structure     | Explicit        | Implicit only   |
| Multi-analyte learning | Yes             | No              |
| Interpretability       | High            | Low             |

---

## 4. Uncertainty: what it means and what it does not mean

Both modeling approaches quantify uncertainty, but the **source, interpretation, and behavior of that uncertainty differ fundamentally**. Understanding these differences is essential before comparing interval widths or imputed values.

* **Bayesian uncertainty reflects what is unknown about the system**, including how uncertainty should evolve over time.
* **ML uncertainty reflects how wrong predictions have been in similar cases**, without learning a temporal or causal process.

Intervals from the two approaches should therefore **not be interpreted interchangeably**, even when their nominal coverage is similar.


### 4.1 Bayesian uncertainty

The Bayesian model produces **posterior predictive distributions** derived from an explicit probabilistic data-generating model. These distributions integrate multiple sources of uncertainty simultaneously:

* **parameter uncertainty**, where regression coefficients and effects are random variables,
* **latent process uncertainty**, including year effects and multi-analyte structure,
* **observation-level noise**, explicitly modeled rather than assumed,
* **temporal extrapolation uncertainty**, learned via the Gaussian process.

A key feature of the Bayesian workflow is that **uncertainty is reduced by adding structure**, not by limiting parameters. Hierarchical modeling introduces shared variance components (e.g., group-level σ parameters) that enable **partial pooling**, allowing sparse groups or years to borrow strength from the full dataset.

As a result:

* posterior intervals may *shrink* after introducing hierarchical structure,
* overfitting is controlled through priors and pooling rather than parameter omission,
* uncertainty **grows naturally as information decreases**, such as when imputing missing years farther from observed data.

Model comparison tools such as **PSIS-LOO cross-validation and WAIC** reinforce this behavior by penalizing models based on their *effective* number of parameters rather than raw parameter count. Hierarchical models with learned variance components often generalize better under these criteria, despite being more complex.


### 4.2 Machine learning uncertainty

The ML workflow produces **conformal prediction intervals**, which are:

* empirically calibrated to achieve nominal coverage,
* derived from held-out residuals rather than an explicit stochastic model,
* agnostic to causal structure and temporal dependence unless encoded as features.

Uncertainty in the ML approach is controlled through **algorithmic regularization** (e.g., tree depth, learning rate, subsampling), not through explicit modeling of variance components.

Monte Carlo “draws” generated from ML prediction intervals are therefore a **propagation device**, not posterior samples. They allow uncertainty to be carried forward into aggregated quantities (e.g., annual loads), but they do not represent a learned distribution over latent processes or future states.

Importantly, ML uncertainty does **not** automatically increase with forecasting distance. Missing years receive intervals comparable in width to observed years with similar covariates, reflecting empirical coverage rather than epistemic uncertainty growth.



### 4.3 How uncertainty is controlled

| Mechanism                 | Bayesian workflow               | ML workflow                   |
| ------------------------- | ------------------------------- | ----------------------------- |
| Primary control           | Priors and hierarchical pooling | Algorithmic regularization    |
| Variance modeling         | Explicit (σ parameters)         | Implicit or post hoc          |
| Overfitting penalty       | PSIS-LOO, WAIC                  | Cross-validation              |
| Effect of added structure | Can reduce uncertainty          | Usually increases complexity  |
| Treatment of sparse data  | Borrowing strength              | Limited by feature similarity |


### 4.4 Uncertainty comparison summary

| Aspect                   | Bayesian             | ML                   |
| ------------------------ | -------------------- | -------------------- |
| Type                     | Posterior predictive | Conformal predictive |
| Encodes epistemic growth | Yes                  | No                   |
| Time-distance awareness  | Yes                  | No                   |
| Draws represent          | Possible futures     | Interval sampling    |
| Interpretation           | Probabilistic        | Coverage-based       |

---


## 5. Missing years and imputation behavior

This difference is central to Chapter 3.

| Feature               | Bayesian                  | ML                 |
| --------------------- | ------------------------- | ------------------ |
| Missing data handling | Native                    | Explicit step      |
| Interval behavior     | Widens with distance      | Roughly stationary |
| Temporal coherence    | Learned                   | Not learned        |
| Assumption            | Structured latent process | Exchangeability    |

ML can fill gaps efficiently, but it does not learn how uncertainty should evolve over time.

---

## 6. Computational expense

Measured wall-clock runtimes on the same dataset:

| Model                       | Runtime         |
| --------------------------- | --------------- |
| Bayesian hierarchical model | 5,184 seconds (1.44 h) |
| CatBoost Model              | 6,460 seconds (1.80 h)  |

Additional considerations:

* Bayesian runtime scales with model complexity, number of parameters, and sampler efficiency, but benefits from partial pooling and shared structure across analytes and years.
* ML runtime scales approximately linearly with the number of LOYO folds and must refit separate concentration and volume models for each held-out year.
* When uncertainty propagation and leave-one-year-out validation are performed rigorously, ML is **not substantially cheaper** than the Bayesian approach at this scale.
* ML impute-only mode, which reuses saved models without refitting, is extremely fast (seconds), making it suitable for rapid deployment or gap-filling once models are trained.

---

## 7. Accuracy and evaluation goals

| Aspect              | Bayesian            | ML                  |
| ------------------- | ------------------- | ------------------- |
| Optimization target | Coherent inference  | Predictive error    |
| Evaluation focus    | Process realism     | RMSE, MAE, coverage |
| Overfitting control | Priors, pooling     | Cross-validation    |
| Strength            | Uncertainty realism | Point prediction    |

ML often achieves lower pointwise error; Bayesian models provide more defensible uncertainty under sparse data.

---

## 8. Real-world applicability

### Bayesian workflow is preferable when:

* causal effects are of interest,
* uncertainty must be defensible under missing data,
* long-term trends matter,
* interpretability is required for policy or management decisions.

### ML workflow is preferable when:

* prediction is the primary goal,
* relationships are highly nonlinear,
* computational speed and deployment matter,
* causal interpretation is not required.

---

## 9. Complementarity rather than competition

In this project, the ML workflow serves as:

* a predictive benchmark,
* a sensitivity check on Bayesian assumptions,
* a practical reference for applied users.

Disagreement between methods is informative and highlights where assumptions matter.

---

## 10. Summary comparison table

| Dimension             | Bayesian hierarchical model | ML CatBoost + conformal   |
| --------------------- | --------------------------- | ------------------------- |
| Variable selection    | DAG-based                   | Feature-rich              |
| Causal interpretation | Yes                         | No                        |
| Temporal modeling     | Explicit                    | Implicit                  |
| Missing data          | Native                      | Imputed                   |
| Uncertainty behavior  | Expands with distance       | Stationary                |
| Interpretability      | High                        | Low                       |
| Computational cost    | High upfront                | Moderate                  |
| Deployment            | Research-focused            | Production-friendly       |
| Best use              | Understanding processes     | Filling gaps, forecasting |
---

## 11. Quantitative model evaluation metrics

To complement the qualitative comparison above, Bayesian and machine learning results were evaluated using **point-based and distribution-aware performance metrics** computed at the annual scale for each analyte and treatment. Metrics were calculated only for years with observed annual loads and then summarized across years and treatments.

The intent of these metrics is **not to declare a single best model**, but to quantify how Bayesian and ML approaches differ in predictive accuracy, calibration, and uncertainty representation.

### 11.1 Root Mean Squared Error (RMSE)

RMSE measures average pointwise deviation between modeled and observed annual loads:

$$
\mathrm{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 }
$$

where:
- $y_i$ is the observed annual load,
- $\hat{y}_i$ is the modeled central estimate (posterior mean for Bayes, predictive mean for ML),
- $N$ is the number of observed years.

**Interpretation**
- Lower RMSE indicates better point prediction accuracy.
- RMSE disproportionately penalizes large errors.
- RMSE ignores uncertainty width and distributional shape.

---

### 11.2 Normalized RMSE (NRMSE)

To enable comparison across analytes with different magnitudes, RMSE was normalized in two ways.

**Mean-normalized RMSE**

$$
\mathrm{NRMSE}_{\mu} = \frac{\mathrm{RMSE}}{\overline{|y|}}
$$

**Range-normalized RMSE**

$$
\mathrm{NRMSE}_{\text{range}} = \frac{\mathrm{RMSE}}{\max(y) - \min(y)}
$$

**Interpretation**
- NRMSE expresses error relative to the scale of the data.
- Useful for cross-analyte comparisons.
- Like RMSE, NRMSE evaluates only point estimates.

---

### 11.3 Interval coverage

Interval coverage evaluates how often observed values fall within modeled uncertainty intervals:

$$
\mathrm{Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(y_i \in [L_i, U_i])
$$

where $L_i$ and $U_i$ are lower and upper uncertainty bounds.

**Interpretation**
- Coverage near the nominal level (for example 0.95) indicates good calibration.
- Coverage alone does not penalize overly wide intervals.
- High coverage can be achieved by conservative uncertainty bounds.

Coverage is therefore a **necessary but insufficient** uncertainty diagnostic.

---

### 11.4 Continuous Ranked Probability Score (CRPS)

CRPS is a **proper scoring rule** that evaluates the entire predictive distribution against an observed value, rather than only a point estimate or interval.

For a predictive cumulative distribution function $F$ and an observation $y$:

$$
\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} [F(z) - \mathbb{I}(z \ge y)]^2 \, dz
$$

When predictive distributions are represented by Monte Carlo draws $x_1, \ldots, x_S$, CRPS can be approximated as:

$$
\mathrm{CRPS} \approx
\frac{1}{S} \sum_{s=1}^{S} |x_s - y|
- \frac{1}{2S^2} \sum_{s=1}^{S} \sum_{s'=1}^{S} |x_s - x_{s'}|
$$

This formulation follows Gneiting and Raftery (2007).

**Interpretation**
- Lower CRPS indicates a better probabilistic forecast.
- CRPS rewards sharp distributions only when they are well calibrated.
- Overconfident and overly diffuse distributions are both penalized.
- CRPS directly reflects distributional spread, skewness, and variance.

---

### 11.5 Why CRPS is essential for Bayes versus ML comparison

CRPS is particularly important in this study because:

- The Bayesian model produces posterior predictive distributions with epistemic uncertainty that evolves through time.
- The ML model produces empirically calibrated predictive distributions without explicit temporal uncertainty propagation.
- RMSE and NRMSE collapse both approaches to a single central estimate.
- Coverage ignores uncertainty sharpness.

CRPS directly evaluates the **quality of uncertainty itself**, making it the most appropriate metric for comparing Bayesian and ML approaches when uncertainty realism is a central research objective.

Gneiting, T., & Raftery, A. E. (2007).
Strictly proper scoring rules, prediction, and estimation.
Journal of the American Statistical Association, 102(477), 359–378.
https://doi.org/10.1198/016214506000001437

---

### 11.6 Metric interpretation summary

| Metric   | Uses uncertainty | Penalizes overconfidence | Penalizes wide intervals | Scale-aware |
|---------|------------------|--------------------------|--------------------------|-------------|
| RMSE    | No               | No                       | No                       | No          |
| NRMSE   | No               | No                       | No                       | Yes         |
| Coverage| Partially        | No                       | No                       | No          |
| CRPS    | Yes              | Yes                      | Yes                      | Yes         |

In this project, RMSE and NRMSE quantify point prediction accuracy, while CRPS quantifies probabilistic forecast quality. Together, these metrics provide a defensible and transparent comparison of Bayesian and machine learning approaches.
