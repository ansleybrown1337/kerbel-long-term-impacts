# Bayesian v3p3 prior audit

v3p3 inherits the complete v3p2 prior specification and adds one coefficient:
the direct furrow tire-compaction effect on standardized runoff volume,
`beta_tire_comp_V ~ normal(0, 1)`. The binary exposure is not assigned a
directional constraint. No existing v3p2 prior was changed.

Predictors and modeled concentrations and volumes remain on their existing
scales unless noted otherwise.

## New v3p3 prior

| Parameter | Prior | Reason |
|---|---:|---|
| Furrow tire-compaction effect on runoff volume, `beta_tire_comp_V` | `normal(0, 1)` | Match the accepted unit-scale convention for a zero-centered coefficient on standardized runoff volume while allowing either direction |

## Prior changes inherited from v3p2

| Parameter or family | v3p1 | v3p3 | Reason |
|---|---:|---:|---|
| Global runoff-volume effect on concentration | Hierarchical: `mu_beta_vol ~ normal(0, 0.3)`, `sigma_beta_vol ~ exponential(2)`, 10 standard-normal deviations | One `beta_vol ~ normal(0, 1)` | Remove the nonidentified analyte hierarchy while preserving the direct DAG pathway |
| Volume intercept, STIR effect on volume, inflow-volume effect | `normal(0, 0.7)` | `normal(0, 1)` | Unit-scale prior for standardized regression terms |
| Irrigation and duplicate hierarchy means | `normal(0, 0.3)` | `normal(0, 1)` | Unit-scale prior for standardized hierarchy locations |
| Residue effects on volume and concentration | `normal(0, 0.7)` | `normal(0, 1)` | Unit-scale prior for standardized regression terms |
| Shared analyte-effect locations `mu_A` | `normal(0, 0.7)` | `normal(0, 1)` | Unit-scale prior for standardized analyte-level intercept/STIR/inflow locations |
| STIR effect on residue | `normal(0, 0.7)` | `normal(0, 1)` | Unit-scale prior on the residue model's linear predictor |
| Log observation-error location | `normal(log(0.2), 0.7)` | `normal(log(0.2), 1)` | Widen the prior SD while retaining the scientifically distinct log-error center |
| GP, process, hierarchy, crop, sampler, flume, block, residue, and observation scale priors | `exponential(2)`, except runoff-volume observation error `exponential(5)` and year length scale `exponential(1)` | `exponential(1)` | Use one unit-rate exponential convention |

For an exponential distribution parameterized by rate, changing rate 2 to 1
changes its prior mean from 0.5 to 1; changing rate 5 to 1 changes its prior
mean from 0.2 to 1. These are broader priors, not merely cosmetic notation
changes.

## Priors deliberately retained

| Prior | Why it remains |
|---|---|
| `std_normal()` latent innovations and imputation states | Already exactly `normal(0, 1)` |
| `normal(0, sigma_group)` conditional effects | The SD is an estimated hierarchy scale, not a fixed sub-unit prior |
| `normal(mu_res_unit, sigma_res_obs)` residue likelihood | Observation model, not a coefficient prior |
| Volume and concentration observation likelihoods | Data likelihoods are not prior-regularization constants |
| `lkj_corr_cholesky(2)` and `lkj_corr_cholesky(3)` | Correlation-matrix priors; their shape parameters are whole-number, family-specific choices |
| `res_base ~ beta(2, 10)` | Bounded baseline residue proportion requires a distribution on 0 to 1 |

## Interpretation boundary

`normal(0, 1)` is broad on a standardized regression scale, but it is not
automatically noninformative and does not guarantee convergence. Likewise,
wider exponential scale priors can expose weak identification more strongly.
v3p3 therefore treats this as an explicit prior specification to evaluate,
not as proof that prior sensitivity has been resolved. Posterior diagnostics
and scientifically plausible prior/posterior predictive ranges must still be
reviewed after the run.
