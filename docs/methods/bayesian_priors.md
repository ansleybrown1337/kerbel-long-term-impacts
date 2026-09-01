# Bayesian prior specification

The Bayes v3p3 model uses weakly regularizing priors on standardized scales.
The complete executable specification is
`code/bayes/m_stir_mogp_v3p3_physical_event.stan`; this page is a readable
summary.

| Parameter family | Prior |
| --- | --- |
| Standardized zero-centered regression coefficients | `normal(0, 1)` |
| Positive scale parameters | `exponential(1)` |
| Log observation-error location | `normal(log(0.2), 1)` |
| Baseline residue proportion | `beta(2, 10)` |
| Concentration event-correlation Cholesky factor | `lkj_corr_cholesky(2)` |

The runoff-volume effect on concentration is one global standardized
coefficient, `beta_vol ~ normal(0, 1)`. Furrow tire compaction enters only the
runoff-volume process through `beta_tire_comp_V ~ normal(0, 1)`. It does not
enter the residue or concentration process directly.

The saved-fit diagnostics and prior audit tables in
`results/bayes/v3p3_physical_event/` should be reviewed with posterior
estimates. The released fit contains 28 divergent transitions and 20 finite
parameters with R-hat above 1.04 (maximum 1.068), so the posterior is not fully
converged.
