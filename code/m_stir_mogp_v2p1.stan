// m_stir_mogp_v2p1.stan
// v2p1 (event-level latent volume; row-level concentration):
// - Applies the outflow-volume observation likelihood once per event-volume unit.
// - Represents inflow volume once per physical plot-event, including missing-value imputation.
// - Applies the residue likelihood/imputation once per planting-season plot unit.
// - Maps event-level volume and residue-unit states back to analyte rows.
// - Retains the v1p8 scientific structure and scalar beta_vol.
// - Residue handled with a unit-level logit-normal regression on logit(residue).
// - Missing residue units are imputed as bounded parameters.
// - Uses row aliases of shared RES_star_unit values in volume + concentration.
//
// Notes:
// - VOL_event and C are assumed z-scored upstream (as in your pipeline).
// - RES_unit is passed as a proportion (0,1); upstream R clamps with epsilon.

functions {

  matrix cov_GPL2(matrix x, real sq_alpha, real sq_rho, real delta) {
    int N = dims(x)[1];
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = sq_alpha + delta;
      for (j in (i + 1):N) {
        K[i, j] = sq_alpha * exp(-sq_rho * square(x[i, j]));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sq_alpha + delta;
    return K;
  }

  vector merge_missing(array[] int miss_indexes, vector x_obs, vector x_miss) {
    int N = num_elements(x_obs);
    int N_miss = num_elements(x_miss);
    vector[N] merged;
    merged = x_obs;
    for (i in 1:N_miss) {
      merged[miss_indexes[i]] = x_miss[i];
    }
    return merged;
  }
}

data {
  int Y_n;
  int D_n;  // retained for compatibility; not used directly
  int F_n;
  int S_n;
  int B_n;
  int A_n;
  int Cr_n;
  int N;

  // Observed outcomes (rows listed in *_missidx are excluded from obs likelihood)
  vector[N] C;
  int<lower=0> N_C_miss;
  array[N_C_miss] int<lower=1, upper=N> C_missidx;

  // Concentration censoring (left-censor at reporting limit on SAME z-scale as C)
  // C_cens[i]=1 indicates the true observed concentration is < C_cens_limit[i]
  array[N] int<lower=0, upper=1> C_cens;
  vector[N] C_cens_limit;

  // Event-volume units. E maps analyte rows to event units; E_rep_row supplies
  // one row's event-level predictors to the volume process.
  int<lower=1> E_n;
  array[N] int<lower=1, upper=E_n> E;
  array[E_n] int<lower=1, upper=N> E_rep_row;
  vector[E_n] VOL_event;
  int<lower=0> N_VOL_event_miss;
  array[N_VOL_event_miss] int<lower=1, upper=E_n> VOL_event_missidx;

  // Inflow volume is shared at the physical plot-event level. VIN_E maps each
  // volume-measurement unit to its physical VIN event.
  int<lower=1> VIN_n;
  array[E_n] int<lower=1, upper=VIN_n> VIN_E;
  vector[VIN_n] VIN_event;
  int<lower=0> N_VIN_event_miss;
  array[N_VIN_event_miss] int<lower=1, upper=VIN_n> VIN_event_missidx;

  // Residue is measured once per planting-season plot unit.
  int<lower=1> R_n;
  array[N] int<lower=1, upper=R_n> R;
  vector[R_n] RES_unit;
  vector[R_n] STIR_res_unit;
  array[R_n] int<lower=1, upper=Cr_n> PrevCr_res_unit;
  int<lower=0> N_RES_unit_miss;
  array[N_RES_unit_miss] int<lower=1, upper=R_n> RES_unit_missidx;

  // indices
  array[N] int<lower=1, upper=Y_n>  Y;
  array[N] int<lower=1, upper=F_n>  Fu;
  array[N] int<lower=1, upper=S_n>  S;
  array[N] int<lower=1, upper=B_n>  B;
  array[N] int<lower=1, upper=A_n>  A;
  array[N] int<lower=1, upper=Cr_n> Cr;
  array[N] int<lower=1, upper=Cr_n> PrevCr;  // previous year's crop (same coding as Cr)
  array[N] int DUP;  // 0/1
  vector[N] IRR_z;  // standardized upstream

  // predictors
  vector[N] STIR;
  vector[N] CIN;
  int<lower=0> N_CIN_miss;
  array[N_CIN_miss] int<lower=1, upper=N> CIN_missidx;

  // year distance matrix for GP
  matrix[Y_n, Y_n] D;
}



parameters {
  // analyte-level MVN (non-centered)
  matrix[4, A_n] Z_A;
  vector[4] mu_A;
  cholesky_factor_corr[4] L_A;
  vector<lower=0>[4] sigma_A;

  // analyte by block MVN
  matrix[B_n, A_n] Z_B;
  cholesky_factor_corr[B_n] L_B;
  vector<lower=0>[B_n] sigma_B;

  // analyte by sampler MVN
  matrix[S_n, A_n] Z_S;
  cholesky_factor_corr[S_n] L_S;
  vector<lower=0>[S_n] sigma_S;

  // analyte by flume MVN
  matrix[F_n, A_n] Z_F;
  cholesky_factor_corr[F_n] L_F;
  vector<lower=0>[F_n] sigma_F;

  // fixed effects
  real beta_vol;
  real beta_vin;  // inflow volume effect on outflow volume
  // irrigation effects on concentration (analyte-specific MVN)
real mu_beta_irr;
cholesky_factor_corr[A_n] L_beta_irr;
vector<lower=0>[A_n] sigma_beta_irr;
vector[A_n] z_beta_irr;

  // Residue effects in outcome models (use RES_star on proportion scale)
  real beta_res_V;
  vector[A_n] beta_res_C;

  // volume model STIR effect
  real a_V;
  real b_V;

  // crop effects (volume and concentration)
  vector[Cr_n] gamma_Cr_V;
  matrix[A_n, Cr_n] gamma_Cr;
  real<lower=0> sigma_Cr_V;
  vector<lower=0>[A_n] sigma_Cr;

  // process residual scales (latent truth variability)
  real<lower=0> sigma_analyte;  // global process SD for C_true around mu_C (z scale)
  real<lower=0> sigma_V;               // process SD for V_true around mu_V

  // multi-output GP hyperparameters (analyte covariance)
  cholesky_factor_corr[A_n] L_corr_Agp;
  vector<lower=0>[A_n] sigma_Agp;

  // GP kernel hyperparameters (year)
  real<lower=0> etasq_year;
  real<lower=0> rhosq_year;

  // non-centered latent GP
  matrix[Y_n, A_n] Z_gp;

  // missing CIN imputation
  vector[N_CIN_miss] CIN_impute;

  // missing event-level VIN imputation
  vector[N_VIN_event_miss] VIN_event_impute;

  // latent true outcomes
  // Event-level volume and row-level concentration, non-centered
  vector[E_n] z_V_event;
  vector[N] z_C;

  // ---------------------------------------------------------------------------
// Residue submodel: logit-normal regression with in-model imputation
//
// Mean on logit scale:
//   mu_res_unit[u] = logit(res_base) + b_res_stir * STIR_res_unit[u]
//                    + gamma_PCr_res[PrevCr_res_unit[u]]
//
// Observations:
//   logit(RES_unit[u]) ~ Normal(mu_res_unit[u], sigma_res_obs)
//
// Missing values:
//   RES_unit_miss_logit[j] are unconstrained logit-scale parameters with:
//     RES_unit_miss_logit[j] ~ Normal(mu_res_unit[idx], sigma_res_obs)
// ---------------------------------------------------------------------------

  real<lower=0, upper=1> res_base;
  real b_res_stir;
  vector[Cr_n] gamma_PCr_res;
  real<lower=0> sigma_PCr_res;
  real<lower=0> sigma_res_obs;

  // Imputed missing residue units on the logit scale.
  vector[N_RES_unit_miss] RES_unit_miss_logit;

  // Observation error (estimated, not supplied as data)
  // Concentration observation (measurement) error: analyte-specific with partial pooling (z scale)
  real mu_log_sigma_C_obs;
  real<lower=0> tau_log_sigma_C_obs;
  vector[A_n] z_log_sigma_C_obs;
  real<lower=0> sigma_VOL_obs;       // volume measurement SD (z scale)
}

transformed parameters {
  // analyte-level effects
  vector[A_n] alpha;
  vector[A_n] beta_stir;
  vector[A_n] beta_cin;
  vector[A_n] beta_dup;
  vector[A_n] beta_irr;
  matrix[A_n, 4] v_A;

  // random effects
  matrix[A_n, B_n] gamma_B;
  matrix[A_n, S_n] gamma_S;
  matrix[A_n, F_n] gamma_F;

  // multi-output GP structures
  matrix[Y_n, Y_n] K_year;
  matrix[Y_n, Y_n] L_t;
  matrix[A_n, A_n] L_Agp;
  matrix[Y_n, A_n] F_year;

  // analyte-specific concentration observation SD (z scale)
  vector<lower=0>[A_n] sigma_C_obs;

  // residue on proportion scale (used in outcome models)

  // ---------------------------------------------------------------------------
  // Deterministic linear predictors and non-centered event/row latent truths
  // (exposed here so they are also available to generated quantities)
  // ---------------------------------------------------------------------------
  vector[N] CIN_merge;   // CIN with missing values imputed via CIN_impute
  vector[VIN_n] VIN_event_merge; // physical event VIN with missing values imputed
  vector[N] VIN_merge;   // row-level alias of event VIN
  vector[R_n] mu_res_unit; // residue-unit model mean on logit scale
  vector[R_n] RES_star_unit; // residue-unit observed/imputed predictor
  vector[N] mu_res;      // row-level alias of residue model mean
  vector[N] RES_star;    // row-level alias of residue predictor
  vector[E_n] mu_V_event; // event-level volume process mean (z scale)
  vector[E_n] V_true_event; // event-level latent true volume (z scale)
  vector[N] mu_V;        // row-level alias of event volume mean (z scale)
  vector[N] mu_C;        // concentration process mean (z scale)
  vector[N] V_true;      // row-level alias of event latent true volume (z scale)
  vector[N] C_true;      // latent true concentration (z scale)

  // non-centered MVNs for random effects
  gamma_F = (diag_pre_multiply(sigma_F, L_F) * Z_F)';
  gamma_S = (diag_pre_multiply(sigma_S, L_S) * Z_S)';
  gamma_B = (diag_pre_multiply(sigma_B, L_B) * Z_B)';
  v_A     = (diag_pre_multiply(sigma_A, L_A) * Z_A)';

  beta_dup  = mu_A[4] + v_A[, 4];
  beta_cin  = mu_A[3] + v_A[, 3];
  beta_stir = mu_A[2] + v_A[, 2];
  alpha     = mu_A[1] + v_A[, 1];


  // analyte-specific irrigation slopes (MVN with LKJ correlation)
  beta_irr = mu_beta_irr + (diag_pre_multiply(sigma_beta_irr, L_beta_irr) * z_beta_irr);

  // hierarchical analyte-specific measurement error for concentration
  sigma_C_obs = exp(mu_log_sigma_C_obs + tau_log_sigma_C_obs * z_log_sigma_C_obs);

  // multi-output GP: separable covariance Σ_A ⊗ K_year
  K_year = cov_GPL2(D, etasq_year, rhosq_year, 0.01);
  L_t    = cholesky_decompose(K_year);
  L_Agp  = diag_pre_multiply(sigma_Agp, L_corr_Agp);

  // non-centered: F_year = L_t * Z_gp * L_Agp'
  F_year = L_t * Z_gp * L_Agp';
  // Missing-value merges.
  CIN_merge = merge_missing(CIN_missidx, CIN, CIN_impute);
  VIN_event_merge = merge_missing(VIN_event_missidx, VIN_event, VIN_event_impute);


  // Residue-unit mean on logit scale (parents: planting-time STIR, previous crop).
  for (u in 1:R_n) {
    mu_res_unit[u] = logit(res_base) +
      b_res_stir * STIR_res_unit[u] +
      gamma_PCr_res[PrevCr_res_unit[u]];
  }

  // Merge observed and missing residue once per residue unit.
  RES_star_unit = RES_unit;
  if (N_RES_unit_miss > 0) {
    for (j in 1:N_RES_unit_miss) {
      RES_star_unit[RES_unit_missidx[j]] = inv_logit(RES_unit_miss_logit[j]);
    }
  }

  // Row-level aliases preserve downstream extraction and concentration code.
  for (i in 1:N) {
    VIN_merge[i] = VIN_event_merge[VIN_E[E[i]]];
    mu_res[i] = mu_res_unit[R[i]];
    RES_star[i] = RES_star_unit[R[i]];
  }

  for (e in 1:E_n) {
    int r = E_rep_row[e];
    mu_V_event[e] = a_V +
      b_V * STIR[r] +
      beta_vin * VIN_event_merge[VIN_E[e]] +
      beta_res_V * RES_star[r] +
      gamma_Cr_V[Cr[r]];
  }

  // Non-centered event-level latent truth.
  V_true_event = mu_V_event + sigma_V * z_V_event;

  // Row-level aliases preserve downstream extraction and concentration code.
  for (i in 1:N) {
    mu_V[i] = mu_V_event[E[i]];
    V_true[i] = V_true_event[E[i]];
  }

  // Concentration process mean (parents: STIR, CIN, latent VOL, IRR, DUP, residue proportion, crop, REs, GP)
  for (i in 1:N) {
    mu_C[i] =
      alpha[A[i]] +
      beta_stir[A[i]] * STIR[i] +
      beta_cin[A[i]]  * CIN_merge[i] +
      beta_vol        * V_true[i] +
      beta_irr[A[i]]  * IRR_z[i] +
      beta_dup[A[i]]  * DUP[i] +
      beta_res_C[A[i]] * RES_star[i] +
      gamma_Cr[A[i], Cr[i]] +
      gamma_B[A[i], B[i]] +
      gamma_S[A[i], S[i]] +
      gamma_F[A[i], Fu[i]] +
      F_year[Y[i], A[i]];
  }

  // Non-centered latent truth with analyte-specific SDs:
  for (i in 1:N) {
    C_true[i] = mu_C[i] + sigma_analyte * z_C[i];
  }

}

model {

  // missingness masks
  array[N] int is_C_miss;
  array[E_n] int is_VOL_event_miss;
  array[R_n] int is_RES_unit_miss;

  // initialize masks
  for (i in 1:N) {
    is_C_miss[i] = 0;
  }
  for (e in 1:E_n) is_VOL_event_miss[e] = 0;
  for (u in 1:R_n) is_RES_unit_miss[u] = 0;
  for (k in 1:N_C_miss)   is_C_miss[C_missidx[k]] = 1;
  for (k in 1:N_VOL_event_miss) is_VOL_event_miss[VOL_event_missidx[k]] = 1;
  for (k in 1:N_RES_unit_miss) is_RES_unit_miss[RES_unit_missidx[k]] = 1;

  // Priors (gently tightened; consistent with z-scored outcomes)
  // ---------------------------------------------------------------------------

  // GP priors
  etasq_year ~ exponential(2);
  rhosq_year ~ exponential(1);
  sigma_Agp  ~ exponential(2);
  L_corr_Agp ~ lkj_corr_cholesky(3);
  to_vector(Z_gp) ~ normal(0, 1);

  // process residual scales
  sigma_V       ~ exponential(2);
  sigma_analyte ~ exponential(2);

  // observation (measurement) error priors (z scale)
  sigma_VOL_obs ~ exponential(5);   // mean 0.2
  // concentration measurement SD (z scale): hierarchical, analyte-specific
  z_log_sigma_C_obs  ~ std_normal();
  mu_log_sigma_C_obs ~ normal(log(0.2), 0.7);
  tau_log_sigma_C_obs ~ exponential(2);


  // fixed effects (regularized slopes on standardized predictors)
  b_V      ~ normal(0, 0.7);
  a_V      ~ normal(0, 0.7);
    beta_vol ~ normal(0, 0.7);
  beta_vin ~ normal(0, 0.7);

// irrigation slopes: MVN across analytes (on standardized IRR_z scale)
mu_beta_irr ~ normal(0, 0.3);          // conservative prior for mean slope
sigma_beta_irr ~ exponential(2);       // regularizes (z-scale)
L_beta_irr ~ lkj_corr_cholesky(2);
z_beta_irr ~ normal(0, 1);

  beta_res_V ~ normal(0, 0.7);
  beta_res_C ~ normal(0, 0.7);

  // crop effects
  sigma_Cr_V ~ exponential(2);
  gamma_Cr_V ~ normal(0, sigma_Cr_V);

  sigma_Cr ~ exponential(2);
  for (a in 1:A_n) {
    gamma_Cr[a] ~ normal(0, sigma_Cr[a]);
  }

  // random effects
  sigma_F ~ exponential(2);
  L_F     ~ lkj_corr_cholesky(3);
  to_vector(Z_F) ~ normal(0, 1);

  sigma_S ~ exponential(2);
  L_S     ~ lkj_corr_cholesky(3);
  to_vector(Z_S) ~ normal(0, 1);

  sigma_B ~ exponential(2);
  L_B     ~ lkj_corr_cholesky(3);
  to_vector(Z_B) ~ normal(0, 1);

  // analyte-level MVN
  sigma_A ~ exponential(2);
  L_A     ~ lkj_corr_cholesky(3);
  mu_A    ~ normal(0, 0.7);
  to_vector(Z_A) ~ normal(0, 1);
  // CIN missing data model (imputed values in CIN_impute; CIN_merge is deterministic in transformed parameters)
  // Imputation prior for missing CIN values (CIN is z-standardized in the data)
  CIN_impute ~ normal(0, 1);
  VIN_event_impute ~ normal(0, 1);

  // ---------------------------------------------------------------------------
  // Residue submodel: logit-normal latent process + observation
  // Parents: STIR, Crop Type
  // ---------------------------------------------------------------------------
  sigma_PCr_res   ~ exponential(2);
  gamma_PCr_res   ~ normal(0, sigma_PCr_res);

  // Baseline residue proportion; informative but mild
  res_base     ~ beta(2, 10);
  b_res_stir   ~ normal(0, 0.7);

  // Residue process and observation scales (logit scale)
  // These priors discourage extreme curvature while allowing meaningful variation.
  sigma_res_obs  ~ exponential(2);   // mean 0.5 on logit scale
// Residue observation model (one likelihood/imputation per residue unit).
// ---------------------------------------------------------------------------
// Observed residue units
for (u in 1:R_n) {
  if (is_RES_unit_miss[u] == 0) {
    target += normal_lpdf(logit(RES_unit[u]) | mu_res_unit[u], sigma_res_obs);
  }
}
// Missing residue units (imputation parameters)
if (N_RES_unit_miss > 0) {
  for (j in 1:N_RES_unit_miss) {
    int idx = RES_unit_missidx[j];
    RES_unit_miss_logit[j] ~ normal(mu_res_unit[idx], sigma_res_obs);
  }
}
// ---------------------------------------------------------------------------
  // Latent PROCESS model for volume truth (z scale), non-centered
  // V_true_event is defined in transformed parameters as:
  // V_true_event = mu_V_event + sigma_V * z_V_event
  // ---------------------------------------------------------------------------
  z_V_event ~ std_normal();
  // ---------------------------------------------------------------------------
  // Latent PROCESS model for concentration truth (z scale), non-centered
  // C_true is defined in transformed parameters as: C_true[i] = mu_C[i] + sigma_analyte * z_C[i]
  // ---------------------------------------------------------------------------
  z_C ~ std_normal();


  // ---------------------------------------------------------------------------
  // OBSERVATION models (measurement error) for event VOL and row-level C.
  // ---------------------------------------------------------------------------
  for (e in 1:E_n) {
    if (is_VOL_event_miss[e] == 0) {
      target += normal_lpdf(VOL_event[e] | V_true_event[e], sigma_VOL_obs);
    }
  }

  for (i in 1:N) {
    if (is_C_miss[i] == 0) {
      if (C_cens[i] == 1) {
        target += normal_lcdf(C_cens_limit[i] | C_true[i], sigma_C_obs[A[i]]);
      } else {
        target += normal_lpdf(C[i] | C_true[i], sigma_C_obs[A[i]]);
      }
    }
  }
}

generated quantities {
  vector[E_n] VOL_event_rep;
  vector[R_n] RES_unit_rep01;
  vector[N] VOL_rep;
  vector[N] C_rep;
  vector[N] RES_rep01;

  for (e in 1:E_n) {
    VOL_event_rep[e] = normal_rng(V_true_event[e], sigma_VOL_obs);
  }

  for (u in 1:R_n) {
    RES_unit_rep01[u] = inv_logit(normal_rng(mu_res_unit[u], sigma_res_obs));
  }

  for (i in 1:N) {
    // replicate observed outcomes
    VOL_rep[i] = VOL_event_rep[E[i]];
    C_rep[i]   = normal_rng(C_true[i], sigma_C_obs[A[i]]);

    // row-level alias of one replicated residue value per residue unit
    RES_rep01[i] = RES_unit_rep01[R[i]];
  }
}
