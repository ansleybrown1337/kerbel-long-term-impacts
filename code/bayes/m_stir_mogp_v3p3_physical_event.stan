// m_stir_mogp_v3p3_physical_event.stan
//
// Scientific identities:
// - one latent outflow volume per physical event;
// - one latent outflow concentration per physical event x study analyte;
// - one inflow concentration per physical event x study analyte;
// - sampler, flume/collection method, and duplicate effects live only in the
//   concentration observation layer;
// - correlated event-level analyte innovations allow any observed study
//   analyte (for example TSS) to inform missing values of the others.
// v3p3 uses one global standardized runoff-volume effect on concentration,
// adds a documented event-level furrow tire-compaction effect to runoff
// volume only, and retains the unit-scale normal/exponential prior convention
// documented in docs/methods/bayesian_prior_audit_v3p3.md.

functions {
  matrix cov_GPL2_corr(matrix x, real sq_rho, real delta) {
    int N = dims(x)[1];
    matrix[N, N] K;
    for (i in 1:(N - 1)) {
      K[i, i] = 1 + delta;
      for (j in (i + 1):N) {
        K[i, j] = exp(-sq_rho * square(x[i, j]));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = 1 + delta;
    return K;
  }

  vector merge_missing(array[] int miss_indexes, vector x_obs, vector x_miss) {
    int N = num_elements(x_obs);
    vector[N] merged = x_obs;
    for (i in 1:num_elements(x_miss)) {
      merged[miss_indexes[i]] = x_miss[i];
    }
    return merged;
  }
}

data {
  int<lower=1> Y_n;
  int<lower=1> D_n;  // retained for data-contract compatibility
  int<lower=1> F_n;
  int<lower=1> S_n;
  int<lower=1> B_n;
  int<lower=1> A_n;
  int<lower=1> Cr_n;
  int<lower=1> N;

  vector[N] C;
  int<lower=0> N_C_miss;
  array[N_C_miss] int<lower=1, upper=N> C_missidx;
  array[N] int<lower=0, upper=1> C_cens;
  vector[N] C_cens_limit;

  // Physical events and the complete E_n x A_n event-analyte scaffold.
  int<lower=1> E_n;
  int<lower=1> EA_n;
  array[N] int<lower=1, upper=E_n> E;
  array[N] int<lower=1, upper=EA_n> EA;
  array[E_n] int<lower=1, upper=N> E_rep_row;
  array[E_n] int<lower=0, upper=1> TIRE_COMPACTION_E;

  int<lower=0> J_VOL;
  vector[J_VOL] VOL_obs;
  array[J_VOL] int<lower=1, upper=E_n> VOL_event_id;

  int<lower=1> VIN_n;
  array[E_n] int<lower=1, upper=VIN_n> VIN_E;
  vector[VIN_n] VIN_event;
  int<lower=0> N_VIN_event_miss;
  array[N_VIN_event_miss] int<lower=1, upper=VIN_n> VIN_event_missidx;

  int<lower=1> R_n;
  array[N] int<lower=1, upper=R_n> R;
  vector[R_n] RES_unit;
  vector[R_n] STIR_res_unit;
  array[R_n] int<lower=1, upper=Cr_n> PrevCr_res_unit;
  int<lower=0> N_RES_unit_miss;
  array[N_RES_unit_miss] int<lower=1, upper=R_n> RES_unit_missidx;

  array[N] int<lower=1, upper=Y_n> Y;
  array[N] int<lower=1, upper=F_n> Fu;
  array[N] int<lower=1, upper=S_n> S;
  array[N] int<lower=1, upper=B_n> B;
  array[N] int<lower=1, upper=A_n> A;
  array[N] int<lower=1, upper=Cr_n> Cr;
  array[N] int<lower=1, upper=Cr_n> PrevCr;
  array[N] int<lower=0, upper=1> DUP;
  vector[N] IRR_z;
  vector[N] STIR;

  // Inflow concentration is on the complete event x analyte scaffold.
  vector[EA_n] CIN_EA;
  int<lower=0> N_CIN_EA_miss;
  array[N_CIN_EA_miss] int<lower=1, upper=EA_n> CIN_EA_missidx;

  matrix[Y_n, Y_n] D;
}

parameters {
  // Correlated analyte coefficients: intercept, STIR, and inflow concentration.
  matrix[3, A_n] Z_A;
  vector[3] mu_A;
  cholesky_factor_corr[3] L_A;
  vector<lower=0>[3] sigma_A;

  // Physical block effects.
  matrix[B_n, A_n] Z_B;
  cholesky_factor_corr[B_n] L_B;
  vector<lower=0>[B_n] sigma_B;

  // Observation-layer sampler and flume/collection effects.
  matrix[S_n, A_n] Z_S;
  cholesky_factor_corr[S_n] L_S;
  vector<lower=0>[S_n] sigma_S;
  matrix[F_n, A_n] Z_F;
  cholesky_factor_corr[F_n] L_F;
  vector<lower=0>[F_n] sigma_F;

  // One global standardized runoff-volume effect on concentration. The v3p1
  // analyte hierarchy was not identifiable: its shared scale and every
  // derived analyte slope failed the prespecified R-hat review threshold.
  real beta_vol;
  real mu_beta_irr;
  real<lower=0> sigma_beta_irr;
  vector[A_n] z_beta_irr;
  real mu_beta_dup;
  real<lower=0> sigma_beta_dup;
  vector[A_n] z_beta_dup;

  real beta_vin;
  real beta_res_V;
  real beta_tire_comp_V;
  vector[A_n] beta_res_C;
  real a_V;
  real b_V;

  vector[Cr_n] gamma_Cr_V;
  matrix[A_n, Cr_n] gamma_Cr;
  real<lower=0> sigma_Cr_V;
  vector<lower=0>[A_n] sigma_Cr;

  real<lower=0> sigma_V;

  // Correlated event-level concentration innovations.
  vector<lower=0>[A_n] sigma_C_process;
  cholesky_factor_corr[A_n] L_corr_C_event;
  matrix[E_n, A_n] Z_C_event;

  // Multi-output temporal GP. K_year is a unit-amplitude correlation kernel;
  // sigma_Agp is the single analyte-specific amplitude.
  cholesky_factor_corr[A_n] L_corr_Agp;
  vector<lower=0>[A_n] sigma_Agp;
  real<lower=0> rhosq_year;
  matrix[Y_n, A_n] Z_gp;

  vector[N_CIN_EA_miss] CIN_EA_impute;
  vector[N_VIN_event_miss] VIN_event_impute;
  vector[E_n] z_V_event;

  real<lower=0, upper=1> res_base;
  real b_res_stir;
  vector[Cr_n] gamma_PCr_res;
  real<lower=0> sigma_PCr_res;
  real<lower=0> sigma_res_obs;
  vector[N_RES_unit_miss] RES_unit_miss_logit;

  real mu_log_sigma_C_obs;
  real<lower=0> tau_log_sigma_C_obs;
  vector[A_n] z_log_sigma_C_obs;
  real<lower=0> sigma_VOL_obs;
}

transformed parameters {
  vector[A_n] alpha;
  vector[A_n] beta_stir;
  vector[A_n] beta_cin;
  vector[A_n] beta_irr;
  vector[A_n] beta_dup;
  matrix[A_n, 3] v_A;

  matrix[A_n, B_n] gamma_B;
  matrix[A_n, S_n] gamma_S;
  matrix[A_n, F_n] gamma_F;

  matrix[Y_n, Y_n] K_year;
  matrix[Y_n, Y_n] L_t;
  matrix[A_n, A_n] L_Agp;
  matrix[Y_n, A_n] F_year;
  matrix[A_n, A_n] L_C_event;

  vector<lower=0>[A_n] sigma_C_obs;
  vector[EA_n] CIN_EA_merge;
  vector[VIN_n] VIN_event_merge;
  vector[R_n] mu_res_unit;
  vector[R_n] RES_star_unit;
  vector[E_n] mu_V_event;
  vector[E_n] V_true_event;
  matrix[E_n, A_n] mu_C_event;
  matrix[E_n, A_n] C_true_event;

  // Deterministic row aliases retained for observation diagnostics and legacy
  // figure code. They do not create additional latent states.
  vector[N] CIN_merge;
  vector[N] VIN_merge;
  vector[N] mu_res;
  vector[N] RES_star;
  vector[N] mu_V;
  vector[N] V_true;
  vector[N] mu_C;
  vector[N] C_true;
  real etasq_year = 1;
  real sigma_analyte = mean(sigma_C_process);

  gamma_F = (diag_pre_multiply(sigma_F, L_F) * Z_F)';
  gamma_S = (diag_pre_multiply(sigma_S, L_S) * Z_S)';
  gamma_B = (diag_pre_multiply(sigma_B, L_B) * Z_B)';
  v_A = (diag_pre_multiply(sigma_A, L_A) * Z_A)';

  alpha = mu_A[1] + v_A[, 1];
  beta_stir = mu_A[2] + v_A[, 2];
  beta_cin = mu_A[3] + v_A[, 3];
  beta_irr = mu_beta_irr + sigma_beta_irr * z_beta_irr;
  beta_dup = mu_beta_dup + sigma_beta_dup * z_beta_dup;

  sigma_C_obs = exp(
    mu_log_sigma_C_obs + tau_log_sigma_C_obs * z_log_sigma_C_obs
  );

  K_year = cov_GPL2_corr(D, rhosq_year, 0.01);
  L_t = cholesky_decompose(K_year);
  L_Agp = diag_pre_multiply(sigma_Agp, L_corr_Agp);
  F_year = L_t * Z_gp * L_Agp';

  L_C_event = diag_pre_multiply(sigma_C_process, L_corr_C_event);
  CIN_EA_merge = merge_missing(
    CIN_EA_missidx, CIN_EA, CIN_EA_impute
  );
  VIN_event_merge = merge_missing(
    VIN_event_missidx, VIN_event, VIN_event_impute
  );

  for (u in 1:R_n) {
    mu_res_unit[u] = logit(res_base)
      + b_res_stir * STIR_res_unit[u]
      + gamma_PCr_res[PrevCr_res_unit[u]];
  }
  RES_star_unit = RES_unit;
  for (j in 1:N_RES_unit_miss) {
    RES_star_unit[RES_unit_missidx[j]] = inv_logit(
      RES_unit_miss_logit[j]
    );
  }

  for (e in 1:E_n) {
    int r = E_rep_row[e];
    mu_V_event[e] = a_V
      + b_V * STIR[r]
      + beta_vin * VIN_event_merge[VIN_E[e]]
      + beta_res_V * RES_star_unit[R[r]]
      + beta_tire_comp_V * TIRE_COMPACTION_E[e]
      + gamma_Cr_V[Cr[r]];
  }
  V_true_event = mu_V_event + sigma_V * z_V_event;

  for (e in 1:E_n) {
    int r = E_rep_row[e];
    for (a in 1:A_n) {
      int ea = (e - 1) * A_n + a;
      mu_C_event[e, a] = alpha[a]
        + beta_stir[a] * STIR[r]
        + beta_cin[a] * CIN_EA_merge[ea]
        + beta_vol * V_true_event[e]
        + beta_irr[a] * IRR_z[r]
        + beta_res_C[a] * RES_star_unit[R[r]]
        + gamma_Cr[a, Cr[r]]
        + gamma_B[a, B[r]]
        + F_year[Y[r], a];
    }
  }
  C_true_event = mu_C_event + Z_C_event * L_C_event';
  for (i in 1:N) {
    CIN_merge[i] = CIN_EA_merge[EA[i]];
    VIN_merge[i] = VIN_event_merge[VIN_E[E[i]]];
    mu_res[i] = mu_res_unit[R[i]];
    RES_star[i] = RES_star_unit[R[i]];
    mu_V[i] = mu_V_event[E[i]];
    V_true[i] = V_true_event[E[i]];
    mu_C[i] = mu_C_event[E[i], A[i]];
    C_true[i] = C_true_event[E[i], A[i]];
  }
}

model {
  array[N] int is_C_miss = rep_array(0, N);
  array[R_n] int is_RES_unit_miss = rep_array(0, R_n);

  for (k in 1:N_C_miss) {
    is_C_miss[C_missidx[k]] = 1;
  }
  for (k in 1:N_RES_unit_miss) {
    is_RES_unit_miss[RES_unit_missidx[k]] = 1;
  }

  rhosq_year ~ exponential(1);
  sigma_Agp ~ exponential(1);
  L_corr_Agp ~ lkj_corr_cholesky(3);
  to_vector(Z_gp) ~ std_normal();

  sigma_V ~ exponential(1);
  sigma_C_process ~ exponential(1);
  L_corr_C_event ~ lkj_corr_cholesky(2);
  to_vector(Z_C_event) ~ std_normal();

  sigma_VOL_obs ~ exponential(1);
  z_log_sigma_C_obs ~ std_normal();
  // Retain the measurement-error location on its log-SD scale, but use a
  // unit prior SD. Centering this at zero would imply a median observation
  // error of one full standardized outcome SD.
  mu_log_sigma_C_obs ~ normal(log(0.2), 1);
  tau_log_sigma_C_obs ~ exponential(1);

  a_V ~ normal(0, 1);
  b_V ~ normal(0, 1);
  beta_vin ~ normal(0, 1);

  beta_vol ~ normal(0, 1);
  mu_beta_irr ~ normal(0, 1);
  sigma_beta_irr ~ exponential(1);
  z_beta_irr ~ std_normal();
  mu_beta_dup ~ normal(0, 1);
  sigma_beta_dup ~ exponential(1);
  z_beta_dup ~ std_normal();

  beta_res_V ~ normal(0, 1);
  beta_tire_comp_V ~ normal(0, 1);
  beta_res_C ~ normal(0, 1);

  sigma_Cr_V ~ exponential(1);
  gamma_Cr_V ~ normal(0, sigma_Cr_V);
  sigma_Cr ~ exponential(1);
  for (a in 1:A_n) {
    gamma_Cr[a] ~ normal(0, sigma_Cr[a]);
  }

  sigma_F ~ exponential(1);
  L_F ~ lkj_corr_cholesky(3);
  to_vector(Z_F) ~ std_normal();
  sigma_S ~ exponential(1);
  L_S ~ lkj_corr_cholesky(3);
  to_vector(Z_S) ~ std_normal();
  sigma_B ~ exponential(1);
  L_B ~ lkj_corr_cholesky(3);
  to_vector(Z_B) ~ std_normal();

  sigma_A ~ exponential(1);
  L_A ~ lkj_corr_cholesky(3);
  mu_A ~ normal(0, 1);
  to_vector(Z_A) ~ std_normal();

  CIN_EA_impute ~ std_normal();
  VIN_event_impute ~ std_normal();

  sigma_PCr_res ~ exponential(1);
  gamma_PCr_res ~ normal(0, sigma_PCr_res);
  res_base ~ beta(2, 10);
  b_res_stir ~ normal(0, 1);
  sigma_res_obs ~ exponential(1);

  for (u in 1:R_n) {
    if (is_RES_unit_miss[u] == 0) {
      target += normal_lpdf(
        logit(RES_unit[u]) | mu_res_unit[u], sigma_res_obs
      );
    }
  }
  for (j in 1:N_RES_unit_miss) {
    int idx = RES_unit_missidx[j];
    RES_unit_miss_logit[j] ~ normal(mu_res_unit[idx], sigma_res_obs);
  }

  z_V_event ~ std_normal();
  for (j in 1:J_VOL) {
    VOL_obs[j] ~ normal(
      V_true_event[VOL_event_id[j]], sigma_VOL_obs
    );
  }

  for (i in 1:N) {
    if (is_C_miss[i] == 0) {
      real obs_mu = C_true_event[E[i], A[i]]
        + beta_dup[A[i]] * DUP[i]
        + gamma_S[A[i], S[i]]
        + gamma_F[A[i], Fu[i]];
      if (C_cens[i] == 1) {
        target += normal_lcdf(
          C_cens_limit[i] | obs_mu, sigma_C_obs[A[i]]
        );
      } else {
        target += normal_lpdf(C[i] | obs_mu, sigma_C_obs[A[i]]);
      }
    }
  }
}

generated quantities {
  corr_matrix[A_n] Cor_C_event = multiply_lower_tri_self_transpose(
    L_corr_C_event
  );
  vector[J_VOL] VOL_obs_rep;
  vector[R_n] RES_unit_rep01;
  vector[N] C_rep;
  vector[N] RES_rep01;

  for (j in 1:J_VOL) {
    VOL_obs_rep[j] = normal_rng(
      V_true_event[VOL_event_id[j]], sigma_VOL_obs
    );
  }
  for (u in 1:R_n) {
    RES_unit_rep01[u] = inv_logit(
      normal_rng(mu_res_unit[u], sigma_res_obs)
    );
  }
  for (i in 1:N) {
    real obs_mu = C_true_event[E[i], A[i]]
      + beta_dup[A[i]] * DUP[i]
      + gamma_S[A[i], S[i]]
      + gamma_F[A[i], Fu[i]];
    C_rep[i] = normal_rng(obs_mu, sigma_C_obs[A[i]]);
    RES_rep01[i] = RES_unit_rep01[R[i]];
  }
}
