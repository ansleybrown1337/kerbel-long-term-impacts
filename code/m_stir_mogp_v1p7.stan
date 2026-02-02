// m_stir_mogp_v1p7.stan
// Model 1p7: adds explicit measurement error via latent "true" V and C,
// and adds a logit-normal latent "true" residue cover model (with STIR + Crop Type parents).

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
    int N = dims(x_obs)[1];
    int N_miss = dims(x_miss)[1];
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
  int D_n;  // retained for compatibility, not used directly
  int F_n;
  int S_n;
  int B_n;
  int A_n;
  int Cr_n;
  int N;

  // Observed outcomes (can contain arbitrary values at missing rows; those rows are excluded from obs likelihood)
  vector[N] C;
  int<lower=0> N_C_miss;
  array[N_C_miss] int<lower=1, upper=N> C_missidx;

  vector[N] VOL;
  int<lower=0> N_VOL_miss;
  array[N_VOL_miss] int<lower=1, upper=N> VOL_missidx;

  // Per-row observation SDs for measurement error (same scale as C and VOL)
  // For missing rows, values can be anything (they are ignored).

  // Residue cover observed as proportion in (0,1). Use NA in R for missing and pass missidx.
  // For observed rows, RES must be strictly between 0 and 1 (apply an epsilon adjustment in R).
  vector[N] RES;
  int<lower=0> N_RES_miss;
  array[N_RES_miss] int<lower=1, upper=N> RES_missidx;

  // indices
  array[N] int<lower=1, upper=Y_n>  Y;
  array[N] int<lower=1, upper=F_n>  Fu;
  array[N] int<lower=1, upper=S_n>  S;
  array[N] int<lower=1, upper=B_n>  B;
  array[N] int<lower=1, upper=A_n>  A;
  array[N] int<lower=1, upper=Cr_n> Cr;
  array[N] int DUP;  // 0/1
  array[N] int IRR;  // count, treated as numeric

  // predictors
  vector[N] STIR;
  vector[N] CIN;
  int<lower=0> N_CIN_miss;
  array[N_CIN_miss] int<lower=1, upper=N> CIN_missidx;

  // year distance matrix for GP
  matrix[Y_n, Y_n] D;
}

parameters {
  // analyte level MVN (non centered)
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
  real beta_irr;

  // Residue effects in outcome models (now applied on logit scale of residue)
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
  vector<lower=0>[A_n] sigma_analyte;  // process SD for C_true around mu_C
  real<lower=0> sigma_V;               // process SD for V_true around mu_V

  // multi output GP hyperparameters (analyte covariance)
  cholesky_factor_corr[A_n] L_corr_Agp;
  vector<lower=0>[A_n] sigma_Agp;

  // GP kernel hyperparameters (time / year)
  real<lower=0> etasq_year;
  real<lower=0> rhosq_year;

  // non centered latent GP
  matrix[Y_n, A_n] Z_gp;

  // missing CIN imputation
  vector[N_CIN_miss] CIN_impute;

  // latent true outcomes (all rows)
  vector[N] V_true;
  vector[N] C_true;

  // Residue: latent true residue proportion in (0,1)
  vector<lower=0, upper=1>[N] RES_true;

  // Residue submodel (mean on logit scale via STIR + Crop)
  real<lower=0, upper=1> res_base; // baseline residue proportion
  real b_res_stir;            // effect of STIR on residue
  vector[Cr_n] gamma_Cr_res;  // crop partial pooling
  real<lower=0> sigma_Cr_res;

  // Process concentration: how tightly RES_true concentrates around mean
  real<lower=0> phi_res_proc;

  // Observation concentration: how tightly RES observations concentrate around RES_true
  real<lower=0> phi_res_obs;

  // Observation error (estimated, not supplied as data)
  vector<lower=0>[A_n] sigma_C_obs;   // analyte-specific concentration measurement SD (on cout_z scale)
  real<lower=0> sigma_VOL_obs;       // volume measurement SD (on volume_z scale)
}

transformed parameters {
  // analyte level effects
  vector[A_n] alpha;
  vector[A_n] beta_stir;
  vector[A_n] beta_cin;
  vector[A_n] beta_dup;
  matrix[A_n, 4] v_A;

  // random effects
  matrix[A_n, B_n] gamma_B;
  matrix[A_n, S_n] gamma_S;
  matrix[A_n, F_n] gamma_F;

  // multi output GP structures
  matrix[Y_n, Y_n] K_year;
  matrix[Y_n, Y_n] L_t;
  matrix[A_n, A_n] L_Agp;
  matrix[Y_n, A_n] F_year;

  // non centered MVNs for random effects
  gamma_F = (diag_pre_multiply(sigma_F, L_F) * Z_F)';
  gamma_S = (diag_pre_multiply(sigma_S, L_S) * Z_S)';
  gamma_B = (diag_pre_multiply(sigma_B, L_B) * Z_B)';
  v_A     = (diag_pre_multiply(sigma_A, L_A) * Z_A)';

  beta_dup  = mu_A[4] + v_A[, 4];
  beta_cin  = mu_A[3] + v_A[, 3];
  beta_stir = mu_A[2] + v_A[, 2];
  alpha     = mu_A[1] + v_A[, 1];

  // multi output GP: separable covariance Σ_A ⊗ K_year
  K_year = cov_GPL2(D, etasq_year, rhosq_year, 0.01);
  L_t    = cholesky_decompose(K_year);
  L_Agp  = diag_pre_multiply(sigma_Agp, L_corr_Agp);

  // non centered: F_year = L_t * Z_gp * L_Agp'
  F_year = L_t * Z_gp * L_Agp';
}

model {
  vector[N] mu_C;
  vector[N] mu_V;
  vector[N] CIN_merge;

  // residue helper quantities
  vector[N] mu_res;
  vector[N] RES_used_logit;

  // missingness masks
  array[N] int is_C_miss;
  array[N] int is_VOL_miss;
  array[N] int is_RES_miss;

  // initialize masks
  for (i in 1:N) {
    is_C_miss[i] = 0;
    is_VOL_miss[i] = 0;
    is_RES_miss[i] = 0;
  }
  for (k in 1:N_C_miss)   is_C_miss[C_missidx[k]] = 1;
  for (k in 1:N_VOL_miss) is_VOL_miss[VOL_missidx[k]] = 1;
  for (k in 1:N_RES_miss) is_RES_miss[RES_missidx[k]] = 1;

  // GP priors
  etasq_year ~ exponential(2);
  rhosq_year ~ exponential(0.5);
  sigma_Agp  ~ exponential(1);
  L_corr_Agp ~ lkj_corr_cholesky(2);
  to_vector(Z_gp) ~ normal(0, 1);

  // priors (existing)
  sigma_V       ~ exponential(1);
  sigma_analyte ~ exponential(1);

  b_V      ~ normal(0, 1);
  a_V      ~ normal(0, 1);
  beta_irr ~ normal(0, 1);
  beta_vol ~ normal(0, 1);

  beta_res_V ~ normal(0, 1);
  beta_res_C ~ normal(0, 1);

  sigma_Cr_V ~ exponential(1);
  gamma_Cr_V ~ normal(0, sigma_Cr_V);

  sigma_Cr ~ exponential(1);
  for (a in 1:A_n) {
    gamma_Cr[a] ~ normal(0, sigma_Cr[a]);
  }

  sigma_F ~ exponential(1);
  L_F     ~ lkj_corr_cholesky(2);
  to_vector(Z_F) ~ normal(0, 1);

  sigma_S ~ exponential(1);
  L_S     ~ lkj_corr_cholesky(2);
  to_vector(Z_S) ~ normal(0, 1);

  sigma_B ~ exponential(1);
  L_B     ~ lkj_corr_cholesky(2);
  to_vector(Z_B) ~ normal(0, 1);

  sigma_A ~ exponential(1);
  L_A     ~ lkj_corr_cholesky(2);
  mu_A    ~ normal(0, 1);
  to_vector(Z_A) ~ normal(0, 1);

  // CIN missing data model
  CIN_merge = merge_missing(CIN_missidx, CIN, CIN_impute);
  CIN_merge ~ normal(0, 1);

  // -------------------------
  // Residue submodel (latent truth + observation model)
  // Parents in DAG: STIR and Crop Type
  //
  // Data: RES is observed residue proportion in (0,1). Missing rows are indexed by RES_missidx.
  // Latent: RES_true is a per-row latent "true" residue proportion.
  //
  // Process model:
  //   logit(mean_res) = logit(res_base) + b_res_stir * STIR + gamma_Cr_res[Crop]
  //   RES_true ~ Beta(mean_res * phi_res_proc, (1-mean_res) * phi_res_proc)
  //
  // Observation model (when RES observed):
  //   RES ~ Beta(RES_true * phi_res_obs, (1-RES_true) * phi_res_obs)
  //
  // Baseline prior: res_base ~ Beta(2,10). If you want mean ~10%, consider Beta(2,18).
  // -------------------------
  sigma_Cr_res  ~ exponential(1);
  gamma_Cr_res  ~ normal(0, sigma_Cr_res);

  res_base    ~ beta(2, 10);
  b_res_stir  ~ normal(0, 1);

  phi_res_proc ~ exponential(1);
  phi_res_obs  ~ exponential(1);

  for (i in 1:N) {
    mu_res[i] = logit(res_base) + b_res_stir * STIR[i] + gamma_Cr_res[Cr[i]];
    {
      real mean_res = inv_logit(mu_res[i]);
      // Latent "true" residue
      RES_true[i] ~ beta(mean_res * phi_res_proc, (1 - mean_res) * phi_res_proc);
    }
  }

  // Observation model for residue (skip missing)
  for (i in 1:N) {
    if (is_RES_miss[i] == 0) {
      // RES is already in (0,1); ensure upstream epsilon adjustment so Beta is valid.
      RES[i] ~ beta(RES_true[i] * phi_res_obs, (1 - RES_true[i]) * phi_res_obs);
    }
  }

  // Use residue on logit scale in outcome models
  for (i in 1:N) {
    RES_used_logit[i] = logit(RES_true[i]);
  }

  // -------------------------
  // Latent PROCESS model for volume truth
  // -------------------------
  for (i in 1:N) {
    mu_V[i] = a_V + b_V * STIR[i] + beta_res_V * RES_used_logit[i] + gamma_Cr_V[Cr[i]];
  }
  V_true ~ normal(mu_V, sigma_V);

  // -------------------------
  // Latent PROCESS model for concentration truth (uses latent V_true)
  // -------------------------
  for (i in 1:N) {
    mu_C[i] =
      alpha[A[i]] +
      beta_stir[A[i]] * STIR[i] +
      beta_cin[A[i]]  * CIN_merge[i] +
      beta_vol        * V_true[i] +
      beta_irr        * IRR[i] +
      beta_dup[A[i]]  * DUP[i] +
      beta_res_C[A[i]] * RES_used_logit[i] +
      gamma_Cr[A[i], Cr[i]] +
      gamma_B[A[i], B[i]] +
      gamma_S[A[i], S[i]] +
      gamma_F[A[i], Fu[i]] +
      F_year[Y[i], A[i]];
  }
  for (i in 1:N) {
    C_true[i] ~ normal(mu_C[i], sigma_analyte[A[i]]);
  }

  // -------------------------
  // OBSERVATION models (measurement error) for VOL and C
  // Only apply likelihood to observed rows.
  // -------------------------
  for (i in 1:N) {
    if (is_VOL_miss[i] == 0) {
      target += normal_lpdf(VOL[i] | V_true[i], sigma_VOL_obs);
    }
    if (is_C_miss[i] == 0) {
      target += normal_lpdf(C[i] | C_true[i], sigma_C_obs[A[i]]);
    }
  }
}

generated quantities {
  vector[N] VOL_rep;
  vector[N] C_rep;
  vector[N] RES_rep01;

  for (i in 1:N) {
    VOL_rep[i] = normal_rng(V_true[i], sigma_VOL_obs);
    C_rep[i]   = normal_rng(C_true[i], sigma_C_obs[A[i]]);

    // Replicated residue observation (on proportion scale)
    // For missing rows, this is a prior predictive replicate.
    RES_rep01[i] = beta_rng(RES_true[i] * phi_res_obs, (1 - RES_true[i]) * phi_res_obs);
  }
}
