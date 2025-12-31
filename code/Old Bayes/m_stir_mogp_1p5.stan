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
    int D_n;  // kept for compatibility with existing R code, not used
    int F_n;
    int S_n;
    int B_n;
    int A_n;
    int N;

    // outcome with missingness
    vector[N] C;
    int<lower=0> N_C_miss;
    array[N_C_miss] int<lower=1, upper=N> C_missidx;

    // indices
    array[N] int<lower=1, upper=Y_n>  Y;
    array[N] int<lower=1, upper=F_n>  Fu;
    array[N] int<lower=1, upper=S_n>  S;
    array[N] int<lower=1, upper=B_n>  B;
    array[N] int<lower=1, upper=A_n>  A;
    array[N] int DUP;  // 0 or 1, treated as numeric
    array[N] int IRR;  // irrigation count, treated as numeric

    // volume with missingness
    vector[N] VOL;
    int<lower=0> N_VOL_miss;
    array[N_VOL_miss] int<lower=1, upper=N> VOL_missidx;

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
    vector<lower = 0>[4] sigma_A;

    // analyte by block MVN
    matrix[B_n, A_n] Z_B;
    cholesky_factor_corr[B_n] L_B;
    vector<lower = 0>[B_n] sigma_B;

    // analyte by sampler MVN
    matrix[S_n, A_n] Z_S;
    cholesky_factor_corr[S_n] L_S;
    vector<lower = 0>[S_n] sigma_S;

    // analyte by flume MVN
    matrix[F_n, A_n] Z_F;
    cholesky_factor_corr[F_n] L_F;
    vector<lower = 0>[F_n] sigma_F;

    // fixed effects
    real beta_vol;
    real beta_irr;
    real a_V;
    real b_V;

    // residual scales
    vector<lower = 0>[A_n] sigma_analyte;
    real<lower = 0> sigma_V;

    // multi output GP hyperparameters (analyte covariance)
    cholesky_factor_corr[A_n] L_corr_Agp;
    vector<lower = 0>[A_n] sigma_Agp;

    // GP kernel hyperparameters (time / year)
    real<lower = 0> etasq_year;
    real<lower = 0> rhosq_year;

    // non centered latent GP
    matrix[Y_n, A_n] Z_gp;

    // missing data parameters
    vector[N_VOL_miss] VOL_impute;
    vector[N_CIN_miss] CIN_impute;
    vector[N_C_miss]   C_impute;
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

    beta_dup   = mu_A[4] + v_A[, 4];
    beta_cin   = mu_A[3] + v_A[, 3];
    beta_stir  = mu_A[2] + v_A[, 2];
    alpha      = mu_A[1] + v_A[, 1];

    // multi output GP: separable covariance Σ_A ⊗ K_year
    K_year = cov_GPL2(D, etasq_year, rhosq_year, 0.01);
    L_t    = cholesky_decompose(K_year);
    L_Agp  = diag_pre_multiply(sigma_Agp, L_corr_Agp);

    // non centered: F_year = L_t * Z_gp * L_Agp'
    F_year = L_t * Z_gp * L_Agp';
}

model {
    vector[N] mu_C;
    vector[N] VOL_merge;
    vector[N] mu_V;
    vector[N] CIN_merge;
    vector[N] C_merge;

    // GP hyperpriors
    etasq_year  ~ exponential(2);
    rhosq_year  ~ exponential(0.5);

    sigma_Agp   ~ exponential(1);
    L_corr_Agp  ~ lkj_corr_cholesky(2);

    // non centered latent GP
    to_vector(Z_gp) ~ normal(0, 1);

    // existing priors
    sigma_V       ~ exponential(1);
    sigma_analyte ~ exponential(1);

    b_V      ~ normal(0, 1);
    a_V      ~ normal(0, 1);
    beta_irr ~ normal(0, 1);
    beta_vol ~ normal(0, 1);

    sigma_F      ~ exponential(1);
    L_F          ~ lkj_corr_cholesky(2);
    to_vector(Z_F) ~ normal(0, 1);

    sigma_S      ~ exponential(1);
    L_S          ~ lkj_corr_cholesky(2);
    to_vector(Z_S) ~ normal(0, 1);

    sigma_B      ~ exponential(1);
    L_B          ~ lkj_corr_cholesky(2);
    to_vector(Z_B) ~ normal(0, 1);

    sigma_A      ~ exponential(1);
    L_A          ~ lkj_corr_cholesky(2);
    mu_A         ~ normal(0, 1);
    to_vector(Z_A) ~ normal(0, 1);

    // CIN missing data model
    CIN_merge = merge_missing(CIN_missidx, CIN, CIN_impute);
    CIN_merge ~ normal(0, 1);

    // Volume model with missing data
    for (i in 1:N) {
        mu_V[i] = a_V + b_V * STIR[i];
    }
    VOL_merge = merge_missing(VOL_missidx, VOL, VOL_impute);
    VOL_merge ~ normal(mu_V, sigma_V);

    // Main concentration model
    for (i in 1:N) {
        mu_C[i] =
            alpha[A[i]] +
            beta_stir[A[i]] * STIR[i] +
            beta_cin[A[i]] * CIN_merge[i] +
            beta_vol * VOL_merge[i] +
            beta_irr * IRR[i] +
            beta_dup[A[i]] * DUP[i] +
            gamma_B[A[i], B[i]] +
            gamma_S[A[i], S[i]] +
            gamma_F[A[i], Fu[i]] +
            F_year[Y[i], A[i]];
    }

    // Outcome with missing data imputation
    C_merge = merge_missing(C_missidx, C, C_impute);
    C_merge ~ normal(mu_C, sigma_analyte[A]);
}
