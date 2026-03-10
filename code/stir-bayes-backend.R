################################################################################
# stir-bayes-backend.R
# Bayesian models linking STIR and water-quality concentrations using rethinking
# Kerbel Long-Term Tillage Impacts Project
# Created by AJ Brown
#
# This script:
#   1. Loads long-format water-quality data with attached STIR metrics.
#   2. Cleans and formats the data for modeling (per-analyte structure).
#   3. Specifies and fits a simple hierarchical ulam model that regresses
#      log(concentration) on standardized STIR with analyte-specific slopes
#      and intercepts.
################################################################################

# ---- 0. Setup ----

# Load required packages
library(rethinking)
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(blastula)

# If you prefer not to rely on setwd, open the Rproj at repo root and run there.
# Then all paths below are relative to the project root.

# ---- 1. Import WQ + STIR data ----

#' Load merged WQ × STIR data
#'
#' @param path Path to wq_with_stir_by_season.csv relative to project root.
#' @param year_max Optional upper bound on Year (e.g., 2023). Use Inf to keep all.
#' @return A tibble with raw merged WQ × STIR data.
load_wq_stir <- function(path = "out/pipeline_csvs/wq_with_stir_by_season.csv",
                         year_max = Inf) {
  
  # Resolve path robustly relative to current working directory
  if (!file.exists(path)) {
    alt <- file.path("..", path)
    if (file.exists(alt)) {
      path <- alt
    } else {
      stop(
        "File '", path, "' not found in '", getwd(), 
        "' or its parent directory."
      )
    }
  }
  
  dat <- readr::read_csv(path, show_col_types = FALSE)
  
  if (is.finite(year_max)) {
    dat <- dat %>% dplyr::filter(Year < year_max)
  }
  
  dat
}

# ---- 2. Data Cleaning ----

#' Clean merged WQ × STIR data
#'
#' Handles WQ flags ("U", "NA", "NA.IRR"), standardizes types,
#' and enforces factor/Date/numeric structure for modeling.
#'
#' @param wq_stir Tibble returned by load_wq_stir()
#' @return Cleaned tibble ready for modeling.
clean_wq_stir <- function(wq_stir) {
  
  wq_stir %>%
    # STEP 2A. Handle WQ flags systematically
    #   "NA"      = sample existed but was not measured      → keep as NA
    #   "U"       = nondetect (below detection limit)        → set to 0 for now
    #   "NA.IRR"  = no runoff occurred                       → drop rows
    dplyr::mutate(
      Result_mg_L        = ifelse(Result_mg_L == "u", "U", Result_mg_L),
      Inflow_Result_mg_L = ifelse(Inflow_Result_mg_L == "u", "U", Inflow_Result_mg_L)
    ) %>%
    # remove no-runoff cases entirely, but keep pure NAs
    dplyr::filter(is.na(Result_mg_L) | Result_mg_L != "NA.IRR") %>%
    # then your case_when and type conversions as before
    dplyr::mutate(
      Result_mg_L = dplyr::case_when(
        Result_mg_L == "U"  ~ "0",
        Result_mg_L == "NA" ~ NA_character_,
        TRUE                ~ Result_mg_L
      ),
      Inflow_Result_mg_L = dplyr::case_when(
        Inflow_Result_mg_L == "U"  ~ "0",
        Inflow_Result_mg_L == "NA" ~ NA_character_,
        TRUE                       ~ Inflow_Result_mg_L
      )
    ) %>%
    
    # STEP 2B. Explicit column type enforcement
    dplyr::mutate(
      # ---- Core numeric concentration fields ----
      Result_mg_L        = as.numeric(Result_mg_L),
      Inflow_Result_mg_L = as.numeric(Inflow_Result_mg_L),
      
      # ---- Identifiers ----
      Treatment  = factor(toupper(Treatment), levels = c("CT", "MT", "ST")),
      Rep        = factor(Rep),
      Analyte    = factor(Analyte),
      Irrigation = factor(Irrigation),
      InflowOutflow = factor(InflowOutflow),
      
      # ---- Dates ----
      Date        = as.Date(Date),
      PlantDate   = as.Date(PlantDate),
      HarvestDate = as.Date(HarvestDate),
      
      # ---- Season metadata ----
      SeasonYear = as.integer(SeasonYear),
      Crop       = factor(Crop),
      
      # ---- STIR predictors ----
      Season_STIR_toDate = as.numeric(Season_STIR_toDate),
      CumAll_STIR_toDate = as.numeric(CumAll_STIR_toDate),
      
      # ---- Boolean fields stored as text in pipeline ----
      Has_Inflow = dplyr::case_when(
        Has_Inflow == "TRUE"  ~ TRUE,
        Has_Inflow == "FALSE" ~ FALSE,
        TRUE                  ~ NA
      ),
      
      NoRunoff = dplyr::case_when(
        NoRunoff == "TRUE"  ~ TRUE,
        NoRunoff == "FALSE" ~ FALSE,
        TRUE                ~ NA
      ),
      
      # ---- Characterize other level variables as categorical ----
      Flag          = factor(Flag),
      Inflow_Flag   = factor(Inflow_Flag),
      FlumeMethod   = factor(FlumeMethod),
      MeasureMethod = factor(MeasureMethod),
      IrrMethod     = factor(IrrMethod),
      TSSMethod     = factor(TSSMethod),
      Lab           = factor(Lab),
      SampleMethod  = factor(SampleMethod),

      # ---- Flow/Volume ----
      Volume = as.numeric(Volume)
    ) %>%
    
    # STEP 2C. Create analyte_abbr for graphing + modeling
    dplyr::mutate(
      analyte_abbr = dplyr::case_when(
        Analyte == "Ammonium(NH4)"  ~ "NH4",
        Analyte == "ICP"            ~ "ICP",
        Analyte == "Nitrate"        ~ "NO3",
        Analyte == "NitrateNitrite" ~ "NOx",
        Analyte == "Nitrite"        ~ "NO2",
        Analyte == "NPOC"           ~ "NPOC",
        Analyte == "OrthoP"         ~ "OP",
        Analyte == "Selenium"       ~ "Se",
        Analyte == "TDS"            ~ "TDS",
        Analyte == "TKN"            ~ "TKN",
        Analyte == "TotalN"         ~ "TN",
        Analyte == "TotalP"         ~ "TP",
        Analyte == "TSP"            ~ "TSP",
        Analyte == "TSS"            ~ "TSS",
        TRUE                        ~ NA_character_
      ),
      analyte_abbr = factor(analyte_abbr,
                            levels = c("NH4","ICP","NO3","NOx","NO2","NPOC",
                                       "OP","Se","TDS","TKN","TN","TP","TSP",
                                       "TSS")
      )
    ) %>%
    
    # STEP 2D. Standardize OUT/INFLOW per-analyte concentrations
    dplyr::group_by(Analyte) %>%
    dplyr::mutate(
      cout_z = rethinking::standardize(Result_mg_L),
      cin_z  = rethinking::standardize(Inflow_Result_mg_L)
    ) %>%
    dplyr::ungroup() %>%
    
    # STEP 2E. Standardize STIR and volume metrics globally
    dplyr::mutate(
      stir_season_z = rethinking::standardize(Season_STIR_toDate),
      stir_cumall_z = rethinking::standardize(CumAll_STIR_toDate),
      volume_z      = rethinking::standardize(Volume)
    )
  
  
}

# ---- Use the functions ----

if (interactive()) {
  # Only run this section in interactive sessions
  # Step 1: load raw merged data
  wq_stir <- load_wq_stir(
    path = "out/pipeline_csvs/wq_with_stir_by_season.csv",
    year_max = Inf
  )
  
  # Quick peek at raw
  glimpse(wq_stir)
  
  # Step 2: clean and type-enforce
  wq_stir_clean <- clean_wq_stir(wq_stir)
}



# ---- Helper functions ----

# Create distance matrix for year effects
# Usage:
#   make_year_dist_mat(2011, 2025)              # contiguous years (old behavior)
#   make_year_dist_mat(unique(d_mod$Year))      # only observed years (recommended)
make_year_dist_mat <- function(start_or_years, end_year = NULL) {
  if (is.null(end_year)) {
    # Case 1: user passed a vector of years
    years <- sort(unique(start_or_years))
  } else {
    # Case 2: user passed start_year, end_year
    years <- seq(start_or_years, end_year)
  }
  
  n_years <- length(years)
  
  # pairwise absolute differences using outer
  dist_matrix <- abs(outer(years, years, "-"))
  
  rownames(dist_matrix) <- years
  colnames(dist_matrix) <- years
  
  dist_matrix
}


# cmdstan dashboard for convergence
cmdstan_dashboard <- function(
    fit,
    warmup = FALSE,
    plot = TRUE,
    trank = TRUE,
    vars = NULL,
    ess_col = c("ess_bulk","ess_tail")
) {
  
  if (!inherits(fit, "CmdStanMCMC")) {
    stop("cmdstan_dashboard(): 'fit' must be a CmdStanMCMC object from cmdstanr.")
  }
  ess_col <- match.arg(ess_col)
  
  # ---- sampler diagnostics (fast) ----
  x_raw <- fit$sampler_diagnostics(inc_warmup = warmup)
  
  # Force to plain base array (prevents S3 '[' dropping dims unexpectedly)
  x <- if (is.array(x_raw)) x_raw else tryCatch(unclass(x_raw), error = function(e) x_raw)
  if (!is.array(x) || length(dim(x)) < 2) {
    stop("cmdstan_dashboard(): sampler_diagnostics did not return an array-like object.")
  }
  
  # Expect dims: iter x chain x diag (but we guard)
  d <- dim(x)
  n_iter  <- d[1]
  n_chain <- d[2]
  n_samps <- n_iter * n_chain
  
  diag_names <- dimnames(x)[[3]]
  
  # Extract diag by name and force to n_iter x n_chain matrix
  diag_matrix <- function(name) {
    if (is.null(diag_names) || !(name %in% diag_names) || length(d) < 3) return(NULL)
    
    k <- match(name, diag_names)
    a <- x[, , k, drop = FALSE]   # iter x chain x 1 (ideally)
    a <- a[, , 1, drop = TRUE]    # should be iter x chain OR vector
    
    if (is.matrix(a) && all(dim(a) == c(n_iter, n_chain))) return(a)
    
    # If it collapsed to a vector, try to reshape if lengths match
    if (is.atomic(a) && length(a) == n_samps) {
      return(matrix(a, nrow = n_iter, ncol = n_chain))
    }
    
    NULL
  }
  
  div_mat   <- diag_matrix("divergent__")
  energy    <- diag_matrix("energy__")
  treedepth <- diag_matrix("treedepth__")
  n_leap    <- diag_matrix("n_leapfrog__")
  stepsize  <- diag_matrix("stepsize__")
  
  n_div <- if (!is.null(div_mat)) sum(div_mat) else NA_integer_
  
  # E-BFMI per chain (only if energy is truly matrix)
  ebfmi <- NULL
  if (is.matrix(energy)) {
    ebfmi <- sapply(seq_len(n_chain), function(ch) {
      e <- as.numeric(energy[, ch])
      num <- mean(diff(e)^2)
      den <- stats::var(e)
      if (is.finite(num) && is.finite(den) && den > 0) num / den else NA_real_
    })
    names(ebfmi) <- paste0("chain", seq_len(n_chain))
  }
  
  # ---- summary (can be slow if vars is NULL) ----
  s <- if (is.null(vars)) fit$summary() else fit$summary(variables = vars)
  
  rhat_vals <- if ("rhat" %in% names(s)) s$rhat else rep(NA_real_, nrow(s))
  ess_vals  <- if (ess_col %in% names(s)) s[[ess_col]] else rep(NA_real_, nrow(s))
  
  # ---- plotting ----
  if (plot) {
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar), add = TRUE)
    par(mfrow = c(2,2), cex.axis = 0.9, cex.lab = 1)
    
    # Panel 1: ESS vs Rhat
    ok <- is.finite(rhat_vals) & is.finite(ess_vals) & ess_vals > 0
    y_rhat <- rhat_vals[ok]
    x_ess  <- ess_vals[ok]
    plot(
      x_ess, y_rhat,
      xlab = sprintf("%s effective sample size", ess_col),
      ylab = "Rhat",
      ylim = c(1, max(1.1, y_rhat, na.rm = TRUE)),
      log = "y",
      pch = 1,        # (1) hollow points
      cex = 0.6
    )
    abline(v = 0.1 * n_samps, col = "red")
    abline(v = n_samps, col = "gray60")
    abline(h = 1, lty = 2, col = "gray40")
    abline(h = 1.04, lty = 2, col = "gray40")  # (2) guide at 1.04
    
    # Panel 2: energy density (only if matrix)
    if (is.matrix(energy)) {
      e_all <- as.numeric(energy)
      d2 <- stats::density(
        e_all,
        adjust = 0.2,  # (3) less smoothing (was 0.8)
        na.rm = TRUE
      )
      plot(d2, main = "HMC energy", xlab = "energy__", ylab = "density")
      mu <- mean(e_all, na.rm = TRUE)
      sig <- stats::sd(e_all, na.rm = TRUE)
      if (is.finite(mu) && is.finite(sig) && sig > 0) {
        curve(stats::dnorm(x, mu, sig), add = TRUE, lwd = 2)
      }
      if (!is.null(ebfmi)) {
        mtext(
          sprintf("E-BFMI: %s", paste(sprintf("%.2f", ebfmi), collapse = ", ")),
          side = 3, line = 0.2, cex = 0.8
        )
      }
    } else {
      plot.new(); title("HMC energy")
      text(0.5, 0.5, "energy__ not available as iter×chain matrix", cex = 1.0)
    }
    
    # Panel 3: divergences
    plot.new(); title("Sampler issues")
    if (is.na(n_div)) {
      text(0.5, 0.75, "divergent__ not available", cex = 1.1)
    } else {
      text(0.5, 0.78, n_div, cex = 4)
      text(0.5, 0.58, "Divergent transitions", cex = 1.1)
    }
    
    # ---- Panel 4: lp__ trace plot (CmdStanR-safe, base R) ----
    trace_pars <- "lp__"
    
    if (!requireNamespace("posterior", quietly = TRUE)) {
      plot.new()
      title("Trace plot")
      text(0.5, 0.5, "Install the 'posterior' package to enable trace plots.", cex = 0.9)
    } else {
      
      dr <- fit$draws(
        variables = trace_pars,
        inc_warmup = warmup,
        format = "draws_array"
      )
      
      d_draws <- dim(dr)
      n_iter_draws  <- d_draws[1]
      n_chain_draws <- d_draws[2]
      var_names <- dimnames(dr)$variable
      
      v <- var_names[1]
      mat <- dr[, , v, drop = TRUE]
      if (!is.matrix(mat)) {
        mat <- matrix(mat, nrow = n_iter_draws, ncol = n_chain_draws)
      }
      
      chain_cols <- grDevices::hcl.colors(n_chain_draws, "Dark 3")
      ylim <- range(mat, finite = TRUE)
      
      plot(
        NA, xlim = c(1, n_iter_draws), ylim = ylim,
        xlab = "Iteration", ylab = v,
        main = paste0("Trace: ", v),
        bty = "l"
      )
      
      for (ch in seq_len(n_chain_draws)) {
        lines(seq_len(n_iter_draws), mat[, ch], col = chain_cols[ch], lwd = 0.8)
      }
      
      legend(
        "topright",
        legend = paste0("chain ", seq_len(n_chain_draws)),
        col = chain_cols,
        lty = 1,
        cex = 0.7,
        bty = "n"
      )
    }
  }
  
  invisible(list(
    sampler_diagnostics = x_raw,
    summary = s,
    n_samples = n_samps,
    n_divergent = n_div,
    ebfmi = ebfmi,
    stepsize = stepsize,
    n_leapfrog = n_leap
  ))
}

# email notification helper (requires mailR package and SMTP setup)
# Optional email notification after long sampling runs.
# To enable, define these environment variables before knitting/running:
#   STAN_NOTIFY_TO
#   STAN_NOTIFY_FROM
#   STAN_NOTIFY_USER
#   STAN_NOTIFY_PASS
#   STAN_NOTIFY_HOST   (for example: smtp.gmail.com)
#   STAN_NOTIFY_PORT   (for example: 465 or 587)
# This function is safe to leave in place when no SMTP credentials are set.

notify_sampling_done <- function(subject = NULL, body = NULL) {
  
  required <- c(
    "STAN_NOTIFY_TO",
    "STAN_NOTIFY_FROM",
    "STAN_NOTIFY_USER",
    "STAN_NOTIFY_PASS",
    "STAN_NOTIFY_HOST",
    "STAN_NOTIFY_PORT"
  )
  
  vals <- Sys.getenv(required, unset = "")
  names(vals) <- required
  
  if (any(vals == "")) {
    missing_vars <- names(vals)[vals == ""]
    message(
      "Sampling-finished email skipped: missing environment variable(s): ",
      paste(missing_vars, collapse = ", ")
    )
    return(invisible(FALSE))
  }
  
  if (!requireNamespace("blastula", quietly = TRUE)) {
    message("Sampling-finished email skipped: install.packages('blastula') is required.")
    return(invisible(FALSE))
  }
  
  smtp_port <- suppressWarnings(as.integer(Sys.getenv("STAN_NOTIFY_PORT")))
  if (is.na(smtp_port)) {
    message("Sampling-finished email skipped: STAN_NOTIFY_PORT is not a valid integer.")
    return(invisible(FALSE))
  }
  
  use_ssl <- identical(smtp_port, 465L)
  
  if (is.null(subject)) {
    subject <- sprintf("Stan sampling finished: %s", model_version)
  }
  
  if (is.null(body)) {
    body <- paste0(
      "Your CmdStan fit finished sampling.\n\n",
      "Model version: ", model_version, "\n",
      "Time finished: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
    )
  }
  
  email_obj <- blastula::compose_email(
    body = blastula::md(body)
  )
  
  tryCatch({
    blastula::smtp_send(
      email = email_obj,
      from = Sys.getenv("STAN_NOTIFY_FROM"),
      to = Sys.getenv("STAN_NOTIFY_TO"),
      subject = subject,
      credentials = blastula::creds_envvar(
        user = Sys.getenv("STAN_NOTIFY_USER"),
        pass_envvar = "STAN_NOTIFY_PASS",
        host = Sys.getenv("STAN_NOTIFY_HOST"),
        port = smtp_port,
        use_ssl = use_ssl
      )
    )
    
    message("Sampling-finished email sent.")
    invisible(TRUE)
    
  }, error = function(e) {
    message("Sampling-finished email failed: ", conditionMessage(e))
    invisible(FALSE)
  })
}


