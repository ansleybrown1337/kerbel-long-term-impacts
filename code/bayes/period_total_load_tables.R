# Reusable saved-output post-processing for Bayesian period-total load tables.
#
# The input is the replicate-aware annual draw export created by the physical-
# event workflow. Period totals therefore sum annual mean-per-treatment-plot
# loads within each posterior draw; treatment contrasts are then calculated
# draw-wise before their 95% intervals are summarized.

period_table_fmt_ci <- function(mean, low, high, digits = 2L) {
  ifelse(
    is.finite(mean) & is.finite(low) & is.finite(high),
    paste0(
      formatC(mean, format = "f", digits = digits),
      "\n(",
      formatC(low, format = "f", digits = digits),
      ", ",
      formatC(high, format = "f", digits = digits),
      ")"
    ),
    NA_character_
  )
}

period_table_fmt_pct_ci <- function(mean, low, high, digits = 1L) {
  ifelse(
    is.finite(mean) & is.finite(low) & is.finite(high),
    paste0(
      formatC(mean, format = "f", digits = digits),
      "%\n(",
      formatC(low, format = "f", digits = digits),
      "%, ",
      formatC(high, format = "f", digits = digits),
      "%)"
    ),
    NA_character_
  )
}

validate_period_annual_draws <- function(annual_draws) {
  required <- c(
    "draw", "draw_id", "Year", "analyte", "treatment", "load_g"
  )
  missing <- setdiff(required, names(annual_draws))
  if (length(missing) > 0L) {
    stop(
      "Annual draw table is missing required column(s): ",
      paste(missing, collapse = ", ")
    )
  }

  checked <- annual_draws %>%
    dplyr::transmute(
      draw = as.integer(.data$draw),
      draw_id = as.integer(.data$draw_id),
      Year = as.integer(.data$Year),
      analyte = as.character(.data$analyte),
      treatment = as.character(.data$treatment),
      load_g = as.numeric(.data$load_g)
    )
  if (anyNA(checked)) {
    stop("Annual draw table contains missing required values.")
  }
  if (any(!is.finite(checked$load_g)) || any(checked$load_g < 0)) {
    stop("Annual draw loads must be finite and nonnegative.")
  }
  if (anyDuplicated(
    checked[c("draw", "draw_id", "Year", "analyte", "treatment")]
  )) {
    stop("Annual draw keys are not unique.")
  }
  if (!setequal(unique(checked$treatment), c("CT", "MT", "ST"))) {
    stop("Annual draw table must contain exactly CT, MT, and ST treatments.")
  }
  checked
}

build_period_total_load_tables <- function(
    annual_draws,
    start_year,
    end_year,
    prob_ci = 0.95
) {
  start_year <- as.integer(start_year)
  end_year <- as.integer(end_year)
  if (is.na(start_year) || is.na(end_year) || start_year > end_year) {
    stop("Period start_year and end_year must define a valid inclusive range.")
  }
  if (!is.finite(prob_ci) || prob_ci <= 0 || prob_ci >= 1) {
    stop("prob_ci must be strictly between 0 and 1.")
  }

  checked <- validate_period_annual_draws(annual_draws)
  period_draws <- checked %>%
    dplyr::filter(.data$Year >= start_year, .data$Year <= end_year)
  expected_years <- seq.int(start_year, end_year)
  observed_years <- sort(unique(period_draws$Year))
  if (!identical(observed_years, expected_years)) {
    stop(
      "Requested period is not fully represented in annual draws. Expected ",
      paste(expected_years, collapse = ", "), "; observed ",
      paste(observed_years, collapse = ", "), "."
    )
  }

  support <- period_draws %>%
    dplyr::group_by(
      .data$draw, .data$draw_id, .data$analyte, .data$treatment
    ) %>%
    dplyr::summarise(
      n_years = dplyr::n_distinct(.data$Year),
      .groups = "drop"
    )
  if (any(support$n_years != length(expected_years))) {
    stop("At least one draw/analyte/treatment has incomplete annual support.")
  }

  alpha_ci <- 1 - prob_ci
  load_draws <- period_draws %>%
    dplyr::mutate(
      treatment = factor(
        .data$treatment, levels = c("CT", "MT", "ST")
      )
    ) %>%
    dplyr::group_by(.data$draw, .data$draw_id, .data$analyte, .data$treatment) %>%
    dplyr::summarise(
      load_sum_kg = sum(.data$load_g) / 1000,
      .groups = "drop"
    )

  load_draws_wide <- load_draws %>%
    tidyr::pivot_wider(
      names_from = "treatment",
      values_from = "load_sum_kg"
    )
  if (any(!is.finite(load_draws_wide$CT)) ||
      any(load_draws_wide$CT <= 0) ||
      any(!is.finite(load_draws_wide$MT)) ||
      any(!is.finite(load_draws_wide$ST))) {
    stop(
      "Draw-wise reductions require positive CT totals and finite MT/ST totals."
    )
  }
  load_draws_wide <- load_draws_wide %>%
    dplyr::mutate(
      MT_pct_reduction_from_CT = 100 * (1 - .data$MT / .data$CT),
      ST_pct_reduction_from_CT = 100 * (1 - .data$ST / .data$CT)
    )

  load_summary <- load_draws %>%
    dplyr::group_by(.data$analyte, .data$treatment) %>%
    dplyr::summarise(
      load_sum_mean_kg = mean(.data$load_sum_kg),
      load_sum_low_kg = stats::quantile(
        .data$load_sum_kg, probs = alpha_ci / 2, names = FALSE
      ),
      load_sum_high_kg = stats::quantile(
        .data$load_sum_kg, probs = 1 - alpha_ci / 2, names = FALSE
      ),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      treatment = factor(
        as.character(.data$treatment), levels = c("CT", "MT", "ST")
      )
    ) %>%
    dplyr::arrange(.data$analyte, .data$treatment)

  numeric <- load_summary %>%
    dplyr::mutate(col = paste0(as.character(.data$treatment), "_mod")) %>%
    dplyr::select(
      "analyte", "col",
      "load_sum_mean_kg",
      "load_sum_low_kg",
      "load_sum_high_kg"
    ) %>%
    tidyr::pivot_wider(
      names_from = "col",
      values_from = c(
        "load_sum_mean_kg",
        "load_sum_low_kg",
        "load_sum_high_kg"
      ),
      names_glue = "{col}_{.value}"
    ) %>%
    dplyr::arrange(.data$analyte)

  pct_summary <- load_draws_wide %>%
    dplyr::group_by(.data$analyte) %>%
    dplyr::summarise(
      MT_pct_red_mean = mean(.data$MT_pct_reduction_from_CT),
      MT_pct_red_low = stats::quantile(
        .data$MT_pct_reduction_from_CT,
        probs = alpha_ci / 2,
        names = FALSE
      ),
      MT_pct_red_high = stats::quantile(
        .data$MT_pct_reduction_from_CT,
        probs = 1 - alpha_ci / 2,
        names = FALSE
      ),
      ST_pct_red_mean = mean(.data$ST_pct_reduction_from_CT),
      ST_pct_red_low = stats::quantile(
        .data$ST_pct_reduction_from_CT,
        probs = alpha_ci / 2,
        names = FALSE
      ),
      ST_pct_red_high = stats::quantile(
        .data$ST_pct_reduction_from_CT,
        probs = 1 - alpha_ci / 2,
        names = FALSE
      ),
      .groups = "drop"
    )
  numeric <- numeric %>%
    dplyr::left_join(pct_summary, by = "analyte")

  pub <- numeric %>%
    dplyr::transmute(
      analyte = .data$analyte,
      CT = period_table_fmt_ci(
        .data$CT_mod_load_sum_mean_kg,
        .data$CT_mod_load_sum_low_kg,
        .data$CT_mod_load_sum_high_kg,
        digits = 2L
      ),
      MT = period_table_fmt_ci(
        .data$MT_mod_load_sum_mean_kg,
        .data$MT_mod_load_sum_low_kg,
        .data$MT_mod_load_sum_high_kg,
        digits = 2L
      ),
      ST = period_table_fmt_ci(
        .data$ST_mod_load_sum_mean_kg,
        .data$ST_mod_load_sum_low_kg,
        .data$ST_mod_load_sum_high_kg,
        digits = 2L
      ),
      `MT reduction from CT` = period_table_fmt_pct_ci(
        .data$MT_pct_red_mean,
        .data$MT_pct_red_low,
        .data$MT_pct_red_high,
        digits = 1L
      ),
      `ST reduction from CT` = period_table_fmt_pct_ci(
        .data$ST_pct_red_mean,
        .data$ST_pct_red_low,
        .data$ST_pct_red_high,
        digits = 1L
      )
    )

  list(
    start_year = start_year,
    end_year = end_year,
    n_years = length(expected_years),
    numeric = numeric,
    pub = pub
  )
}

period_table_write_csv_retry <- function(
    x,
    file,
    attempts = 12L,
    wait_seconds = 0.75
) {
  last_error <- NULL
  for (attempt in seq_len(attempts)) {
    wrote <- tryCatch({
      readr::write_csv(x, file)
      TRUE
    }, error = function(e) {
      last_error <<- e
      FALSE
    })
    if (wrote) return(invisible(file))
    if (attempt < attempts) Sys.sleep(wait_seconds)
  }
  stop(
    "Could not write CSV after ", attempts, " attempts: ", file, "\n",
    conditionMessage(last_error)
  )
}

export_period_total_load_tables <- function(
    annual_draws,
    period_specs,
    results_dir,
    model_version,
    writer = period_table_write_csv_retry
) {
  required_spec_columns <- c(
    "start_year", "end_year", "numeric_stem", "pub_stem"
  )
  missing <- setdiff(required_spec_columns, names(period_specs))
  if (length(missing) > 0L) {
    stop(
      "Period specification is missing column(s): ",
      paste(missing, collapse = ", ")
    )
  }

  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  exports <- vector("list", nrow(period_specs))
  for (i in seq_len(nrow(period_specs))) {
    products <- build_period_total_load_tables(
      annual_draws = annual_draws,
      start_year = period_specs$start_year[[i]],
      end_year = period_specs$end_year[[i]]
    )
    numeric_file <- file.path(
      results_dir,
      paste0(period_specs$numeric_stem[[i]], "_", model_version, ".csv")
    )
    pub_file <- file.path(
      results_dir,
      paste0(period_specs$pub_stem[[i]], "_", model_version, ".csv")
    )
    writer(products$numeric, numeric_file)
    writer(products$pub, pub_file)
    message("[OK] Wrote: ", numeric_file)
    message("[OK] Wrote: ", pub_file)
    products$numeric_file <- numeric_file
    products$pub_file <- pub_file
    exports[[i]] <- products
  }
  names(exports) <- if ("period_id" %in% names(period_specs)) {
    as.character(period_specs$period_id)
  } else {
    paste0(period_specs$start_year, "_", period_specs$end_year)
  }
  exports
}
