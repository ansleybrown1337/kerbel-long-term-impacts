#!/usr/bin/env Rscript

# Targeted, read-only diagnostic audit for an already sampled Bayes v3p1 fit.
# This does not compile, sample, or alter the serialized fit or CmdStan CSVs.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", tail(file_arg, 1L)), winslash = "/")
} else {
  normalizePath("code/bayes/diagnose_saved_fit_v3p1.R", winslash = "/")
}
repo_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."),
  winslash = "/"
)
setwd(repo_root)

suppressPackageStartupMessages(library(cmdstanr))
suppressPackageStartupMessages(library(posterior))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(tidyr))

fit_path <- file.path(
  repo_root, "results", "bayes", "v3p1_physical_event",
  "fit_mogp_v3p1_physical_event.rds"
)
if (!file.exists(fit_path)) stop("Saved fit not found: ", fit_path)
fit <- readRDS(fit_path)
results_dir <- dirname(fit_path)
figures_dir <- file.path(
  repo_root, "figures", "bayes", "v3p1_physical_event"
)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

analyte_labels <- c(
  "NH4", "NO3", "NO2", "OP", "Se", "TDS", "TKN", "TN", "TP", "TSS"
)
residue_variables <- c(
  "beta_res_V", "beta_res_C", "res_base", "b_res_stir",
  "gamma_PCr_res", "sigma_PCr_res", "sigma_res_obs",
  "RES_unit_miss_logit"
)

residue_draw_matrix <- posterior::as_draws_matrix(
  fit$draws(variables = residue_variables)
)
probability_positive <- tibble::tibble(
  variable = colnames(residue_draw_matrix),
  probability_positive = colMeans(residue_draw_matrix > 0)
)
residue_summary <- fit$summary(variables = residue_variables) %>%
  dplyr::left_join(probability_positive, by = "variable") %>%
  dplyr::mutate(
    parameter_family = sub("\\[.*$", "", .data$variable),
    parameter_index = suppressWarnings(
      as.integer(sub("^.*\\[([0-9]+)\\]$", "\\1", .data$variable))
    ),
    analyte = dplyr::if_else(
      .data$parameter_family == "beta_res_C" &
        !is.na(.data$parameter_index),
      analyte_labels[.data$parameter_index],
      NA_character_
    ),
    rhat_threshold = 1.04,
    rhat_status = dplyr::case_when(
      !is.finite(.data$rhat) ~ "unavailable",
      .data$rhat <= .data$rhat_threshold ~ "at_or_below_1.04",
      TRUE ~ "above_1.04"
    )
  )

cat("[RESIDUE_PARAMETER_SUMMARY]\n")
print(
  residue_summary %>%
    dplyr::select(
      "variable", "analyte", "mean", "median", "sd", "q5", "q95",
      "probability_positive", "rhat", "ess_bulk", "ess_tail", "rhat_status"
    ),
  n = Inf
)
residue_summary_path <- file.path(
  results_dir, "residue_parameter_diagnostics_v3p1_physical_event.csv"
)
readr::write_csv(residue_summary, residue_summary_path)

chain_variables <- c(
  "beta_res_V", "b_res_stir", "res_base",
  "sigma_PCr_res", "sigma_res_obs"
)
chain_draws <- fit$draws(
  variables = chain_variables,
  format = "draws_df"
) %>%
  tibble::as_tibble()
chain_summary <- dplyr::bind_rows(lapply(chain_variables, function(variable) {
  tibble::tibble(
    chain = chain_draws$.chain,
    value = as.numeric(chain_draws[[variable]])
  ) %>%
    dplyr::group_by(.data$chain) %>%
    dplyr::summarize(
      parameter = variable,
      mean = mean(.data$value),
      sd = stats::sd(.data$value),
      q05 = stats::quantile(.data$value, 0.05, names = FALSE),
      median = stats::median(.data$value),
      q95 = stats::quantile(.data$value, 0.95, names = FALSE),
      probability_positive = mean(.data$value > 0),
      .groups = "drop"
    )
}))

cat("\n[RESIDUE_CHAIN_SUMMARY]\n")
print(chain_summary, n = Inf)
chain_summary_path <- file.path(
  results_dir, "residue_chain_diagnostics_v3p1_physical_event.csv"
)
readr::write_csv(chain_summary, chain_summary_path)

sampler <- posterior::as_draws_df(fit$sampler_diagnostics()) %>%
  tibble::as_tibble()
divergence_summary <- tibble::tibble(
  chain = sampler$.chain,
  divergent = as.integer(sampler$divergent__)
) %>%
  dplyr::group_by(.data$chain) %>%
  dplyr::summarize(
    divergences = sum(.data$divergent),
    iterations = dplyr::n(),
    divergence_percent = 100 * .data$divergences / .data$iterations,
    .groups = "drop"
  )

cat("\n[DIVERGENCES_BY_CHAIN]\n")
print(divergence_summary, n = Inf)
divergence_path <- file.path(
  results_dir, "sampler_divergences_by_chain_v3p1_physical_event.csv"
)
readr::write_csv(divergence_summary, divergence_path)

divergence_residue_summary <- tibble::tibble(
  chain = sampler$.chain,
  iteration = sampler$.iteration,
  divergent = as.integer(sampler$divergent__)
) %>%
  dplyr::left_join(
    chain_draws %>%
      dplyr::transmute(
        chain = .data$.chain,
        iteration = .data$.iteration,
        beta_res_V = as.numeric(.data$beta_res_V)
      ),
    by = c("chain", "iteration"),
    relationship = "one-to-one"
  ) %>%
  dplyr::mutate(
    divergence_group = dplyr::if_else(
      .data$divergent == 1L, "divergent", "nondivergent"
    )
  ) %>%
  dplyr::group_by(.data$divergence_group) %>%
  dplyr::summarize(
    draws = dplyr::n(),
    mean_beta_res_V = mean(.data$beta_res_V),
    median_beta_res_V = stats::median(.data$beta_res_V),
    q05_beta_res_V = stats::quantile(
      .data$beta_res_V, 0.05, names = FALSE
    ),
    q95_beta_res_V = stats::quantile(
      .data$beta_res_V, 0.95, names = FALSE
    ),
    .groups = "drop"
  )
divergence_residue_path <- file.path(
  results_dir,
  "residue_volume_by_divergence_v3p1_physical_event.csv"
)
readr::write_csv(divergence_residue_summary, divergence_residue_path)

correlation_variables <- c(
  "beta_res_V", "a_V", "b_V", "beta_vin", "sigma_V",
  "b_res_stir", "res_base", "sigma_PCr_res",
  "beta_res_C", "gamma_Cr_V"
)
correlation_draws <- posterior::as_draws_matrix(
  fit$draws(variables = correlation_variables)
)
residue_volume_correlations <- stats::cor(
  correlation_draws[, "beta_res_V"],
  correlation_draws,
  use = "pairwise.complete.obs"
)[1, ] %>%
  tibble::enframe(name = "parameter", value = "correlation_with_beta_res_V") %>%
  dplyr::filter(.data$parameter != "beta_res_V") %>%
  dplyr::mutate(
    absolute_correlation = abs(.data$correlation_with_beta_res_V)
  ) %>%
  dplyr::arrange(dplyr::desc(.data$absolute_correlation))
residue_volume_correlation_path <- file.path(
  results_dir,
  "residue_volume_parameter_correlations_v3p1_physical_event.csv"
)
readr::write_csv(
  residue_volume_correlations,
  residue_volume_correlation_path
)

beta_res_v_by_chain <- split(
  as.numeric(chain_draws$beta_res_V),
  chain_draws$.chain
)
chain_colors <- c("#D55E00", "#0072B2", "#009E73", "#CC79A7")
residue_volume_plot <- file.path(
  figures_dir, "residue_volume_chain_diagnostics_v3p1_physical_event.png"
)
grDevices::png(
  residue_volume_plot, width = 1800, height = 900, res = 180
)
oldpar <- graphics::par(no.readonly = TRUE)
on.exit({
  graphics::par(oldpar)
  if (grDevices::dev.cur() > 1L) grDevices::dev.off()
}, add = TRUE)
graphics::par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
density_list <- lapply(beta_res_v_by_chain, stats::density)
x_range <- range(vapply(density_list, function(x) range(x$x), numeric(2)))
y_range <- c(
  0, max(vapply(density_list, function(x) max(x$y), numeric(1)))
)
graphics::plot(
  NA, xlim = x_range, ylim = y_range,
  xlab = "Residue effect on runoff volume", ylab = "Density",
  main = "Posterior density by chain"
)
for (i in seq_along(density_list)) {
  graphics::lines(
    density_list[[i]], col = chain_colors[i], lwd = 2
  )
}
graphics::abline(v = 0, lty = 2)
graphics::legend(
  "topright", legend = paste("Chain", seq_along(density_list)),
  col = chain_colors, lwd = 2, bty = "n"
)

trace_matrix <- do.call(cbind, beta_res_v_by_chain)
graphics::matplot(
  trace_matrix, type = "l", lty = 1, col = chain_colors,
  xlab = "Post-warmup iteration",
  ylab = "Residue effect on runoff volume",
  main = "Trace by chain"
)
graphics::abline(h = 0, lty = 2)
graphics::legend(
  "topright", legend = paste("Chain", seq_len(ncol(trace_matrix))),
  col = chain_colors, lwd = 2, bty = "n"
)
grDevices::dev.off()
on.exit(NULL, add = FALSE)

message("Wrote residue diagnostics: ", residue_summary_path)
message("Wrote residue chain diagnostics: ", chain_summary_path)
message("Wrote divergence diagnostics: ", divergence_path)
message("Wrote residue-by-divergence diagnostics: ", divergence_residue_path)
message(
  "Wrote residue-volume parameter correlations: ",
  residue_volume_correlation_path
)
message("Wrote residue chain figure: ", residue_volume_plot)
