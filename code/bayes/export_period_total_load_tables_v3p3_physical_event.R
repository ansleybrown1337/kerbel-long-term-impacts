#!/usr/bin/env Rscript

# Regenerate only the Bayesian v3p3 period-total load tables from the saved,
# replicate-aware annual posterior draws. This script never opens the Stan fit,
# compiles the model, samples, or rebuilds event-level predictions.

.script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(.script_arg) == 0L) {
  stop("Could not identify this script's path from commandArgs().")
}
.script_path <- normalizePath(
  sub("^--file=", "", tail(.script_arg, 1L)),
  winslash = "/",
  mustWork = TRUE
)
repo_root <- normalizePath(
  file.path(dirname(.script_path), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})

config <- jsonlite::read_json(
  file.path(repo_root, "config", "physical_event_v3p3.json"),
  simplifyVector = TRUE
)
model_version <- config$versions$bayesian
if (!identical(model_version, "v3p3_physical_event")) {
  stop("Unexpected Bayesian model version: ", model_version)
}
results_dir <- file.path(repo_root, config$output_roots$bayesian_results)
draws_file <- file.path(
  results_dir,
  paste0("annual_load_draws_bayes_", model_version, ".csv")
)
if (!file.exists(draws_file)) {
  stop("Saved annual draw file was not found: ", draws_file)
}

source(file.path(repo_root, "code", "bayes", "period_total_load_tables.R"))
message("[period totals] Reading saved annual draws: ", draws_file)
annual_draws <- readr::read_csv(
  draws_file,
  show_col_types = FALSE,
  progress = interactive()
)
study_start_year <- min(as.integer(annual_draws$Year), na.rm = TRUE)
study_end_year <- max(as.integer(annual_draws$Year), na.rm = TRUE)
if (study_start_year > 2020L || study_end_year < 2025L) {
  stop(
    "Saved annual draws do not span both requested periods: ",
    study_start_year, "-", study_end_year
  )
}

period_specs <- tibble::tribble(
  ~period_id, ~start_year, ~end_year, ~numeric_stem, ~pub_stem,
  "all_study_years",
  study_start_year,
  study_end_year,
  "study_period_total_loads_kg_with_pct_reductions",
  "study_period_total_loads_kg_pub",
  "pre_tire_compaction_era",
  study_start_year,
  2020L,
  paste0(
    "pre_tire_compaction_era_", study_start_year,
    "_2020_total_loads_kg_with_pct_reductions"
  ),
  paste0(
    "pre_tire_compaction_era_", study_start_year,
    "_2020_total_loads_kg_pub"
  ),
  "tire_compaction_era",
  2021L,
  2025L,
  "tire_compaction_era_2021_2025_total_loads_kg_with_pct_reductions",
  "tire_compaction_era_2021_2025_total_loads_kg_pub"
)

exports <- export_period_total_load_tables(
  annual_draws = annual_draws,
  period_specs = period_specs,
  results_dir = results_dir,
  model_version = model_version
)
message(
  "[period totals] Finished without loading or recalibrating the Bayesian model."
)
invisible(exports)
