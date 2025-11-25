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

# If you prefer not to rely on setwd, open the Rproj at repo root and run there.
# Then all paths below are relative to the project root.

# ---- 1. Import WQ + STIR data ----

#' Load merged WQ × STIR data
#'
#' @param path Path to wq_with_stir_by_season.csv relative to project root.
#' @param year_max Optional upper bound on Year (e.g., 2023). Use Inf to keep all.
#' @return A tibble with raw merged WQ × STIR data.
load_wq_stir <- function(path = "out/wq_with_stir_by_season.csv",
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
      # Standardize "u" to "U" for nondetects
      Result_mg_L        = ifelse(Result_mg_L == "u", "U", Result_mg_L),
      Inflow_Result_mg_L = ifelse(Inflow_Result_mg_L == "u", "U", Inflow_Result_mg_L)
    ) %>%
    # remove no-runoff cases entirely
    dplyr::filter(Result_mg_L != "NA.IRR") %>%
    
    # Replace nondetects with zero, preserve missing as NA
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
    
    # STEP 2E. Standardize STIR metrics globally
    dplyr::mutate(
      stir_season_z = rethinking::standardize(Season_STIR_toDate),
      stir_cumall_z = rethinking::standardize(CumAll_STIR_toDate)
    )
  
  
}

# ---- Use the functions ----

if (interactive()) {
  # Only run this section in interactive sessions
  # Step 1: load raw merged data
  wq_stir <- load_wq_stir(
    path = "out/wq_with_stir_by_season.csv",
    year_max = Inf
  )
  
  # Quick peek at raw
  glimpse(wq_stir)
  
  # Step 2: clean and type-enforce
  wq_stir_clean <- clean_wq_stir(wq_stir)
}

# ---- 3. Prepare data for modeling ----
