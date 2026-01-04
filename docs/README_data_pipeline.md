# Kerbel Long-Term Impacts
## End-to-End Water Quality × STIR × Residue Data Pipeline

This repository provides a reproducible, **Python-based** pipeline that transforms raw edge-of-field water-quality monitoring data into analysis-ready datasets used in the Kerbel Long-Term Tillage Impacts project. The pipeline integrates:

- Water-quality observations (OUT, long format)
- Tillage operations processed into STIR metrics
- Crop-season metadata (plant/harvest windows)
- **Residue cover / residue dry-mass measurements** (optional, but supported with full missingness)
- Bayesian-analysis-specific assumptions and standardized predictors

**Final pipeline product (used by Stan/Rmd models):**
```
out/wq_cleaned.csv
```

---

## What the pipeline does

At a high level, the pipeline:

1. Converts raw WQ data from wide → long format (one row per Date × Rep × Treatment × Analyte).
2. Computes STIR at the tillage-operation level, then cumulative totals through time.
3. Merges WQ observations with cumulative STIR by crop-season windows.
4. **Merges residue measurements** onto the experimental unit (Year × Treatment × Rep) and broadcasts them to all analyte rows.
5. Applies Bayesian-model “backend” cleaning and transformations (flag handling, factor enforcement, standardized predictors) and writes the final dataset.

Each step is implemented as a standalone script for transparency and reuse, and orchestrated by a single runner.

---

## How to run (Python)

From the repository root:

```bash
python code/run_pipeline.py --debug
```

### Optional: specify a residue file
By default, the runner looks for:
```
data/residue_dummy_2011_2025.csv
```

To point to a different residue file:
```bash
python code/run_pipeline.py --debug --residue data/residue_myfield.csv
```

### Optional: skip residue merge
If you do not have residue data available yet:
```bash
python code/run_pipeline.py --debug --skip-residue
```

**Important:** the pipeline is designed to run end-to-end each time (no shortcutting via existing CSVs), to avoid version drift and ensure reproducibility.

---

## Pipeline steps and outputs

### Step A. Water-quality longification
**Script:** `code/wq_longify.py`

**Purpose:** Convert the raw master WQ table (wide analyte columns) into a long format where each row represents:
`Date × Treatment × Irrigation × Rep × Analyte`.

**Output:**
```
out/pipeline_csvs/kerbel_master_concentrations_long.csv
```

---

### Step B. STIR processing
**Script:** `code/stir_pipeline.py`

**Purpose:** Convert tillage operation logs to event-level STIR and cumulative STIR totals.

**Output:**
```
out/pipeline_csvs/stir_events_long.csv
```

---

### Step C. Seasonal merge (WQ × STIR)
**Script:** `code/merge_wq_stir_by_season.py`

**Purpose:** Attach crop-season windows and merge cumulative STIR values to each WQ observation.

**Outputs:**
```
out/pipeline_csvs/wq_with_stir_by_season.csv
out/pipeline_csvs/wq_with_stir_unmatched.csv
out/pipeline_csvs/wq_outside_crop_windows.csv
```

---

### Step D. Residue merge (WQ × STIR × Residue)
**Script:** `code/merge_residue.py`

**Purpose:** Merge residue measurements at the experimental-unit level and broadcast to all analyte rows.

**Merge keys:**
- `Year`
- `Treatment`
- `Rep`

Residue data are often collected as multiple spatial subsamples (e.g., Location = N/M/S) and/or multiple throws. This script aggregates (means) across those subsamples within `Year × Treatment × Rep` before merging.

**Outputs:**
```
out/pipeline_csvs/wq_with_stir_by_season_with_residue.csv
out/pipeline_csvs/residue_agg_by_year_trt_rep.csv
```

---

### Step E. Bayes-specific backend cleaning (final)
**Script:** `code/stir_bayes_backend.py`

**Purpose:** Apply analysis-specific assumptions required for Bayesian modeling. This step replaces the legacy `stir-bayes-backend.R` cleaning logic.

Key operations:
- Handles WQ flags consistently:
  - `"U"` (nondetect) → 0 (current modeling assumption)
  - `"NA"` → missing
  - `"NA.IRR"` (no runoff) → row removed
- Enforces types (dates, factors, numeric)
- Creates `analyte_abbr`
- Standardizes predictors used by the models:
  - Per analyte: `cout_z`, `cin_z`
  - Global: `stir_season_z`, `stir_cumall_z`, `volume_z`
- Produces residue proportion for Beta modeling (when residue is present):
  - `residue_prop = clamp(Residue_PercentCover / 100, eps, 1-eps)`
  - `residue_obs` indicator (1 = observed, 0 = missing)

**Final output (single source of truth for modeling):**
```
out/wq_cleaned.csv
```

---

## Inputs and outputs summary

### Inputs (data/)
- `Master_WaterQuality_Kerbel_LastUpdated_*.csv` (raw WQ master table)
- `tillage_records.csv` (tillage operations)
- `tillage_mapper_input.csv` (operation → STIR mapping)
- `crop records.csv` (plant/harvest windows)
- `residue_*.csv` (optional, residue cover/dry mass)

### Intermediate outputs (out/pipeline_csvs/)
- `kerbel_master_concentrations_long.csv`
- `stir_events_long.csv`
- `wq_with_stir_by_season.csv` (+ diagnostics)
- `wq_with_stir_by_season_with_residue.csv`
- `residue_agg_by_year_trt_rep.csv`

### Final output (out/)
- **`wq_cleaned.csv`**

---

## `wq_cleaned.csv` column dictionary

This table documents the columns expected in `out/wq_cleaned.csv`. Some residue columns are **conditional**: they appear once the residue merge step is enabled and residue data exist.

| Column | Type | Meaning / notes |
|---|---:|---|
| `_wq_idx` | int | Stable index carried from the longified WQ dataset (useful for row tracking/joins). |
| `orig_row` | int | Row number within `wq_cleaned.csv` written at export time (debug/tracking). |
| `Date` | date | Runoff sample date. |
| `Year` | int | Water year / season year identifier used throughout the dataset (matches crop/STIR records). |
| `SeasonYear` | int | Crop-season year attached from crop records (typically equals `Year`). |
| `Irrigation` | str/int | Irrigation event identifier within year. |
| `Rep` | int | Plot replicate ID. |
| `Treatment` | factor/str | Tillage treatment (`CT`, `MT`, `ST`). |
| `Crop` | factor/str | Crop type for the season (from crop records). |
| `PlantDate` | date | Crop planting date used to define the season window. |
| `HarvestDate` | date | Crop harvest date used to define the season window. |
| `InflowOutflow` | str | Indicates OUT rows are retained in the longified dataset (pipeline currently outputs OUT only). |
| `SampleID` | str | Sample identifier. |
| `FF` | bool | QA/QC field flag (as provided). |
| `Composite` | str | Composite sample designation (as provided). |
| `Duplicate` | bool | Duplicate status flag (as provided). |
| `Flag` | str | Water-quality lab flag (as provided). |
| `NoRunoff` | bool/NA | “No runoff” indicator (derived; rows with `NA.IRR` are removed). |
| `Analyte` | factor/str | Full analyte name. |
| `analyte_abbr` | factor/str | Abbreviation used for plotting/model indexing (e.g., `TP`, `TSS`). |
| `Result_mg_L` | float | OUT concentration (mg/L) after flag handling (`U` → 0, `NA` → missing). |
| `Inflow_Result_mg_L` | float | Inflow concentration paired to OUT sample (mg/L) after flag handling. |
| `Inflow_Flag` | str | Inflow lab flag (as provided). |
| `Has_Inflow` | bool/NA | Indicator that inflow pairing exists for the OUT row (pipeline-derived). |
| `Volume` | float | OUTflow volume for the event (units as provided in source; treated consistently within analysis). |
| `Inflow_Volume` | float | Inflow volume paired to OUT sample (if available). |
| `SampleMethod` | str | Sampler type/method covariate (DAG adjustment). |
| `MeasureMethod` | str | Depth/measurement method covariate (DAG adjustment). |
| `IrrMethod` | str | Irrigation method covariate. |
| `FlumeMethod` | str | Flume type / rating method covariate (DAG adjustment). |
| `TSSMethod` | str | TSS method metadata (as provided). |
| `Lab` | str | Lab name/identifier. |
| `Notes` | str | Notes from source table. |
| `MDL_Provided` | float | Provided method detection limit from source table (if available). |
| `RL_Provided` | float | Provided reporting limit from source table (if available). |
| `RLMDL_Provided_Units` | str | Units for provided RL/MDL fields (as provided). |
| `RLMDL_Method` | str | Analytical method for RL/MDL fields (as provided). |
| `MDL_mg_L` | float | MDL converted/normalized to mg/L (if possible). |
| `RL_mg_L` | float | RL converted/normalized to mg/L (if possible). |
| `RLMDL_Source` | str | Source used to populate RL/MDL when assumed. |
| `RLMDL_Assumed` | bool | TRUE if RL/MDL were filled using assumptions rather than directly provided. |
| `Season_STIR_toDate` | float | Cumulative STIR within the current crop season up to the sample date. |
| `CumAll_STIR_toDate` | float | Long-term cumulative STIR (all years) up to the sample date. |
| `cout_z` | float | Per-analyte standardized OUT concentration (z-score within analyte). |
| `cin_z` | float | Per-analyte standardized inflow concentration (z-score within analyte). |
| `stir_season_z` | float | Global standardized seasonal STIR. |
| `stir_cumall_z` | float | Global standardized cumulative STIR. |
| `volume_z` | float | Global standardized volume. |
| `Residue_PercentCover`* | float | Mean residue cover (%) aggregated within `Year × Treatment × Rep` (averaged across Location/subsamples). |
| `Residue_DryMass_kg_m2`* | float | Mean residue dry mass (kg/m²) aggregated within `Year × Treatment × Rep`. |
| `Residue_n`* | int | Number of residue subsamples contributing to the aggregate mean (non-missing counts). |
| `residue_prop`* | float | Residue cover proportion in (0,1) for Beta modeling: `Residue_PercentCover / 100` (clamped away from 0/1). |
| `residue_obs`* | int | 1 if residue cover was observed for the row, else 0 (for missingness-aware models). |

\* Residue columns appear once Step D is enabled and residue data exist for a given `Year × Treatment × Rep`.

---

**Last updated:** January 2026  
**Maintainer:** CSU Agricultural Water Quality Program  
