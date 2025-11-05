Excellent — since your STIR documentation is already complete and lives in `docs/STIR calculations.md`, the **main repository README** should now pivot from “explaining STIR math” to highlighting the **pipeline architecture** and **automated data processing** (what the repo *does*).

Here’s a clean, non-redundant rewrite ready to replace your current `README.md`:

---

# Kerbel Long-Term Impacts Project

**Principal Investigator:** AJ Brown – Colorado State University, Agricultural Data Scientist


📍 Department of Soil and Crop Sciences, Colorado State University

---

## Overview

This repository hosts the **data-processing and analysis pipeline** for the **Kerbel Long-Term Tillage Impacts Study**, a 14-year field experiment examining how contrasting tillage systems—**Conventional (CT), Minimum (MT), and Strip Till (ST)**—affect runoff water quality, sediment transport, and nutrient load uncertainty in irrigated hay meadows of western Colorado.

The project integrates:

* detailed **field-scale water-quality monitoring**,
* **crop management and irrigation timing**, and
* quantified **soil-disturbance metrics** (STIR)

into a reproducible, fully automated Python workflow for evaluating both **annual effects** and **long-term legacy impacts** of tillage on agricultural water quality.

---

## Repository Structure

```bash
kerbel-long-term-impacts/
│
├── LICENSE
├── README.md
│
├── code/
│   ├── main.py                      # Master runner: executes all steps in order (WQ → STIR → Merge)
│   ├── wq_longify.py                # Converts master WQ file to tidy long format
│   ├── stir_pipeline.py             # Computes tillage STIR values from mapper + records
│   ├── merge_wq_stir_by_season.py   # Merges WQ with STIR by crop-season windows
│   ├── dag.r, dag2_23July2025.r     # Causal/Bayesian DAG modeling scripts (in development)
│
├── data/
│   ├── Master_WaterQuality_Kerbel_LastUpdated_10272025.csv
│   ├── tillage_records.csv
│   ├── tillage_mapper_input.csv
│   ├── crop records.csv
│   └── STIR_values_MOSES_2023_data.csv
│
├── docs/
│   ├── STIR calculations.md          # Detailed STIR formulation and workflow
│   ├── README_final_outputs.md       # Column documentation for final merged dataset
│   └── Farming Implements presentation slides.pdf
│
├── figs/                             # Figures and DAG diagrams
│
└── out/                              # All generated outputs
    ├── kerbel_master_concentrations_long.csv
    ├── stir_events_long.csv
    ├── wq_with_stir_by_season.csv
    └── wq_with_stir_unmatched.csv
```

---

## 🔁 Pipeline Summary

### Step A – WQ Longification (`wq_longify.py`)

Reads the master water-quality file (`Master_WaterQuality_Kerbel_LastUpdated_10272025.csv`)
and restructures it to **tidy long format**, one row per analyte per sample event.

### Step B – STIR Computation (`stir_pipeline.py`)

Uses **MOSES-based parameters** (NRCS RUSLE2 dataset via *SoilManageR*) and field tillage logs to calculate:

* event-level STIR intensity,
* daily and cumulative disturbance totals per treatment.

*(See `docs/STIR calculations.md` for the complete formulation and references.)*

### Step C – Season Merge (`merge_wq_stir_by_season.py`)

Attaches each WQ record to its crop-season window (`PlantDate → HarvestDate`)
and merges both **seasonal** and **all-years cumulative** STIR totals.

---

## 🚀 Running the Full Pipeline

From the Anaconda Prompt:

```bash
cd "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts"
python code/main.py --debug
```

The runner automatically performs all three stages, writes outputs to `/out`,
and displays a colorized summary of run times and generated files.

Key outputs:

* `out/kerbel_master_concentrations_long.csv` – tidy long water-quality dataset
* `out/stir_events_long.csv` – event-level STIR data
* `out/wq_with_stir_by_season.csv` – final merged analytical table
* `out/wq_with_stir_unmatched.csv` – samples without STIR match (QC)

---

## 📊 Data Documentation

Detailed column definitions for the final output are provided in
[`docs/README_final_outputs.md`](docs/README_final_outputs.md).
That file explains all WQ, crop, and STIR fields—along with schema notes
for `"NA"`, `"U"`, `"NA.IRR"`, and `"None"` values.

---

## 🧮 Analytical Extensions

Post-processing notebooks and R scripts (`dag.r`, `dag2_23July2025.r`) explore:

* causal links between tillage, flow, and nutrient concentration,
* Bayesian uncertainty propagation in load estimates,
* model calibration for dissertation Chapters 2 and 3.

---

## References

* USDA-NRCS (2023). *Revised Universal Soil Loss Equation, Version 2 (RUSLE2), Official Database V 2023-02-24.*
  [https://fargo.nserl.purdue.edu/rusle2_dataweb](https://fargo.nserl.purdue.edu/rusle2_dataweb)

* Harmel R.D. et al. (2006). *Cumulative uncertainty in measured streamflow and water-quality data for small watersheds.* *Transactions of the ASABE,* 49(3): 689-701.

* Heller, O., Chervet, A., Durand‐Maniclas, F., Guillaume, T., Häfner, F., Müller, M., ... & Keller, T. (2025). SoilManageR—An R Package for Deriving Soil Management Indicators to Harmonise Agricultural Practice Assessments. European Journal of Soil Science, 76(2), e70102. GitLab Repository: [https://gitlab.com/nrcs-soil/soilmanager](https://gitlab.com/nrcs-soil/soilmanager)

---

## License

This repository is released under the **GNU GENERAL PUBLIC LICENSE VERSION 2**.
Use of data and code must credit the *Colorado State University Agricultural Water Quality Program (AWQP)* and collaborating partners.

