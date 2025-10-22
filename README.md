# Kerbel Long-Term Impacts Project  
**Principal Investigator:** Ansle Saunders (Colorado State University, Agricultural Water Quality Program)  
📧 **Contact:** [AgWaterQuality@colostate.edu](mailto:AgWaterQuality@colostate.edu)  
📍 **Affiliation:** Department of Soil and Crop Sciences, Colorado State University  
🧭 **Repository purpose:** Data analysis and modeling framework for assessing long-term tillage impacts on soil disturbance, water quality, and hydrologic response.

---

## Overview

This repository houses the analytical workflow and datasets associated with the **Kerbel Long-Term Tillage Impacts Study**, a 14-year investigation examining how contrasting tillage systems (conventional, strip, and minimum) influence **runoff water quality**, **sediment transport**, and **nutrient load uncertainty** in irrigated hay meadows of western Colorado.

The project integrates **field-scale measurements** of tillage intensity, crop management, and water quality with a reproducible modeling pipeline designed to quantify both **short-term disturbance events** and **long-term legacy effects**.

---

## Repository Structure

```bash
kerbel-long-term-impacts/
│
├── LICENSE
├── README.md
├── .gitignore
├── .gitattributes
│
├── code/
│   ├── stir_pipeline.py                # Primary STIR workflow (MOSES-based)
│   ├── dag.r                           # Draft causal DAG model for tillage–WQ relationships
│   └── dag2_23July2025.r              # Updated Bayesian DAG development script
│
├── data/
│   ├── tillage_mapper_input.csv        # Mapper linking field operations to MOSES STIR parameters
│   ├── tillage_records.csv             # Main 14-year tillage operations log (long format)
│   ├── tillage records.xlsx            # Original wide-format input file (archival)
│   ├── crop records.csv                # Crop and harvest timing dataset
│   └── STIR_values_MOSES_2023_data.csv # Extracted MOSES STIR reference data from SoilManageR
│
├── docs/
│   ├── STIR calculations.md            # Documentation of STIR equation, units, and workflow
│   └── Farming Implements presentation slides.pdf
│
├── figs/
│   ├── STIR_visuals.png                # Visualization of STIR distributions and system differences
│   └── dagitty-model (3).png           # Causal DAG model schematic
│
└── out/
    ├── stir_events_long.csv            # Event-level STIR results with cumulative values
    ├── stir_daily_system_wide.csv      # Daily system-wide STIR summaries
    ├── stir_daily_system_wide_with GRAPHS_22Oct25.xlsx  # Visualization-ready output file
    └── unmapped_ops.csv                # Operations not mapped to MOSES parameters
```


---

## Key Components

### 🔹 Soil Tillage Intensity Rating (STIR)

STIR quantifies the mechanical disturbance imposed on soil during tillage operations.  
The calculation follows the **SoilManageR / RUSLE2 formulation**:

\[
STIR = (0.5 \times \frac{Speed\_{km/h}}{1.609}) \times (3.25 \times TTM) \times \frac{Depth\_{cm}}{2.54} \times \frac{SurfaceDisturbance\_%}{100}
\]

Each term represents:
- **Speed** (km/h) – average operation speed  
- **TTM** – tillage type modifier (0–1)  
- **Depth** (cm) – mixing depth  
- **Surface Disturbance** (%) – fraction of soil surface affected  

The resulting STIR value serves as a unitless intensity index (higher = more soil disturbance).

### 🔹 MOSES Database Integration

The **MOSES (Management Operations and Soil Erosion Simulation)** dataset (NRCS, 2023) provides the official RUSLE2 tillage operation parameters. These have been extracted from the *SoilManageR* package and linked through the `tillage_mapper_input.csv` file.  
All STIR computations are dynamically derived from these parameters for transparency and reproducibility.

### 🔹 Crop and Water Quality Integration

Subsequent modules integrate:
- **Crop timing and rotation data** (e.g., planting, harvest, irrigation windows),
- **Runoff and nutrient concentration data** (e.g., TSS, TP, NO₃-N, TKN),
- and **flow-weighted load estimates** to assess cumulative and legacy effects of tillage on water quality.

These relationships will be evaluated using **Bayesian generative models** that incorporate both measurement and process uncertainty following Harmel et al. (2006) and extensions developed in Chapter 2 of the dissertation.

---

## Example Workflow

```bash
# Example STIR calculation run (Windows PowerShell)
python code/stir_pipeline.py `
  --records data/tillage_records.csv `
  --mapper data/tillage_mapper_input.csv `
  --outdir results `
  --crop data/crop_records.csv
```

This produces two outputs:
- **`stir_events_long.csv`** – full event-level data with cumulative STIR (annual, all-years, and crop-window).  
- **`stir_daily_system_wide.csv`** – daily system-wide aggregates with cumulative STIR per tillage type.

---

## References

- USDA-NRCS (2023). *Revised Universal Soil Loss Equation, Version 2 (RUSLE2), Official NRCS RUSLE2 Program and Database (V 2023-02-24).*  
  [https://fargo.nserl.purdue.edu/rusle2_dataweb/RUSLE2_Index.htm](https://fargo.nserl.purdue.edu/rusle2_dataweb/RUSLE2_Index.htm)

- Brown, A.J., Saunders, A., et al. (2025). *Long-term tillage impacts on runoff water quality: legacy and cumulative effects across irrigated hay systems.* (In preparation)

- Harmel, R.D., Cooper, R.J., Slade, R.M., Haney, R.L., Arnold, J.G. (2006). *Cumulative uncertainty in measured streamflow and water quality data for small watersheds.* *Transactions of the ASABE,* 49(3):689-701.

---

## Other Helpful Links

- [CEAP STIR Conservation Definitions](https://www.nrcs.usda.gov/sites/default/files/2024-11/Conservation_Effects_Assessment_Project_%28CEAP%29_Tillage_Classification_Methodology_508.pdf)

---

## License

This project is distributed under the **MIT License** unless otherwise specified in subfolders.  
Data use and publication must acknowledge the Colorado State University Agricultural Water Quality Program (AWQP) and collaborating partners.
