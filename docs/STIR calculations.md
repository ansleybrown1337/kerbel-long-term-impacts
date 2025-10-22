# STIR Calculations and Pipeline Workflow

## Overview

This document describes the workflow used to calculate **Soil Tillage Intensity Rating (STIR)** values for each field operation in the Kerbel Long-Term Impacts dataset. The updated workflow uses the **`stir_pipeline.py`** script, which replaces legacy approaches that relied on user-supplied mixing efficiencies or surface fractions. All STIR values are now calculated consistently using the **MOSES (Management Operations and Soil Erosion Simulation)** dataset distributed by the **USDA–NRCS** and integrated into the *SoilManageR* package.

- [SoilManageR Repo](https://gitlab.com/SoilManageR/SoilManageR)
- [SoilManageR Article](https://bsssjournals.onlinelibrary.wiley.com/doi/10.1111/ejss.70102)

### What STIR Represents

The **Soil Tillage Intensity Rating (STIR)** quantifies the overall physical disturbance to the soil surface caused by a tillage operation. Higher STIR values indicate greater disturbance and correspond to more energy expended in soil manipulation. STIR serves as a core component of the **RUSLE2 (Revised Universal Soil Loss Equation, version 2)** framework for estimating erosion and residue incorporation effects.

### What MOSES Is

**MOSES (Management Operations and Soil Erosion Simulation)** is the official NRCS database that defines operational parameters for tillage, planting, and residue management within RUSLE2. It specifies standardized attributes such as operating speed, tillage depth, surface disturbance, and tillage-type modifier (TTM). These parameters are used by *SoilManageR* to derive STIR values and residue burial coefficients. The dataset forms the scientific foundation of this workflow, ensuring that all tillage operations are traceable to validated national standards.

---

## Equation and Units

STIR is computed directly from the *SoilManageR* equation:


```math
STIR = \left( 0.5 \times \frac{Speed_{[km/h]}}{1.609} \right) \times \left( 3.25 \times TTM \right) \times \left( \frac{Depth_{[cm]}}{2.54} \right) \times \left( \frac{Surf_{Disturbance}{100} \right)
```


| Term                     | Description                       | Conversion       | Notes                                                                            |
| ------------------------ | --------------------------------- | ---------------- | -------------------------------------------------------------------------------- |
| **Speed [km/h]**         | Average implement speed           | ÷ 1.609 → mph    | Converts from km/h to miles per hour                                             |
| **TTM**                  | *Tillage Type Modifier* [0–1]     | none             | Represents the aggressiveness of the implement (0 = no till, 1 = full inversion) |
| **Depth [cm]**           | Average operating depth           | ÷ 2.54 → inches  | Converts from centimeters to inches                                              |
| **Surf_Disturbance [%]** | Percent of soil surface disturbed | ÷ 100 → fraction | Captures the area of soil affected by the implement                              |
| **STIR**                 | Soil Tillage Intensity Rating     | —                | Dimensionless intensity index                                                    |

This formulation converts all metric units to those used in RUSLE2’s empirical calibration (mph, inches) and scales soil energy through both the mechanical and spatial extent of disturbance.

---

## Inputs

### 1. Tillage Mapper (`tillage_mapper_input.csv`)

This file defines how each operation in the field records maps to an equivalent operation in the MOSES database. It also provides all parameters required to compute STIR. Each record corresponds to one implement or operation type.

**Required columns:**

| Column                      | Description                                   | Units |
| --------------------------- | --------------------------------------------- | ----- |
| Operation (verbatim)        | The exact operation name found in field logs  | text  |
| MOSES Operation             | Equivalent operation name from MOSES database | text  |
| Speed [km/h]                | Average operation speed                       | km/h  |
| Surf_Disturbance [%]        | Percent of surface disturbed                  | %     |
| Depth [cm]                  | Average operating depth                       | cm    |
| TILLAGE_TYPE_Modifier [0-1] | Tillage-type modifier (TTM)                   | —     |

**Optional columns:**

* Diesel_use [l/ha]
* Burial_Coefficient [0-1]
* Source
* Description

### 2. Tillage Records (`tillage_records.csv`)

This dataset contains all field operations across conventional (CT), strip-till (ST), and minimum-till (MT) systems.

Supported formats:

* **Wide** (columns: `CT Operation`, `ST Operation`, `MT Operation`)
* **Long** (column: `Operation` with a `System` column indicating CT/ST/MT)

The script normalizes the file into long format internally.

### 3. Crop Records (`crop_records.csv`, optional)

If provided, crop harvest dates are used to compute cumulative STIR values between harvest events. Each record should contain at least a `Date` column and one field identifying harvest events.

---

## Outputs

All outputs are written to the directory specified by `--outdir`.

| File                              | Description                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------- |
| **stir_events_long.csv**          | Long-format table of every operation with computed STIR and mapper parameters |
| **stir_daily_system_wide.csv**    | Daily STIR sums per system (CT/ST/MT) pivoted to wide format                  |
| **stir_events_long_windowed.csv** | (Optional) Cumulative STIR between harvest events if crop data provided       |
| **unmapped_ops.csv**              | Operations from the records not matched to the mapper table                   |

---

## Running the Script

### Example (Windows Command Prompt)

```
python stir_pipeline.py --records "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\data\tillage_records.csv" --mapper "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\data\tillage_mapper_input.csv" --outdir "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\out" --crop "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\data\crop_records.csv"
```

### Example (PowerShell)

```
python .\stir_pipeline.py `
  --records "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\data\tillage_records.csv" `
  --mapper  "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\data\tillage_mapper_input.csv" `
  --outdir  "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\out" `
  --crop    "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts\data\crop_records.csv"
```

---

## Interpretation and Use

The computed STIR values allow comparison of soil disturbance among management systems, years, or tillage operations. When aggregated by crop year or between harvests, cumulative STIR provides a measure of **mechanical soil impact intensity** that can be related to observed changes in residue cover, infiltration, and water-quality responses.

Because the workflow draws directly from MOSES and RUSLE2 calibration data, the resulting STIR values are standardized and reproducible across studies. This alignment ensures that subsequent analyses (e.g., legacy tillage effects, cumulative disturbance modeling) are directly comparable with national datasets used in NRCS conservation planning and erosion modeling.

---

### References

* USDA-NRCS. 2023. “ Revised Universal Soil Loss Equation, Version 2 (RUSLE2), Official NRCS RUSLE2 Program and Database (V 2023-02-24).” [https://fargo.nserl.purdue.edu/rusle2_dataweb](https://fargo.nserl.purdue.edu/rusle2_dataweb)
* Heller, O., Chervet, A., Durand‐Maniclas, F., Guillaume, T., Häfner, F., Müller, M., ... & Keller, T. (2025). SoilManageR—An R Package for Deriving Soil Management Indicators to Harmonise Agricultural Practice Assessments. European Journal of Soil Science, 76(2), e70102. GitLab Repository: [https://gitlab.com/nrcs-soil/soilmanager](https://gitlab.com/nrcs-soil/soilmanager)
