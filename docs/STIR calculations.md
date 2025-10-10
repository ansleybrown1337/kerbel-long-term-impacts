

# **stircalcs.py – Soil Tillage Intensity Rating (STIR) Calculation Framework**

### Author:

*AJ Brown, Colorado State University Agricultural Water Quality Program (AWQP)*
<br/>**Last updated:** October 2025

---

## **Purpose**

This script automates the derivation of **STIR-style soil disturbance metrics** for multi-year tillage operation records.
It integrates:

* Equipment-level physical disturbance (mixing depth, efficiency, area affected)
* Treatment-specific operation logs (Conventional, Strip, and Minimum tillage systems)
* Crop-year boundaries (planting/harvest windows)

to produce both **per-event** and **cumulative soil disturbance indices** suitable for legacy tillage and water-quality modeling.

---

## **Inputs**

| File                     | Path                            | Description                                                                                                                                         | Required Columns                                                                                                                  |
| ------------------------ | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Tillage Mapper Table** | `data/tillage_mapper_table.csv` | User-supplied lookup table linking each “My Operation” (field-record name) to SWAT/RUSLE2 implement codes and STIR parameters.                      | `My Operation`, `SWAT/RUSLE2 Implement`, `Tillage Code`, `Mixing Depth (mm)`, `Mixing Efficiency`, `Area Fraction`, `Type / Note` |
| **Tillage Records**      | `data/tillage_records.csv`      | Master list of all tillage events with date, tractor, implement, and per-system operation columns (`CT Operation`, `ST Operation`, `MT Operation`). | `Date`, `CT Operation`, `ST Operation`, `MT Operation`, plus optional metadata                                                    |
| **Crop Records**         | `data/crop records.csv`         | Defines planting and harvest windows for each crop year; used to group tillage events by crop cycle.                                                | `plant date`, `harvest date`, `crop`                                                                                              |

---

## **Outputs**

| Output File                                                 | Description                                                                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **`data/derived/mapper_normalized.csv`**                    | Cleaned version of the tillage mapper table (validated, numeric conversions applied).                                    |
| **`data/derived/mapper_dict.json`**                         | JSON dictionary keyed by `"My Operation"` used internally for mapping.                                                   |
| **`data/derived/tillage_records_mapped.csv`**               | Wide-format master tillage record with SWAT/RUSLE2 codes and numeric disturbance parameters mapped for CT/ST/MT systems. |
| **`data/derived/tillage_records_mapped_long.csv`**          | Long-format (tidy) dataset — one record per tillage event × system.                                                      |
| **`data/derived/tillage_records_disturbance_long.csv`**     | Adds computed **Disturbance_Fraction** for each tillage event.                                                           |
| **`data/derived/tillage_records_disturbance_long_cum.csv`** | Adds per-crop-year and all-years cumulative STIR sums per system (`Disturbance_Cum`, `Disturbance_CumAllYears`).         |
| **`data/derived/tillage_records_disturbance_wide_cum.csv`** | Wide-format cumulative dataset for easier plotting or joining with other metrics.                                        |

---

## **Key Calculations**

### **1. Fractional Soil Disturbance (per event)**

[
\text{Disturbance_Fraction} = E_m \times \left( \frac{D_m}{D_\text{ref}} \right) \times A_f
]

where:

* (E_m) = *Mixing Efficiency* (fraction of soil mass disturbed)
* (D_m) = *Mixing Depth (mm)*
* (D_\text{ref}) = *Reference Soil Depth* (default 200 mm)
* (A_f) = *Area Fraction* (fraction of field affected per pass)

Each component comes from the user’s `tillage_mapper_table.csv`.

> 🔹 *If Area Fraction is missing for a mixing operation, the script conservatively assumes 1.0 and flags a `Compute_Warning` in the output.*

---

### **2. Cumulative STIR Between Harvests**

Each tillage event is assigned to a crop-year window:
[
(\text{Previous Harvest}, \text{Current Harvest}]
]
and the cumulative soil disturbance **within that window** is computed **including the current event**:

[
\text{Disturbance_Cum}*{i,t} = \sum*{k=1}^{i} \text{Disturbance_Fraction}_{k,t}
]
for each tillage system *t* (CT, ST, MT).

---

### **3. All-Years Cumulative STIR**

An additional running total is computed that **does not reset** at crop-year boundaries:
[
\text{Disturbance_CumAllYears}*{t} = \sum*{y=1}^{N} \text{Disturbance_Fraction}_{y,t}
]

This provides a continuous “legacy tillage disturbance” metric useful for evaluating long-term soil and water-quality impacts.

---

## **Interpretation and Reference Values**

### Typical Mixing Efficiencies (from SWAT–TAMU 2019, Table 15; DayCent 2023 Manual)

| Implement Type          | Mixing Efficiency | Mixing Depth (mm) |
| ----------------------- | ----------------: | ----------------: |
| Moldboard Plow          |              0.95 |               150 |
| Tandem Disk (Regular)   |              0.60 |                75 |
| Chisel Plow             |              0.30 |               150 |
| Field Cultivator        |              0.30 |               100 |
| Row Cultivator          |              0.25 |                25 |
| Harrow (Tine)           |              0.20 |                25 |
| Deep Ripper / Subsoiler |              0.25 |               350 |

*(Sources: SWAT-TAMU 2019, NRCS RUSLE2 Database, DayCent Manual v2023-02-28)*

---

### Example Area Fractions (user inputs recommended)

| Operation                                 | Typical (A_f) | Rationale / Source                                               |
| ----------------------------------------- | ------------: | ---------------------------------------------------------------- |
| Strip Till                                |          0.27 | 6–8 inch bands on 30-inch rows (Deere ST12, NCSU CES 2022)       |
| Field Cultivator                          |           1.0 | Full-width mixing (RUSLE2/SWAT)                                  |
| Row Cleaners                              |          0.17 | 5-inch clear band / 30-inch rows (Yetter 2967 Rigid Row Cleaner) |
| Furrow Diker                              |          0.30 | Corrugation ~9-inch width / 30-inch spacing (NRCS NEH-15 Ch.4)   |
| Non-tillage (planting, spraying, harvest) |           0.0 | No mixing or disturbance                                         |

---

## **How to Run**

From repository root:

```bash
python code/stircalcs.py
```

Outputs will appear in:

```
data/derived/
```

---

## **References**

1. **SWAT–TAMU (2019)**. *Soil and Water Assessment Tool Theoretical Documentation: Agricultural Management.*
   Texas A&M University, College Station, TX.

2. **Parton, W. et al. (2023).** *DayCent Model Technical Manual (v2023-02-28).* Natural Resource Ecology Laboratory, Colorado State University.

3. **NRCS (2018).** *RUSLE2 Official Database – Tillage and Residue Management Operations.*

4. **USDA NRCS (2010).** *National Engineering Handbook, Part 623, Chapter 4: Furrow and Basin Irrigation.*

5. **Deere & Company (2020).** *ST12 Strip-Till Operator’s Manual.*

---

Would you like me to add a short **“Recommended citation”** and a schematic figure (STIR workflow diagram in Markdown or mermaid)?
That would make it look like a formal methods appendix or repo README.
