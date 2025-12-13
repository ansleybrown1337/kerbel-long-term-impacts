# Kerbel Long-Term Impacts – WQ × STIR by Season

This document describes the **final merged dataset** produced by integrating long-format water-quality (WQ) data with tillage disturbance metrics (STIR values) across defined crop-season windows.
The merge is performed by `code/merge_wq_stir_by_season.py` and executed automatically through the master runner `code/main.py`.

---

## 1. Overview

The file `wq_with_stir_by_season.csv` represents the **fully harmonized dataset** combining WQ and tillage information for each monitored treatment (CT, MT, ST) and sampling date.
Each record corresponds to a unique:

```
Date × Treatment × Rep × Analyte
```

and includes:

* season-level metadata (`PlantDate`, `HarvestDate`, `SeasonYear`),
* cumulative tillage disturbance indices (`Season_STIR_toDate`, `CumAll_STIR_toDate`),
* all relevant WQ sample metadata.

This dataset serves as the **primary analytical product** for long-term evaluation of tillage legacy effects on nutrient and sediment runoff.

---

## 2. File Location

**Path:**

```
out/wq_with_stir_by_season.csv
```

**Produced by:**

```bash
python code/main.py --debug
```

---

## 3. Processing Sequence

1. **Water-quality longification**
   Converts the master water-quality file into tidy long format (`wq_longify.py`).

2. **STIR pipeline**
   Compiles tillage operation records and computes cumulative STIR metrics (`stir_pipeline.py`).

3. **Seasonal merge**
   Attaches each WQ sample to its corresponding crop-season window and merges in cumulative STIR totals (`merge_wq_stir_by_season.py`).

---

## 4. Column Descriptions

| Column                                                           | Description                                                                                                                                                |
| :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SampleID**                                                     | Unique identifier assigned to each outflow water sample.                                                                                                   |
| **Date**                                                         | Sampling date (ISO 8601: `YYYY-MM-DD`).                                                                                                                    |
| **Year**                                                         | Calendar year of sample collection.                                                                                                                        |
| **Irrigation**                                                   | Irrigation event number within a crop season.                                                                                                              |
| **Rep**                                                          | Field replicate number.                                                                                                                                    |
| **Treatment**                                                    | Field treatment identifier: CT = Conventional Till, MT = Minimum Till, ST = Strip Till.                                                                    |
| **Analyte**                                                      | Chemical or physical analyte name (e.g., Nitrate, TKN, OrthoP, TP, TSS).                                                                                   |
| **Result_mg_L**                                                  | Measured analyte concentration in mg/L for the OUT (runoff) sample.                                                                                        |
| **Inflow_Result_mg_L**                                           | Corresponding inflow analyte concentration (mg/L), if available; `"NA"` if not matched.                                                                    |
| **Has_Inflow**                                                   | Boolean (`TRUE`/`FALSE`) indicating whether an inflow value was paired successfully.                                                                       |
| **Flag**, **Inflow_Flag**                                        | QA/QC flags for OUT and INF records, respectively.                                                                                                         |
| **Volume**                                                       | Measured runoff event volume (L).                                                                                                                          |
| **NoRunoff**                                                     | Logical flag for irrigation events where no runoff was observed.                                                                                           |
| **FlumeMethod**, **MeasureMethod**, **IrrMethod**, **TSSMethod** | Field or laboratory method identifiers describing flow measurement and analytical approaches.                                                              |
| **Lab**                                                          | Analytical laboratory responsible for chemical analyses.                                                                                                   |
| **Notes**                                                        | Supplemental comments and contextual information from field staff.                                                                                         |
| **PlantDate**                                                    | Start date of the associated crop season (from crop records).                                                                                              |
| **HarvestDate**                                                  | End date of the associated crop season.                                                                                                                    |
| **SeasonYear**                                                   | Year assigned to the crop season, typically based on `PlantDate`.                                                                                          |
| **Crop**                                                         | Crop name or identifier from crop records, if available.                                                                                                   |
| **CumAll_STIR_toDate**                                           | Cumulative STIR index for the treatment from the **first recorded tillage event (2011)** through the WQ sample date. Reflects long-term tillage intensity. |
| **Season_STIR_toDate**                                           | Cumulative STIR index for the treatment **within the current crop season window** (`PlantDate` → sample date). Reflects within-season tillage disturbance. |
| **System** *(optional)*                                          | Alias for Treatment, inherited from STIR records when present.                                                                                             |
| **STIR_cum_all** *(optional)*                                    | Running total STIR value in the STIR log as of each sample’s date (diagnostic column).                                                                     |

---

## 5. Notes on Data Quality and Schema

* `"NA"` values represent data that occurred but were never measured (missing data), and are preserved as literal strings for schema compatibility.
* `"U"` designates results below analytical detection limits (undetected).
* `"NA.IRR"` represents cases where irrigation occurred but no runoff was observed.
* `"None"` occurs in categorical fields where no method, laboratory, or classification was applicable.
* `"TRUE"` and `"FALSE"` values are stored as literal text for compatibility across R and Python environments.
* Date parsing is strict; unparseable rows are dropped with a warning when `--debug` is enabled.
* All `Treatment` and `System` labels are normalized to uppercase (`CT`, `MT`, `ST`) for consistent joins.
* STIR values are dimensionless indices as defined by NRCS (0–1000 scale); larger values indicate higher surface disturbance intensity.

---

## 6. Example (abridged)

| Date       | Treatment | Analyte | Result_mg_L | CumAll_STIR_toDate | Season_STIR_toDate | PlantDate  | HarvestDate | SeasonYear |
| :--------- | :-------- | :------ | :---------: | :----------------: | :----------------: | :--------- | :---------- | :--------- |
| 2011-06-30 | CT        | OrthoP  |    0.064    |         375        |         112        | 2011-05-01 | 2011-09-15  | 2011       |
| 2011-06-30 | MT        | OrthoP  |    0.087    |         190        |         60         | 2011-05-01 | 2011-09-15  | 2011       |
| 2011-07-01 | ST        | OrthoP  |    0.118    |         135        |         52         | 2011-05-01 | 2011-09-15  | 2011       |
| 2013-07-28 | CT        | Nitrate |     1.54    |         690        |         184        | 2013-05-10 | 2013-09-10  | 2013       |

---

## 7. Reproducibility

To reproduce this dataset from the full pipeline:

```bash
cd path/to/kerbel-long-term-impacts
python code/main.py --debug
```

Output files:

```
out/wq_with_stir_by_season.csv
out/wq_with_stir_unmatched.csv
```

---

**Last Updated:** November 2025
**Author:** AJ Brown
**Maintainer:** Agricultural Water Quality Program (CSU)

