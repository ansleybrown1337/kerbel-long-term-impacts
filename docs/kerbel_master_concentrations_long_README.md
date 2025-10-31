# Kerbel Master Water Quality – Long Format

This document describes the **long-format dataset** generated from the master water-quality data file
(`Master_WaterQuality_Kerbel_LastUpdated_10272025.csv`).  
The transformation was performed using the Python script `code/wq_longify.py`.

---

## 1. Overview

The script restructures the original wide-format master dataset into a **tidy long format**, where each row represents a **single analyte result** for a unique sampling event.  
This long format is more suitable for data analysis, modeling, and visualization in R or Python.

### Transformation Steps

1. **Read source file**  
   The original file has three header rows:
   - Row 1: Category headers (e.g., *Concentrations*, *Flow and Method*, *Load*)
   - Row 2: Actual column names
   - Row 3: Units (ignored for this format)

2. **Drop non-essential Load columns**  
   The block of columns between `FlumeMethod` and `TSSMethod` represents *load calculations* and is not included in the long-format output.

3. **Keep relevant metadata and concentration analytes**  
   Metadata (site, date, method, etc.) and analyte concentrations are retained.

4. **Melt concentrations to long format**  
   Each analyte (e.g., Nitrate, TKN, OrthoP) becomes its own row under the column `Analyte`, with its measured value stored in `Result_mg_L`.

5. **Pair inflow and outflow samples**  
   - Each OUT (CT, MT, ST) sample row is matched with its corresponding INF (inflow) analyte using keys:
     `Year`, `Date`, `Irrigation`, `Rep`, and `Analyte`.
   - The inflow concentration is stored in `Inflow_Result_mg_L`.
   - The inflow flag (if any) is stored in `Inflow_Flag`.
   - A boolean field `Has_Inflow` indicates whether a valid inflow match was found.

6. **Preserve schema tokens**  
   Values like `"NA"`, `"NA.IRR"`, `"U"`, `"None"` are preserved as text rather than converted to nulls.

7. **Drop INF rows**  
   After pairing, all inflow-only rows are removed; only outflow records remain.

---

## 2. Output File

**Path:** `out/kerbel_master_concentrations_long.csv`

Each row represents a single analyte observation for a unique **Date × Irrigation × Rep × Treatment** combination, optionally paired with its inflow value.

---

## 3. Column Descriptions

| Column | Description |
|---------|--------------|
| **Date** | Sampling date (MM/DD/YYYY format). |
| **Year** | Calendar year of sampling. |
| **Irrigation** | Irrigation event number (1–n within a season). |
| **Rep** | Replicate number (1–n). |
| **Treatment** | Field treatment identifier (CT = Conventional Till, MT = Minimum Till, ST = Strip Till). |
| **InflowOutflow** | Indicates whether sample was taken from inflow (INF) or outflow (OUT). In the long format, only OUT rows remain. |
| **SampleID** | Unique sample identifier assigned during field collection. |
| **FF** | Indicates if flow was filtered (`TRUE`/`FALSE`). |
| **Composite** | Indicates if the sample represents a composite of multiple subsamples. |
| **Duplicate** | Indicates if the sample is a duplicate measurement. |
| **Flag** | Data flag for outflow record (e.g., QA/QC codes). |
| **NoRunoff** | Logical flag indicating whether runoff was observed. |
| **Volume** | Measured runoff volume (liters). |
| **SampleMethod** | Method used to collect sample (e.g., 7 L Bucket, Transducer, etc.). |
| **MeasureMethod** | Equipment type or procedure used to measure flow or depth (e.g., Siphon, Gated Pipe). |
| **IrrMethod** | Irrigation method (e.g., 7 V, 10 V, Weir). |
| **FlumeMethod** | Flume or weir type used to measure flow. |
| **TSSMethod** | Method used to determine Total Suspended Solids (TSS). |
| **Lab** | Analytical laboratory performing chemical analyses. |
| **Notes** | Additional sample notes and context. |
| **Analyte** | Chemical or physical analyte name (e.g., OrthoP, TKN, Nitrate). |
| **Result_mg_L** | Measured concentration of the analyte in mg/L for the OUT (field runoff) sample. |
| **Inflow_Result_mg_L** | Corresponding inflow analyte concentration for the same event. `"NA"` if not available. |
| **Inflow_Flag** | Data flag associated with the inflow sample. `"NA"` if not available. |
| **Has_Inflow** | Boolean (`TRUE`/`FALSE`) indicating if an inflow value was successfully paired. |

---

## 4. Notes on Data Quality and Schema

- `"NA"` values represent missing data and are preserved as literal strings for schema compatibility.  
- `"U"` designates results below detection limits (undetected).  
- `"NA.IRR"` represents cases where no runoff occurred (irrigation-only events).  
- `"None"` occurs in categorical fields where no method or lab information was recorded.

---

## 5. Example

| Date | Year | Irrigation | Rep | Treatment | Analyte | Result_mg_L | Inflow_Result_mg_L | Has_Inflow |
|------|------|-------------|-----|------------|----------|---------------|--------------------|-------------|
| 6/30/2011 | 2011 | 1 | 1 | CT | OrthoP | 0.064 | 0.01 | TRUE |
| 6/30/2011 | 2011 | 1 | 1 | MT | OrthoP | 0.087 | 0.01 | TRUE |
| 7/1/2011 | 2011 | 1 | 2 | ST | OrthoP | 0.118 | 0.01 | TRUE |
| 7/28/2011 | 2011 | 3 | 1 | CT | OrthoP | NA | NA | FALSE |

---

## 6. Reproducibility

To regenerate this dataset:

```bash
cd path/to/kerbel-long-term-impacts
python code/wq_longify.py
```

Output file will be saved to:

```
out/kerbel_master_concentrations_long.csv
```

---

**Last Updated:** October 2025  
**Author:** A. Kerbel  
**Maintainer:** Agricultural Water Quality Program (CSU)
