#!/usr/bin/env python3
"""
merge_wq_stir_by_season_v2.py

Purpose
-------
Merge OUT water-quality samples (long format) with STIR disturbance metrics
computed from daily STIR (wide by treatment: CT_STIR, MT_STIR, ST_STIR) and
true crop-season windows defined by Plant/Harvest dates in `data/crop records.csv`.
For every OUT sample row, attach:

- STIR_daily_at_date         : daily STIR on (or last prior to) the sample Date
- STIR_cum_all_to_date       : cumulative STIR from all available history up to sample Date
- STIR_season_cum_to_date    : cumulative STIR during the active crop season
                               (PlantDate..sample Date), computed as:
                               cum_all_at_sample - cum_all_at_plant_baseline

Inputs (relative to repo root)
-----------------------------
- out/kerbel_master_concentrations_long.csv  (long WQ; produced by wq_longify.py)
- data/stir_daily_system_wide.csv            (daily STIR, wide by treatment)
- data/crop records.csv                      (Plant and Harvest dates per year)

Output
------
- out/wq_with_stir_merged_by_season.csv

Notes
-----
- STIR files do not include Rep; we merge on Treatment only and carry Rep through.
- We only keep OUT rows from WQ for modeling.
- Special tokens ("NA","NA.IRR","U","None") in WQ columns are preserved as text;
  new numeric STIR columns are left blank if they cannot be computed.

"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WQ_LONG = REPO_ROOT / "out" / "kerbel_master_concentrations_long.csv"
STIR_DAILY = REPO_ROOT / "out" / "stir_daily_system_wide.csv"
CROP_RECORDS = REPO_ROOT / "data" / "crop records.csv"
OUTFILE = REPO_ROOT / "out" / "wq_with_stir_merged_by_season.csv"

PRESERVE_TOKENS = {"NA", "NA.IRR", "U", "None", ""}

def to_date(s: pd.Series) -> pd.Series:
    # Coerce to datetime; tokens become NaT
    return pd.to_datetime(s, errors="coerce")

def read_wq():
    if not WQ_LONG.exists():
        raise FileNotFoundError(f"WQ long not found: {WQ_LONG}")
    df = pd.read_csv(WQ_LONG, keep_default_na=False)
    # Keep only OUT samples
    df = df[df["InflowOutflow"].astype(str).str.upper() == "OUT"].copy()
    # Dates
    df["Date_dt"] = to_date(df["Date"])
    # Enforce canonical Treatment labels (strip/upper)
    df["Treatment"] = df["Treatment"].astype(str).str.strip().str.upper()
    # Preserve tokens in WQ value columns by leaving them as-is (already strings)
    return df

def read_crop_records():
    if not CROP_RECORDS.exists():
        raise FileNotFoundError(f"Crop records not found: {CROP_RECORDS}")
    cr = pd.read_csv(CROP_RECORDS, keep_default_na=False)
    # Expect columns: 'plant date', 'harvest date', 'crop', yields...
    # Normalize header case/whitespace for dates
    cols = {c.lower().strip(): c for c in cr.columns}
    pcol = cols.get("plant date")
    hcol = cols.get("harvest date")
    if pcol is None or hcol is None:
        raise KeyError("Could not find 'plant date'/'harvest date' columns in crop records.")
    cr = cr.rename(columns={pcol: "PlantDate", hcol: "HarvestDate"})
    cr["PlantDate_dt"] = to_date(cr["PlantDate"])
    cr["HarvestDate_dt"] = to_date(cr["HarvestDate"])
    # Add Year convenience (plant year)
    cr["Year"] = cr["PlantDate_dt"].dt.year
    # We'll later cross with Treatment (CT/MT/ST); crop records are field-level.
    return cr[["PlantDate", "HarvestDate", "PlantDate_dt", "HarvestDate_dt", "Year", "crop"]]

def read_stir_daily():
    if not STIR_DAILY.exists():
        raise FileNotFoundError(f"STIR daily not found: {STIR_DAILY}")
    sd = pd.read_csv(STIR_DAILY, keep_default_na=False)
    # Required columns we saw when inspecting:
    required = ["Date", "Year",
                "CT_STIR", "MT_STIR", "ST_STIR",
                "CT_STIR_Sum_All", "MT_STIR_Sum_All", "ST_STIR_Sum_All"]
    missing = [c for c in required if c not in sd.columns]
    if missing:
        raise KeyError(f"Missing columns in stir_daily_system_wide.csv: {missing}")
    sd["Date_dt"] = to_date(sd["Date"])
    # Pivot wide->long by Treatment
    long_parts = []
    for trt in ["CT", "MT", "ST"]:
        part = sd[["Date", "Date_dt", "Year",
                   f"{trt}_STIR", f"{trt}_STIR_Sum_All"]].copy()
        part = part.rename(columns={f"{trt}_STIR": "STIR_daily",
                                    f"{trt}_STIR_Sum_All": "STIR_cum_all"})
        part["Treatment"] = trt
        long_parts.append(part)
    sdl = pd.concat(long_parts, ignore_index=True)
    # Ensure numeric
    for c in ["STIR_daily", "STIR_cum_all"]:
        sdl[c] = pd.to_numeric(sdl[c], errors="coerce")
    # Sort for merge_asof
    sdl = sdl.sort_values(["Treatment", "Date_dt"])
    return sdl

def attach_season_windows(out_df: pd.DataFrame, crop: pd.DataFrame) -> pd.DataFrame:
    """
    For each OUT sample row, find the crop season containing its Date_dt:
    PlantDate_dt <= Date_dt <= HarvestDate_dt.
    Because crop records are field-level (not per Treatment), we reuse same window for all treatments.
    """
    # Cartesian-merge lite: merge all samples with all crop windows of the same Year or adjacent,
    # then filter to the single window that contains the sample date.
    # We'll allow a +-180 day tolerance around sample date to reduce candidate set.
    # Create minimal crop seasons table
    seasons = crop[["PlantDate_dt", "HarvestDate_dt", "Year", "crop"]].copy()
    # For matching windows, allow sample year to match either plant year or harvest year
    out_df = out_df.copy()
    out_df["SampleYear"] = out_df["Date_dt"].dt.year

    # Pair everything, then filter to window containment
    merged = out_df.merge(seasons, how="cross")
    mask = (merged["Date_dt"] >= merged["PlantDate_dt"]) & (merged["Date_dt"] <= merged["HarvestDate_dt"])
    merged = merged[mask].copy()

    # If multiple seasons overlap a date (unlikely), take the closest PlantDate prior to sample
    merged["plant_gap_days"] = (merged["Date_dt"] - merged["PlantDate_dt"]).dt.days
    merged = merged.sort_values(["Date_dt", "Treatment", "Rep", "plant_gap_days"])
    merged = merged.groupby(["Date", "Treatment", "Rep", "Analyte", "SampleID"], as_index=False).first()

    return merged

def merge_stir(out_df: pd.DataFrame, sdl: pd.DataFrame, seasons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Using merge_asof within each Treatment:
    - Get STIR_cum_all at PlantDate (baseline) and at sample Date
    - Get STIR_daily at sample Date (or last prior)
    """
    # Prepare per-treatment sorted frames
    result = seasons_df.copy()

    # asof merge for cum_all at sample
    result = pd.merge_asof(
        result.sort_values(["Treatment", "Date_dt"]),
        sdl.sort_values(["Treatment", "Date_dt"]),
        by="Treatment",
        left_on="Date_dt",
        right_on="Date_dt",
        direction="backward",
        suffixes=("", "_sdl"),
    )

    # Rename daily at sample for clarity
    result = result.rename(columns={"STIR_daily": "STIR_daily_at_date",
                                    "STIR_cum_all": "STIR_cum_all_to_date"})

    # Baseline at plant date
    base = pd.merge_asof(
        seasons_df.sort_values(["Treatment", "PlantDate_dt"]),
        sdl.sort_values(["Treatment", "Date_dt"]),
        by="Treatment",
        left_on="PlantDate_dt",
        right_on="Date_dt",
        direction="backward",
        suffixes=("", "_base"),
    )[["Date", "Treatment", "Rep", "Analyte", "SampleID", "PlantDate_dt", "STIR_cum_all"]].rename(
        columns={"STIR_cum_all": "STIR_cum_all_at_plant"}
    )

    result = result.merge(
        base,
        on=["Date", "Treatment", "Rep", "Analyte", "SampleID", "PlantDate_dt"],
        how="left",
    )

    # Season cumulative to date
    result["STIR_season_cum_to_date"] = result["STIR_cum_all_to_date"] - result["STIR_cum_all_at_plant"]

    return result

def main():
    print("[INFO] Reading inputs...")
    wq = read_wq()
    sdl = read_stir_daily()
    crop = read_crop_records()

    # Keep only columns we need from WQ to cut payload
    keep_cols = [
        "Date","Year","Irrigation","Rep","Treatment","InflowOutflow","SampleID","FF","Composite","Duplicate",
        "Flag","NoRunoff","Volume","SampleMethod","MeasureMethod","IrrMethod","FlumeMethod","TSSMethod","Lab","Notes",
        "Analyte","Result_mg_L","Inflow_Result_mg_L","Date_dt"
    ]
    wq = wq[keep_cols].copy()

    print("[INFO] Attaching season windows via Plant/Harvest...")
    wq_in_season = attach_season_windows(wq, crop)

    # If no season window found for some rows, drop (cannot compute season cum)
    missing = set(wq.index) - set(wq_in_season.index)
    if missing:
        print(f"[WARN] {len(missing)} OUT rows did not fall within any Plant–Harvest window and will not have season metrics.")

    print("[INFO] Merging STIR metrics...")
    merged = merge_stir(wq, sdl, wq_in_season)

    # Final column order
    final_cols = [
        "Date","Year","Irrigation","Rep","Treatment","InflowOutflow","SampleID","FF","Composite","Duplicate",
        "Flag","NoRunoff","Volume","SampleMethod","MeasureMethod","IrrMethod","FlumeMethod","TSSMethod","Lab","Notes",
        "Analyte","Result_mg_L","Inflow_Result_mg_L",
        "PlantDate_dt","HarvestDate_dt","crop",
        "STIR_daily_at_date","STIR_cum_all_to_date","STIR_cum_all_at_plant","STIR_season_cum_to_date"
    ]

    # Bring back human-readable Plant/Harvest strings for convenience
    # Join minimal window info
    season_info = wq_in_season[["Date","Treatment","Rep","Analyte","SampleID","PlantDate_dt","HarvestDate_dt","crop"]].copy().drop_duplicates()
    merged = merged.merge(season_info, on=["Date","Treatment","Rep","Analyte","SampleID","PlantDate_dt"], how="left")

    # Reorder/select
    merged = merged[final_cols]

    # Write CSV (preserve WQ tokens by not letting pandas convert them)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTFILE, index=False)

    print(f"[OK] Wrote merged file → {OUTFILE}")
    print(f"[INFO] Rows (OUT with season match): {len(merged)}")

if __name__ == "__main__":
    main()
