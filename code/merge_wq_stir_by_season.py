#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_wq_stir_by_season.py

Purpose
-------
Merge water-quality (WQ) long-format data with STIR event data (long format),
and compute two "to date" STIR metrics for each WQ sample:
  1) Season_STIR_toDate: cumulative STIR from the season PlantDate up to the sample Date
  2) CumAll_STIR_toDate: cumulative STIR from the first available STIR event up to the sample Date

Season windows are defined by crop records (PlantDate -> HarvestDate), joined by Treatment.

Assumptions / Inputs
--------------------
- WQ long CSV (e.g., out/kerbel_master_concentrations_long.csv) contains at minimum:
    Date, Treatment (CT/MT/ST), SampleID, Analyte
  Additional columns are preserved.
- STIR events long CSV (e.g., out/stir_events_long.csv) contains:
    Date, System (CT/MT/ST), STIR   (per-event value)
  A cumulative column STIR_cum_all is created upstream by compute_cumulative_stir().
- Crop records CSV (e.g., data/crop records.csv) contains:
    PlantDate, HarvestDate
  SeasonYear may be present or will be derived from PlantDate when absent.

Outputs
-------
- out/wq_with_stir_by_season.csv
    All WQ rows with season context and STIR metrics:
      * Season_STIR_toDate
      * CumAll_STIR_toDate
      * PlantDate, HarvestDate, SeasonYear (season window columns)
- out/wq_with_stir_unmatched.csv
    Subset of WQ rows that fell inside a season window but had no STIR value up to their Date
    (e.g., no prior operations yet) — useful for QC.

CLI
---
Example:
  python code/merge_wq_stir_by_season.py --wq out/kerbel_master_concentrations_long.csv --stir out/stir_events_long.csv --crops "data/crop records.csv" --out out --debug

Notes
-----
- Per-treatment merge_asof on sorted Date keys avoids cross-group sort issues.
- Dates are parsed strictly; rows with non-parsable Date/PlantDate/HarvestDate are dropped (warned in --debug).
- Treatment/System labels are normalized to uppercase {CT, MT, ST} for joining; other labels pass through but only match when equal on both sides.
"""


from __future__ import annotations

import argparse
import os
from typing import Tuple
import pandas as pd
import numpy as np


def _to_datetime(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    bad = s.isna() & series.notna()
    if bad.any():
        print(f"[WARN] {name}: {bad.sum()} rows had unparsable dates; dropping later if required.")
    return s


def read_wq(path: str, debug: bool=False) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize expected columns
    for req in ["Date", "Treatment"]:
        if req not in df.columns:
            raise KeyError(f"[WQ] Required column '{req}' not found in {path}. cols={list(df.columns)}")

    # Clean
    df["Date"] = _to_datetime(df["Date"], "WQ.Date")
    df = df[df["Date"].notna()].copy()
    df["Treatment"] = df["Treatment"].astype(str).str.strip().str.upper()

    # Optional: ensure SampleID & Analyte exist (if not, make placeholders)
    if "SampleID" not in df.columns:
        df["SampleID"] = np.arange(len(df))
    if "Analyte" not in df.columns:
        df["Analyte"] = "NA"

    if debug:
        tts = sorted(df["Treatment"].dropna().unique().tolist())
        print(f"[INFO] WQ rows: {len(df)}  Treatments: {tts}")
    return df


def read_stir(path: str, debug: bool=False) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Try to detect the treatment/system column (commonly 'System')
    cand_sys = [c for c in df.columns if c.lower() in ("system", "treatment")]
    if not cand_sys:
        raise KeyError(f"[STIR] Could not find 'System' or 'Treatment' column in {path}. cols={list(df.columns)}")

    sys_col = cand_sys[0]
    if "Date" not in df.columns:
        raise KeyError(f"[STIR] Required column 'Date' not found in {path}. cols={list(df.columns)}")

    # Try to find STIR value column
    cand_stir = [c for c in df.columns if c.strip().upper() in ("STIR", "STIR_EVENT", "STIR_VALUE")]
    if not cand_stir:
        cand_stir = [c for c in df.columns if "stir" in c.lower()]
    if not cand_stir:
        raise KeyError(f"[STIR] Could not locate STIR value column in {path}. cols={list(df.columns)}")
    stir_col = cand_stir[0]

    df["Date"] = _to_datetime(df["Date"], "STIR.Date")
    df = df[df["Date"].notna()].copy()
    df["System"] = df[sys_col].astype(str).str.strip().str.upper()
    df["STIR_val"] = pd.to_numeric(df[stir_col], errors="coerce")
    if df["STIR_val"].isna().all():
        raise ValueError(f"[STIR] All STIR values are NaN after parsing from column '{stir_col}'.")

    keep = ["Date", "System", "STIR_val"]
    extra = [c for c in df.columns if c not in keep]
    df = df[keep + extra].sort_values(["System", "Date"], kind="mergesort")

    if debug:
        tts = sorted(df["System"].dropna().unique().tolist())
        print(f"[INFO] STIR events rows: {len(df)} | Treatments: {tts}")
    return df


def read_crop_records(path: str, debug: bool=False) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for need in ["plant date", "harvest date"]:
        if need not in cols_lower:
            raise KeyError(f"[CROPS] Missing required column '{need}' in {path}. "
                           f"Available: {list(df.columns)}")

    plant_col = cols_lower["plant date"]
    harvest_col = cols_lower["harvest date"]

    # Normalize to the names attach_season_windows expects
    out = pd.DataFrame({
        "PlantDate": pd.to_datetime(df[plant_col], errors="coerce"),
        "HarvestDate": pd.to_datetime(df[harvest_col], errors="coerce"),
    })
    out["SeasonYear"] = out["PlantDate"].dt.year

    # Optional passthroughs if present
    if "treatment" in cols_lower:
        out["Treatment"] = df[cols_lower["treatment"]].astype(str).str.strip().str.upper()
    if "crop" in cols_lower:
        out["Crop"] = df[cols_lower["crop"]]

    out = out.loc[out["PlantDate"].notna() & out["HarvestDate"].notna()].sort_values(
        ["PlantDate", "HarvestDate"], kind="mergesort"
    )

    if debug and not out.empty:
        y0, y1 = int(out["PlantDate"].dt.year.min()), int(out["PlantDate"].dt.year.max())
        print(f"[INFO] Crop season rows (valid): {len(out)}  Year span: {y0}–{y1}")

    return out


def attach_season_windows(dfw: pd.DataFrame, dfc: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Attach crop-season windows (PlantDate, HarvestDate, SeasonYear, Crop, etc.) to the
    water-quality table by Treatment and Date. Robust to multiple crop-record schemas:
      - Season_Plant / Season_Harvest / Season_Year
      - 'plant date' / 'harvest date' (any case/spacing)
      - Optional Treatment in crop file (will be replicated across WQ treatments if missing)

    Expected in dfw (WQ): at minimum ['Treatment', 'Date'].
    Returns dfw with season columns appended.
    """
    dfw = dfw.copy()
    dfc = dfc.copy()

    # --------------------------
    # Normalize WQ 'Date' column
    # --------------------------
    # Accept legacy/case variants
    wq_cols_lower = {c.lower().strip(): c for c in dfw.columns}
    if "date" not in wq_cols_lower and "sampledate" in wq_cols_lower:
        # back-compat: promote 'SampleDate' to 'Date'
        dfw = dfw.rename(columns={wq_cols_lower["sampledate"]: "Date"})
    elif "date" in wq_cols_lower and wq_cols_lower["date"] != "Date":
        dfw = dfw.rename(columns={wq_cols_lower["date"]: "Date"})

    if "Date" not in dfw.columns:
        raise KeyError("[WQ] Required 'Date' column not found (nor 'SampleDate').")

    # Coerce Date to datetime
    dfw["Date"] = pd.to_datetime(dfw["Date"], errors="coerce", utc=False)
    if dfw["Date"].isna().any():
        bad = int(dfw["Date"].isna().sum())
        raise ValueError(f"[WQ] {bad} rows have non-parsable Date values.")

    # Normalize Treatment casing/whitespace
    if "Treatment" not in dfw.columns:
        raise KeyError("[WQ] Required 'Treatment' column not found.")
    dfw["Treatment"] = (
        dfw["Treatment"].astype(str).str.strip().str.upper()
    )

    # -----------------------------------------
    # Harmonize crop-records column name schema
    # -----------------------------------------
    # Direct Season_* to canonical names
    rename_map = {}
    if {"Season_Plant", "Season_Harvest"}.issubset(dfc.columns):
        rename_map.update({"Season_Plant": "PlantDate", "Season_Harvest": "HarvestDate"})
    if "Season_Year" in dfc.columns and "SeasonYear" not in dfc.columns:
        rename_map["Season_Year"] = "SeasonYear"
    if rename_map:
        dfc = dfc.rename(columns=rename_map)

    # If not Season_* pattern, look for 'plant date'/'harvest date' (any case/spacing)
    crop_cols_lower = {c.lower().strip(): c for c in dfc.columns}
    if "plantdate" not in {c.lower().replace(" ", "") for c in dfc.columns}:
        # Try to detect by friendly names
        plant_key = None
        harvest_key = None
        for k, orig in crop_cols_lower.items():
            k_clean = k.replace(" ", "")
            if plant_key is None and k_clean in ("plantdate", "plantingdate", "plant"):
                plant_key = orig
            if harvest_key is None and k_clean in ("harvestdate", "harvestingdate", "harvest"):
                harvest_key = orig
        if plant_key and harvest_key:
            dfc = dfc.rename(columns={plant_key: "PlantDate", harvest_key: "HarvestDate"})

    # Coerce Plant/Harvest to datetime where present
    for col in ("PlantDate", "HarvestDate"):
        if col in dfc.columns:
            dfc[col] = pd.to_datetime(dfc[col], errors="coerce", utc=False)

    # Ensure SeasonYear exists (derive from PlantDate if missing)
    if "SeasonYear" not in dfc.columns:
        if "PlantDate" in dfc.columns and dfc["PlantDate"].notna().any():
            dfc["SeasonYear"] = dfc["PlantDate"].dt.year
        elif "HarvestDate" in dfc.columns and dfc["HarvestDate"].notna().any():
            dfc["SeasonYear"] = dfc["HarvestDate"].dt.year
        else:
            # As a last resort, try an integer-like 'year' column
            year_like = None
            for k, orig in crop_cols_lower.items():
                if k in ("year", "seasonyear", "cropyear"):
                    year_like = orig
                    break
            if year_like is not None:
                dfc = dfc.rename(columns={year_like: "SeasonYear"})
            else:
                raise KeyError("[CROPS] Could not infer 'SeasonYear' (need PlantDate/HarvestDate or a year column).")

    # Optional columns: Crop present?
    if "Crop" not in dfc.columns:
        # create empty if not available
        dfc["Crop"] = pd.NA

    # ----------------------------------------------------------------
    # Replicate season windows across treatments if Treatment missing
    # ----------------------------------------------------------------
    if "Treatment" not in dfc.columns or dfc["Treatment"].isna().all():
        trts = (
            dfw["Treatment"].dropna().astype(str).str.strip().str.upper().unique()
        )
        dfc["_k"] = 1
        tx = pd.DataFrame({"Treatment": trts, "_k": 1})
        dfc = dfc.merge(tx, on="_k", how="left").drop(columns="_k")
    else:
        # Normalize Treatment values in crops too
        dfc["Treatment"] = dfc["Treatment"].astype(str).str.strip().str.upper()

    # --------------------------------
    # Validate required crop columns
    # --------------------------------
    req = ["PlantDate", "HarvestDate", "SeasonYear", "Treatment"]
    missing = [c for c in req if c not in dfc.columns]
    if missing:
        raise KeyError(f"[CROPS] Missing required columns post-normalization: {missing}. "
                       f"Have: {list(dfc.columns)}")

    # Drop rows with invalid/empty dates
    dfc = dfc.loc[dfc["PlantDate"].notna() & dfc["HarvestDate"].notna()].copy()
    if dfc.empty:
        raise ValueError("[CROPS] No valid rows with both PlantDate and HarvestDate after parsing.")

    # Ensure PlantDate <= HarvestDate (swap if user accidentally flipped)
    swapped = dfc["PlantDate"] > dfc["HarvestDate"]
    if swapped.any():
        if debug:
            print(f"[WARN] {int(swapped.sum())} crop rows had PlantDate > HarvestDate; swapping.")
        tmp = dfc.loc[swapped, "PlantDate"].copy()
        dfc.loc[swapped, "PlantDate"] = dfc.loc[swapped, "HarvestDate"]
        dfc.loc[swapped, "HarvestDate"] = tmp

    # -----------------------------
    # Attach windows to WQ by join
    # -----------------------------
    dfw = dfw.reset_index(drop=False).rename(columns={"index": "_wq_idx"})
    merged = dfw.merge(dfc, on="Treatment", how="left", suffixes=("", "_crop"))

    # Keep only rows whose WQ Date falls inside the crop window
    in_window = merged["Date"].between(merged["PlantDate"], merged["HarvestDate"], inclusive="both")
    win = merged.loc[in_window].copy()

    if win.empty:
        if debug:
            # Provide a short diagnostic of ranges by treatment to help troubleshoot
            rng = (
                dfc.groupby("Treatment", dropna=False)
                   .agg(PlantMin=("PlantDate", "min"),
                        HarvestMax=("HarvestDate", "max"),
                        NSeasons=("SeasonYear", "nunique"))
                   .reset_index()
            )
            print("[DEBUG] No WQ dates fell into crop windows. Crop ranges by Treatment:")
            print(rng.to_string(index=False))
        raise ValueError("[ATTACH] No WQ rows fall within any crop-season window. "
                         "Check date ranges and Treatment labels between WQ and crops.")

    # If multiple windows match a single WQ row (should be rare), keep the one with the
    # latest PlantDate (i.e., most recent season start before/at the WQ Date).
    win["__plant_rank"] = win.groupby("_wq_idx")["PlantDate"].rank(method="first", ascending=False)
    chosen = win.loc[win["__plant_rank"] == 1].drop(columns="__plant_rank", errors="ignore")

    # Restore original order and provide a clean sort
    chosen = chosen.sort_values(["Treatment", "Date"]).reset_index(drop=True)

    if debug:
        y0 = int(dfc["PlantDate"].dt.year.min())
        y1 = int(dfc["HarvestDate"].dt.year.max())
        print(f"[INFO] Attached crop windows to WQ: {len(chosen)}/{len(dfw)} rows matched "
              f"across {y0}–{y1} seasons.")

    return chosen


def compute_cumulative_stir(stir: pd.DataFrame, debug: bool=False) -> pd.DataFrame:
    df = stir.sort_values(["System", "Date"], kind="mergesort").copy()
    df["STIR_cum_all"] = df.groupby("System", sort=False)["STIR_val"].cumsum()
    return df


def merge_stir_with_wq(wq_seasoned: pd.DataFrame, stir_cum: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Attach two STIR 'to date' metrics to season-tagged WQ rows:
      - CumAll_STIR_toDate: cumulative STIR for Treatment from earliest STIR event up to WQ Date (inclusive)
      - Season_STIR_toDate: cumulative STIR within the current crop season window
                            i.e., CumAll_at_Date - CumAll_just_before_PlantDate

    Requirements in wq_seasoned:
      ['Treatment','Date','PlantDate','HarvestDate','SeasonYear'] (already attached upstream)

    Requirements in stir_cum:
      ['System','Date','STIR_cum_all']  (cumulative across all years per system)
    """
    # Defensive copies
    wq = wq_seasoned.copy()
    sc_all = stir_cum.copy()

    # Hard requirements
    for need in ["Treatment", "Date", "PlantDate", "HarvestDate", "SeasonYear"]:
        if need not in wq.columns:
            raise KeyError(f"[WQ] Missing required column: {need}")
    for need in ["System", "Date", "STIR_cum_all"]:
        if need not in sc_all.columns:
            raise KeyError(f"[STIR] Missing required column: {need}")

    # Datetime coercions (drop rows with bad dates rather than letting merge_asof choke)
    for col in ["Date", "PlantDate", "HarvestDate"]:
        wq[col] = pd.to_datetime(wq[col], errors="coerce")
    sc_all["Date"] = pd.to_datetime(sc_all["Date"], errors="coerce")

    bad_wq = wq[["Date","PlantDate","HarvestDate"]].isna().any(axis=1)
    if bad_wq.any():
        if debug:
            print(f"[WARN] Dropping {int(bad_wq.sum())} WQ rows with NaT in Date/PlantDate/HarvestDate before merge.")
        wq = wq.loc[~bad_wq].copy()

    # Normalize labels for safe matching
    wq["Treatment"] = wq["Treatment"].astype(str).str.strip().str.upper()
    sc_all["System"]  = sc_all["System"].astype(str).str.strip().str.upper()

    # Prepare outputs
    wq["CumAll_STIR_toDate"] = np.nan
    wq["Season_STIR_toDate"] = np.nan

    # Per-treatment asof merges: this avoids 'left keys must be sorted' across mixed groups
    eps = pd.to_timedelta(1, unit="us")
    out_pieces = []

    trts = wq["Treatment"].dropna().unique().tolist()
    for trt in trts:
        wq_t = wq.loc[wq["Treatment"] == trt].copy()
        sc_t = sc_all.loc[sc_all["System"] == trt].copy()

        # If we have no STIR for this treatment, leave NaN metrics and move on
        if sc_t.empty:
            out_pieces.append(wq_t)
            continue

        # Strict sort on the asof key within the group (critical!)
        wq_t = wq_t.sort_values("Date", kind="mergesort").reset_index(drop=True)
        sc_t = sc_t.sort_values("Date", kind="mergesort").reset_index(drop=True)

        # CumAll_STIR_toDate at the sample Date
        merge_all = pd.merge_asof(
            wq_t[["Date"]], sc_t[["Date","STIR_cum_all"]],
            left_on="Date", right_on="Date",
            direction="backward", allow_exact_matches=True
        )
        wq_t["CumAll_STIR_toDate"] = merge_all["STIR_cum_all"].to_numpy()

        # Baseline cumulative just BEFORE PlantDate (epsilon step back)
        plant_minus = (wq_t["PlantDate"] - eps).rename("Date")
        base_all = pd.merge_asof(
            plant_minus.to_frame(), sc_t[["Date","STIR_cum_all"]],
            left_on="Date", right_on="Date",
            direction="backward", allow_exact_matches=True
        )["STIR_cum_all"]

        # Season value = current cumAll - baseline at season start
        wq_t["Season_STIR_toDate"] = wq_t["CumAll_STIR_toDate"] - base_all.to_numpy()

        out_pieces.append(wq_t)

    merged = pd.concat(out_pieces, ignore_index=True)

    if debug:
        n_nan_all = int(merged["CumAll_STIR_toDate"].isna().sum())
        n_nan_season = int(merged["Season_STIR_toDate"].isna().sum())
        print(f"[INFO] CumAll_STIR_toDate NaN rows: {n_nan_all} | Season_STIR_toDate NaN rows: {n_nan_season}")

    return merged


def write_unmatched_csv(
    wq_in: pd.DataFrame,
    merged: pd.DataFrame,
    out_path: str,
    keys=("SampleID", "Date", "Treatment"),
    keep_cols=("SampleID", "Date", "Treatment", "Analyte"),
) -> pd.DataFrame:
    for k in keys:
        if k not in wq_in.columns:
            raise KeyError(f"[unmatched] key '{k}' missing in wq_in columns: {list(wq_in.columns)}")
        if k not in merged.columns:
            raise KeyError(f"[unmatched] key '{k}' missing in merged columns: {list(merged.columns)}")

    if "Date" in keys:
        wq_in = wq_in.copy()
        merged = merged.copy()
        wq_in["Date"] = pd.to_datetime(wq_in["Date"], errors="coerce")
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")

    if merged.duplicated(list(keys)).any():
        merged_keys_only = merged[list(keys)].drop_duplicates()
    else:
        merged_keys_only = merged[list(keys)].drop_duplicates()

    base_cols = [c for c in keep_cols if c in wq_in.columns]
    if not base_cols:
        base_cols = list(keys)

    left = wq_in[base_cols].copy()
    anti = left.merge(merged_keys_only, on=list(keys), how="left", indicator=True)
    unmatched = anti[anti["_merge"] == "left_only"].drop(columns=["_merge"]).drop_duplicates(list(keys))

    unmatched.to_csv(out_path, index=False)
    print(f"[INFO] Unmatched rows written: {len(unmatched)} -> {out_path}")
    return unmatched


def main():
    ap = argparse.ArgumentParser(description="Merge WQ (long) with STIR events (long) using crop season windows.")
    ap.add_argument("--wq", default=os.path.join("out", "kerbel_master_concentrations_long.csv"))
    ap.add_argument("--stir", default=os.path.join("out", "stir_events_long.csv"))
    ap.add_argument("--crops", default=os.path.join("data", "crop records.csv"))
    ap.add_argument("--out", default=os.path.join("out"))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    debug = args.debug

    # 1) Read inputs
    if debug: print("[INFO] Reading WQ long-format data ...")
    wq = read_wq(args.wq, debug=debug)

    if debug: print("[INFO] Reading STIR long-format data ...")
    stir = read_stir(args.stir, debug=debug)

    if debug: print("[INFO] Reading crop records ...")
    crops = read_crop_records(args.crops, debug=debug)

    # 2) Attach crop-season windows to each WQ row
    wq_seasoned = attach_season_windows(wq, crops, debug=debug)
    if debug:
        print(f"[INFO] Attached crop windows to WQ: {len(wq_seasoned)}/{len(wq)} rows matched.")

    # 3) Prepare cumulative STIR (all-years running total per Treatment/System)
    stir_cum = compute_cumulative_stir(stir)

    # 4) Merge STIR cumulatives to the season-tagged WQ table
    merged = merge_stir_with_wq(wq_seasoned, stir_cum, debug=debug)

    # 5) Write outputs
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    merged_out = os.path.join(out_dir, "wq_with_stir_by_season.csv")
    unmatched_out = os.path.join(out_dir, "wq_with_stir_unmatched.csv")

    merged.to_csv(merged_out, index=False)
    if debug:
        print(f"[INFO] Merged rows written: {len(merged)} -> {merged_out}")

    # Unmatched list (WQ rows inside a season but with no CumAll STIR to date)
    in_season_mask = merged["PlantDate"].notna()
    wq_in = merged.loc[in_season_mask, ["SampleID", "Date", "Treatment", "Analyte"]].copy()

    merged_keys = merged.loc[in_season_mask, ["SampleID", "Date", "Treatment"]].copy()
    merged_keys["__has_cum"] = merged.loc[in_season_mask, "CumAll_STIR_toDate"].notna().astype(int)
    unmatched_keys = merged_keys.loc[merged_keys["__has_cum"] == 0, ["SampleID", "Date", "Treatment"]].copy()

    _ = write_unmatched_csv(
        wq_in=wq_in,
        merged=unmatched_keys.assign(Analyte="NA"),
        out_path=unmatched_out,
        keys=("SampleID", "Date", "Treatment"),
        keep_cols=("SampleID", "Date", "Treatment", "Analyte"),
    )

    if debug:
        tts = sorted(merged["Treatment"].dropna().unique().tolist())
        print(f"[INFO] Done. Treatments present in merged: {tts}")


if __name__ == "__main__":
    main()
