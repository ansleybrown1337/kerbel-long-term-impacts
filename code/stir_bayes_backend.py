#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stir_bayes_backend.py

Python equivalent of `stir-bayes-backend.R::clean_wq_stir()`.

This script is intentionally *analysis-specific* and should be called as an
additional pipeline step AFTER `wq_with_stir_by_season.csv` is created.

Implements (as in the R code):
- Handle WQ flag tokens: "U", "NA", "NA.IRR" (and lowercase "u")
- Drop "NA.IRR" rows (no runoff), but preserve true missing values
- Enforce types (dates, numeric columns, key categorical columns)
- Create analyte_abbr (matching your mapping)
- Standardize:
  * cout_z, cin_z per analyte
  * stir_season_z, stir_cumall_z, volume_z globally
Notes:
- Z-scores use sample SD (ddof=1) to match R's sd() / rethinking::standardize().
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd




def _clamp01(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Clamp numeric values to (eps, 1-eps). Useful for Beta models."""
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    return pd.Series(np.clip(vals, eps, 1.0 - eps), index=series.index)
ANALYTE_ABBR = {
    "Ammonium(NH4)": "NH4",
    "ICP": "ICP",
    "Nitrate": "NO3",
    "NitrateNitrite": "NOx",
    "Nitrite": "NO2",
    "NPOC": "NPOC",
    "OrthoP": "OP",
    "Selenium": "Se",
    "TDS": "TDS",
    "TKN": "TKN",
    "TotalN": "TN",
    "TotalP": "TP",
    "TSP": "TSP",
    "TSS": "TSS",
}
ANALYTE_ABBR_LEVELS = ["NH4","ICP","NO3","NOx","NO2","NPOC","OP","Se","TDS","TKN","TN","TP","TSP","TSS"]


def _zscore(x: pd.Series) -> pd.Series:
    """Sample z-score ignoring missing, like rethinking::standardize()."""
    xv = pd.to_numeric(x, errors="coerce").astype(float)
    mu = np.nanmean(xv)
    sd = np.nanstd(xv, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return (xv - mu) * np.nan
    return (xv - mu) / sd


def _token_series(s: pd.Series) -> pd.Series:
    """
    Convert series to cleaned token strings while preserving true missing.
    We want to keep literal tokens "U", "NA", "NA.IRR" for logic, and preserve NA as missing.
    """
    # If already numeric, convert to string but keep NaN as NaN
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float").astype("string")

    # Otherwise treat as string, but preserve missing
    ss = s.astype("string")
    ss = ss.str.strip()
    # normalize lowercase u -> U
    ss = ss.replace({"u": "U"})
    return ss


def _token_to_numeric(s: pd.Series) -> pd.Series:
    """Apply token rules: U -> 0, NA -> missing, otherwise numeric."""
    ss = _token_series(s)
    # Preserve true missing as missing
    # Replace tokens
    ss = ss.replace({"U": "0", "NA": pd.NA})
    return pd.to_numeric(ss, errors="coerce")


def _to_bool_best_effort(s: pd.Series) -> pd.Series:
    """
    Mirror R case_when that expects "TRUE"/"FALSE", but be more robust:
    accepts TRUE/FALSE in any case, 1/0, yes/no, and already-boolean.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    ss = s.astype("string").str.strip()
    up = ss.str.upper()

    mapping = {
        "TRUE": True, "FALSE": False,
        "T": True, "F": False,
        "1": True, "0": False,
        "YES": True, "NO": False,
        "Y": True, "N": False,
    }
    out = up.map(mapping)
    return out.astype("boolean")


def clean_wq_stir(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning + feature engineering for the Bayes STIR pipeline.

    Backwards compatible behavior:
      - Legacy numeric columns (Result_mg_L, Inflow_Result_mg_L) are preserved exactly as before,
        including any nondetect-to-zero mapping that _token_to_numeric() performs.
      - Existing z-score columns (cout_z, cin_z, stir_season_z, stir_cumall_z, volume_z) are preserved.

    New outputs (for future left-censoring / LOD modeling):
      - Result_is_nd (0/1) and Result_mg_L_cens (Float64 with NDs as NA)
      - Inflow_is_nd (0/1) and Inflow_Result_mg_L_cens (Float64 with NDs as NA)
      - Result_lod_mg_L and Inflow_lod_mg_L (RL preferred, else MDL)

    New outputs (to move standardization out of Stan):
      - IRR_z: global z-score of Irrigation
      - inflow_volume_z: global z-score of Inflow_Volume
    """
    out = df.copy()

    # --- STEP 2A: Handle WQ flags ("u","U","NA","NA.IRR") ---
    for col in ["Result_mg_L", "Inflow_Result_mg_L"]:
        if col in out.columns:
            out[col] = _token_series(out[col])

    # --- STEP 2A.1: Preserve ND information for censoring (backwards compatible) ---
    # We DO NOT modify legacy numeric columns here. We only create parallel *_cens columns and flags.
    def _nd_mask_from_tokens(s: pd.Series) -> pd.Series:
        # Match typical lab "U" qualifier. Adjust here if your token grammar changes.
        ss = s.astype("string").str.strip()
        return ss.str.upper().str.startswith("U")

    # Outflow ND flag + LOD
    if "Result_mg_L" in out.columns:
        tok = out["Result_mg_L"].astype("string").str.strip()
        nd_mask = _nd_mask_from_tokens(tok)
        out["Result_is_nd"] = nd_mask.fillna(False).astype(int)
        out["Result_mg_L_cens"] = pd.Series(pd.NA, index=out.index, dtype="Float64")

        # LOD: prefer RL, then MDL (both already in mg/L in the cleaned table)
        rl = pd.to_numeric(out["RL_mg_L"], errors="coerce") if "RL_mg_L" in out.columns else pd.Series(pd.NA, index=out.index)
        mdl = pd.to_numeric(out["MDL_mg_L"], errors="coerce") if "MDL_mg_L" in out.columns else pd.Series(pd.NA, index=out.index)
        out["Result_lod_mg_L"] = rl.combine_first(mdl).astype("Float64")

    # Inflow ND flag + LOD
    if "Inflow_Result_mg_L" in out.columns:
        tok = out["Inflow_Result_mg_L"].astype("string").str.strip()
        nd_mask = _nd_mask_from_tokens(tok)
        out["Inflow_is_nd"] = nd_mask.fillna(False).astype(int)
        out["Inflow_Result_mg_L_cens"] = pd.Series(pd.NA, index=out.index, dtype="Float64")

        rl = pd.to_numeric(out["RL_mg_L"], errors="coerce") if "RL_mg_L" in out.columns else pd.Series(pd.NA, index=out.index)
        mdl = pd.to_numeric(out["MDL_mg_L"], errors="coerce") if "MDL_mg_L" in out.columns else pd.Series(pd.NA, index=out.index)
        out["Inflow_lod_mg_L"] = rl.combine_first(mdl).astype("Float64")

    # Drop no-runoff cases entirely: Result_mg_L == "NA.IRR"
    if "Result_mg_L" in out.columns:
        rm = out["Result_mg_L"].astype("string").str.strip()
        # drop only explicit NA.IRR (case-insensitive, allowing stray spaces)
        drop_mask = rm.str.upper().eq("NA.IRR")
        out = out.loc[~drop_mask].copy()

    # Apply token -> numeric conversions (legacy behavior preserved)
    if "Result_mg_L" in out.columns:
        out["Result_mg_L"] = _token_to_numeric(out["Result_mg_L"])
        # Fill detected values into cens column (NDs remain NA)
        if "Result_mg_L_cens" in out.columns and "Result_is_nd" in out.columns:
            det = out["Result_is_nd"].fillna(0).astype(int).eq(0)
            out.loc[det, "Result_mg_L_cens"] = pd.to_numeric(out.loc[det, "Result_mg_L"], errors="coerce").astype("Float64")

    if "Inflow_Result_mg_L" in out.columns:
        out["Inflow_Result_mg_L"] = _token_to_numeric(out["Inflow_Result_mg_L"])
        if "Inflow_Result_mg_L_cens" in out.columns and "Inflow_is_nd" in out.columns:
            det = out["Inflow_is_nd"].fillna(0).astype(int).eq(0)
            out.loc[det, "Inflow_Result_mg_L_cens"] = pd.to_numeric(out.loc[det, "Inflow_Result_mg_L"], errors="coerce").astype("Float64")

    # --- STEP 2B: Type enforcement (best-effort, only if columns exist) ---
    if "Treatment" in out.columns:
        trt = out["Treatment"].astype("string").str.strip().str.upper()
        # Keep only expected levels; unknown -> NA
        trt = trt.where(trt.isin(["CT", "MT", "ST"]))
        out["Treatment"] = pd.Categorical(trt, categories=["CT", "MT", "ST"], ordered=True)

    for c in ["Rep", "Analyte", "Irrigation", "InflowOutflow", "Crop"]:
        if c in out.columns:
            out[c] = out[c].astype("string")

    for c in ["Date", "PlantDate", "HarvestDate"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date

    if "SeasonYear" in out.columns:
        out["SeasonYear"] = pd.to_numeric(out["SeasonYear"], errors="coerce")

    # Numeric columns (include Irrigation because we compute IRR_z here)
    for c in ["Season_STIR_toDate", "CumAll_STIR_toDate", "Volume", "Inflow_Volume", "Irrigation"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["Has_Inflow", "NoRunoff"]:
        if c in out.columns:
            out[c] = _to_bool_best_effort(out[c])

    # --- STEP 2C: analyte_abbr ---
    if "Analyte" in out.columns:
        out["analyte_abbr"] = out["Analyte"].astype("string").map(ANALYTE_ABBR)
        out["analyte_abbr"] = pd.Categorical(out["analyte_abbr"], categories=ANALYTE_ABBR_LEVELS, ordered=True)

    # --- STEP 2D: Per-analyte standardization ---
    if "Analyte" in out.columns:
        if "Result_mg_L" in out.columns:
            out["cout_z"] = out.groupby("Analyte", dropna=False)["Result_mg_L"].transform(_zscore)
        if "Inflow_Result_mg_L" in out.columns:
            out["cin_z"] = out.groupby("Analyte", dropna=False)["Inflow_Result_mg_L"].transform(_zscore)

    # --- STEP 2E: Global standardization ---
    if "Season_STIR_toDate" in out.columns:
        out["stir_season_z"] = _zscore(out["Season_STIR_toDate"])
    if "CumAll_STIR_toDate" in out.columns:
        out["stir_cumall_z"] = _zscore(out["CumAll_STIR_toDate"])
    if "Volume" in out.columns:
        out["volume_z"] = _zscore(out["Volume"])

    # NEW: global irrigation z-score (replaces Stan transformed-data IRR_z)
    if "Irrigation" in out.columns:
        out["IRR_z"] = _zscore(out["Irrigation"])

    # NEW: global inflow volume z-score (candidate covariate for future model)
    if "Inflow_Volume" in out.columns:
        out["inflow_volume_z"] = _zscore(out["Inflow_Volume"])

    # --- Residue: percent cover to proportion for Beta modeling ---
    # Input expected on 0–100 scale as Residue_PercentCover.
    # We produce residue_prop in (0,1) (clamped away from 0/1 for Beta support).
    if "Residue_PercentCover" in out.columns:
        rp = pd.to_numeric(out["Residue_PercentCover"], errors="coerce") / 100.0
        out["residue_prop"] = _clamp01(rp)
        # 1 if residue was observed in the input table, else 0
        out["residue_obs"] = out["Residue_PercentCover"].notna().astype(int)

    return out
def main() -> None:
    ap = argparse.ArgumentParser(description="Bayes-specific cleaning for Kerbel STIR × WQ merged data.")
    ap.add_argument("--in", dest="src", default="out/pipeline_csvs/wq_with_stir_by_season.csv",
                    help="Input merged WQ×STIR CSV.")
    ap.add_argument("--out", dest="dst", default="out/wq_cleaned.csv",
                    help="Output cleaned CSV for Bayes modeling.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    # keep_default_na=False avoids pandas treating literal 'NA' as missing;
    # we handle tokens explicitly.
    df = pd.read_csv(src, keep_default_na=False)

    cleaned = clean_wq_stir(df)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(dst, index=False)

    print(f"[OK] Wrote Bayes-cleaned file → {dst}")
    if args.debug:
        print(f"[INFO] Rows in:  {len(df)}")
        print(f"[INFO] Rows out: {len(cleaned)}")
        for c in ["Has_Inflow", "NoRunoff"]:
            if c in cleaned.columns:
                vc = cleaned[c].value_counts(dropna=False)
                print(f"[INFO] {c} value_counts (incl NA):")
                print(vc.to_string())
        for c in ["cout_z", "stir_season_z", "volume_z"]:
            if c in cleaned.columns:
                x = cleaned[c].astype(float)
                print(f"[INFO] {c}: mean≈{np.nanmean(x):.4f} sd≈{np.nanstd(x, ddof=1):.4f}")


if __name__ == "__main__":
    main()
