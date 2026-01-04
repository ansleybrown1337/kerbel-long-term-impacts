#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_residue.py

Merge residue measurements (or dummy residue) onto the merged WQ×STIR dataset.

Design goals
------------
- Keep this step simple and transparent.
- Aggregate residue across within-rep spatial subsamples (e.g., Location = N/M/S)
  so that the merge happens at the experimental-unit level.
- Merge keys are chosen to match the WQ/STIR dataset schema:
    Year × Treatment × Rep

Expected residue input schema (recommended)
-------------------------------------------
Year, Treatment, Rep, Location, Residue_DryMass_kg_m2, Residue_PercentCover

Behavior
--------
1) Reads residue CSV
2) Aggregates (mean) across Location (and any replicate subsamples per Location)
3) Left-joins onto WQ×STIR merged dataset
4) Writes the merged dataset

Outputs
-------
- <out> (merged WQ×STIR with residue columns)
- <out_dir>/residue_agg_by_year_trt_rep.csv (QC / transparency)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQ_RESIDUE_COLS = {
    "Year",
    "Treatment",
    "Rep",
    "Residue_DryMass_kg_m2",
    "Residue_PercentCover",
}


def _norm_trt(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NA": pd.NA, "NAN": pd.NA})
    )


def _read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na=False to preserve literal "NA" if present; we coerce explicitly.
    return pd.read_csv(path, keep_default_na=False)


def aggregate_residue(res: pd.DataFrame) -> pd.DataFrame:
    missing = REQ_RESIDUE_COLS.difference(res.columns)
    if missing:
        raise ValueError(f"Residue file missing required columns: {sorted(missing)}")

    out = res.copy()

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Treatment"] = _norm_trt(out["Treatment"])
    out["Rep"] = pd.to_numeric(out["Rep"], errors="coerce").astype("Int64")

    out["Residue_DryMass_kg_m2"] = pd.to_numeric(out["Residue_DryMass_kg_m2"], errors="coerce")
    out["Residue_PercentCover"] = pd.to_numeric(out["Residue_PercentCover"], errors="coerce")

    # Drop rows with missing merge keys (cannot be merged deterministically)
    out = out.dropna(subset=["Year", "Treatment", "Rep"]).copy()

    # Aggregate over Location (and any additional within-location subsamples)
    agg = (
        out.groupby(["Year", "Treatment", "Rep"], dropna=False)
        .agg(
            Residue_PercentCover=("Residue_PercentCover", "mean"),
            Residue_DryMass_kg_m2=("Residue_DryMass_kg_m2", "mean"),
            Residue_n=("Residue_PercentCover", lambda x: int(np.sum(pd.notna(x)))),
        )
        .reset_index()
    )

    return agg


def merge_residue(wq: pd.DataFrame, res_agg: pd.DataFrame) -> pd.DataFrame:
    out = wq.copy()

    # normalize keys on WQ side (non-destructive)
    if "Year" not in out.columns or "Treatment" not in out.columns or "Rep" not in out.columns:
        raise ValueError("WQ×STIR dataset must contain Year, Treatment, and Rep columns for residue merge.")

    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Treatment"] = _norm_trt(out["Treatment"])

    # Rep sometimes arrives as str; coerce to Int64 safely
    out["Rep"] = pd.to_numeric(out["Rep"], errors="coerce").astype("Int64")

    merged = out.merge(res_agg, how="left", on=["Year", "Treatment", "Rep"])

    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge residue cover/drymass onto merged WQ×STIR dataset.")
    ap.add_argument("--wq", required=True, help="Input merged WQ×STIR CSV (e.g., out/pipeline_csvs/wq_with_stir_by_season.csv).")
    ap.add_argument("--residue", required=True, help="Residue CSV (observed or dummy).")
    ap.add_argument("--out", required=True, help="Output CSV path for WQ×STIR merged with residue.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    wq_path = Path(args.wq)
    res_path = Path(args.residue)
    out_path = Path(args.out)

    if not wq_path.exists():
        raise FileNotFoundError(f"WQ input not found: {wq_path}")
    if not res_path.exists():
        raise FileNotFoundError(f"Residue input not found: {res_path}")

    wq = _read_csv(wq_path)
    res = _read_csv(res_path)

    res_agg = aggregate_residue(res)
    merged = merge_residue(wq, res_agg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # QC output alongside the merged file
    qc_path = out_path.parent / "residue_agg_by_year_trt_rep.csv"
    res_agg.to_csv(qc_path, index=False)

    print(f"[OK] Wrote WQ×STIR with residue → {out_path}")
    print(f"[OK] Wrote residue aggregation QC → {qc_path}")

    if args.debug:
        # quick diagnostics
        n_wq = len(wq)
        n_merged = len(merged)
        n_nonmiss = int(merged["Residue_PercentCover"].notna().sum()) if "Residue_PercentCover" in merged.columns else 0
        print(f"[INFO] WQ rows: {n_wq} | merged rows: {n_merged}")
        print(f"[INFO] Rows with non-missing Residue_PercentCover: {n_nonmiss}")
        print("[INFO] Residue_PercentCover summary (merged):")
        if "Residue_PercentCover" in merged.columns:
            print(merged["Residue_PercentCover"].describe(include="all").to_string())


if __name__ == "__main__":
    main()
