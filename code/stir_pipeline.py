#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stir_pipeline.py
----------------
Streamlined STIR calculation pipeline using MOSES/NRCS parameters and the
SoilManageR STIR formulation.

Key behavior
============
- Uses a single unit-aware mapper (tillage_mapper_input.csv).
- Reads operations log (wide CT/ST/MT or long), normalizes to long.
- Direct join by verbatim operation name (no heuristics).
- ALWAYS computes STIR with the SoilManageR/RUSLE2 equation:
    STIR = (0.5 * (Speed[km/h]/1.609)) * (3.25 * TTM) * (Depth[cm]/2.54) * (Surf[%]/100)
- Outputs exactly TWO files:
    1) stir_events_long.csv : preserves all mapper columns + cumulative columns
    2) stir_daily_system_wide.csv : daily aggregates (sum/mean) + cumulative STIR columns
- Optional crop file allows computing window-based cumulative STIR in long output
  (added as STIR_Cum_CropWindow with a Window_ID).

Author: generated for the PhD Ch2 - tillage impacts on WQ project
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np


# -----------------------------
# Utility functions
# -----------------------------

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce listed columns to numeric, preserving NaN on failure."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_mapper(mapper_path: Path) -> pd.DataFrame:
    """Read the tillage mapper and validate required columns."""
    req_cols = [
        "Operation (verbatim)",
        "MOSES Operation",
        "Speed [km/h]",
        "Surf_Disturbance [%]",
        "Depth [cm]",
        "TILLAGE_TYPE_Modifier [0-1]",
    ]
    opt_cols = [
        "Speed_MIN [km/h]","Speed_MAX [km/h]",
        "Depth_MIN [cm]","Depth_MAX [cm]",
        "TILLAGE_TYPE","STIR","Diesel_use [l/ha]","Burial_Coefficient [0-1]",
        "Source","Description"
    ]

    mapper = pd.read_csv(mapper_path)
    missing = [c for c in req_cols if c not in mapper.columns]
    if missing:
        raise ValueError(f"Mapper is missing required columns: {missing}")

    # numeric coercion
    num_cols = [
        "Speed [km/h]","Speed_MIN [km/h]","Speed_MAX [km/h]",
        "Surf_Disturbance [%]",
        "Depth [cm]","Depth_MIN [cm]","Depth_MAX [cm]",
        "TILLAGE_TYPE_Modifier [0-1]","STIR","Diesel_use [l/ha]","Burial_Coefficient [0-1]"
    ]
    mapper = _coerce_numeric(mapper, num_cols)

    mapper["Operation (verbatim)"] = mapper["Operation (verbatim)"].astype(str).str.strip()

    if mapper["Operation (verbatim)"].duplicated().any():
        dupes = mapper[mapper["Operation (verbatim)"].duplicated(keep=False)] \
                    .sort_values("Operation (verbatim)")
        print("[WARN] Duplicate keys in mapper; keeping first occurrence:\n",
              dupes[["Operation (verbatim)","MOSES Operation"]].to_string(index=False),
              file=sys.stderr)
        mapper = mapper.drop_duplicates(subset=["Operation (verbatim)"], keep="first")

    return mapper


def _read_records(records_path: Path) -> pd.DataFrame:
    """Read operations log; normalize to long format with columns:
       Date, System (CT/ST/MT or NA), Operation (verbatim), plus any metadata.
    """
    rec = pd.read_csv(records_path)
    if "Date" not in rec.columns:
        raise ValueError("Records CSV must contain a 'Date' column.")
    rec["Date"] = pd.to_datetime(rec["Date"], errors="coerce")
    if rec["Date"].isna().any():
        n_bad = rec["Date"].isna().sum()
        print(f"[WARN] {n_bad} rows with unparseable Date; they will be dropped.", file=sys.stderr)
        rec = rec.dropna(subset=["Date"])

    op_cols = [c for c in rec.columns if c.lower() in {"operation","ct operation","st operation","mt operation"}]
    meta_cols = [c for c in rec.columns if c not in op_cols]

    if "Operation" in rec.columns:
        long = rec.rename(columns={"Operation":"Operation (verbatim)"})
        long["System"] = "NA"
    else:
        rename_map = {}
        for c in rec.columns:
            lc = c.lower()
            if lc == "ct operation":
                rename_map[c] = "CT"
            elif lc == "st operation":
                rename_map[c] = "ST"
            elif lc == "mt operation":
                rename_map[c] = "MT"
        wide = rec.rename(columns=rename_map)
        for k in ["CT","ST","MT"]:
            if k not in wide.columns: wide[k] = np.nan
        keep = meta_cols + ["CT","ST","MT"]
        wide = wide[keep]
        long = wide.melt(id_vars=meta_cols, value_vars=["CT","ST","MT"],
                         var_name="System", value_name="Operation (verbatim)")

    long["Operation (verbatim)"] = long["Operation (verbatim)"].astype(str).str.strip()
    long = long[long["Operation (verbatim)"].astype(bool)].copy()

    return long


def compute_stir(speed_kmh: float, depth_cm: float, surf_pct: float, ttm: float) -> float:
    """Compute STIR using SoilManageR/RUSLE2 equation with unit conversions."""
    if pd.isna(speed_kmh) or pd.isna(depth_cm) or pd.isna(surf_pct) or pd.isna(ttm):
        return np.nan
    mph = speed_kmh / 1.609
    inches = depth_cm / 2.54
    area_frac = surf_pct / 100.0
    return (0.5 * mph) * (3.25 * ttm) * inches * area_frac


def compute_stir_vectorized(speed_kmh: pd.Series, depth_cm: pd.Series, surf_pct: pd.Series, ttm: pd.Series) -> pd.Series:
    """Vectorized STIR computation across Series inputs."""
    mph = speed_kmh.astype(float) / 1.609
    inches = depth_cm.astype(float) / 2.54
    area = surf_pct.astype(float) / 100.0
    ttm = ttm.astype(float)
    return (0.5 * mph) * (3.25 * ttm) * inches * area


def _compute_windowed_cumulative(events: pd.DataFrame, crop_path: Path) -> pd.DataFrame:
    """Assign each event to (prev harvest, this harvest] windows and cumsum STIR within windows.

    This is a generic implementation and should be tailored to your crop file schema if needed.
    """
    crops = pd.read_csv(crop_path)
    if "Date" not in crops.columns:
        raise ValueError("Crop records must have a 'Date' column.")
    crops["Date"] = pd.to_datetime(crops["Date"], errors="coerce")
    crops = crops.dropna(subset=["Date"]).sort_values("Date")

    # Identify harvest boundary dates (search any string column for 'harvest' by default)
    harvest_mask = np.column_stack([
        crops[c].astype(str).str.contains("harvest", case=False, na=False)
        for c in crops.columns if crops[c].dtype == object
    ]).any(axis=1)
    harvest_dates = crops.loc[harvest_mask, "Date"].sort_values().unique()

    ev = events.copy().sort_values("Date")
    if len(harvest_dates) == 0:
        ev["Window_ID"] = pd.NA
        ev["STIR_Cum_Window"] = ev["STIR_Event"].fillna(0).cumsum()
        return ev

    bins = [-np.inf] + list(pd.to_datetime(harvest_dates, utc=False).astype("int64"))
    ev_int = ev["Date"].astype("int64")
    ev["Window_ID"] = pd.cut(ev_int, bins=bins, labels=range(1, len(bins)), include_lowest=True).astype("Int64")
    ev["STIR_Cum_Window"] = ev.groupby("Window_ID")["STIR_Event"].cumsum(skipna=True)
    return ev


def _add_cumulative_columns_long(events: pd.DataFrame, crop_path: Path | None = None) -> pd.DataFrame:
    """Add cumulative STIR columns to the long table:
       - STIR_Cum_CalendarYear: cumsum within calendar year by System
       - STIR_Cum_AllYears    : grand running cumsum by System
       - If crop_path provided: STIR_Cum_CropWindow and Window_ID
    """
    ev = events.copy()
    ev = ev.sort_values(["System","Date","Operation (verbatim)"], kind="stable")
    ev["Year"] = ev["Date"].dt.year

    ev["STIR_Cum_CalendarYear"] = ev.groupby(["System","Year"], dropna=False)["STIR_Event"].cumsum()
    ev["STIR_Cum_AllYears"] = ev.groupby(["System"], dropna=False)["STIR_Event"].cumsum()

    if crop_path and Path(crop_path).exists():
        try:
            windowed = _compute_windowed_cumulative(ev[["Date","STIR_Event"]].assign(System=ev["System"]), crop_path)
            ev["Window_ID"] = windowed["Window_ID"]
            ev["STIR_Cum_CropWindow"] = windowed["STIR_Cum_Window"]
        except Exception as e:
            print(f"[WARN] Skipping crop-window cumulative: {e}", file=sys.stderr)

    return ev


def _aggregate_daily_wide(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily x system with sum/mean metrics and cumulative STIR columns.

    Metrics by day x system:
      - Sum: STIR_Event, Diesel_use [l/ha] (if available)
      - Mean: Speed [km/h], Depth [cm], Surf_Disturbance [%], TILLAGE_TYPE_Modifier [0-1]

    Output columns (wide):
      Date,
      <System>_STIR, <System>_Diesel_l_ha,
      <System>_Speed_kmh_mean, <System>_Depth_cm_mean, <System>_SurfDist_pct_mean, <System>_TTM_mean,
      <System>_STIR_Cum_Year, <System>_STIR_Cum_All
    """
    ev = events.copy()

    base_cols = ["Date","System","STIR_Event"]
    if "Diesel_use [l/ha]" in ev.columns:
        base_cols.append("Diesel_use [l/ha]")

    # Means (contextual)
    mean_cols = {}
    for src, out in [
        ("Speed [km/h]","Speed_kmh_mean"),
        ("Depth [cm]","Depth_cm_mean"),
        ("Surf_Disturbance [%]","SurfDist_pct_mean"),
        ("TILLAGE_TYPE_Modifier [0-1]","TTM_mean"),
    ]:
        if src in ev.columns:
            mean_cols[src] = out

    agg_parts = []
    sum_df = ev[base_cols].groupby(["Date","System"], dropna=False).sum(numeric_only=True).reset_index()
    agg_parts.append(sum_df)

    if mean_cols:
        means = ev[["Date","System"] + list(mean_cols.keys())].groupby(["Date","System"], dropna=False).mean(numeric_only=True).reset_index()
        means = means.rename(columns=mean_cols)
        agg_parts.append(means)

    from functools import reduce
    agg = reduce(lambda l,r: pd.merge(l,r,on=["Date","System"], how="outer"), agg_parts)

    pivots = []
    for col in agg.columns:
        if col in ("Date","System"): continue
        wide_piece = agg.pivot(index="Date", columns="System", values=col).rename(columns=lambda s: f"{s}_{col}")
        pivots.append(wide_piece)

    wide = pd.concat(pivots, axis=1).reset_index().sort_values("Date")

    wide["Year"] = wide["Date"].dt.year
    for sys_name in ["CT","ST","MT","NA"]:
        stir_col = f"{sys_name}_STIR_Event"
        if stir_col in wide.columns:
            wide[f"{sys_name}_STIR_Cum_Year"] = wide.groupby("Year")[stir_col].cumsum()
            wide[f"{sys_name}_STIR_Cum_All"] = wide[stir_col].cumsum()

    wide = wide.rename(columns={
        "CT_STIR_Event":"CT_STIR",
        "ST_STIR_Event":"ST_STIR",
        "MT_STIR_Event":"MT_STIR",
        "NA_STIR_Event":"NA_STIR",
        "CT_Diesel_use [l/ha]":"CT_Diesel_l_ha",
        "ST_Diesel_use [l/ha]":"ST_Diesel_l_ha",
        "MT_Diesel_use [l/ha]":"MT_Diesel_l_ha",
        "NA_Diesel_use [l/ha]":"NA_Diesel_l_ha",
    })

    return wide


def run_pipeline(records_path: Path, mapper_path: Path, outdir: Path, crop_path: Path | None = None) -> None:
    """End-to-end run producing exactly two outputs (long & wide) with cumulatives."""
    outdir.mkdir(parents=True, exist_ok=True)

    mapper = _read_mapper(mapper_path)
    ops = _read_records(records_path)

    events = ops.merge(mapper, on="Operation (verbatim)", how="left", indicator=True)
    events["Map_Status"] = np.where(events["_merge"] == "both", "OK", "CHECK")
    events.drop(columns=["_merge"], inplace=True)

    events["STIR_Event"] = compute_stir_vectorized(
        speed_kmh=events["Speed [km/h]"],
        depth_cm=events["Depth [cm]"],
        surf_pct=events["Surf_Disturbance [%]"],
        ttm=events["TILLAGE_TYPE_Modifier [0-1]"],
    )

    # Mirror to a plain 'STIR' column for user convenience
    events["STIR"] = events["STIR_Event"]

    # Diagnostics: flag rows missing any required inputs for STIR
    req_cols = ["Speed [km/h]","Depth [cm]","Surf_Disturbance [%]","TILLAGE_TYPE_Modifier [0-1]"]
    def _missing_fields(row):
        missing = [c for c in req_cols if pd.isna(row.get(c))]
        return ",".join(missing) if missing else ""
    events["STIR_Missing_Fields"] = events.apply(_missing_fields, axis=1)
    events["STIR_Inputs_Complete"] = events["STIR_Missing_Fields"].eq("")

    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
    events_long = _add_cumulative_columns_long(events, crop_path=crop_path)
    events_long = events_long.sort_values(["Date","System","Operation (verbatim)"], kind="stable")
    events_long.to_csv(outdir / "stir_events_long.csv", index=False)

    wide = _aggregate_daily_wide(events_long)
    wide.to_csv(outdir / "stir_daily_system_wide.csv", index=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute STIR from operations using a MOSES/NRCS mapper (two-output mode).")
    p.add_argument("--records", required=True, type=Path, help="Path to tillage records CSV (wide CT/ST/MT or long).")
    p.add_argument("--mapper", required=True, type=Path, help="Path to tillage_mapper_input.csv (unit-aware).")
    p.add_argument("--outdir", required=True, type=Path, help="Directory for outputs.")
    p.add_argument("--crop", required=False, type=Path, help="Optional path to crop records CSV for crop-window cumulatives in LONG file.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_pipeline(records_path=args.records, mapper_path=args.mapper, outdir=args.outdir, crop_path=args.crop)


if __name__ == "__main__":
    main()
