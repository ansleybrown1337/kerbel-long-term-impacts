#!/usr/bin/env python3
"""
stircalcs.py

Step 1: Prepare mapper artifacts.
- Loads the tillage mapper + records CSVs from ./data
- Validates & normalizes the mapper
- Builds a mapping dict keyed by 'My Operation'
- Scans records (CT/ST/MT/My Operation columns) for unmapped strings
- Writes artifacts to ./data/derived

Step 2: Apply mapper to records.
- Maps assumptions to each of CT/ST/MT operation columns
- Writes wide and long mapped datasets to ./data/derived

Step 3: Compute per-row disturbance (STIR-style) metrics.
- Uses Mixing Depth, Mixing Efficiency, and Area Fraction
- Writes long and wide outputs with Disturbance_Fraction

Usage:
  python code/stircalcs.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

MAPPER_CSV = DATA_DIR / "tillage_mapper_table.csv"
RECORDS_CSV = DATA_DIR / "tillage_records.csv"
CROP_CSV = DATA_DIR / "crop records.csv"

# Reference depth for fractional disturbance calc
REF_DEPTH_MM = 200.0  # change if you want a different reference layer

REQUIRED_MAPPER_COLS = [
    "My Operation",
    "SWAT/RUSLE2 Implement",
    "Tillage Code",
    "Mixing Depth (mm)",
    "Mixing Efficiency",
    "Area Fraction",
    "Type / Note",
]

RECORD_OP_COLS = ["CT Operation", "ST Operation", "MT Operation"]  # columns to map

# ---------- Helpers ----------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def load_mapper(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_cols(df)

    # validate required columns
    missing = [c for c in REQUIRED_MAPPER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Mapper missing required columns: {missing}")

    # trim whitespace in key string fields
    for c in ["My Operation", "SWAT/RUSLE2 Implement", "Tillage Code", "Type / Note"]:
        df[c] = df[c].astype(str).str.strip()

    # coerce numerics
    df["Mixing Depth (mm)"] = pd.to_numeric(df["Mixing Depth (mm)"], errors="coerce")
    df["Mixing Efficiency"] = pd.to_numeric(df["Mixing Efficiency"], errors="coerce")
    df["Area Fraction"] = pd.to_numeric(df["Area Fraction"], errors="coerce")

    # ensure key column is string
    df["My Operation"] = df["My Operation"].astype(str)

    # optional area fraction column
    if "Area Fraction" in df.columns:
        df["Area Fraction"] = pd.to_numeric(df["Area Fraction"], errors="coerce")

    # check for duplicate keys
    dups = df["My Operation"][df["My Operation"].duplicated(keep=False)]
    if not dups.empty:
        dup_list = sorted(
            df.loc[df["My Operation"].duplicated(keep=False), "My Operation"].unique().tolist()
        )
        raise ValueError(f"Duplicate 'My Operation' keys in mapper: {dup_list}")

    return df


def load_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_cols(df)
    return df


def build_mapping_dict(mapper_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Keyed by 'My Operation' -> dict with:
    'SWAT/RUSLE2 Implement', 'Tillage Code', 'Mixing Depth (mm)', 'Mixing Efficiency', 'Type / Note', (opt) 'Area Fraction'
    """
    cols = [
        "SWAT/RUSLE2 Implement",
        "Tillage Code",
        "Mixing Depth (mm)",
        "Mixing Efficiency",
        "Type / Note",
    ]
    if "Area Fraction" in mapper_df.columns:
        cols.append("Area Fraction")

    idx = mapper_df["My Operation"].astype(str)
    subset = mapper_df.set_index(idx)[cols]
    out: Dict[str, Dict[str, Any]] = subset.to_dict(orient="index")  # type: ignore[assignment]
    return out


def collect_unique_ops(records_df: pd.DataFrame) -> List[str]:
    candidates = RECORD_OP_COLS + (["My Operation"] if "My Operation" in records_df.columns else [])
    cols = [c for c in candidates if c in records_df.columns]
    if not cols:
        return []

    uniques: set[str] = set()
    for c in cols:
        s = records_df[c]
        s = s.dropna().astype(str).str.strip()
        for val in s:
            if val:
                uniques.add(val)

    return sorted(uniques)


def map_one_column(
    df: pd.DataFrame,
    op_col: str,
    mapping: Dict[str, Dict[str, Any]],
    prefix: str,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    mapped_impl, mapped_code, mapped_depth, mapped_eff, mapped_note, mapped_af = [], [], [], [], [], []
    still_unmapped: set[str] = set()

    for val in df.get(op_col, pd.Series([None]*len(df))):
        if pd.isna(val):
            mapped_impl.append(None); mapped_code.append(None); mapped_depth.append(None)
            mapped_eff.append(None); mapped_note.append(None); mapped_af.append(None)
            continue

        key = str(val).strip()
        info = mapping.get(key)

        if info is None:
            still_unmapped.add(key)
            mapped_impl.append(None); mapped_code.append(None); mapped_depth.append(None)
            mapped_eff.append(None); mapped_note.append(None); mapped_af.append(None)
        else:
            mapped_impl.append(info.get("SWAT/RUSLE2 Implement"))
            mapped_code.append(info.get("Tillage Code"))
            mapped_depth.append(info.get("Mixing Depth (mm)"))
            mapped_eff.append(info.get("Mixing Efficiency"))
            mapped_note.append(info.get("Type / Note"))
            mapped_af.append(info.get("Area Fraction"))

    df[f"{prefix}SWAT Implement"] = mapped_impl
    df[f"{prefix}Tillage Code"] = mapped_code
    df[f"{prefix}Mixing Depth (mm)"] = mapped_depth
    df[f"{prefix}Mixing Efficiency"] = mapped_eff
    df[f"{prefix}Area Fraction"] = mapped_af
    df[f"{prefix}Type / Note"] = mapped_note

    return df, sorted(still_unmapped)


def to_long_format(mapped_df: pd.DataFrame) -> pd.DataFrame:
    df = mapped_df.copy()

    per_prefix_cols = {
        "CT": [f"CT {x}" for x in ["SWAT Implement", "Tillage Code", "Mixing Depth (mm)", "Mixing Efficiency", "Area Fraction", "Type / Note"]],
        "ST": [f"ST {x}" for x in ["SWAT Implement", "Tillage Code", "Mixing Depth (mm)", "Mixing Efficiency", "Area Fraction", "Type / Note"]],
        "MT": [f"MT {x}" for x in ["SWAT Implement", "Tillage Code", "Mixing Depth (mm)", "Mixing Efficiency", "Area Fraction", "Type / Note"]],
    }
    op_cols = ["CT Operation", "ST Operation", "MT Operation"]
    mapped_cols = per_prefix_cols["CT"] + per_prefix_cols["ST"] + per_prefix_cols["MT"]
    id_vars = [c for c in df.columns if c not in (op_cols + mapped_cols)]

    long_frames = []
    for prefix, cols in per_prefix_cols.items():
        part = df[id_vars + [f"{prefix} Operation"] + cols].copy()
        part = part.rename(columns={
            f"{prefix} Operation": "Operation",
            f"{prefix} SWAT Implement": "SWAT Implement",
            f"{prefix} Tillage Code": "Tillage Code",
            f"{prefix} Mixing Depth (mm)": "Mixing Depth (mm)",
            f"{prefix} Mixing Efficiency": "Mixing Efficiency",
            f"{prefix} Area Fraction": "Area Fraction",
            f"{prefix} Type / Note": "Type / Note",
        })
        part["Op System"] = prefix  # CT/ST/MT
        long_frames.append(part)

    out = pd.concat(long_frames, axis=0, ignore_index=True)
    # Drop rows with no operation value
    out = out[~out["Operation"].isna() & (out["Operation"].astype(str).str.strip() != "")]
    return out


def compute_disturbance_fraction(df_long: pd.DataFrame, ref_depth_mm: float = REF_DEPTH_MM) -> pd.DataFrame:
    df = df_long.copy()

    # Safe numerics
    for c in ["Mixing Depth (mm)", "Mixing Efficiency", "Area Fraction"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Identify rows that truly need AF (depth>0 and eff>0)
    needs_af = (df["Mixing Depth (mm)"].fillna(0) > 0) & (df["Mixing Efficiency"].fillna(0) > 0)
    missing_af = needs_af & df["Area Fraction"].isna()

    # Add/append warnings
    if "Compute_Warning" not in df.columns:
        df["Compute_Warning"] = np.nan

    if missing_af.any():
        # Conservative fallback: assume full-width if user forgot AF for a mixing pass
        df.loc[missing_af, "Area Fraction"] = 1.0
        msg = "Missing Area Fraction in mapper; assumed 1.0 (full-width). Edit 'Area Fraction' in tillage_mapper_table.csv."
        df.loc[missing_af, "Compute_Warning"] = df.loc[missing_af, "Compute_Warning"].fillna("").astype(str)
        df.loc[missing_af, "Compute_Warning"] = df.loc[missing_af, "Compute_Warning"].mask(
            df.loc[missing_af, "Compute_Warning"] == "",
            msg
        ).fillna(df.loc[missing_af, "Compute_Warning"] + " | " + msg)

    # For non-tillage ops (depth==0 or eff==0), AF doesn't affect result; NaN is fine
    df["Disturbance_Fraction"] = (
        df["Mixing Efficiency"].fillna(0.0) *
        (df["Mixing Depth (mm)"].fillna(0.0) / float(ref_depth_mm)) *
        df["Area Fraction"].fillna(0.0)   # if AF missing and it's a non-mixing op -> 0
    )

    df["Disturbance_Fraction"] = df["Disturbance_Fraction"].clip(lower=0.0)
    return df


def _load_crop_windows(path: Path) -> pd.DataFrame:
    """
    Expect a CSV like:
      plant date | harvest date | crop
    or variants:
      Planting_Date | Harvest_Date | Crop

    Returns a DataFrame with columns:
      ['Crop', 'Planting_Date', 'Harvest_Date', 'Window_ID', 'Window_Start', 'Window_End']
    where Window_Start is the previous harvest (global, sorted by Harvest_Date).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    # Resolve flexible column names
    plant_col = lower.get("plant date", lower.get("planting_date", None))
    harvest_col = lower.get("harvest date", lower.get("harvest_date", None))
    crop_col = lower.get("crop", None)

    if not plant_col or not harvest_col:
        raise ValueError("crop records.csv must have 'plant date' and 'harvest date' (or Planting_Date/Harvest_Date).")

    if not crop_col:
        crop_col = "Crop"
        if crop_col not in df.columns:
            # make a placeholder crop column if absent
            df[crop_col] = "Unknown"

    out = df[[crop_col, plant_col, harvest_col]].copy()
    out = out.rename(columns={crop_col: "Crop", plant_col: "Planting_Date", harvest_col: "Harvest_Date"})

    out["Planting_Date"] = pd.to_datetime(out["Planting_Date"], errors="coerce")
    out["Harvest_Date"]  = pd.to_datetime(out["Harvest_Date"], errors="coerce")

    # Sort by harvest (global), build window starts as previous harvest
    out = out.sort_values(["Harvest_Date", "Planting_Date"]).reset_index(drop=True)
    out["Window_ID"] = np.arange(1, len(out) + 1)
    out["Window_Start"] = out["Harvest_Date"].shift(1)  # previous harvest
    out["Window_End"]   = out["Harvest_Date"]

    # For the very first row, set a wide-open start so events before the first recorded harvest are included if needed
    if not out["Window_Start"].isna().all():
        first_idx = out.index.min()
        out.loc[first_idx, "Window_Start"] = pd.Timestamp.min

    return out


def _assign_windows_by_harvest(dist_long: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    events = dist_long.copy()
    events = events.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 🔧 Drop any pre-existing window fields to avoid duplicate column names
    for col in ["Window_ID", "Window_Start", "Window_End", "Crop"]:
        if col in events.columns:
            events = events.drop(columns=[col])

    wins = (
        windows[["Window_ID", "Window_Start", "Window_End", "Crop"]]
        .dropna(subset=["Window_End"])
        .sort_values("Window_End")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        left=events,
        right=wins,
        left_on="Date",
        right_on="Window_End",
        direction="forward",
        allow_exact_matches=True,
        suffixes=("", "_win"),
    )

    # Rename the right-side window columns back to base names
    rename_map = {
        "Window_ID_win": "Window_ID",
        "Window_Start_win": "Window_Start",
        "Window_End_win": "Window_End",
        "Crop_win": "Crop",
    }
    keep = {k: v for k, v in rename_map.items() if k in merged.columns}
    if keep:
        merged = merged.rename(columns=keep)

    # Keep only events within (prev harvest, this harvest]
    mask = (merged["Date"] > merged["Window_Start"]) & (merged["Date"] <= merged["Window_End"])
    merged = merged[mask].copy()
    return merged


def _compute_cumprior_per_system(assigned: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Window_ID, Op System), compute cumulative STIR INCLUDING the current row.
    (Default: Disturbance_Cum)
    """
    df = assigned.copy()  # <-- make df first

    # sanity check
    required_cols = {"Window_ID", "Op System", "Date", "Operation", "Disturbance_Fraction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"_compute_cumprior_per_system: missing columns: {sorted(missing)}; "
            f"have={sorted(df.columns)}"
        )

    df["Disturbance_Fraction"] = pd.to_numeric(df["Disturbance_Fraction"], errors="coerce").fillna(0.0)

    # stable order
    df = df.sort_values(["Window_ID", "Op System", "Date", "Operation"]).reset_index(drop=True)

    # Cumulative INCLUDING current event
    df["Disturbance_Cum"] = df.groupby(["Window_ID", "Op System"])["Disturbance_Fraction"].cumsum()
    return df


def _long_to_wide_from_values(long_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    Build the wide table directly from the long dataset so we never duplicate calculations.
    Uses max() per (Date, Op System) to handle multiple ops per system on the same date.
    """
    df = long_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Op System"]).sort_values(["Date", "Op System"])

    # collapse possibly multiple same-day events per system
    grouped = (df.groupby(["Date", "Op System"], as_index=False)[value_cols].max())

    # pivot to wide
    wide = grouped.pivot(index="Date", columns="Op System", values=value_cols)

    # flatten MultiIndex columns: (value_col, system) -> "SYS value_col"
    wide.columns = [f"{sys} {val}" for val, sys in wide.columns]
    wide = wide.reset_index().sort_values("Date")
    return wide


# ---------- Main ----------
def main() -> None:
    print("== stircalcs.py : prepare mapper artifacts, apply to records, compute disturbance ==")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Reading mapper:  {MAPPER_CSV}")
    print(f"Reading records: {RECORDS_CSV}")

    # STEP 1: prepare artifacts
    mapper_df = load_mapper(MAPPER_CSV)
    records_df = load_records(RECORDS_CSV)

    mapping_dict = build_mapping_dict(mapper_df)
    unique_ops = collect_unique_ops(records_df)
    unmapped_ops_initial = sorted([op for op in unique_ops if op not in mapping_dict])

    # Write prep artifacts
    mapper_norm_out = DERIVED_DIR / "mapper_normalized.csv"
    mapper_df.to_csv(mapper_norm_out, index=False)

    mapper_json_out = DERIVED_DIR / "mapper_dict.json"
    with mapper_json_out.open("w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)

    unmapped_out = DERIVED_DIR / "unmapped_ops.csv"
    pd.DataFrame({"unmapped_operation": unmapped_ops_initial}).to_csv(unmapped_out, index=False)

    # STEP 2: apply mapper to CT/ST/MT columns
    mapped_df = records_df.copy()
    all_still_unmapped: set[str] = set()

    for prefix, col in zip(["CT", "ST", "MT"], RECORD_OP_COLS):
        if col in mapped_df.columns:
            mapped_df, still_unmapped = map_one_column(mapped_df, col, mapping_dict, prefix=prefix + " ")
            all_still_unmapped.update(still_unmapped)

    # Make sure we have "{prefix} Operation" columns
    for prefix, col in zip(["CT", "ST", "MT"], RECORD_OP_COLS):
        if f"{prefix} Operation" not in mapped_df.columns and col in mapped_df.columns:
            mapped_df[f"{prefix} Operation"] = mapped_df[col]

    # Write wide mapped dataset
    mapped_wide_out = DERIVED_DIR / "tillage_records_mapped.csv"
    mapped_df.to_csv(mapped_wide_out, index=False)

    # Build long version (one row per record × operation)
    mapped_long_df = to_long_format(mapped_df)
    mapped_long_out = DERIVED_DIR / "tillage_records_mapped_long.csv"
    mapped_long_df.to_csv(mapped_long_out, index=False)

    # Ensure Date is datetime in both wide and long before any window logic
    if "Date" in mapped_df.columns:
        mapped_df["Date"] = pd.to_datetime(mapped_df["Date"], errors="coerce")
    if "Date" in mapped_long_df.columns:
        mapped_long_df["Date"] = pd.to_datetime(mapped_long_df["Date"], errors="coerce")

    # STEP 3: compute disturbance per row
    disturbed_long_df = compute_disturbance_fraction(mapped_long_df, ref_depth_mm=REF_DEPTH_MM)
    disturbed_long_out = DERIVED_DIR / "tillage_records_disturbance_long.csv"
    disturbed_long_df.to_csv(disturbed_long_out, index=False)
    # Give each long row a stable id for clean merges
    disturbed_long_df = disturbed_long_df.copy()
    disturbed_long_df["RowID"] = np.arange(len(disturbed_long_df))

    # STEP 4: add Crop to LONG via harvest windows, then compute cumulative STIR

    # Load crop windows
    crop_windows = _load_crop_windows(CROP_CSV)

    # Assign each event to (prev harvest, this harvest] window
    assigned = _assign_windows_by_harvest(disturbed_long_df, crop_windows)

    # Merge Crop (and Window_ID) back onto the full long table
    disturbed_long_df = disturbed_long_df.merge(
        assigned[["RowID", "Window_ID", "Crop"]],
        on="RowID",
        how="left"
    )

    # assign each event to (prev harvest, this harvest] window
    assigned = _assign_windows_by_harvest(disturbed_long_df, crop_windows)

    # per-window, per-system cumulative INCLUDING current event
    cum_long = _compute_cumprior_per_system(assigned)

    # add never-reset cumulative across ALL years (per system) — on LONG
    cum_long = cum_long.sort_values(["Op System", "Date"]).reset_index(drop=True)
    cum_long["Disturbance_CumAllYears"] = (
        cum_long.groupby("Op System")["Disturbance_Fraction"].cumsum()
    )

    # save LONG with cumulative columns
    disturbed_long_cum_out = DERIVED_DIR / "tillage_records_disturbance_long_cum.csv"
    cum_long.to_csv(disturbed_long_cum_out, index=False)

    # Derive WIDE directly from LONG (single source of truth)
    value_cols = ["Disturbance_Fraction", "Disturbance_Cum", "Disturbance_CumAllYears"]
    wide_cum = _long_to_wide_from_values(cum_long, value_cols=value_cols)

    disturbed_wide_cum_out = DERIVED_DIR / "tillage_records_disturbance_wide_cum.csv"
    wide_cum.to_csv(disturbed_wide_cum_out, index=False)

    # Small console summary
    cum_summary = {
        "windows_rows": int(crop_windows.shape[0]),
        "events_assigned": int(cum_long.shape[0]),
        "long_cum_csv": str(disturbed_long_cum_out.relative_to(REPO_ROOT)),
        "wide_cum_csv": str(disturbed_wide_cum_out.relative_to(REPO_ROOT)),
    }
    print(json.dumps(cum_summary, indent=2))


    # Console summaries
    prep_summary = {
        "mapper_shape": mapper_df.shape,
        "records_shape": records_df.shape,
        "num_unique_ops_in_records": len(unique_ops),
        "num_unmapped_ops_initial": len(unmapped_ops_initial),
        "artifacts": {
            "mapper_normalized_csv": str(mapper_norm_out.relative_to(REPO_ROOT)),
            "mapper_dict_json": str(mapper_json_out.relative_to(REPO_ROOT)),
            "unmapped_ops_csv": str(unmapped_out.relative_to(REPO_ROOT)),
        },
    }
    apply_summary = {
        "mapped_wide_csv": str(mapped_wide_out.relative_to(REPO_ROOT)),
        "mapped_long_csv": str(mapped_long_out.relative_to(REPO_ROOT)),
        "num_still_unmapped_after_apply": len(all_still_unmapped),
        "still_unmapped_csv": str((DERIVED_DIR / "still_unmapped_ops_after_apply.csv").relative_to(REPO_ROOT))
                            if (DERIVED_DIR / "still_unmapped_ops_after_apply.csv").exists()
                            else "not written",
    }

    disturb_summary = {
        "disturbance_long_csv": str(disturbed_long_out.relative_to(REPO_ROOT)),
        "disturbance_wide_csv": str(disturbed_wide_cum_out.relative_to(REPO_ROOT)),  # <-- was disturbed_wide_out
        "ref_depth_mm": REF_DEPTH_MM,
    }


    print(json.dumps(prep_summary, indent=2))
    print(json.dumps(apply_summary, indent=2))
    print(json.dumps(disturb_summary, indent=2))

    if all_still_unmapped:
        # Also write them out for convenience
        still_unmapped_out = DERIVED_DIR / "still_unmapped_ops_after_apply.csv"
        pd.DataFrame({"still_unmapped_operation": sorted(all_still_unmapped)}).to_csv(still_unmapped_out, index=False)
        print("\nSample of still-unmapped ops (up to 25):")
        for op in sorted(all_still_unmapped)[:25]:
            print(f"  - {op}")


if __name__ == "__main__":
    main()
