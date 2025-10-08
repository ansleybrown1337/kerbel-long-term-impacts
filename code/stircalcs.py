#!/usr/bin/env python3
"""
stircalcs.py

Step 1: Prepare mapper artifacts.
- Loads the tillage mapper + records CSVs from ./data
- Validates & normalizes the mapper
- Builds a mapping dict keyed by 'My Operation'
- Scans records (CT/ST/MT/My Operation columns) for unmapped strings
- Writes artifacts to ./data/derived

Usage:
  python code/stircalcs.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

MAPPER_CSV = DATA_DIR / "tillage_mapper_table.csv"
RECORDS_CSV = DATA_DIR / "tillage_records.csv"

REQUIRED_MAPPER_COLS = [
    "My Operation",
    "SWAT/RUSLE2 Implement",
    "Tillage Code",
    "Mixing Depth (mm)",
    "Mixing Efficiency",
    "Type / Note",
]

RECORD_OP_COLS_CANDIDATES = [
    "CT Operation",
    "ST Operation",
    "MT Operation",
    "My Operation",
]


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

    # ensure key column is string (helps static typing of dict keys)
    df["My Operation"] = df["My Operation"].astype(str)

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
    'SWAT/RUSLE2 Implement', 'Tillage Code', 'Mixing Depth (mm)', 'Mixing Efficiency', 'Type / Note'
    """
    idx = mapper_df["My Operation"].astype(str)
    subset = mapper_df.set_index(idx)[
        [
            "SWAT/RUSLE2 Implement",
            "Tillage Code",
            "Mixing Depth (mm)",
            "Mixing Efficiency",
            "Type / Note",
        ]
    ]
    # Pylance-friendly: keys are str, values are Dict[str, Any]
    out: Dict[str, Dict[str, Any]] = subset.to_dict(orient="index")  # type: ignore[assignment]
    return out


def collect_unique_ops(records_df: pd.DataFrame) -> List[str]:
    """
    Collect unique operation strings across CT/ST/MT/My Operation columns using
    a pure-Python set (avoid pandas .unique/.drop_duplicates to keep static
    type checkers happy).
    """
    cols = [c for c in RECORD_OP_COLS_CANDIDATES if c in records_df.columns]
    if not cols:
        cols = ["My Operation"] if "My Operation" in records_df.columns else []
    if not cols:
        return []

    uniques: set[str] = set()
    for c in cols:
        s = records_df[c]
        # Ensure string, strip, filter NaN/empty
        s = s.dropna().astype(str).str.strip()
        for val in s:
            if val:
                uniques.add(val)

    return sorted(uniques)


# ---------- Main ----------
def main() -> None:
    print("== stircalcs.py : prepare mapper artifacts ==")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Reading mapper:  {MAPPER_CSV}")
    print(f"Reading records: {RECORDS_CSV}")

    mapper_df = load_mapper(MAPPER_CSV)
    records_df = load_records(RECORDS_CSV)

    mapping_dict = build_mapping_dict(mapper_df)
    unique_ops = collect_unique_ops(records_df)

    # Determine unmapped
    unmapped_ops = sorted([op for op in unique_ops if op not in mapping_dict])

    # Write artifacts
    mapper_norm_out = DERIVED_DIR / "mapper_normalized.csv"
    mapper_df.to_csv(mapper_norm_out, index=False)

    mapper_json_out = DERIVED_DIR / "mapper_dict.json"
    with mapper_json_out.open("w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)

    unmapped_out = DERIVED_DIR / "unmapped_ops.csv"
    pd.DataFrame({"unmapped_operation": unmapped_ops}).to_csv(unmapped_out, index=False)

    # Console summary
    summary = {
        "mapper_shape": mapper_df.shape,
        "records_shape": records_df.shape,
        "num_unique_ops_in_records": len(unique_ops),
        "num_unmapped_ops": len(unmapped_ops),
        "artifacts": {
            "mapper_normalized_csv": str(mapper_norm_out.relative_to(REPO_ROOT)),
            "mapper_dict_json": str(mapper_json_out.relative_to(REPO_ROOT)),
            "unmapped_ops_csv": str(unmapped_out.relative_to(REPO_ROOT)),
        },
    }
    print(json.dumps(summary, indent=2))

    if unmapped_ops:
        print("\nSample of unmapped ops (up to 25):")
        for op in unmapped_ops[:25]:
            print(f"  - {op}")


if __name__ == "__main__":
    main()
