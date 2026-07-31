#!/usr/bin/env python3
"""Assign spring residue measurements to planting-season plots.

Residue is a planting-time covariate, not a calendar-year covariate.  Each
annual residue record is therefore assigned only to the first crop planted in
that calendar year.  This prevents the 2018 bean and 2023 silage-corn
measurements from being reused for the winter-wheat crops planted later in
those years.

The residue submodel's STIR covariate is accumulated through the actual
measurement date when that date is known.  Otherwise PlantDate is the
documented proxy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from merge_wq_stir_by_season import (
    compute_cumulative_stir,
    merge_stir_with_wq,
    read_stir,
)


REQ_RESIDUE_COLS = {
    "Year",
    "Treatment",
    "Rep",
    "Residue_DryMass_kg_m2",
    "Residue_PercentCover",
}
SEASON_KEY = [
    "Treatment",
    "PlantDate",
    "HarvestDate",
    "SeasonYear",
    "Crop",
    "PreviousHarvestDate",
    "previous_crop",
]


def _norm_trt(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NA": pd.NA, "NAN": pd.NA})
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def aggregate_residue(res: pd.DataFrame) -> pd.DataFrame:
    missing = REQ_RESIDUE_COLS.difference(res.columns)
    if missing:
        raise ValueError(f"Residue file missing required columns: {sorted(missing)}")

    out = res.copy()
    out["ResidueYear"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Treatment"] = _norm_trt(out["Treatment"])
    out["Rep"] = pd.to_numeric(out["Rep"], errors="coerce").astype("Int64")
    out["Residue_DryMass_kg_m2"] = pd.to_numeric(
        out["Residue_DryMass_kg_m2"], errors="coerce"
    )
    out["Residue_PercentCover"] = pd.to_numeric(
        out["Residue_PercentCover"], errors="coerce"
    )
    out = out.dropna(subset=["ResidueYear", "Treatment", "Rep"]).copy()

    return (
        out.groupby(["ResidueYear", "Treatment", "Rep"], dropna=False)
        .agg(
            Residue_PercentCover=("Residue_PercentCover", "mean"),
            Residue_DryMass_kg_m2=("Residue_DryMass_kg_m2", "mean"),
            Residue_n=("Residue_PercentCover", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )


def read_measurement_dates(path: Path) -> pd.DataFrame:
    dates = _read_csv(path)
    required = {
        "ResidueYear",
        "ResidueMeasurementDate",
        "AssignedCrop",
        "AssignedPlantDate",
        "DateSource",
    }
    missing = required.difference(dates.columns)
    if missing:
        raise ValueError(
            f"Residue measurement-date file missing columns: {sorted(missing)}"
        )
    dates = dates.copy()
    dates["ResidueYear"] = pd.to_numeric(
        dates["ResidueYear"], errors="coerce"
    ).astype("Int64")
    dates["ResidueMeasurementDate"] = pd.to_datetime(
        dates["ResidueMeasurementDate"], errors="coerce"
    )
    dates["AssignedPlantDate"] = pd.to_datetime(
        dates["AssignedPlantDate"], errors="coerce"
    )
    dates["AssignedCrop"] = dates["AssignedCrop"].astype("string").str.strip().str.lower()
    if dates[list(required - {"DateSource", "AssignedCrop"})].isna().any().any():
        raise ValueError("Residue measurement-date file contains invalid required values.")
    if dates["ResidueYear"].duplicated().any():
        dup = dates.loc[dates["ResidueYear"].duplicated(False), "ResidueYear"].tolist()
        raise ValueError(f"Duplicate residue measurement-date years: {dup}")
    return dates


def build_season_table(wq: pd.DataFrame, measurement_dates: pd.DataFrame) -> pd.DataFrame:
    missing = set(SEASON_KEY).difference(wq.columns)
    if missing:
        raise ValueError(f"WQ x STIR data missing season columns: {sorted(missing)}")

    seasons = wq[SEASON_KEY].drop_duplicates().copy()
    for col in ["PlantDate", "HarvestDate", "PreviousHarvestDate"]:
        seasons[col] = pd.to_datetime(seasons[col], errors="coerce")
    seasons["Treatment"] = _norm_trt(seasons["Treatment"])
    seasons["Crop"] = seasons["Crop"].astype("string").str.strip().str.lower()
    seasons["SeasonYear"] = pd.to_numeric(
        seasons["SeasonYear"], errors="coerce"
    ).astype("Int64")
    if seasons[["Treatment", "PlantDate", "HarvestDate"]].isna().any().any():
        raise ValueError("Season table contains invalid Treatment/PlantDate/HarvestDate.")

    seasons["PlantCalendarYear"] = seasons["PlantDate"].dt.year.astype("Int64")
    first_plant = seasons.groupby(
        ["Treatment", "PlantCalendarYear"], dropna=False
    )["PlantDate"].transform("min")
    seasons["ResidueAssignmentEligible"] = seasons["PlantDate"].eq(first_plant)

    dates = measurement_dates.rename(
        columns={
            "AssignedCrop": "ConfirmedCrop",
            "AssignedPlantDate": "ConfirmedPlantDate",
        }
    )
    seasons = seasons.merge(
        dates,
        how="left",
        left_on="PlantCalendarYear",
        right_on="ResidueYear",
        validate="many_to_one",
    )

    confirmed = seasons["ResidueMeasurementDate"].notna()
    mismatch = confirmed & (
        ~seasons["PlantDate"].eq(seasons["ConfirmedPlantDate"])
        | ~seasons["Crop"].eq(seasons["ConfirmedCrop"])
    )
    # A confirmed date belongs only to its explicitly named season; other
    # seasons in that year retain their PlantDate proxy.
    seasons.loc[mismatch, [
        "ResidueMeasurementDate",
        "ConfirmedCrop",
        "ConfirmedPlantDate",
        "DateSource",
    ]] = [pd.NaT, pd.NA, pd.NaT, pd.NA]

    exact = seasons["ResidueMeasurementDate"].notna()
    if (exact & ~seasons["ResidueAssignmentEligible"]).any():
        bad = seasons.loc[
            exact & ~seasons["ResidueAssignmentEligible"],
            ["PlantCalendarYear", "Crop", "PlantDate"],
        ]
        raise ValueError(
            "A confirmed residue measurement is not assigned to the first crop "
            f"planted that year:\n{bad.to_string(index=False)}"
        )

    seasons["ResidueModelDate"] = seasons["PlantDate"]
    seasons.loc[exact, "ResidueModelDate"] = seasons.loc[
        exact, "ResidueMeasurementDate"
    ]
    seasons["ResidueModelDateSource"] = "PlantDate proxy"
    seasons.loc[exact, "ResidueModelDateSource"] = seasons.loc[exact, "DateSource"]
    return seasons


def add_residue_stir(
    seasons: pd.DataFrame, stir_path: Path, debug: bool = False
) -> pd.DataFrame:
    model_rows = seasons.copy()
    model_rows["Date"] = model_rows["ResidueModelDate"]
    stir = compute_cumulative_stir(read_stir(str(stir_path), debug=debug), debug=debug)
    modeled = merge_stir_with_wq(
        model_rows,
        stir,
        debug=debug,
        season_anchor="postharvest",
    )
    modeled = modeled.rename(
        columns={
            "Season_STIR_toDate": "Residue_STIR_toMeasurement",
            "Season_STIR_LeftCensored": "Residue_STIR_LeftCensored",
            "Season_STIR_StartDate": "Residue_STIR_StartDate",
            "Season_STIR_BoundaryDayCarryover": (
                "Residue_STIR_BoundaryDayCarryover"
            ),
        }
    )
    return modeled


def merge_residue(
    wq: pd.DataFrame,
    res_agg: pd.DataFrame,
    measurement_dates: pd.DataFrame,
    stir_path: Path,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = wq.copy()
    for col in ["PlantDate", "HarvestDate", "PreviousHarvestDate"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out["Treatment"] = _norm_trt(out["Treatment"])
    out["Rep"] = pd.to_numeric(out["Rep"], errors="coerce").astype("Int64")
    out["Crop"] = out["Crop"].astype("string").str.strip().str.lower()
    out["SeasonYear"] = pd.to_numeric(
        out["SeasonYear"], errors="coerce"
    ).astype("Int64")

    seasons = add_residue_stir(
        build_season_table(out, measurement_dates),
        stir_path,
        debug=debug,
    )
    attach_cols = SEASON_KEY + [
        "PlantCalendarYear",
        "ResidueAssignmentEligible",
        "ResidueMeasurementDate",
        "ResidueModelDate",
        "ResidueModelDateSource",
        "Residue_STIR_toMeasurement",
        "Residue_STIR_LeftCensored",
        "Residue_STIR_StartDate",
        "Residue_STIR_BoundaryDayCarryover",
    ]
    out = out.merge(
        seasons[attach_cols],
        how="left",
        on=SEASON_KEY,
        validate="many_to_one",
    )

    eligible = out["ResidueAssignmentEligible"].fillna(False)
    assigned = out.loc[eligible].merge(
        res_agg,
        how="left",
        left_on=["PlantCalendarYear", "Treatment", "Rep"],
        right_on=["ResidueYear", "Treatment", "Rep"],
        validate="many_to_one",
    )
    residue_cols = [
        "ResidueYear",
        "Residue_PercentCover",
        "Residue_DryMass_kg_m2",
        "Residue_n",
    ]
    out[residue_cols] = np.nan
    out.loc[eligible, residue_cols] = assigned[residue_cols].to_numpy()

    audit_cols = attach_cols + ["Rep"] + residue_cols
    audit = out[audit_cols].drop_duplicates().sort_values(
        ["PlantDate", "Treatment", "Rep"], kind="mergesort"
    )
    return out, audit


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assign residue to planting-season plots and compute planting-time STIR."
    )
    ap.add_argument("--wq", required=True, help="Merged WQ x seasonal-STIR CSV.")
    ap.add_argument("--residue", required=True, help="Residue measurement CSV.")
    ap.add_argument("--stir", required=True, help="Long-format STIR event CSV.")
    ap.add_argument(
        "--measurement-dates",
        default="data/residue_measurement_dates_v3p1.csv",
        help="Confirmed residue measurement dates and crop assignments.",
    )
    ap.add_argument("--out", required=True, help="Output CSV.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paths = {
        "wq": Path(args.wq),
        "residue": Path(args.residue),
        "stir": Path(args.stir),
        "measurement_dates": Path(args.measurement_dates),
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} input not found: {path}")

    res_agg = aggregate_residue(_read_csv(paths["residue"]))
    merged, audit = merge_residue(
        _read_csv(paths["wq"]),
        res_agg,
        read_measurement_dates(paths["measurement_dates"]),
        paths["stir"],
        debug=args.debug,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    res_agg.to_csv(out_path.parent / "residue_agg_by_year_trt_rep.csv", index=False)
    audit_path = out_path.parent / "residue_assignment_audit_v3p1.csv"
    audit.to_csv(audit_path, index=False)

    n_units = len(audit)
    n_obs = int(audit["Residue_PercentCover"].notna().sum())
    print(f"[OK] Wrote WQ x STIR with planting-season residue -> {out_path}")
    print(f"[OK] Wrote residue assignment audit -> {audit_path}")
    print(f"[INFO] Planting-season plot units: {n_units} | observed residue: {n_obs}")


if __name__ == "__main__":
    main()
