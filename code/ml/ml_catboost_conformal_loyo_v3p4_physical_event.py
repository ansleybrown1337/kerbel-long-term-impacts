#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Physical-event v3p4 CatBoost workflow for the Kerbel monitoring record.

The concentration model retains every legitimate analyte row. The runoff-volume
model retains every genuine VolumeObservationID, mapped to:

    Year + Irrigation + Rep + Treatment

Date remains observation metadata and a deterministic EventDate supplies the
event-level time predictors. The primary full-record products contain model
predictions for every eligible concentration row and every physical-event
volume, whether the outcome was observed or missing. Observed outcomes train
and evaluate the models and appear as reference markers; they never replace
the primary modeled values. The optional ``strict_prediction`` mode removes
same-event observed co-outcomes from the feature set and requires an explicit
output directory so it cannot overwrite the primary full-record outputs.

Primary ML load centers are deterministic physical-event point totals.
Uncertainty draws resample the weighted signed log-scale residuals from each
physical-event-grouped split-conformal calibration set; conformal endpoints are
not treated as a uniform predictive distribution.

Default inputs and outputs remain compatible with the original workflow:

    input:  out/wq_cleaned.csv
    output: results/ml/v3p4_physical_event/
    figures: figures/ml/v3p4_physical_event/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.physical_event import (  # noqa: E402
    CORRECTED_VERSION as DATA_CONTRACT_VERSION,
    PHYSICAL_EVENT_KEY,
    add_concentration_observation_id,
    add_physical_event_id,
    aggregate_replicate_mean,
    build_event_date_audit,
    build_event_analyte_load_ledger,
    build_event_analyte_point_load_ledger,
    build_volume_observation_table,
    event_balanced_weights,
    loyo_masks,
    resolve_prediction_draws,
    split_event_groups,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "physical_event_v3p4.json"
)
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"ML v3p4 configuration is absent: {CONFIG_PATH}")
PHYSICAL_EVENT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
WORKFLOW_VERSION = str(PHYSICAL_EVENT_CONFIG["workflow_version"])

EVENT_KEY_PREFERRED = PHYSICAL_EVENT_KEY
PREVIOUS_CROP_CANDIDATES = [
    "previous_crop", "PreviousCrop", "Previous_Crop", "PrevCrop", "prev_crop"
]
RESIDUE_COVER_CANDIDATES = ["residue_prop", "Residue_PercentCover"]

CONCENTRATION_LEAKAGE_COLUMNS = {
    "Result_mg_L", "Result_mg_L_cens", "cout_z", "Result_is_nd", "y_logC"
}
VOLUME_LEAKAGE_COLUMNS = {
    "Volume", "volume_z", "y_logV", "Load", "Load_mg", "AnnualLoad_mg",
    "Result_mg_L_filled", "Volume_filled",
}

CATEGORICAL_CANDIDATES = {
    "Treatment", "Analyte", "Flag", "Inflow_Flag", "FlumeMethod",
    "MeasureMethod", "IrrMethod", "IrrigationInfrastructure", "TSSMethod",
    "Lab", "Crop", "previous_crop", "PreviousCrop", "Previous_Crop",
    "PrevCrop", "prev_crop", "SampleMethod", "RLMDL_Method", "RLMDL_Source",
    "RLMDL_Assumed", "FF", "Composite", "Duplicate", "SeasonYear",
}


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(15):
        if (
            (cur / "README.md").is_file()
            and (cur / "code").is_dir()
            and (cur / "config" / "physical_event_v3p4.json").is_file()
        ):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find the repository root containing README.md, code/, and "
        "config/physical_event_v3p4.json. "
        "Run inside the repository or pass --repo."
    )


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_log1p(series: pd.Series) -> pd.Series:
    values = as_numeric(series)
    return np.log1p(values.where(values >= 0))


def inverse_log1p(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(np.expm1(arr), 0.0, np.inf)


def true_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def nonblank_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip().str.lower()
    return ~text.isin({"", "na", "nan", "none", "<na>"})


def choose_first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def parse_and_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None, str | None]:
    """Standardize required fields and add time/infrastructure features."""
    out = df.copy()
    if "_wq_idx" not in out.columns:
        out.insert(0, "_wq_idx", np.arange(len(out), dtype=int))

    for column in ["Date", "PlantDate", "HarvestDate"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "Date" in out.columns:
        out["ObservationDate"] = out["Date"]

    if "Treatment" not in out.columns and "System" in out.columns:
        out["Treatment"] = out["System"]
    if "Year" not in out.columns:
        if "Date" not in out.columns:
            raise ValueError("The input must contain Year or Date.")
        out["Year"] = out["Date"].dt.year
    out["Year"] = as_numeric(out["Year"])

    if "Date" in out.columns:
        out["DayOfYear"] = out["Date"].dt.dayofyear
    elif "DayOfYear" not in out.columns:
        out["DayOfYear"] = np.nan
    if "Date" in out.columns and "PlantDate" in out.columns:
        out["DaysSincePlant"] = (out["Date"] - out["PlantDate"]).dt.days
    elif "DaysSincePlant" not in out.columns:
        out["DaysSincePlant"] = np.nan
    if "Date" in out.columns and "HarvestDate" in out.columns:
        out["DaysUntilHarvest"] = (out["HarvestDate"] - out["Date"]).dt.days
    elif "DaysUntilHarvest" not in out.columns:
        out["DaysUntilHarvest"] = np.nan

    previous_crop = choose_first_present(out.columns, PREVIOUS_CROP_CANDIDATES)
    residue_cover = choose_first_present(out.columns, RESIDUE_COVER_CANDIDATES)
    if residue_cover is None:
        residue_cover = next(
            (c for c in out.columns if "residue" in c.lower() and "cover" in c.lower()),
            None,
        )

    # IrrMethod is primary when it contains usable monitoring-record values.
    infrastructure_source = "year_fallback"
    if "IrrMethod" in out.columns and out["IrrMethod"].notna().any():
        raw = out["IrrMethod"].astype("string")
        clean = pd.Series(pd.NA, index=out.index, dtype="string")
        clean.loc[raw.str.contains("gated", case=False, na=False)] = "Gated Pipe"
        clean.loc[raw.str.contains("siphon", case=False, na=False)] = "Siphon"
        clean = clean.fillna(raw.str.strip())
        out["IrrigationInfrastructure"] = clean
        infrastructure_source = "IrrMethod"
    else:
        warn("IrrMethod is absent or entirely missing; deriving infrastructure from monitoring year.")
        out["IrrigationInfrastructure"] = np.where(
            out["Year"] <= 2022, "Siphon", np.where(out["Year"] >= 2023, "Gated Pipe", "Unknown")
        )

    return out, infrastructure_source, previous_crop, residue_cover


def resolve_event_key(df: pd.DataFrame) -> list[str]:
    key = list(EVENT_KEY_PREFERRED)
    missing = [column for column in key if column not in df.columns]
    if missing:
        raise ValueError(f"Physical-event key is incomplete: {missing}")
    return key


def build_physical_event_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build row-level concentration and genuine volume-observation tables.

    A placeholder row is retained for a physical event with no volume target so
    the full-record model can later predict it. It has no role in volume fitting.
    """

    rows = add_concentration_observation_id(add_physical_event_id(df))
    volume_observations, reports = build_volume_observation_table(rows, strict=True)
    event_dates, multi_date_audit, predictor_conflicts = build_event_date_audit(
        rows, volume_observations
    )
    if event_dates["EventDate"].isna().any():
        raise ValueError("At least one physical event has no defensible EventDate.")
    rows = rows.merge(
        event_dates[["PhysicalEventID", "EventDate"]],
        on="PhysicalEventID",
        how="left",
        validate="many_to_one",
    )
    rows["EventDate"] = pd.to_datetime(rows["EventDate"], errors="raise")
    rows["Date"] = rows["EventDate"]
    rows["DayOfYear"] = rows["EventDate"].dt.dayofyear
    if "PlantDate" in rows:
        rows["DaysSincePlant"] = (rows["EventDate"] - rows["PlantDate"]).dt.days
    if "HarvestDate" in rows:
        rows["DaysUntilHarvest"] = (rows["HarvestDate"] - rows["EventDate"]).dt.days
    reports["event_date_audit"] = event_dates
    reports["multi_date_event_audit"] = multi_date_audit
    reports["event_level_predictor_conflicts"] = predictor_conflicts
    volume_observations = volume_observations.rename(
        columns={"Date": "ObservationDate"}
    )

    # Volume-process features must be event-level. Measurement provenance comes
    # from the genuine observation table rather than an arbitrary analyte row.
    provenance = {"MeasureMethod", "FlumeMethod", "Volume", "NoRunoff"}
    feature_columns = [
        column for column in rows.columns
        if column not in provenance
        and column not in {
            "Result_mg_L", "Result_mg_L_cens", "cout_z", "Result_is_nd",
            "ConcentrationObservationID", "SampleID", "Duplicate", "Analyte",
            "analyte_abbr", "Flag", "Lab", "TSSMethod", "SampleMethod",
            "ObservationDate",
        }
    ]
    event_features = (
        rows.sort_values("_wq_idx" if "_wq_idx" in rows else "PhysicalEventID")
        .drop_duplicates("PhysicalEventID", keep="first")
        [["PhysicalEventID", *[c for c in feature_columns if c != "PhysicalEventID"]]]
    )
    observation_columns = [
        column for column in [
            "VolumeObservationID", "PhysicalEventID", "Volume",
            "MeasureMethod", "FlumeMethod", "source_row_count", "_confirmed_zero",
            "ObservationDate",
        ] if column in volume_observations
    ]
    volume_rows = volume_observations[observation_columns].merge(
        event_features, on="PhysicalEventID", how="left", validate="many_to_one"
    )
    absent = event_features.loc[
        ~event_features["PhysicalEventID"].isin(volume_rows["PhysicalEventID"])
    ].copy()
    absent["VolumeObservationID"] = pd.NA
    absent["Volume"] = np.nan
    for column in ["MeasureMethod", "FlumeMethod", "source_row_count", "_confirmed_zero"]:
        if column not in absent:
            absent[column] = pd.NA
    absent_for_concat = (
        absent.reindex(columns=volume_rows.columns)
        .dropna(axis=1, how="all")
    )
    volume_rows = pd.concat(
        [volume_rows, absent_for_concat],
        ignore_index=True,
    )
    volume_rows["y_logV"] = safe_log1p(volume_rows["Volume"])
    rows["y_logC"] = safe_log1p(rows["Result_mg_L"])
    rows["y_logV"] = safe_log1p(rows["Volume"])
    blocking_sources = [
        reports["ambiguous_volume_observations"].assign(finding="ambiguous_volume"),
        reports["zero_missing_conflicts"].assign(finding="zero_conflict"),
        reports["missing_physical_key"].assign(finding="missing_physical_key"),
        predictor_conflicts.assign(finding="event_level_predictor_conflict"),
    ]
    populated_blocking_sources = [
        source for source in blocking_sources if not source.empty
    ]
    if populated_blocking_sources:
        blocking = pd.concat(
            populated_blocking_sources,
            ignore_index=True,
            sort=False,
        )
    else:
        blocking_columns = list(
            dict.fromkeys(
                column
                for source in blocking_sources
                for column in source.columns
            )
        )
        blocking = pd.DataFrame(columns=blocking_columns)
    return volume_rows, blocking, rows, reports


APPROVED_COMPACTION_ASSIGNMENTS = {
    (2021, "ST"),
    (2022, "MT"),
    (2022, "ST"),
    (2023, "MT"),
    (2023, "ST"),
    (2024, "MT"),
    (2024, "ST"),
    (2025, "MT"),
    (2025, "ST"),
}


def attach_furrow_tire_compaction(
    events: pd.DataFrame,
    repo: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach the reviewed post-residue compaction exposure to volume events.

    The returned event table is the authoritative event-level exposure source.
    ``attach_compaction_to_concentration_rows`` propagates that same reviewed
    value to concentration observations without creating a second assignment
    rule.
    """

    config = PHYSICAL_EVENT_CONFIG["furrow_tire_compaction"]
    source_path = repo / str(config["source_file"])
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Furrow tire-compaction source is absent: {source_path}"
        )
    records = pd.read_csv(source_path, keep_default_na=True)
    required = {
        "Year", "Treatment", "RepScope", "FurrowTireCompaction",
        "AnchorDate", "AnchorOperation", "Timing", "DateStatus",
        "ProvenanceNote",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(
            f"Furrow tire-compaction source is missing columns: {missing}"
        )
    records["Year"] = pd.to_numeric(records["Year"], errors="raise").astype(int)
    records["Treatment"] = records["Treatment"].astype(str).str.strip()
    records["FurrowTireCompaction"] = pd.to_numeric(
        records["FurrowTireCompaction"], errors="raise"
    ).astype(int)
    records["AnchorDate"] = pd.to_datetime(records["AnchorDate"], errors="coerce")
    if records.duplicated(["Year", "Treatment"]).any():
        raise ValueError(
            "Furrow tire-compaction assignments duplicate Year + Treatment."
        )
    found_assignments = set(
        records[["Year", "Treatment"]].itertuples(index=False, name=None)
    )
    if (
        found_assignments != APPROVED_COMPACTION_ASSIGNMENTS
        or not records["FurrowTireCompaction"].eq(1).all()
        or not records["RepScope"].astype(str).str.lower().eq("both").all()
        or not records["Timing"].astype(str).eq(
            "after_residue_before_first_runoff"
        ).all()
    ):
        raise ValueError(
            "Furrow tire-compaction assignments do not match the approved "
            "2021 ST and 2022-2025 MT/ST treatment-year roster."
        )

    output = events.copy()
    output["Year"] = pd.to_numeric(output["Year"], errors="raise").astype(int)
    output["Treatment"] = output["Treatment"].astype(str).str.strip()
    output = output.merge(
        records,
        on=["Year", "Treatment"],
        how="left",
        validate="many_to_one",
    )
    output["FurrowTireCompaction"] = (
        pd.to_numeric(output["FurrowTireCompaction"], errors="coerce")
        .fillna(int(config["default_unexposed_value"]))
        .astype(int)
    )
    if not output["FurrowTireCompaction"].isin([0, 1]).all():
        raise ValueError("FurrowTireCompaction must be a complete binary field.")
    output["CompactionExposureSource"] = np.where(
        output["FurrowTireCompaction"].eq(1),
        source_path.name,
        "unlisted_default_zero",
    )
    output["EventDate"] = pd.to_datetime(output["EventDate"], errors="raise")
    dated_timing_failure = (
        output["FurrowTireCompaction"].eq(1)
        & output["AnchorDate"].notna()
        & output["EventDate"].lt(output["AnchorDate"])
    )
    if dated_timing_failure.any():
        raise ValueError(
            "At least one compacted event precedes its documented operational "
            "anchor date."
        )
    undated_exposure = (
        output["FurrowTireCompaction"].eq(1)
        & output["AnchorDate"].isna()
    )
    if not output.loc[
        undated_exposure, "DateStatus"
    ].astype(str).eq("user_confirmed_before_first_runoff").all():
        raise ValueError(
            "Undated compaction exposure lacks user-confirmed pre-runoff timing."
        )

    compacted_events = int(
        output.loc[
            output["FurrowTireCompaction"].eq(1), "PhysicalEventID"
        ].nunique()
    )
    compacted_with_volume = int(
        output.loc[
            output["FurrowTireCompaction"].eq(1)
            & output["VolumeObservationID"].notna()
            & output["y_logV"].notna(),
            "PhysicalEventID",
        ].nunique()
    )
    expected_events = int(config["expected_compacted_events"])
    expected_with_volume = int(
        config["expected_compacted_events_with_genuine_volume"]
    )
    if (
        compacted_events != expected_events
        or compacted_with_volume != expected_with_volume
    ):
        raise ValueError(
            "Unexpected compaction support: "
            f"{compacted_events} exposed events and "
            f"{compacted_with_volume} exposed genuine volume observations; "
            f"expected {expected_events} and {expected_with_volume}."
        )

    audit_columns = [
        "PhysicalEventID", "Year", "Irrigation", "Rep", "Treatment",
        "EventDate", "VolumeObservationID", "FurrowTireCompaction",
        "RepScope", "AnchorDate", "AnchorOperation", "Timing", "DateStatus",
        "ProvenanceNote", "CompactionExposureSource",
    ]
    volume_support = (
        output.assign(
            has_genuine_volume_observation=(
                output["VolumeObservationID"].notna() & output["y_logV"].notna()
            )
        )
        .groupby("PhysicalEventID", as_index=False)
        .agg(
            n_genuine_volume_observations=(
                "has_genuine_volume_observation", "sum"
            ),
            has_genuine_volume_observation=(
                "has_genuine_volume_observation", "any"
            ),
        )
    )
    audit = (
        output[[column for column in audit_columns if column in output]]
        .sort_values(["PhysicalEventID", "VolumeObservationID"], na_position="last")
        .drop_duplicates("PhysicalEventID", keep="first")
        .merge(
            volume_support,
            on="PhysicalEventID",
            how="left",
            validate="one_to_one",
        )
    )
    summary = (
        audit.groupby(["Year", "Treatment"], as_index=False, dropna=False)
        .agg(
            physical_events=("PhysicalEventID", "size"),
            compacted_events=("FurrowTireCompaction", "sum"),
            events_with_genuine_volume=(
                "has_genuine_volume_observation", "sum"
            ),
            first_event_date=("EventDate", "min"),
            last_event_date=("EventDate", "max"),
        )
    )
    compacted_genuine = (
        audit.assign(
            compacted_genuine=lambda frame: (
                frame["FurrowTireCompaction"].eq(1)
                & frame["has_genuine_volume_observation"]
            )
        )
        .groupby(["Year", "Treatment"], as_index=False)["compacted_genuine"]
        .sum()
        .rename(
            columns={
                "compacted_genuine":
                    "compacted_events_with_genuine_volume"
            }
        )
    )
    summary = summary.merge(
        compacted_genuine,
        on=["Year", "Treatment"],
        how="left",
        validate="one_to_one",
    )
    return output, audit, summary


def attach_compaction_to_concentration_rows(
    rows: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Propagate one validated compaction value to every concentration row."""

    event_exposure = events[
        ["PhysicalEventID", "FurrowTireCompaction"]
    ].drop_duplicates()
    if event_exposure["PhysicalEventID"].duplicated().any():
        raise ValueError(
            "FurrowTireCompaction is not unique within PhysicalEventID."
        )
    output = rows.drop(
        columns=["FurrowTireCompaction"], errors="ignore"
    ).merge(
        event_exposure,
        on="PhysicalEventID",
        how="left",
        validate="many_to_one",
    )
    output["FurrowTireCompaction"] = pd.to_numeric(
        output["FurrowTireCompaction"], errors="raise"
    ).astype(int)
    if not output["FurrowTireCompaction"].isin([0, 1]).all():
        raise ValueError(
            "Concentration-row FurrowTireCompaction must be complete and binary."
        )
    expected_events = int(
        PHYSICAL_EVENT_CONFIG["furrow_tire_compaction"][
            "expected_compacted_events"
        ]
    )
    compacted_events = output.loc[
        output["FurrowTireCompaction"].eq(1), "PhysicalEventID"
    ].nunique()
    if compacted_events != expected_events:
        raise ValueError(
            "Unexpected concentration-model compaction support: "
            f"{compacted_events} exposed events; expected {expected_events}."
        )
    return output


def select_features(
    df_rows: pd.DataFrame,
    events: pd.DataFrame,
    mode: str,
    previous_crop: str | None,
    residue_cover: str | None,
    infrastructure_source: str,
    dry_mass_missing_threshold: float,
) -> tuple[list[str], list[str]]:
    concentration_desired = [
        "Year", "Irrigation", "Rep", "Treatment", "Analyte",
        "Inflow_Result_mg_L", "Inflow_Volume", "Flag", "Inflow_Flag",
        "FlumeMethod", "MeasureMethod", "IrrMethod", "TSSMethod", "Lab",
        "SeasonYear", "Crop", "CumAll_STIR_toDate", "Season_STIR_toDate",
        "DaysSincePlant", "DaysUntilHarvest", "DayOfYear", "SampleMethod",
        "FF", "Composite", "Duplicate", "RLMDL_Method", "RLMDL_Source",
        "RLMDL_Assumed", "MDL_mg_L", "RL_mg_L",
        "FurrowTireCompaction",
    ]
    if mode == "reconstruction":
        concentration_desired.insert(5, "Volume")
    if previous_crop:
        concentration_desired.insert(concentration_desired.index("CumAll_STIR_toDate"), previous_crop)
    if residue_cover:
        concentration_desired.append(residue_cover)
    if (
        "Residue_DryMass_kg_m2" in df_rows.columns
        and df_rows["Residue_DryMass_kg_m2"].isna().mean() <= dry_mass_missing_threshold
    ):
        concentration_desired.append("Residue_DryMass_kg_m2")
    if infrastructure_source != "IrrMethod" and "IrrigationInfrastructure" in df_rows.columns:
        concentration_desired.append("IrrigationInfrastructure")

    volume_desired = [
        "Year", "Irrigation", "Rep", "Treatment", "Crop", "SeasonYear",
        "CumAll_STIR_toDate", "Season_STIR_toDate", "Inflow_Volume",
        "MeasureMethod", "FlumeMethod", "IrrMethod", "DayOfYear",
        "DaysSincePlant", "FurrowTireCompaction",
    ]
    if previous_crop:
        volume_desired.insert(volume_desired.index("CumAll_STIR_toDate"), previous_crop)
    if residue_cover:
        volume_desired.append(residue_cover)
    if (
        "Residue_DryMass_kg_m2" in events.columns
        and events["Residue_DryMass_kg_m2"].isna().mean() <= dry_mass_missing_threshold
    ):
        volume_desired.append("Residue_DryMass_kg_m2")
    if infrastructure_source != "IrrMethod" and "IrrigationInfrastructure" in events.columns:
        volume_desired.append("IrrigationInfrastructure")
    # Never create event-level concentration averages before fitting the volume
    # model. Row-level concentration information is retained only in its own model.

    concentration = list(dict.fromkeys(
        column for column in concentration_desired
        if column in df_rows.columns and column not in CONCENTRATION_LEAKAGE_COLUMNS
    ))
    volume = list(dict.fromkeys(
        column for column in volume_desired
        if column in events.columns and column not in VOLUME_LEAKAGE_COLUMNS
    ))
    return concentration, volume


def prepare_feature_frame(df: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    X = df.loc[:, list(features)].copy()
    categorical: list[str] = []
    for column in X.columns:
        should_be_categorical = column in CATEGORICAL_CANDIDATES
        if column == "SeasonYear":
            should_be_categorical = as_numeric(X[column]).notna().mean() < 0.95
        if should_be_categorical or pd.api.types.is_object_dtype(X[column]) or isinstance(X[column].dtype, pd.StringDtype):
            categorical.append(column)
            X[column] = X[column].astype("string").fillna("__MISSING__").astype(str)
        else:
            X[column] = as_numeric(X[column])
    return X, categorical


def build_feature_audit(
    row_df: pd.DataFrame,
    event_df: pd.DataFrame,
    concentration_features: Sequence[str],
    volume_features: Sequence[str],
) -> pd.DataFrame:
    records: list[dict] = []
    for target, frame, selected, leakage in [
        ("logC", row_df, concentration_features, CONCENTRATION_LEAKAGE_COLUMNS),
        ("logV", event_df, volume_features, VOLUME_LEAKAGE_COLUMNS),
    ]:
        for feature in selected:
            records.append({
                "Target": target,
                "feature": feature,
                "included": True,
                "reason": "selected",
                "missing_fraction": float(frame[feature].isna().mean()),
                "n_unique_nonmissing": int(frame[feature].nunique(dropna=True)),
            })
        for feature in sorted(leakage):
            if feature in frame.columns and feature not in selected:
                records.append({
                    "Target": target,
                    "feature": feature,
                    "included": False,
                    "reason": "target_or_target_derived_leakage",
                    "missing_fraction": float(frame[feature].isna().mean()),
                    "n_unique_nonmissing": int(frame[feature].nunique(dropna=True)),
                })
    return pd.DataFrame(records)


def catboost_params(args: argparse.Namespace) -> dict:
    return {
        "loss_function": "RMSE",
        "iterations": int(args.cb_iterations),
        "learning_rate": float(args.cb_lr),
        "depth": int(args.cb_depth),
        "l2_leaf_reg": float(args.cb_l2),
        "subsample": float(args.cb_subsample),
        "colsample_bylevel": float(args.cb_colsample_bylevel),
        "eval_metric": "RMSE",
        "thread_count": int(args.threads),
        "verbose": int(args.cb_verbose_every) if int(args.cb_verbose_every) > 0 else False,
        "allow_writing_files": False,
    }


def make_model(params: dict, seed: int, X: pd.DataFrame, categorical: Sequence[str]) -> CatBoostRegressor:
    cat_indices = [X.columns.get_loc(column) for column in categorical if column in X.columns]
    return CatBoostRegressor(**params, random_seed=int(seed), cat_features=cat_indices)


def residual_quantile(
    residuals: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    values = np.asarray(residuals, dtype=float)
    weights = (
        np.ones(len(values), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        raise ValueError("No finite calibration residuals were available.")
    order = np.argsort(values)
    values = values[order]
    cumulative_weight = np.cumsum(weights[order]) / weights.sum()
    position = min(
        int(np.searchsorted(cumulative_weight, 1.0 - float(alpha), side="left")),
        len(values) - 1,
    )
    return float(values[position])


def fit_conformal_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: Sequence[str],
    groups: pd.Series,
    alpha: float,
    calibration: str,
    calibration_size: float,
    seed: int,
    params: dict,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> tuple[CatBoostRegressor, float, int, np.ndarray, np.ndarray]:
    if len(y) < 4:
        raise ValueError(f"At least four training observations are required; got {len(y)}.")
    if calibration != "physical-event":
        raise ValueError("v3p4 requires --calibration physical-event.")
    proper_positions, calibration_positions = split_event_groups(
        groups.reset_index(drop=True), float(calibration_size), int(seed)
    )
    if len(proper_positions) < 4:
        raise ValueError("Physical-event split left fewer than four proper-training rows.")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    model = make_model(params, seed, X, categorical)
    fit_kwargs: dict[str, object] = {}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights[proper_positions]
    model.fit(X.iloc[proper_positions], y.iloc[proper_positions], **fit_kwargs)
    calibration_prediction = np.asarray(model.predict(X.iloc[calibration_positions]), dtype=float)
    signed_residuals = (
        y.iloc[calibration_positions].to_numpy(dtype=float) - calibration_prediction
    )
    residuals = np.abs(signed_residuals)
    calibration_weights = None if weights is None else weights[calibration_positions]
    sampling_weights = (
        np.ones(len(residuals), dtype=float)
        if calibration_weights is None
        else np.asarray(calibration_weights, dtype=float)
    )
    valid = (
        np.isfinite(signed_residuals)
        & np.isfinite(sampling_weights)
        & (sampling_weights > 0)
    )
    signed_residuals = signed_residuals[valid]
    sampling_weights = sampling_weights[valid]
    if not len(signed_residuals):
        raise ValueError("No calibration residuals were available for Monte Carlo propagation.")
    sampling_probabilities = sampling_weights / sampling_weights.sum()
    return (
        model,
        residual_quantile(residuals, alpha, calibration_weights),
        int(len(residuals)),
        signed_residuals,
        sampling_probabilities,
    )


def predict_log_interval(model: CatBoostRegressor, q: float, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(model.predict(X), dtype=float)
    return prediction, prediction - float(q), prediction + float(q)


def metric_record(
    year: int,
    target: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    q: float,
    calibration_n: int,
    mode: str,
    calibration: str,
) -> dict:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    covered = (truth >= low) & (truth <= high)
    return {
        "Year_Test": int(year),
        "Target": target,
        "n_test": int(len(truth)),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "R2": float(r2_score(truth, prediction)) if len(truth) > 2 else np.nan,
        "q_conformal": float(q),
        "coverage": float(np.mean(covered)) if len(covered) else np.nan,
        "mean_interval_width_log": float(np.mean(high - low)) if len(covered) else np.nan,
        "n_calibration": int(calibration_n),
        "mode": mode,
        "calibration": calibration,
    }


def empirical_calibration_residual_draws(
    prediction_log: np.ndarray,
    signed_residuals_log: np.ndarray,
    sampling_probabilities: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the empirical split-conformal calibration residual distribution."""

    predictions = np.asarray(prediction_log, dtype=float)
    residuals = np.asarray(signed_residuals_log, dtype=float)
    probabilities = np.asarray(sampling_probabilities, dtype=float)
    if not len(residuals) or len(residuals) != len(probabilities):
        raise ValueError("Calibration residuals and probabilities must be nonempty and aligned.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Calibration sampling probabilities must be finite and nonnegative.")
    probability_sum = probabilities.sum()
    if not np.isfinite(probability_sum) or probability_sum <= 0:
        raise ValueError("Calibration sampling probabilities must have positive mass.")
    probabilities = probabilities / probability_sum
    sampled_positions = rng.choice(
        len(residuals),
        size=(len(predictions), int(draws)),
        replace=True,
        p=probabilities,
    )
    return predictions[:, None] + residuals[sampled_positions]


def summarize_annual_draws(draws: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if draws.empty:
        return pd.DataFrame(columns=["Year", "Treatment", "Analyte", "mean", "median", "low", "high", "n_draws"])
    q_low, q_high = alpha / 2.0, 1.0 - alpha / 2.0
    return (
        draws.groupby(["Year", "Treatment", "Analyte"], as_index=False)["AnnualLoad_mg"]
        .agg(
            mean="mean",
            median="median",
            low=lambda values: float(np.quantile(values, q_low)),
            high=lambda values: float(np.quantile(values, q_high)),
            n_draws="count",
        )
    )


def _distinct_text(values: pd.Series) -> str:
    labels = sorted({
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip()
    })
    return " | ".join(labels)


def _resolution_source_audit(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    observation_id: str,
    prefix: str,
    method_columns: Sequence[str],
) -> pd.DataFrame:
    """Describe every row resolved into one scientific prediction unit."""

    grouped = frame.groupby(list(group_columns), dropna=False)
    audit = grouped.size().rename(f"n_{prefix}_prediction_rows").reset_index()
    if observation_id in frame:
        counts = (
            grouped[observation_id]
            .nunique(dropna=True)
            .rename(f"n_{prefix}_observation_ids")
            .reset_index()
        )
        audit = audit.merge(counts, on=list(group_columns), validate="one_to_one")
    for column in method_columns:
        if column not in frame:
            continue
        method_count = (
            grouped[column]
            .nunique(dropna=True)
            .rename(f"n_distinct_{prefix}_{column}")
            .reset_index()
        )
        method_values = (
            grouped[column]
            .agg(_distinct_text)
            .rename(f"{prefix}_{column}_values")
            .reset_index()
        )
        audit = audit.merge(
            method_count, on=list(group_columns), validate="one_to_one"
        ).merge(
            method_values, on=list(group_columns), validate="one_to_one"
        )
    if prefix == "volume" and "source_row_count" in frame:
        copied_rows = (
            grouped["source_row_count"]
            .sum(min_count=1)
            .fillna(0)
            .rename("n_volume_source_rows_before_copy_deduplication")
            .reset_index()
        )
        audit = audit.merge(
            copied_rows, on=list(group_columns), validate="one_to_one"
        )
    return audit


def point_load_products(
    concentration_predictions: pd.DataFrame,
    volume_predictions: pd.DataFrame,
    *,
    concentration_value: str,
    volume_value: str,
    concentration_resolution: str,
    volume_resolution: str,
    method_priority: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve point predictions and return event, annual, and audit products."""

    analyte_column = (
        "analyte_abbr" if "analyte_abbr" in concentration_predictions else "Analyte"
    )
    concentration_groups = [
        "PhysicalEventID", "Year", "Treatment", "Rep", analyte_column,
    ]
    volume_groups = ["PhysicalEventID"]
    c_source = concentration_predictions.copy()
    v_source = volume_predictions.copy()
    c_source["Concentration_mg_L"] = pd.to_numeric(
        c_source[concentration_value], errors="raise"
    )
    v_source["Volume_L"] = pd.to_numeric(v_source[volume_value], errors="raise")
    if c_source["Concentration_mg_L"].isna().any():
        raise ValueError("A concentration point prediction is missing.")
    if v_source["Volume_L"].isna().any():
        raise ValueError("A volume point prediction is missing.")

    c_audit = _resolution_source_audit(
        c_source,
        concentration_groups,
        observation_id="ConcentrationObservationID",
        prefix="concentration",
        method_columns=["SampleMethod", "MeasureMethod", "Lab", "Duplicate"],
    )
    v_audit = _resolution_source_audit(
        v_source,
        volume_groups,
        observation_id="VolumeObservationID",
        prefix="volume",
        method_columns=["MeasureMethod", "FlumeMethod"],
    )
    c_resolved = resolve_prediction_draws(
        c_source,
        group_columns=concentration_groups,
        value_column="Concentration_mg_L",
        method=concentration_resolution,
        method_column="SampleMethod" if "SampleMethod" in c_source else None,
        method_priority=method_priority,
    )
    if analyte_column != "Analyte":
        c_resolved = c_resolved.rename(columns={analyte_column: "Analyte"})
        c_audit = c_audit.rename(columns={analyte_column: "Analyte"})
    v_resolved = resolve_prediction_draws(
        v_source,
        group_columns=volume_groups,
        value_column="Volume_L",
        method=volume_resolution,
        method_column="MeasureMethod" if "MeasureMethod" in v_source else None,
        method_priority=method_priority,
    )
    ledger = build_event_analyte_point_load_ledger(c_resolved, v_resolved)
    ledger = ledger.merge(
        c_audit,
        on=["PhysicalEventID", "Year", "Treatment", "Rep", "Analyte"],
        how="left",
        validate="one_to_one",
    ).merge(
        v_audit,
        on="PhysicalEventID",
        how="left",
        validate="many_to_one",
    )
    key = ["PhysicalEventID", "Analyte"]
    if ledger.duplicated(key).any():
        raise AssertionError("PhysicalEventID x Analyte point loads are not unique.")
    ledger["central_estimate_type"] = "physical_event_point_total"
    ledger["concentration_resolution"] = concentration_resolution
    ledger["volume_resolution"] = volume_resolution
    plot_totals, annual = aggregate_replicate_mean(
        ledger,
        value_column="Load_kg",
        group_columns=["Year", "Treatment", "Analyte"],
        plot_total_column="PlotAnnualLoad_kg",
        treatment_mean_column="AnnualLoad_kg",
    )
    event_counts = (
        ledger.groupby(["Year", "Treatment", "Analyte"], as_index=False)
        .agg(n_physical_events=("PhysicalEventID", "nunique"))
    )
    annual = annual.merge(
        event_counts,
        on=["Year", "Treatment", "Analyte"],
        validate="one_to_one",
    )
    annual["AnnualLoad_mg"] = annual["AnnualLoad_kg"] * 1_000_000.0
    annual["central_estimate_type"] = (
        "mean_of_replicate_annual_plot_totals"
    )
    audit_columns = [
        column for column in ledger.columns
        if column in {"PhysicalEventID", "Year", "Treatment", "Rep", "Analyte"}
        or column.startswith("n_")
        or column.endswith("_values")
        or column in {"concentration_resolution", "volume_resolution"}
    ]
    audit = ledger[audit_columns].copy()
    if audit.duplicated(key).any():
        raise AssertionError("Point-load resolution audit is not event-analyte unique.")
    return ledger, annual, audit


def full_record_point_products(
    rows: pd.DataFrame,
    volume_prediction_rows: pd.DataFrame,
    *,
    use_observed: bool,
    concentration_resolution: str,
    volume_resolution: str,
    method_priority: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic full-record point totals without using draw medians."""

    analyte_column = "analyte_abbr" if "analyte_abbr" in rows else "Analyte"
    c_source = rows.loc[
        rows["Year"].notna()
        & rows["Treatment"].notna()
        & rows[analyte_column].notna()
    ].copy()
    if use_observed:
        c_source["_point_concentration"] = pd.to_numeric(
            c_source["Result_mg_L"], errors="coerce"
        ).fillna(pd.to_numeric(c_source["Result_mg_L_model_pred"], errors="raise"))
    else:
        c_source["_point_concentration"] = pd.to_numeric(
            c_source["Result_mg_L_model_pred"], errors="raise"
        )

    v_source = volume_prediction_rows.copy()
    if use_observed:
        v_source["_point_volume"] = pd.to_numeric(
            v_source["Volume"], errors="coerce"
        ).fillna(pd.to_numeric(v_source["Volume_model_pred"], errors="raise"))
    else:
        v_source["_point_volume"] = pd.to_numeric(
            v_source["Volume_model_pred"], errors="raise"
        )
    return point_load_products(
        c_source,
        v_source,
        concentration_value="_point_concentration",
        volume_value="_point_volume",
        concentration_resolution=concentration_resolution,
        volume_resolution=volume_resolution,
        method_priority=method_priority,
    )


def feature_importance_summary(records: list[pd.DataFrame]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
    combined = pd.concat(records, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)["importance"]
        .agg(importance_mean="mean", importance_sd="std")
        .sort_values("importance_mean", ascending=False)
    )


def save_model(
    model: CatBoostRegressor,
    path: Path,
    metadata_path: Path,
    target: str,
    q: float,
    features: Sequence[str],
    categorical: Sequence[str],
    params: dict,
    args: argparse.Namespace,
    signed_residuals_log: np.ndarray,
    sampling_probabilities: np.ndarray,
    event_key: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    metadata = {
        "target": target,
        "unit_of_analysis": "analyte_row" if target == "logC" else "event_volume",
        "alpha": float(args.alpha),
        "q_conformal": float(q),
        "calibration_signed_residuals_log": np.asarray(
            signed_residuals_log, dtype=float
        ).tolist(),
        "calibration_sampling_probabilities": np.asarray(
            sampling_probabilities, dtype=float
        ).tolist(),
        "monte_carlo_propagation": "empirical_signed_calibration_residual_resampling",
        "feature_cols": list(features),
        "cat_cols": list(categorical),
        "cb_params": params,
        "mode": args.mode,
        "calibration": args.calibration,
        "event_key": list(event_key or []),
        "workflow_version": WORKFLOW_VERSION,
        "event_unit": "PhysicalEventID",
        "furrow_tire_compaction_feature_scope": (
            "concentration_and_runoff_volume"
        ),
        "uses_furrow_tire_compaction": "FurrowTireCompaction" in features,
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "workflow": "ml_catboost_conformal_loyo_v3p4_physical_event.py",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(path: Path, metadata_path: Path) -> tuple[CatBoostRegressor, dict]:
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing saved model or metadata: {path}, {metadata_path}")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model, json.loads(metadata_path.read_text(encoding="utf-8"))


def full_record_annual_products(
    rows: pd.DataFrame,
    volume_prediction_rows: pd.DataFrame,
    draws: int,
    alpha: float,
    seed: int,
    *,
    use_observed: bool,
    concentration_resolution: str,
    volume_resolution: str,
    method_priority: Sequence[str],
    concentration_signed_residuals_log: np.ndarray,
    concentration_sampling_probabilities: np.ndarray,
    volume_signed_residuals_log: np.ndarray,
    volume_sampling_probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a unique full-record ledger and annual summary.

    ``use_observed=False`` is the primary all-row model prediction. ``True`` is
    the explicitly opt-in observed-plus-imputed sensitivity. In either case all
    predictions first exist at row/observation level before within-draw
    resolution.
    """

    if draws <= 0:
        raise ValueError("Full-record prediction requires at least one draw.")
    rng = np.random.default_rng(seed)
    D = int(draws)
    analyte_column = "analyte_abbr" if "analyte_abbr" in rows else "Analyte"
    c_source = rows.loc[
        rows["Year"].notna() & rows["Treatment"].notna() & rows[analyte_column].notna()
    ].reset_index(drop=True)
    c_prediction_log = np.log1p(
        pd.to_numeric(c_source["Result_mg_L_model_pred"], errors="raise").to_numpy(dtype=float)
    )
    c_matrix = inverse_log1p(empirical_calibration_residual_draws(
        c_prediction_log,
        concentration_signed_residuals_log,
        concentration_sampling_probabilities,
        D,
        rng,
    ))
    if use_observed:
        observed_c = pd.to_numeric(c_source["Result_mg_L"], errors="coerce")
        observed_positions = observed_c.notna().to_numpy()
        c_matrix[observed_positions] = observed_c.loc[observed_positions].to_numpy(
            dtype=float
        )[:, None]
    # Repeat only the columns required for resolution. Repeating the complete
    # cleaned row table creates tens of millions of object cells and can require
    # several unnecessary GiB of RAM.
    c_long_columns = [
        "PhysicalEventID", "Year", "Treatment", "Rep", analyte_column,
        *(["SampleMethod"] if "SampleMethod" in c_source else []),
    ]
    c_long_base = c_source[c_long_columns]
    c_long = c_long_base.loc[c_long_base.index.repeat(D)].reset_index(drop=True)
    c_long["Draw"] = np.tile(np.arange(D), len(c_source))
    c_long["Concentration_mg_L"] = c_matrix.reshape(-1)
    c_resolved = resolve_prediction_draws(
        c_long,
        group_columns=[
            "PhysicalEventID", "Year", "Treatment", "Rep", analyte_column, "Draw"
        ],
        value_column="Concentration_mg_L",
        method=concentration_resolution,
        method_column="SampleMethod" if "SampleMethod" in c_long else None,
        method_priority=method_priority,
    )
    if analyte_column != "Analyte":
        c_resolved = c_resolved.rename(columns={analyte_column: "Analyte"})

    v_source = volume_prediction_rows.reset_index(drop=True)
    v_prediction_log = np.log1p(
        pd.to_numeric(v_source["Volume_model_pred"], errors="raise").to_numpy(dtype=float)
    )
    v_matrix = inverse_log1p(empirical_calibration_residual_draws(
        v_prediction_log,
        volume_signed_residuals_log,
        volume_sampling_probabilities,
        D,
        rng,
    ))
    if use_observed:
        observed_v = pd.to_numeric(v_source["Volume"], errors="coerce")
        observed_positions = observed_v.notna().to_numpy()
        v_matrix[observed_positions] = observed_v.loc[observed_positions].to_numpy(
            dtype=float
        )[:, None]
    v_long_columns = [
        "PhysicalEventID",
        *(["MeasureMethod"] if "MeasureMethod" in v_source else []),
    ]
    v_long_base = v_source[v_long_columns]
    v_long = v_long_base.loc[v_long_base.index.repeat(D)].reset_index(drop=True)
    v_long["Draw"] = np.tile(np.arange(D), len(v_source))
    v_long["Volume_L"] = v_matrix.reshape(-1)
    v_resolved = resolve_prediction_draws(
        v_long,
        group_columns=["PhysicalEventID", "Draw"],
        value_column="Volume_L",
        method=volume_resolution,
        method_column="MeasureMethod" if "MeasureMethod" in v_long else None,
        method_priority=method_priority,
    )
    ledger = build_event_analyte_load_ledger(c_resolved, v_resolved)
    ledger["AnnualLoad_mg"] = ledger["Load_kg"] * 1_000_000.0
    _, annual_draws = aggregate_replicate_mean(
        ledger,
        value_column="AnnualLoad_mg",
        group_columns=["Year", "Treatment", "Analyte"],
        draw_column="Draw",
        plot_total_column="PlotAnnualLoad_mg",
        treatment_mean_column="AnnualLoad_mg",
    )
    summary = summarize_annual_draws(annual_draws, alpha)
    return ledger, summary


def refit_and_impute(
    df: pd.DataFrame,
    concentration_rows: pd.DataFrame,
    events: pd.DataFrame,
    X_c: pd.DataFrame,
    X_v: pd.DataFrame,
    cat_c: Sequence[str],
    cat_v: Sequence[str],
    features_c: Sequence[str],
    features_v: Sequence[str],
    event_key: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_dir = output_dir / args.model_subdir
    c_model_path = model_dir / "model_logC.cbm"
    c_meta_path = model_dir / "model_logC_meta.json"
    v_model_path = model_dir / "model_logV.cbm"
    v_meta_path = model_dir / "model_logV_meta.json"

    observed_c = concentration_rows["y_logC"].notna()
    observed_v = events["y_logV"].notna()

    if args.impute_only:
        model_c, meta_c = load_model(c_model_path, c_meta_path)
        model_v, meta_v = load_model(v_model_path, v_meta_path)
        if meta_c.get("feature_cols") != list(features_c):
            raise ValueError("Saved logC feature columns do not match the current input.")
        if meta_v.get("feature_cols") != list(features_v):
            raise ValueError("Saved logV feature columns do not match the current event table.")
        if meta_c.get("mode", "reconstruction") != args.mode or meta_v.get("mode", "reconstruction") != args.mode:
            raise ValueError("Saved model mode does not match --mode.")
        q_c, q_v = float(meta_c["q_conformal"]), float(meta_v["q_conformal"])
        required_propagation_fields = [
            "calibration_signed_residuals_log",
            "calibration_sampling_probabilities",
        ]
        if any(field not in meta_c for field in required_propagation_fields) or any(
            field not in meta_v for field in required_propagation_fields
        ):
            raise ValueError(
                "Saved models predate empirical calibration-residual propagation. "
                "Run the complete ML workflow once without --impute_only."
            )
        c_signed_residuals = np.asarray(
            meta_c["calibration_signed_residuals_log"], dtype=float
        )
        c_sampling_probabilities = np.asarray(
            meta_c["calibration_sampling_probabilities"], dtype=float
        )
        v_signed_residuals = np.asarray(
            meta_v["calibration_signed_residuals_log"], dtype=float
        )
        v_sampling_probabilities = np.asarray(
            meta_v["calibration_sampling_probabilities"], dtype=float
        )
    else:
        (
            model_c, q_c, _, c_signed_residuals, c_sampling_probabilities,
        ) = fit_conformal_model(
            X_c.loc[observed_c].reset_index(drop=True),
            concentration_rows.loc[observed_c, "y_logC"].reset_index(drop=True),
            cat_c,
            concentration_rows.loc[observed_c, "PhysicalEventID"].reset_index(drop=True),
            args.alpha, args.calibration, args.calib_size, args.seed + 50001, params,
            concentration_rows.loc[observed_c, "event_weight"].reset_index(drop=True),
        )
        (
            model_v, q_v, _, v_signed_residuals, v_sampling_probabilities,
        ) = fit_conformal_model(
            X_v.loc[observed_v].reset_index(drop=True),
            events.loc[observed_v, "y_logV"].reset_index(drop=True),
            cat_v,
            events.loc[observed_v, "PhysicalEventID"].reset_index(drop=True),
            args.alpha, args.calibration, args.calib_size, args.seed + 60001, params,
            events.loc[observed_v, "event_weight"].reset_index(drop=True),
        )
        if args.save_models:
            save_model(
                model_c, c_model_path, c_meta_path, "logC", q_c,
                features_c, cat_c, params, args,
                c_signed_residuals, c_sampling_probabilities,
            )
            save_model(
                model_v, v_model_path, v_meta_path, "logV", q_v,
                features_v, cat_v, params, args,
                v_signed_residuals, v_sampling_probabilities, event_key,
            )

    for target, residuals, probabilities, q_value in [
        ("logC", c_signed_residuals, c_sampling_probabilities, q_c),
        ("logV", v_signed_residuals, v_sampling_probabilities, q_v),
    ]:
        if len(residuals) != len(probabilities) or not np.isclose(
            np.asarray(probabilities, dtype=float).sum(), 1.0
        ):
            raise AssertionError(
                f"{target} calibration residuals and sampling probabilities are invalid."
            )
        pd.DataFrame({
            "Target": target,
            "CalibrationResidualLog": np.asarray(residuals, dtype=float),
            "SamplingProbability": np.asarray(probabilities, dtype=float),
            "AbsoluteResidualConformalQ": float(q_value),
            "PropagationMethod": "weighted_resampling_of_signed_log_scale_calibration_residuals",
        }).to_csv(
            output_dir / f"calibration_residual_distribution_{target}.csv",
            index=False,
        )

    if args.fit_final_models_only:
        return pd.DataFrame(), pd.DataFrame()

    pred_c_log, lo_c_log, hi_c_log = predict_log_interval(model_c, q_c, X_c)
    pred_v_log, lo_v_log, hi_v_log = predict_log_interval(model_v, q_v, X_v)

    concentration_prediction = pd.DataFrame({
        "_wq_idx": concentration_rows["_wq_idx"].to_numpy(),
        "Result_mg_L_model_pred": inverse_log1p(pred_c_log),
        "Result_mg_L_model_pi_low": inverse_log1p(lo_c_log),
        "Result_mg_L_model_pi_high": inverse_log1p(hi_c_log),
    })
    event_predictions = events.copy()
    event_predictions["Volume_model_pred"] = inverse_log1p(pred_v_log)
    event_predictions["Volume_model_pi_low"] = inverse_log1p(lo_v_log)
    event_predictions["Volume_model_pi_high"] = inverse_log1p(hi_v_log)
    event_predictions["q_conformal_logV"] = q_v
    event_predictions["mode"] = args.mode
    event_predictions["unit_of_analysis"] = "volume_observation_or_prediction_event"

    imputed = df.copy()
    if "PhysicalEventID" not in imputed.columns:
        imputed = add_physical_event_id(imputed)
    imputed = imputed.merge(concentration_prediction, on="_wq_idx", how="left")
    volume_map = event_predictions[[
        "PhysicalEventID", "Volume_model_pred", "Volume_model_pi_low", "Volume_model_pi_high"
    ]]
    # Several volume-observation rows may predict the same event. Resolve only
    # now, after prediction, before mapping one volume back to concentration rows.
    volume_map = resolve_prediction_draws(
        volume_map, group_columns=["PhysicalEventID"], value_column="Volume_model_pred",
        method="median",
    ).merge(
        resolve_prediction_draws(
            event_predictions, group_columns=["PhysicalEventID"],
            value_column="Volume_model_pi_low", method="median",
        ), on="PhysicalEventID", validate="one_to_one"
    ).merge(
        resolve_prediction_draws(
            event_predictions, group_columns=["PhysicalEventID"],
            value_column="Volume_model_pi_high", method="median",
        ), on="PhysicalEventID", validate="one_to_one"
    )
    imputed = imputed.merge(volume_map, on="PhysicalEventID", how="left")

    missing_c = imputed["Result_mg_L"].isna() & imputed["Result_mg_L_model_pred"].notna()
    missing_v = imputed["Volume"].isna() & imputed["Volume_model_pred"].notna()
    imputed["Result_mg_L_pred"] = np.where(missing_c, imputed["Result_mg_L_model_pred"], np.nan)
    imputed["Result_mg_L_pi_low"] = np.where(missing_c, imputed["Result_mg_L_model_pi_low"], np.nan)
    imputed["Result_mg_L_pi_high"] = np.where(missing_c, imputed["Result_mg_L_model_pi_high"], np.nan)
    imputed["q_conformal_logC"] = np.where(missing_c, q_c, np.nan)
    imputed["Volume_pred"] = np.where(missing_v, imputed["Volume_model_pred"], np.nan)
    imputed["Volume_pi_low"] = np.where(missing_v, imputed["Volume_model_pi_low"], np.nan)
    imputed["Volume_pi_high"] = np.where(missing_v, imputed["Volume_model_pi_high"], np.nan)
    imputed["q_conformal_logV"] = np.where(missing_v, q_v, np.nan)
    imputed["Result_mg_L_filled"] = as_numeric(imputed["Result_mg_L"]).fillna(imputed["Result_mg_L_pred"])
    imputed["Volume_filled"] = as_numeric(imputed["Volume"]).fillna(imputed["Volume_pred"])
    if "NoRunoff" in imputed.columns:
        imputed.loc[true_mask(imputed["NoRunoff"]), "Volume_filled"] = 0.0

    if args.observed_plus_imputed_sensitivity:
        imputed.to_csv(
            output_dir / "row_values_observed_plus_imputed_sensitivity.csv",
            index=False,
        )
    imputed[[
        column for column in [
            "_wq_idx", "PhysicalEventID", "ConcentrationObservationID", "Year",
            "Treatment", "Analyte", "Result_mg_L_model_pred",
            "Result_mg_L_model_pi_low", "Result_mg_L_model_pi_high",
            "Volume_model_pred", "Volume_model_pi_low", "Volume_model_pi_high",
        ] if column in imputed
    ]].to_csv(output_dir / "row_predictions_full_record_model_only.csv", index=False)
    event_predictions.to_csv(output_dir / "volume_observation_and_event_predictions.csv", index=False)

    model_point_ledger, model_point_annual, model_point_audit = full_record_point_products(
        imputed,
        event_predictions,
        use_observed=False,
        concentration_resolution=args.concentration_resolution,
        volume_resolution=args.volume_resolution,
        method_priority=args.method_priority,
    )
    summary_key = ["Year", "Treatment", "Analyte"]
    model_point_ledger.to_csv(
        output_dir / "event_analyte_point_ledger_full_record_model_only.csv",
        index=False,
    )
    model_point_annual.to_csv(
        output_dir / "annual_load_point_totals_full_record_model_only.csv",
        index=False,
    )
    model_point_audit.to_csv(
        output_dir / "point_load_resolution_audit_full_record_model_only.csv",
        index=False,
    )
    if args.observed_plus_imputed_sensitivity:
        (
            sensitivity_point_ledger,
            sensitivity_point_annual,
            sensitivity_point_audit,
        ) = full_record_point_products(
            imputed,
            event_predictions,
            use_observed=True,
            concentration_resolution=args.concentration_resolution,
            volume_resolution=args.volume_resolution,
            method_priority=args.method_priority,
        )
        sensitivity_point_ledger.to_csv(
            output_dir / "event_analyte_point_ledger_observed_plus_imputed_sensitivity.csv",
            index=False,
        )
        sensitivity_point_annual.to_csv(
            output_dir / "annual_load_point_totals_observed_plus_imputed_sensitivity.csv",
            index=False,
        )
        sensitivity_point_audit.to_csv(
            output_dir / "point_load_resolution_audit_observed_plus_imputed_sensitivity.csv",
            index=False,
        )

    # Write the primary all-row prediction ledger before any optional
    # observed-plus-imputed sensitivity so both are never retained in memory.
    model_ledger, model_summary = full_record_annual_products(
        imputed, event_predictions, int(args.impute_draws), float(args.alpha), args.seed + 80001,
        use_observed=False,
        concentration_resolution=args.concentration_resolution,
        volume_resolution=args.volume_resolution,
        method_priority=args.method_priority,
        concentration_signed_residuals_log=c_signed_residuals,
        concentration_sampling_probabilities=c_sampling_probabilities,
        volume_signed_residuals_log=v_signed_residuals,
        volume_sampling_probabilities=v_sampling_probabilities,
    )
    model_summary = model_summary.merge(
        model_point_annual[summary_key + ["AnnualLoad_mg"]].rename(
            columns={"AnnualLoad_mg": "point_total_mg"}
        ),
        on=summary_key,
        how="left",
        validate="one_to_one",
    )
    if model_summary["point_total_mg"].isna().any():
        raise AssertionError("A full-record model-only annual point total is missing.")
    model_ledger.to_csv(
        output_dir / "event_analyte_draw_ledger_full_record_model_only.csv", index=False
    )
    model_summary.to_csv(output_dir / "annual_load_summary_full_record_model_only.csv", index=False)
    del model_ledger

    if args.observed_plus_imputed_sensitivity:
        sensitivity_ledger, sensitivity_summary = full_record_annual_products(
            imputed,
            event_predictions,
            int(args.impute_draws),
            float(args.alpha),
            args.seed + 90001,
            use_observed=True,
            concentration_resolution=args.concentration_resolution,
            volume_resolution=args.volume_resolution,
            method_priority=args.method_priority,
            concentration_signed_residuals_log=c_signed_residuals,
            concentration_sampling_probabilities=c_sampling_probabilities,
            volume_signed_residuals_log=v_signed_residuals,
            volume_sampling_probabilities=v_sampling_probabilities,
        )
        sensitivity_summary = sensitivity_summary.merge(
            sensitivity_point_annual[summary_key + ["AnnualLoad_mg"]].rename(
                columns={"AnnualLoad_mg": "point_total_mg"}
            ),
            on=summary_key,
            how="left",
            validate="one_to_one",
        )
        if sensitivity_summary["point_total_mg"].isna().any():
            raise AssertionError(
                "An observed-plus-imputed annual point total is missing."
            )
        sensitivity_ledger.to_csv(
            output_dir
            / "event_analyte_draw_ledger_observed_plus_imputed_sensitivity.csv",
            index=False,
        )
        sensitivity_summary.to_csv(
            output_dir
            / "annual_load_summary_observed_plus_imputed_sensitivity.csv",
            index=False,
        )
    return imputed, event_predictions


def plot_legacy_cv_rmse(metrics: pd.DataFrame, figure_dir: Path) -> None:
    if metrics.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    for target in ["logC", "logV"]:
        subset = metrics.loc[metrics["Target"] == target].sort_values("Year_Test")
        if not subset.empty:
            axis.plot(subset["Year_Test"], subset["RMSE"], marker="o", label=target)
    axis.set(
        title="LOYO full-record prediction RMSE",
        xlabel="Held-out year",
        ylabel="RMSE (log1p scale)",
    )
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "cv_rmse_by_year.png", dpi=200)
    plt.close(figure)


def generate_postprocess_figure_suite(
    repo: Path,
    output_dir: Path,
    figure_dir: Path,
    data_path: Path,
    args: argparse.Namespace,
) -> None:
    """Create the complete figure suite without refitting either CatBoost model."""
    from ml.ml_postprocess_plots_v3p4_physical_event import generate_figures

    generate_figures(
        repo=repo,
        output_dir=output_dir,
        figure_dir=figure_dir,
        source_path=data_path,
        units="g",
        alpha=float(args.alpha),
        fi_topk=int(args.fi_topk),
        use_imputed=bool(args.observed_plus_imputed_sensitivity),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--data", default=None, help="Default: <repo>/out/wq_cleaned.csv")
    parser.add_argument("--mode", choices=["reconstruction", "strict_prediction"], default="reconstruction")
    parser.add_argument("--calibration", choices=["physical-event"], default="physical-event")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concentration-resolution", choices=["median", "mean", "method_priority"],
        default=PHYSICAL_EVENT_CONFIG["concentration_resolution"],
    )
    parser.add_argument(
        "--volume-resolution", choices=["median", "mean", "method_priority"],
        default=PHYSICAL_EVENT_CONFIG["volume_resolution"],
    )
    parser.add_argument(
        "--method-priority", nargs="*", default=list(PHYSICAL_EVENT_CONFIG["method_priority"]),
        help="Optional explicit hierarchy; empty by design unless reviewed by the user.",
    )
    parser.add_argument(
        "--event-balanced-weights",
        action=argparse.BooleanOptionalAction,
        default=bool(PHYSICAL_EVENT_CONFIG["event_balanced_training_weights"]),
        help="Weight observed concentration rows within event-analyte and volume rows within event to sum to one.",
    )
    parser.add_argument("--calib_size", type=float, default=0.25)
    parser.add_argument("--exclude_flagged", action="store_true")
    parser.add_argument("--dry_mass_missing_threshold", type=float, default=0.60)

    parser.add_argument("--cb_iterations", type=int, default=3000)
    parser.add_argument("--cb_lr", type=float, default=0.05)
    parser.add_argument("--cb_depth", type=int, default=8)
    parser.add_argument("--cb_l2", type=float, default=6.0)
    parser.add_argument("--cb_subsample", type=float, default=0.8)
    parser.add_argument("--cb_colsample_bylevel", type=float, default=0.8)
    parser.add_argument("--cb_verbose_every", type=int, default=0)
    parser.add_argument("--threads", type=int, default=-1)

    parser.add_argument("--output_dir", default=None, help="Explicit output directory; relative paths resolve under repo.")
    parser.add_argument("--fig_dir", default=None, help="Explicit figure directory; relative paths resolve under repo.")
    parser.add_argument("--out_subdir", default=None, help="Optional subdirectory under <repo>/results.")
    parser.add_argument("--fig_subdir", default=None, help="Optional subdirectory under <repo>/figures.")
    parser.add_argument("--model_subdir", default="models")

    parser.add_argument("--impute_missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--impute_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--observed_plus_imputed_sensitivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also create the non-primary sensitivity that substitutes observed "
            "outcomes and predicts only missing outcomes."
        ),
    )
    parser.add_argument(
        "--fit_final_models_only",
        action="store_true",
        help=(
            "Fit and save the final full-record logC/logV models and calibration "
            "residual distributions, then stop before reconstruction."
        ),
    )
    parser.add_argument("--impute_draws", type=int, default=2000)
    parser.add_argument("--save_models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_importance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the ML post-processing figure suite after this stage.",
    )
    parser.add_argument(
        "--fi_topk",
        type=int,
        default=int(
            PHYSICAL_EVENT_CONFIG["ml_feature_contract"][
                "feature_importance_top_k"
            ]
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate physical events, compaction assignments, feature scope, "
            "and audit outputs, then stop before fitting CatBoost."
        ),
    )

    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fast_iterations", type=int, default=600)
    parser.add_argument("--fast_draws", type=int, default=300)
    return parser.parse_args()


def resolve_output_path(repo: Path, explicit: str | None, base: str, default_name: str) -> Path:
    if explicit:
        path = Path(explicit)
        return path.resolve() if path.is_absolute() else (repo / path).resolve()
    return (repo / base / default_name).resolve()


def update_existing_run_manifest(output_dir: Path, **fields: object) -> None:
    """Record separated-stage completion without inventing a LOYO manifest."""

    manifest_path = output_dir / "run_manifest_ml_v3p4_physical_event.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(fields)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")
    if not 0 < args.calib_size < 1:
        raise ValueError("--calib_size must be between 0 and 1.")

    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())
    data_path = Path(args.data).resolve() if args.data else repo / "out" / "wq_cleaned.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Required input not found: {data_path}")

    if args.mode == "strict_prediction" and not (args.output_dir or args.out_subdir):
        raise ValueError(
            "strict_prediction mode requires --output_dir or --out_subdir so it cannot overwrite "
            "the primary reconstruction outputs."
        )
    if args.output_dir or args.out_subdir:
        output_dir = resolve_output_path(
            repo, args.output_dir, "results", args.out_subdir or "ml"
        )
    else:
        output_dir = (repo / PHYSICAL_EVENT_CONFIG["output_roots"]["ml_results"]).resolve()
    default_figure_name = args.fig_subdir or (
        "ml/strict_prediction_v3p4"
        if args.mode == "strict_prediction" else "ml"
    )
    if args.fig_dir or args.fig_subdir or args.mode == "strict_prediction":
        figure_dir = resolve_output_path(repo, args.fig_dir, "figures", default_figure_name)
    else:
        figure_dir = (repo / PHYSICAL_EVENT_CONFIG["output_roots"]["ml_figures"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        args.cb_iterations = int(args.fast_iterations)
        args.draws = int(args.fast_draws)
        print(f"[INFO] FAST mode: iterations={args.cb_iterations}, annual draws={args.draws}")

    print(f"[INFO] Input: {data_path}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Figures: {figure_dir}")
    print(f"[INFO] Mode: {args.mode}; calibration: {args.calibration}")

    raw = pd.read_csv(data_path, na_values=["NA", "NaN", "nan", ""], keep_default_na=True, low_memory=False)
    required = {"Result_mg_L", "Volume", "Treatment", "Analyte"}
    missing_required = required - set(raw.columns)
    if missing_required:
        raise ValueError(f"Input is missing required columns: {sorted(missing_required)}")
    preflight_metadata_path = (
        repo
        / PHYSICAL_EVENT_CONFIG["data_contract"]["preflight_directory"]
        / "preflight_metadata.json"
    )
    if not preflight_metadata_path.is_file():
        raise FileNotFoundError(f"Physical-event preflight is required: {preflight_metadata_path}")
    preflight_metadata = json.loads(preflight_metadata_path.read_text(encoding="utf-8"))
    if (
        preflight_metadata.get("workflow_version") != DATA_CONTRACT_VERSION
        or not preflight_metadata.get("ready_for_model_execution", False)
    ):
        raise ValueError(
            "Physical-event preflight is incompatible, incomplete, or blocked; review BLOCKING_REVIEW.csv."
        )
    df, infrastructure_source, previous_crop, residue_cover = parse_and_engineer(raw)
    event_key = resolve_event_key(df)
    events, conflicts, df, volume_reports = build_physical_event_tables(df)
    (
        events,
        compaction_event_audit,
        compaction_audit_summary,
    ) = attach_furrow_tire_compaction(events, repo)
    df = attach_compaction_to_concentration_rows(df, events)
    analysis_analyte_column = "analyte_abbr" if "analyte_abbr" in df else "Analyte"

    # No-runoff is a volume-state designation, not permission to discard a
    # legitimate chemistry row. Retain every concentration observation.
    concentration_rows = df.copy()
    if args.exclude_flagged:
        flagged = pd.Series(False, index=concentration_rows.index)
        for column in ["Flag", "Inflow_Flag"]:
            if column in concentration_rows.columns:
                flagged |= nonblank_mask(concentration_rows[column])
        concentration_rows = concentration_rows.loc[~flagged].copy()
    concentration_rows["event_weight"] = 0.0
    observed_concentration_rows = concentration_rows["y_logC"].notna()
    if args.event_balanced_weights:
        concentration_rows.loc[observed_concentration_rows, "event_weight"] = event_balanced_weights(
            concentration_rows.loc[observed_concentration_rows],
            ["PhysicalEventID", analysis_analyte_column],
        ).to_numpy()
    else:
        concentration_rows.loc[observed_concentration_rows, "event_weight"] = 1.0
    events["event_weight"] = 0.0
    observed_volume_rows = events["y_logV"].notna()
    if args.event_balanced_weights:
        events.loc[observed_volume_rows, "event_weight"] = event_balanced_weights(
            events.loc[observed_volume_rows], ["PhysicalEventID"]
        ).to_numpy()
    else:
        events.loc[observed_volume_rows, "event_weight"] = 1.0

    features_c, features_v = select_features(
        df, events, args.mode, previous_crop, residue_cover,
        infrastructure_source, args.dry_mass_missing_threshold,
    )
    if not features_c or not features_v:
        raise ValueError("Feature selection produced an empty model feature set.")
    feature_contract = PHYSICAL_EVENT_CONFIG["ml_feature_contract"]
    required_concentration_features = set(
        feature_contract["concentration_required"]
    )
    missing_concentration_features = sorted(
        required_concentration_features - set(features_c)
    )
    if missing_concentration_features:
        raise ValueError(
            "Required concentration features are absent: "
            f"{missing_concentration_features}."
        )
    excluded_duplicate = str(
        feature_contract["concentration_excluded_duplicate"]
    )
    if excluded_duplicate in features_c:
        raise ValueError(
            f"{excluded_duplicate} duplicates RL_mg_L in the current data and "
            "must not enter the v3p4 concentration model."
        )
    if "FurrowTireCompaction" not in features_v:
        raise ValueError(
            "FurrowTireCompaction is required in the runoff-volume model."
        )
    required_volume_timing = str(
        feature_contract["volume_required_plant_timing"]
    )
    excluded_volume_timing = str(
        feature_contract["volume_excluded_harvest_timing"]
    )
    if (
        required_volume_timing not in features_v
        or excluded_volume_timing in features_v
    ):
        raise ValueError(
            "The runoff-volume model must use "
            f"{required_volume_timing} and exclude {excluded_volume_timing}."
        )
    X_c, cat_c = prepare_feature_frame(concentration_rows, features_c)
    X_v, cat_v = prepare_feature_frame(events, features_v)

    # Audits are written before fitting so the physical-unit correction remains inspectable.
    events.loc[events["y_logV"].notna()].to_csv(
        output_dir / "volume_observation_training_table_v3p4.csv", index=False
    )
    concentration_rows.loc[concentration_rows["y_logC"].notna()].to_csv(
        output_dir / "concentration_observation_training_table_v3p4.csv", index=False
    )
    conflicts.to_csv(output_dir / "physical_event_blocking_conflicts.csv", index=False)
    volume_reports["event_date_audit"].to_csv(
        output_dir / "event_date_audit.csv", index=False
    )
    volume_reports["multi_date_event_audit"].to_csv(
        output_dir / "multi_date_event_audit.csv", index=False
    )
    volume_reports["event_level_predictor_conflicts"].to_csv(
        output_dir / "event_level_predictor_conflicts.csv", index=False
    )
    compaction_event_audit.to_csv(
        output_dir
        / "tire_compaction_event_audit_ml_v3p4_physical_event.csv",
        index=False,
    )
    compaction_audit_summary.to_csv(
        output_dir
        / "tire_compaction_audit_summary_ml_v3p4_physical_event.csv",
        index=False,
    )
    build_feature_audit(df, events, features_c, features_v).to_csv(
        output_dir / "feature_audit_summary.csv", index=False
    )
    summary_audit = pd.DataFrame(
        [
            ("workflow_version", WORKFLOW_VERSION, "corrected ML workflow"),
            ("physical_event_key", " + ".join(event_key), "shared event contract"),
            ("comparison_analyte_field", analysis_analyte_column, "normalized ML/Bayes output key"),
            ("concentration_observations", len(df), "all legitimate rows retained"),
            ("physical_events", df["PhysicalEventID"].nunique(), "hydrologic plot events"),
            ("volume_candidate_rows_before_dedup", int(df["Volume"].notna().sum()), "copied raw rows"),
            ("genuine_volume_observations", int(events["VolumeObservationID"].notna().sum()), "volume training rows"),
            ("events_without_volume_observation", int(events["VolumeObservationID"].isna().sum()), "prediction only"),
            (
                "furrow_tire_compaction_events",
                int(compaction_event_audit["FurrowTireCompaction"].sum()),
                "binary predictor used in concentration and runoff volume",
            ),
            (
                "compacted_events_with_genuine_volume",
                int((
                    compaction_event_audit[
                        "FurrowTireCompaction"
                    ].eq(1)
                    & compaction_event_audit[
                        "has_genuine_volume_observation"
                    ]
                ).sum()),
                "observed support for the runoff-volume predictor",
            ),
            ("blocking_conflict_rows", len(conflicts), "must be zero before fitting"),
            ("event_balanced_training_weights", args.event_balanced_weights, "default true; recorded per training row"),
            ("predictors_used_logC", ";".join(features_c), "row model"),
            ("predictors_used_logV", ";".join(features_v), "volume-observation model"),
        ],
        columns=["metric", "value", "details"],
    )
    summary_audit.to_csv(output_dir / "event_volume_audit_summary.csv", index=False)

    print(
        f"[AUDIT] {len(df):,} concentration rows -> {df['PhysicalEventID'].nunique():,} physical events; "
        f"{df['Volume'].notna().sum():,} nonmissing-volume rows -> "
        f"{events['VolumeObservationID'].notna().sum():,} genuine volume observations; conflicts={len(conflicts)}"
    )
    print(f"[AUDIT] Previous crop: {previous_crop or 'not found'}; residue: {residue_cover or 'not found'}")
    print(f"[AUDIT] Irrigation infrastructure source: {infrastructure_source}")
    print(
        "[AUDIT] Furrow tire compaction: "
        f"{int(compaction_event_audit['FurrowTireCompaction'].sum())} exposed events; "
        f"{int((compaction_event_audit['FurrowTireCompaction'].eq(1) & compaction_event_audit['has_genuine_volume_observation']).sum())} "
        "with genuine volume; included in logC and logV."
    )
    if args.preflight_only:
        print(
            "[v3p4 ML] PRE-FLIGHT PASSED. The 528-event contract and "
            "logC/logV compaction feature scope are ready; model fitting "
            "was intentionally skipped."
        )
        return

    params = catboost_params(args)
    if args.fit_final_models_only:
        if args.impute_only:
            raise ValueError("--fit_final_models_only cannot be combined with --impute_only.")
        if not args.save_models:
            raise ValueError("--fit_final_models_only requires --save_models.")
        refit_and_impute(
            df, concentration_rows, events, X_c, X_v, cat_c, cat_v,
            features_c, features_v, event_key, output_dir, args, params,
        )
        update_existing_run_manifest(
            output_dir,
            full_record_models_saved=True,
            final_model_fit_deferred=False,
            full_record_prediction_completed=False,
        )
        print(
            "[DONE] Final full-record CatBoost models and calibration residuals "
            "saved; reconstruction not started."
        )
        return

    if args.impute_only:
        if not args.impute_missing:
            raise ValueError("--impute_only requires --impute_missing.")
        refit_and_impute(
            df, concentration_rows, events, X_c, X_v, cat_c, cat_v,
            features_c, features_v, event_key, output_dir, args, params,
        )
        update_existing_run_manifest(
            output_dir,
            primary_prediction_scope=(
                "all eligible concentration rows and all physical events, "
                "observed and missing"
            ),
            observed_values_role=(
                "training, held-out evaluation, and reference markers only; "
                "never substituted into primary modeled products"
            ),
            primary_uses_observed_value_substitution=False,
            observed_plus_imputed_sensitivity_requested=bool(
                args.observed_plus_imputed_sensitivity
            ),
            full_record_models_saved=True,
            final_model_fit_deferred=False,
            full_record_prediction_completed=True,
        )
        if args.figures:
            generate_postprocess_figure_suite(
                repo, output_dir, figure_dir, data_path, args
            )
        print("[DONE] Full-record predictions regenerated from saved event-level models.")
        return

    years = sorted(int(value) for value in df["Year"].dropna().unique())
    if len(years) < 2:
        raise ValueError("LOYO cross-validation requires at least two years.")

    rng = np.random.default_rng(args.seed)
    metrics_records: list[dict] = []
    prediction_records: list[pd.DataFrame] = []
    annual_draw_records: list[pd.DataFrame] = []
    event_ledger_records: list[pd.DataFrame] = []
    annual_point_records: list[pd.DataFrame] = []
    event_point_ledger_records: list[pd.DataFrame] = []
    importance_c: list[pd.DataFrame] = []
    importance_v: list[pd.DataFrame] = []

    progress = tqdm(years, desc="LOYO physical-event folds", unit="year", ncols=115)
    for held_year in progress:
        started = time.time()
        c_outer_train, c_outer_test = loyo_masks(
            concentration_rows["Year"], concentration_rows["PhysicalEventID"], held_year
        )
        v_outer_train, v_outer_test = loyo_masks(
            events["Year"], events["PhysicalEventID"], held_year
        )
        c_train = c_outer_train & concentration_rows["y_logC"].notna()
        c_test = c_outer_test & concentration_rows["y_logC"].notna()
        v_train = v_outer_train & events["y_logV"].notna()
        v_test = v_outer_test & events["y_logV"].notna()
        legacy_base = ["SampleID", "Date", "Year", "Treatment", "Analyte", "Irrigation", "Rep"]
        c_load = pd.DataFrame(columns=[
            "PhysicalEventID", "Year", "Treatment", "Analyte", "pointC", "loC", "hiC"
        ])
        v_load = pd.DataFrame(columns=["PhysicalEventID", "pointV", "loV", "hiV"])

        if c_test.any():
            Xc_train = X_c.loc[c_train].reset_index(drop=True)
            yc_train = concentration_rows.loc[c_train, "y_logC"].reset_index(drop=True)
            gc_train = concentration_rows.loc[c_train, "PhysicalEventID"].reset_index(drop=True)
            wc_train = concentration_rows.loc[c_train, "event_weight"].reset_index(drop=True)
            Xc_test = X_c.loc[c_test]
            yc_test = concentration_rows.loc[c_test, "y_logC"].to_numpy(dtype=float)
            (
                model_c, q_c, n_cal_c,
                c_signed_residuals, c_sampling_probabilities,
            ) = fit_conformal_model(
                Xc_train, yc_train, cat_c, gc_train, args.alpha, args.calibration,
                args.calib_size, args.seed + held_year, params, wc_train,
            )
            pred_c, lo_c, hi_c = predict_log_interval(model_c, q_c, Xc_test)
            metrics_records.append(metric_record(
                held_year, "logC", yc_test, pred_c, lo_c, hi_c, q_c, n_cal_c,
                args.mode, args.calibration,
            ))
            if args.feature_importance:
                importance_c.append(pd.DataFrame({
                    "Year_Test": held_year, "feature": features_c,
                    "importance": model_c.get_feature_importance(),
                }))
            c_base = concentration_rows.loc[
                c_test, [c for c in legacy_base if c in concentration_rows]
            ].copy()
            for column in legacy_base:
                if column not in c_base.columns:
                    c_base[column] = np.nan
            c_base["PhysicalEventID"] = concentration_rows.loc[c_test, "PhysicalEventID"].to_numpy()
            c_base["ConcentrationObservationID"] = concentration_rows.loc[c_test, "ConcentrationObservationID"].to_numpy()
            c_base["Analyte"] = concentration_rows.loc[c_test, analysis_analyte_column].astype(str).to_numpy()
            c_base["Target"] = "Result_mg_L"
            c_base["y_true"] = inverse_log1p(yc_test)
            c_base["y_pred"] = inverse_log1p(pred_c)
            c_base["pi_low"] = inverse_log1p(lo_c)
            c_base["pi_high"] = inverse_log1p(hi_c)
            c_base["UnitOfAnalysis"] = "analyte_row"
            c_base["mode"] = args.mode
            c_base["calibration"] = args.calibration
            prediction_records.append(c_base)
            c_load = concentration_rows.loc[c_test, [
                "PhysicalEventID", "Year", "Treatment", "Rep", analysis_analyte_column,
                "ConcentrationObservationID", "SampleMethod"
            ]].copy()
            if analysis_analyte_column != "Analyte":
                c_load = c_load.rename(columns={analysis_analyte_column: "Analyte"})
            c_load["pointC"] = inverse_log1p(pred_c)
            c_load["loC"] = lo_c
            c_load["hiC"] = hi_c
        else:
            warn(f"Held-out year {held_year} has no observed concentration targets; logC metrics omitted.")

        if v_test.any():
            Xv_train = X_v.loc[v_train].reset_index(drop=True)
            yv_train = events.loc[v_train, "y_logV"].reset_index(drop=True)
            gv_train = events.loc[v_train, "PhysicalEventID"].reset_index(drop=True)
            wv_train = events.loc[v_train, "event_weight"].reset_index(drop=True)
            Xv_test = X_v.loc[v_test]
            yv_test = events.loc[v_test, "y_logV"].to_numpy(dtype=float)
            (
                model_v, q_v, n_cal_v,
                v_signed_residuals, v_sampling_probabilities,
            ) = fit_conformal_model(
                Xv_train, yv_train, cat_v, gv_train, args.alpha, args.calibration,
                args.calib_size, args.seed + 1000 + held_year, params, wv_train,
            )
            pred_v, lo_v, hi_v = predict_log_interval(model_v, q_v, Xv_test)
            metrics_records.append(metric_record(
                held_year, "logV", yv_test, pred_v, lo_v, hi_v, q_v, n_cal_v,
                args.mode, args.calibration,
            ))
            if args.feature_importance:
                importance_v.append(pd.DataFrame({
                    "Year_Test": held_year, "feature": features_v,
                    "importance": model_v.get_feature_importance(),
                }))
            v_base = events.loc[v_test, [c for c in legacy_base if c in events]].copy()
            for column in legacy_base:
                if column not in v_base.columns:
                    v_base[column] = np.nan
            v_base["PhysicalEventID"] = events.loc[v_test, "PhysicalEventID"].to_numpy()
            v_base["VolumeObservationID"] = events.loc[v_test, "VolumeObservationID"].to_numpy()
            v_base["Target"] = "Volume_L"
            v_base["y_true"] = inverse_log1p(yv_test)
            v_base["y_pred"] = inverse_log1p(pred_v)
            v_base["pi_low"] = inverse_log1p(lo_v)
            v_base["pi_high"] = inverse_log1p(hi_v)
            v_base["UnitOfAnalysis"] = "volume_observation"
            v_base["mode"] = args.mode
            v_base["calibration"] = args.calibration
            prediction_records.append(v_base)
            v_load = events.loc[v_test, ["PhysicalEventID", "MeasureMethod"]].copy()
            v_load["pointV"] = inverse_log1p(pred_v)
            v_load["loV"] = lo_v
            v_load["hiV"] = hi_v
        else:
            warn(f"Held-out year {held_year} has no observed event-volume targets; logV metrics omitted.")

        # Resolve concentration and volume only after every eligible row has a draw.
        paired_events = set(c_load["PhysicalEventID"]) & set(v_load["PhysicalEventID"])
        if paired_events and int(args.draws) > 0:
            c_source = c_load.loc[c_load["PhysicalEventID"].isin(paired_events)].reset_index(drop=True)
            v_source = v_load.loc[v_load["PhysicalEventID"].isin(paired_events)].reset_index(drop=True)
            point_ledger, annual_points, _ = point_load_products(
                c_source,
                v_source,
                concentration_value="pointC",
                volume_value="pointV",
                concentration_resolution=args.concentration_resolution,
                volume_resolution=args.volume_resolution,
                method_priority=args.method_priority,
            )
            event_point_ledger_records.append(point_ledger)
            annual_point_records.append(annual_points)
            c_draw = inverse_log1p(empirical_calibration_residual_draws(
                np.log1p(c_source["pointC"].to_numpy(dtype=float)),
                c_signed_residuals,
                c_sampling_probabilities,
                args.draws,
                rng,
            ))
            v_draw = inverse_log1p(empirical_calibration_residual_draws(
                np.log1p(v_source["pointV"].to_numpy(dtype=float)),
                v_signed_residuals,
                v_sampling_probabilities,
                args.draws,
                rng,
            ))
            c_long = c_source.loc[c_source.index.repeat(int(args.draws))].reset_index(drop=True)
            c_long["Draw"] = np.tile(np.arange(int(args.draws)), len(c_source))
            c_long["Concentration_mg_L"] = c_draw.reshape(-1)
            c_resolved = resolve_prediction_draws(
                c_long,
                group_columns=[
                    "PhysicalEventID", "Year", "Treatment", "Rep", "Analyte", "Draw"
                ],
                value_column="Concentration_mg_L",
                method=args.concentration_resolution,
                method_column="SampleMethod" if "SampleMethod" in c_long else None,
                method_priority=args.method_priority,
            )
            v_long = v_source.loc[v_source.index.repeat(int(args.draws))].reset_index(drop=True)
            v_long["Draw"] = np.tile(np.arange(int(args.draws)), len(v_source))
            v_long["Volume_L"] = v_draw.reshape(-1)
            v_resolved = resolve_prediction_draws(
                v_long,
                group_columns=["PhysicalEventID", "Draw"],
                value_column="Volume_L",
                method=args.volume_resolution,
                method_column="MeasureMethod",
                method_priority=args.method_priority,
            )
            ledger = build_event_analyte_load_ledger(c_resolved, v_resolved)
            ledger["AnnualLoad_mg"] = ledger["Load_kg"] * 1_000_000.0
            if ledger.duplicated(["PhysicalEventID", "Analyte", "Draw"]).any():
                raise AssertionError("PhysicalEventID x Analyte x Draw is not unique.")
            event_ledger_records.append(ledger)
            _, annual_treatment_means = aggregate_replicate_mean(
                ledger,
                value_column="AnnualLoad_mg",
                group_columns=["Year", "Treatment", "Analyte"],
                draw_column="Draw",
                plot_total_column="PlotAnnualLoad_mg",
                treatment_mean_column="AnnualLoad_mg",
            )
            annual_draw_records.append(annual_treatment_means)

        progress.set_postfix_str(
            f"year={held_year} C={int(c_test.sum())} Vevents={int(v_test.sum())} "
            f"paired_events={len(paired_events)} s={time.time() - started:.1f}"
        )

    metrics = pd.DataFrame(metrics_records).sort_values(["Target", "Year_Test"])
    legacy_prediction_columns = legacy_base + [
        "Target", "y_true", "y_pred", "pi_low", "pi_high", "PhysicalEventID",
        "ConcentrationObservationID", "VolumeObservationID",
        "UnitOfAnalysis", "mode", "calibration",
    ]
    predictions = (
        pd.concat(prediction_records, ignore_index=True).reindex(columns=legacy_prediction_columns)
        if prediction_records else pd.DataFrame(columns=legacy_prediction_columns)
    )
    annual_draws = (
        pd.concat(annual_draw_records, ignore_index=True)
        if annual_draw_records else pd.DataFrame(columns=["Year", "Treatment", "Analyte", "Draw", "AnnualLoad_mg"])
    )
    annual_points = (
        pd.concat(annual_point_records, ignore_index=True)
        if annual_point_records else pd.DataFrame(
            columns=["Year", "Treatment", "Analyte", "AnnualLoad_kg", "AnnualLoad_mg"]
        )
    )
    event_point_ledger = (
        pd.concat(event_point_ledger_records, ignore_index=True)
        if event_point_ledger_records else pd.DataFrame(
            columns=["PhysicalEventID", "Analyte", "Concentration_mg_L", "Volume_L", "Load_kg"]
        )
    )
    event_ledger = (
        pd.concat(event_ledger_records, ignore_index=True)
        if event_ledger_records else pd.DataFrame(
            columns=["PhysicalEventID", "Analyte", "Draw", "Concentration_mg_L", "Volume_L", "Load_kg"]
        )
    )
    if not event_ledger.empty and event_ledger.duplicated(
        ["PhysicalEventID", "Analyte", "Draw"]
    ).any():
        raise AssertionError("ML event-analyte-draw ledger is not unique.")
    if not event_point_ledger.empty and event_point_ledger.duplicated(
        ["PhysicalEventID", "Analyte"]
    ).any():
        raise AssertionError("ML event-analyte point ledger is not unique.")
    metrics.to_csv(output_dir / "cv_metrics_by_year.csv", index=False)
    predictions.to_csv(output_dir / "loyo_predictions_row_level.csv", index=False)
    event_ledger.to_csv(output_dir / "event_analyte_draw_ledger_ml_v3p4.csv", index=False)
    event_point_ledger.to_csv(
        output_dir / "event_analyte_point_ledger_ml_v3p4.csv", index=False
    )
    annual_draws.to_csv(output_dir / "annual_load_draws_model_only_loyo.csv", index=False)
    annual_points.to_csv(
        output_dir / "annual_load_point_totals_model_only_loyo.csv", index=False
    )
    loyo_summary = summarize_annual_draws(annual_draws, args.alpha)
    if not loyo_summary.empty:
        loyo_summary = loyo_summary.merge(
            annual_points[["Year", "Treatment", "Analyte", "AnnualLoad_mg"]].rename(
                columns={"AnnualLoad_mg": "point_total_mg"}
            ),
            on=["Year", "Treatment", "Analyte"],
            how="left",
            validate="one_to_one",
        )
    loyo_summary.to_csv(
        output_dir / "annual_load_summary_model_only_loyo.csv", index=False
    )

    diagnostics = predictions.copy()
    diagnostics["residual_observed_minus_predicted"] = (
        as_numeric(diagnostics["y_true"]) - as_numeric(diagnostics["y_pred"])
    )
    diagnostics["discrepancy_standardization_group"] = np.where(
        diagnostics["Target"].eq("Result_mg_L"),
        diagnostics["Target"].astype(str) + "|" + diagnostics["Analyte"].astype(str),
        diagnostics["Target"].astype(str),
    )
    residual_scale = diagnostics.groupby(
        "discrepancy_standardization_group"
    )["residual_observed_minus_predicted"].transform("std")
    diagnostics["standardized_discrepancy"] = (
        diagnostics["residual_observed_minus_predicted"] / residual_scale.replace(0, np.nan)
    )
    diagnostics.to_csv(output_dir / "row_level_residual_diagnostics.csv", index=False)
    original_performance = (
        diagnostics.groupby(["Target", "Year"], dropna=False)
        .apply(
            lambda group: pd.Series({
                "n": len(group),
                "rmse_original_units": float(np.sqrt(np.mean(np.square(group["residual_observed_minus_predicted"])))),
                "nrmse_mean_abs": float(
                    np.sqrt(np.mean(np.square(group["residual_observed_minus_predicted"])))
                    / max(as_numeric(group["y_true"]).abs().mean(), 1e-12)
                ),
                "interval_coverage": float(
                    ((as_numeric(group["y_true"]) >= as_numeric(group["pi_low"]))
                     & (as_numeric(group["y_true"]) <= as_numeric(group["pi_high"]))).mean()
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    original_performance.to_csv(output_dir / "loyo_performance_original_units.csv", index=False)

    coverage = (
        predictions.assign(covered=lambda frame: (
            as_numeric(frame["y_true"]) >= as_numeric(frame["pi_low"])
        ) & (
            as_numeric(frame["y_true"]) <= as_numeric(frame["pi_high"])
        ))
        .groupby(["Target", "Year"], as_index=False)
        .agg(n=("covered", "size"), coverage=("covered", "mean"))
    )
    coverage.to_csv(output_dir / "cv_interval_coverage_by_year.csv", index=False)

    fi_c = feature_importance_summary(importance_c)
    fi_v = feature_importance_summary(importance_v)
    fi_c.to_csv(output_dir / "feature_importance_logC.csv", index=False)
    fi_v.to_csv(output_dir / "feature_importance_logV.csv", index=False)

    if args.impute_missing:
        refit_and_impute(
            df, concentration_rows, events, X_c, X_v, cat_c, cat_v,
            features_c, features_v, event_key, output_dir, args, params,
        )
    else:
        warn("Imputation disabled; compatibility files derived from full-data models were not regenerated.")

    manifest = {
        "workflow_version": WORKFLOW_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "event_unit": "PhysicalEventID",
        "physical_event_key": PHYSICAL_EVENT_KEY,
        "comparison_analyte_field": analysis_analyte_column,
        "years": years,
        "concentration_resolution": args.concentration_resolution,
        "volume_resolution": args.volume_resolution,
        "method_priority": args.method_priority,
        "event_balanced_training_weights": args.event_balanced_weights,
        "heldout_year_excluded": True,
        "calibration_split_unit": "PhysicalEventID",
        "primary_ml_central_estimate": "mean_of_replicate_annual_plot_totals",
        "primary_ml_central_estimate_uses_draw_median": False,
        "primary_prediction_scope": (
            "all eligible concentration rows and all physical events, "
            "observed and missing"
        ),
        "observed_values_role": (
            "training, held-out evaluation, and reference markers only; "
            "never substituted into primary modeled products"
        ),
        "primary_uses_observed_value_substitution": False,
        "observed_plus_imputed_sensitivity_requested": bool(
            args.observed_plus_imputed_sensitivity
        ),
        "event_analyte_point_uniqueness_asserted": True,
        "interval_level": 1.0 - float(args.alpha),
        "row_interval_type": "physical-event-grouped split-conformal prediction interval",
        "interval_evaluation": "outer leave-one-year-out",
        "annual_interval_type": "Monte Carlo empirical calibration-residual prediction interval",
        "annual_interval_is_parameter_confidence_interval": False,
        "monte_carlo_propagation": "weighted_resampling_of_signed_log_scale_calibration_residuals",
        "uniform_between_conformal_bounds_used": False,
        "annual_reporting_unit": "mean_per_treatment_plot",
        "annual_aggregation_hierarchy": PHYSICAL_EVENT_CONFIG[
            "annual_reporting"
        ]["hierarchy"],
        "seasonal_stir": PHYSICAL_EVENT_CONFIG["seasonal_stir"],
        "storm_handling": PHYSICAL_EVENT_CONFIG["storm_handling"],
        "furrow_tire_compaction": {
            "feature": "FurrowTireCompaction",
            "scope": "concentration and runoff volume",
            "included_in_logV": "FurrowTireCompaction" in features_v,
            "included_in_logC": "FurrowTireCompaction" in features_c,
            "compacted_events": int(
                compaction_event_audit["FurrowTireCompaction"].sum()
            ),
            "compacted_events_with_genuine_volume": int(
                (
                    compaction_event_audit["FurrowTireCompaction"].eq(1)
                    & compaction_event_audit[
                        "has_genuine_volume_observation"
                    ]
                ).sum()
            ),
            "source_file": PHYSICAL_EVENT_CONFIG[
                "furrow_tire_compaction"
            ]["source_file"],
        },
        "feature_contract": PHYSICAL_EVENT_CONFIG["ml_feature_contract"],
        "full_record_reconstruction_requested": bool(args.impute_missing),
        "full_record_prediction_completed": bool(args.impute_missing),
        "full_record_models_saved": bool(args.impute_missing and args.save_models),
        "calibration_residual_audit_files": (
            [
                "calibration_residual_distribution_logC.csv",
                "calibration_residual_distribution_logV.csv",
            ]
            if args.impute_missing
            else []
        ),
        "final_model_fit_deferred": not bool(args.impute_missing),
    }
    (output_dir / "run_manifest_ml_v3p4_physical_event.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.figures:
        generate_postprocess_figure_suite(repo, output_dir, figure_dir, data_path, args)
    print(f"[DONE] Revised physical-event workflow outputs written to {output_dir}")


if __name__ == "__main__":
    main()
