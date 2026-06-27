#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Event-level v2 CatBoost workflow for the Kerbel monitoring record.

The concentration model uses analyte rows. The runoff-volume model uses one
row per physical volume event, keyed (when available) by:

    Date + Year + Irrigation + Rep + Treatment + SampleID + MeasureMethod

The default ``reconstruction`` mode intentionally permits same-event observed
co-outcomes. It is a conditional reconstruction/imputation model for the
historical Kerbel record, not a transferable forecasting model. The optional
``strict_prediction`` mode removes those co-outcomes and requires an explicit
output directory so it cannot overwrite the primary reconstruction outputs.

Default inputs and outputs remain compatible with the original workflow:

    input:  out/wq_cleaned.csv
    output: out/ml_catboost_conformal_loyo/
    figures: figs/ml_catboost_conformal_loyo/
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


EVENT_KEY_PREFERRED = [
    "Date", "Year", "Irrigation", "Rep", "Treatment", "SampleID", "MeasureMethod"
]
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
        if (cur / "README.md").exists() and (cur / "out").exists() and (cur / "figs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find the repository root containing README.md, out/, and figs/. "
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


def first_nonmissing(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


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
    key = [column for column in EVENT_KEY_PREFERRED if column in df.columns]
    missing = [column for column in EVENT_KEY_PREFERRED if column not in df.columns]
    if missing:
        warn(f"Event key degraded because these columns are unavailable: {missing}. Actual key: {key}")
    if not key:
        raise ValueError("No event-key columns are available.")
    return key


def event_id_series(df: pd.DataFrame, key_columns: Sequence[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for column in key_columns:
        values = df[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            text = values.dt.strftime("%Y-%m-%dT%H:%M:%S").astype("string")
        else:
            text = values.astype("string").str.strip()
        text = text.fillna("<MISSING>")
        parts.append(column + "=" + text)
    event_id = parts[0]
    for part in parts[1:]:
        event_id = event_id + "|" + part
    return event_id.astype(str)


def clean_analyte_feature_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return text or "UNKNOWN"


def build_event_table(
    df: pd.DataFrame,
    event_key: Sequence[str],
    reconstruction: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse analyte rows to physical events and construct co-outcome summaries."""
    work = df.copy()
    work["EventVolumeID"] = event_id_series(work, event_key)
    work["Volume"] = as_numeric(work["Volume"])

    conflict_records: list[dict] = []
    event_records: list[dict] = []

    numeric_candidates = [
        "Year", "Irrigation", "Rep", "CumAll_STIR_toDate", "Season_STIR_toDate",
        "residue_prop", "Residue_PercentCover", "Residue_DryMass_kg_m2",
        "Inflow_Volume", "DayOfYear", "DaysSincePlant", "DaysUntilHarvest",
    ]
    categorical_candidates = [
        "Treatment", "Crop", "previous_crop", "PreviousCrop", "Previous_Crop",
        "PrevCrop", "prev_crop", "SeasonYear", "MeasureMethod", "FlumeMethod",
        "IrrMethod", "IrrigationInfrastructure",
    ]

    grouped = work.groupby("EventVolumeID", dropna=False, sort=False)
    for event_id, group in grouped:
        volume_values = np.sort(group["Volume"].dropna().unique())
        record: dict = {"EventVolumeID": event_id}
        for column in event_key:
            record[column] = first_nonmissing(group[column])
        record["_event_n_analyte_rows"] = int(len(group))
        record["_event_n_volume_nonmissing"] = int(group["Volume"].notna().sum())
        record["_event_n_unique_volume"] = int(len(volume_values))
        record["Volume"] = float(np.median(volume_values)) if len(volume_values) else np.nan
        record["NoRunoff"] = bool(true_mask(group["NoRunoff"]).any()) if "NoRunoff" in group else False

        if len(volume_values) > 1:
            conflict = {column: record.get(column, np.nan) for column in event_key}
            conflict.update({
                "EventVolumeID": event_id,
                "n_analyte_rows": int(len(group)),
                "n_unique_nonmissing_volume": int(len(volume_values)),
                "volume_values": ";".join(f"{value:.12g}" for value in volume_values),
                "resolution": "median_nonmissing_volume",
            })
            conflict_records.append(conflict)

        for column in numeric_candidates:
            if column in group.columns and column not in event_key:
                values = as_numeric(group[column]).dropna()
                record[column] = float(values.median()) if not values.empty else np.nan
        for column in categorical_candidates:
            if column in group.columns and column not in event_key:
                record[column] = first_nonmissing(group[column])
        event_records.append(record)

    events = pd.DataFrame(event_records)

    # Robust event/analyte summaries: median log concentration for duplicates.
    concentration_features = pd.DataFrame({"EventVolumeID": events["EventVolumeID"]})
    if reconstruction:
        observed = work.loc[as_numeric(work["Result_mg_L"]).ge(0)].copy()
        observed["_event_logC"] = np.log1p(as_numeric(observed["Result_mg_L"]))
        if not observed.empty:
            overall = (
                observed.groupby("EventVolumeID", dropna=False)["_event_logC"]
                .agg(event_mean_logC_all="mean", event_max_logC_all="max", event_n_conc_obs="size")
                .reset_index()
            )
            concentration_features = concentration_features.merge(overall, on="EventVolumeID", how="left")
            analyte_column = "analyte_abbr" if "analyte_abbr" in observed.columns else "Analyte"
            robust = (
                observed.dropna(subset=[analyte_column])
                .groupby(["EventVolumeID", analyte_column], dropna=False)["_event_logC"]
                .median()
                .reset_index()
            )
            robust["_feature"] = "event_logC_" + robust[analyte_column].map(clean_analyte_feature_name)
            pivot = robust.pivot(index="EventVolumeID", columns="_feature", values="_event_logC").reset_index()
            concentration_features = concentration_features.merge(pivot, on="EventVolumeID", how="left")
        else:
            concentration_features["event_mean_logC_all"] = np.nan
            concentration_features["event_max_logC_all"] = np.nan
            concentration_features["event_n_conc_obs"] = 0

    events = events.merge(concentration_features, on="EventVolumeID", how="left")
    events["y_logV"] = safe_log1p(events["Volume"])

    conflict_columns = list(event_key) + [
        "EventVolumeID", "n_analyte_rows", "n_unique_nonmissing_volume",
        "volume_values", "resolution",
    ]
    conflicts = pd.DataFrame(conflict_records, columns=conflict_columns)
    work["y_logC"] = safe_log1p(work["Result_mg_L"])
    work["y_logV"] = safe_log1p(work["Volume"])
    return events, conflicts, work


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
        "RLMDL_Assumed", "MDL_mg_L", "RL_mg_L", "Result_lod_mg_L",
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
        "DaysSincePlant", "DaysUntilHarvest",
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
    if mode == "reconstruction":
        volume_desired.extend(
            column for column in events.columns
            if column.startswith("event_logC_") or column in {
                "event_mean_logC_all", "event_max_logC_all", "event_n_conc_obs"
            }
        )

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


def audit_summary(
    df: pd.DataFrame,
    events: pd.DataFrame,
    conflicts: pd.DataFrame,
    event_key: Sequence[str],
    concentration_features: Sequence[str],
    volume_features: Sequence[str],
    residue_cover: str | None,
    previous_crop: str | None,
    infrastructure_source: str,
    mode: str,
) -> pd.DataFrame:
    rows_with_volume = int(df["Volume"].notna().sum())
    observed_events = int(events["Volume"].notna().sum())
    previous_details = "not_found"
    if previous_crop:
        levels = sorted(df[previous_crop].dropna().astype(str).unique().tolist())
        previous_details = f"missing_fraction={df[previous_crop].isna().mean():.6f}; levels={' | '.join(levels)}"
    residue_details = "not_found"
    if residue_cover:
        residue_details = f"missing_fraction={df[residue_cover].isna().mean():.6f}"
    metrics = [
        ("mode", mode, "conditional reconstruction unless strict_prediction"),
        ("event_key", " + ".join(event_key), "actual key used; missing values retained"),
        ("original_analyte_rows", len(df), "all input rows"),
        ("rows_with_nonmissing_volume", rows_with_volume, "analyte-level rows before collapse"),
        ("unique_event_volume_groups", len(events), "all event groups including missing targets"),
        ("unique_event_volume_groups_with_nonmissing_volume", observed_events, "volume training targets"),
        ("duplicated_volume_rows_removed", rows_with_volume - observed_events, "nonmissing analyte rows minus observed events"),
        ("maximum_rows_per_event_volume_group", int(events["_event_n_analyte_rows"].max()), "analyte rows"),
        ("conflicting_volume_groups", len(conflicts), "groups with >1 nonmissing Volume"),
        ("missing_SampleID_rows_retained", int(df["SampleID"].isna().sum()) if "SampleID" in df else 0, "dropna=False semantics"),
        ("missing_SampleID_event_groups_retained", int(events["SampleID"].isna().sum()) if "SampleID" in events else 0, "event rows"),
        ("predictors_used_logC", ";".join(concentration_features), f"n={len(concentration_features)}"),
        ("predictors_used_logV", ";".join(volume_features), f"n={len(volume_features)}"),
        ("residue_column_used", residue_cover or "not_found", residue_details),
        ("previous_crop_used", str(bool(previous_crop)), previous_details),
        ("irrigation_infrastructure_source", infrastructure_source, "IrrMethod primary; year fallback <=2022/>=2023"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "details"])


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


def residual_quantile(residuals: np.ndarray, alpha: float) -> float:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite calibration residuals were available.")
    # This intentionally retains the v1 random-split quantile definition.
    return float(np.quantile(values, 1.0 - float(alpha)))


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
) -> tuple[CatBoostRegressor, float, int]:
    if len(y) < 4:
        raise ValueError(f"At least four training observations are required; got {len(y)}.")
    if calibration == "random":
        train_positions, calibration_positions = train_test_split(
            np.arange(len(y)), test_size=float(calibration_size), random_state=int(seed)
        )
        model = make_model(params, seed, X, categorical)
        model.fit(X.iloc[train_positions], y.iloc[train_positions])
        calibration_prediction = np.asarray(model.predict(X.iloc[calibration_positions]), dtype=float)
        residuals = np.abs(y.iloc[calibration_positions].to_numpy(dtype=float) - calibration_prediction)
        return model, residual_quantile(residuals, alpha), int(len(residuals))

    unique_groups = sorted(as_numeric(groups).dropna().unique().tolist())
    if len(unique_groups) < 2:
        warn("Grouped-year calibration needs at least two training years; falling back to random split.")
        return fit_conformal_model(
            X, y, categorical, groups, alpha, "random", calibration_size, seed, params
        )

    residual_blocks: list[np.ndarray] = []
    group_values = as_numeric(groups).to_numpy()
    for index, held_group in enumerate(unique_groups):
        calibration_mask = group_values == held_group
        training_mask = ~calibration_mask
        if calibration_mask.sum() == 0 or training_mask.sum() < 4:
            continue
        inner_model = make_model(params, seed + 10000 + index, X, categorical)
        inner_model.fit(X.iloc[np.flatnonzero(training_mask)], y.iloc[np.flatnonzero(training_mask)])
        prediction = np.asarray(inner_model.predict(X.iloc[np.flatnonzero(calibration_mask)]), dtype=float)
        residual_blocks.append(
            np.abs(y.iloc[np.flatnonzero(calibration_mask)].to_numpy(dtype=float) - prediction)
        )
    if not residual_blocks:
        raise ValueError("Grouped-year calibration produced no residuals.")
    residuals = np.concatenate(residual_blocks)
    final_model = make_model(params, seed + 20000, X, categorical)
    final_model.fit(X, y)
    return final_model, residual_quantile(residuals, alpha), int(len(residuals))


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


def uniform_log_draws(low: np.ndarray, high: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    lo = np.minimum(low, high)
    hi = np.maximum(low, high)
    return lo[:, None] + rng.random((len(lo), int(draws))) * (hi - lo)[:, None]


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
    event_key: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    metadata = {
        "target": target,
        "unit_of_analysis": "analyte_row" if target == "logC" else "event_volume",
        "alpha": float(args.alpha),
        "q_conformal": float(q),
        "feature_cols": list(features),
        "cat_cols": list(categorical),
        "cb_params": params,
        "mode": args.mode,
        "calibration": args.calibration,
        "event_key": list(event_key or []),
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "workflow": "ml_catboost_conformal_loyo_v2_eventlevel.py",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(path: Path, metadata_path: Path) -> tuple[CatBoostRegressor, dict]:
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing saved model or metadata: {path}, {metadata_path}")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model, json.loads(metadata_path.read_text(encoding="utf-8"))


def write_imputed_row_draws(
    output_path: Path,
    imputed: pd.DataFrame,
    draws: int,
    seed: int,
) -> None:
    """Preserve the v1 row-draw schema while sharing volume draws by event."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["_wq_idx", "Target", "draws"])
        if draws <= 0:
            return

        missing_c = imputed["Result_mg_L"].isna() & imputed["Result_mg_L_pi_low"].notna()
        for _, row in imputed.loc[missing_c].iterrows():
            log_draw = uniform_log_draws(
                np.array([np.log1p(max(float(row["Result_mg_L_pi_low"]), 0.0))]),
                np.array([np.log1p(max(float(row["Result_mg_L_pi_high"]), 0.0))]),
                draws,
                rng,
            )[0]
            values = inverse_log1p(log_draw)
            writer.writerow([int(row["_wq_idx"]), "Result_mg_L", ",".join(f"{value:.12g}" for value in values)])

        event_draw_strings: dict[str, str] = {}
        missing_v = imputed["Volume"].isna() & imputed["Volume_pi_low"].notna()
        for _, row in imputed.loc[missing_v].iterrows():
            event_id = str(row["EventVolumeID"])
            if event_id not in event_draw_strings:
                log_draw = uniform_log_draws(
                    np.array([np.log1p(max(float(row["Volume_pi_low"]), 0.0))]),
                    np.array([np.log1p(max(float(row["Volume_pi_high"]), 0.0))]),
                    draws,
                    rng,
                )[0]
                values = inverse_log1p(log_draw)
                event_draw_strings[event_id] = ",".join(f"{value:.12g}" for value in values)
            writer.writerow([int(row["_wq_idx"]), "Volume", event_draw_strings[event_id]])


def imputed_annual_summary(
    df: pd.DataFrame,
    draws: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Annual load summaries using observed values and model-filled missing values."""
    work = df.copy()
    work["Year"] = as_numeric(work["Year"])
    no_runoff = true_mask(work["NoRunoff"]) if "NoRunoff" in work.columns else pd.Series(False, index=work.index)
    usable = no_runoff | (work["Result_mg_L_filled"].notna() & work["Volume_filled"].notna())
    work = work.loc[usable & work["Year"].notna() & work["Treatment"].notna() & work["Analyte"].notna()].copy()
    work["_no_runoff"] = no_runoff.loc[work.index]
    rng = np.random.default_rng(seed)
    D = int(draws)
    q_low, q_high = alpha / 2.0, 1.0 - alpha / 2.0
    volume_draw_cache: dict[str, np.ndarray] = {}
    records: list[dict] = []

    for (year, treatment, analyte), group in work.groupby(["Year", "Treatment", "Analyte"], dropna=False):
        point_load = np.where(
            group["_no_runoff"].to_numpy(),
            0.0,
            as_numeric(group["Result_mg_L_filled"]).to_numpy() * as_numeric(group["Volume_filled"]).to_numpy(),
        )
        if D <= 0:
            total = float(np.nansum(point_load))
            records.append({
                "Year": int(year), "Treatment": treatment, "Analyte": analyte,
                "mean": total, "median": total, "low": total, "high": total,
                "n_events": int(len(group)), "n_draws": 0,
            })
            continue

        annual = np.zeros(D, dtype=float)
        for _, row in group.iterrows():
            if bool(row["_no_runoff"]):
                continue
            if pd.notna(row["Result_mg_L"]):
                c_draw = np.full(D, float(row["Result_mg_L"]))
            else:
                lo_c, hi_c = float(row["Result_mg_L_pi_low"]), float(row["Result_mg_L_pi_high"])
                c_draw = inverse_log1p(uniform_log_draws(
                    np.array([np.log1p(max(lo_c, 0.0))]),
                    np.array([np.log1p(max(hi_c, 0.0))]), D, rng,
                )[0])

            if pd.notna(row["Volume"]):
                v_draw = np.full(D, float(row["Volume"]))
            else:
                event_id = str(row["EventVolumeID"])
                if event_id not in volume_draw_cache:
                    lo_v, hi_v = float(row["Volume_pi_low"]), float(row["Volume_pi_high"])
                    volume_draw_cache[event_id] = inverse_log1p(uniform_log_draws(
                        np.array([np.log1p(max(lo_v, 0.0))]),
                        np.array([np.log1p(max(hi_v, 0.0))]), D, rng,
                    )[0])
                v_draw = volume_draw_cache[event_id]
            annual += c_draw * v_draw

        records.append({
            "Year": int(year), "Treatment": treatment, "Analyte": analyte,
            "mean": float(np.mean(annual)), "median": float(np.median(annual)),
            "low": float(np.quantile(annual, q_low)), "high": float(np.quantile(annual, q_high)),
            "n_events": int(len(group)), "n_draws": D,
        })
    return pd.DataFrame(records)


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
    else:
        model_c, q_c, _ = fit_conformal_model(
            X_c.loc[observed_c].reset_index(drop=True),
            concentration_rows.loc[observed_c, "y_logC"].reset_index(drop=True),
            cat_c,
            concentration_rows.loc[observed_c, "Year"].reset_index(drop=True),
            args.alpha, args.calibration, args.calib_size, args.seed + 50001, params,
        )
        model_v, q_v, _ = fit_conformal_model(
            X_v.loc[observed_v].reset_index(drop=True),
            events.loc[observed_v, "y_logV"].reset_index(drop=True),
            cat_v,
            events.loc[observed_v, "Year"].reset_index(drop=True),
            args.alpha, args.calibration, args.calib_size, args.seed + 60001, params,
        )
        if args.save_models:
            save_model(model_c, c_model_path, c_meta_path, "logC", q_c, features_c, cat_c, params, args)
            save_model(model_v, v_model_path, v_meta_path, "logV", q_v, features_v, cat_v, params, args, event_key)

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
    event_predictions["unit_of_analysis"] = "event_volume"

    imputed = df.copy()
    if "EventVolumeID" not in imputed.columns:
        imputed["EventVolumeID"] = event_id_series(imputed, event_key)
    imputed = imputed.merge(concentration_prediction, on="_wq_idx", how="left")
    volume_map = event_predictions[[
        "EventVolumeID", "Volume_model_pred", "Volume_model_pi_low", "Volume_model_pi_high"
    ]]
    imputed = imputed.merge(volume_map, on="EventVolumeID", how="left")

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

    imputed.to_csv(output_dir / "wq_cleaned_ml_imputed.csv", index=False)
    # Retained filename for downstream and saved-model workflows.
    imputed.to_csv(output_dir / "predictions_from_saved_models.csv", index=False)
    event_predictions.to_csv(output_dir / "event_volume_predictions.csv", index=False)
    write_imputed_row_draws(
        output_dir / "imputed_row_draws.csv", imputed, int(args.impute_draws), args.seed + 70001
    )
    annual_imputed = imputed_annual_summary(
        imputed, int(args.impute_draws), float(args.alpha), args.seed + 80001
    )
    annual_imputed.to_csv(output_dir / "annual_load_summary_imputed.csv", index=False)
    return imputed, event_predictions


def plot_legacy_cv_rmse(metrics: pd.DataFrame, figure_dir: Path) -> None:
    if metrics.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 5))
    for target in ["logC", "logV"]:
        subset = metrics.loc[metrics["Target"] == target].sort_values("Year_Test")
        if not subset.empty:
            axis.plot(subset["Year_Test"], subset["RMSE"], marker="o", label=target)
    axis.set(title="LOYO conditional-reconstruction RMSE", xlabel="Held-out year", ylabel="RMSE (log1p scale)")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "cv_rmse_by_year.png", dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--data", default=None, help="Default: <repo>/out/wq_cleaned.csv")
    parser.add_argument("--mode", choices=["reconstruction", "strict_prediction"], default="reconstruction")
    parser.add_argument("--calibration", choices=["random", "grouped-year"], default="random")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--out_subdir", default=None, help="Backward-compatible subdirectory under <repo>/out.")
    parser.add_argument("--fig_subdir", default=None, help="Backward-compatible subdirectory under <repo>/figs.")
    parser.add_argument("--model_subdir", default="models")

    parser.add_argument("--impute_missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--impute_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--impute_draws", type=int, default=2000)
    parser.add_argument("--save_models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_importance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fi_topk", type=int, default=25)

    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--fast_iterations", type=int, default=600)
    parser.add_argument("--fast_draws", type=int, default=300)
    return parser.parse_args()


def resolve_output_path(repo: Path, explicit: str | None, base: str, default_name: str) -> Path:
    if explicit:
        path = Path(explicit)
        return path.resolve() if path.is_absolute() else (repo / path).resolve()
    return (repo / base / default_name).resolve()


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
    output_dir = resolve_output_path(
        repo,
        args.output_dir,
        "out",
        args.out_subdir or "ml_catboost_conformal_loyo",
    )
    default_figure_name = args.fig_subdir or (
        "ml_catboost_conformal_loyo_strict_prediction"
        if args.mode == "strict_prediction" else "ml_catboost_conformal_loyo"
    )
    figure_dir = resolve_output_path(repo, args.fig_dir, "figs", default_figure_name)
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
    df, infrastructure_source, previous_crop, residue_cover = parse_and_engineer(raw)
    event_key = resolve_event_key(df)
    events, conflicts, df = build_event_table(df, event_key, args.mode == "reconstruction")

    no_runoff = true_mask(df["NoRunoff"]) if "NoRunoff" in df.columns else pd.Series(False, index=df.index)
    concentration_rows = df.loc[~no_runoff].copy()
    if args.exclude_flagged:
        flagged = pd.Series(False, index=concentration_rows.index)
        for column in ["Flag", "Inflow_Flag"]:
            if column in concentration_rows.columns:
                flagged |= nonblank_mask(concentration_rows[column])
        concentration_rows = concentration_rows.loc[~flagged].copy()

    features_c, features_v = select_features(
        df, events, args.mode, previous_crop, residue_cover,
        infrastructure_source, args.dry_mass_missing_threshold,
    )
    if not features_c or not features_v:
        raise ValueError("Feature selection produced an empty model feature set.")
    X_c, cat_c = prepare_feature_frame(concentration_rows, features_c)
    X_v, cat_v = prepare_feature_frame(events, features_v)

    # Audits are written before fitting so the physical-unit correction remains inspectable.
    events.to_csv(output_dir / "event_volume_training_table.csv", index=False)
    conflicts.to_csv(output_dir / "event_volume_audit_conflicts.csv", index=False)
    build_feature_audit(df, events, features_c, features_v).to_csv(
        output_dir / "feature_audit_summary.csv", index=False
    )
    summary_audit = audit_summary(
        df, events, conflicts, event_key, features_c, features_v,
        residue_cover, previous_crop, infrastructure_source, args.mode,
    )
    summary_audit.to_csv(output_dir / "event_volume_audit_summary.csv", index=False)

    print(
        f"[AUDIT] {len(df):,} analyte rows -> {len(events):,} event groups; "
        f"{df['Volume'].notna().sum():,} nonmissing-volume rows -> "
        f"{events['Volume'].notna().sum():,} observed-volume events; conflicts={len(conflicts)}"
    )
    print(f"[AUDIT] Previous crop: {previous_crop or 'not found'}; residue: {residue_cover or 'not found'}")
    print(f"[AUDIT] Irrigation infrastructure source: {infrastructure_source}")

    params = catboost_params(args)
    if args.impute_only:
        if not args.impute_missing:
            raise ValueError("--impute_only requires --impute_missing.")
        refit_and_impute(
            df, concentration_rows, events, X_c, X_v, cat_c, cat_v,
            features_c, features_v, event_key, output_dir, args, params,
        )
        print("[DONE] Imputation regenerated from saved event-level models.")
        return

    years = sorted(int(value) for value in df["Year"].dropna().unique())
    if len(years) < 2:
        raise ValueError("LOYO cross-validation requires at least two years.")

    rng = np.random.default_rng(args.seed)
    metrics_records: list[dict] = []
    prediction_records: list[pd.DataFrame] = []
    annual_draw_records: list[pd.DataFrame] = []
    importance_c: list[pd.DataFrame] = []
    importance_v: list[pd.DataFrame] = []

    progress = tqdm(years, desc="LOYO event-level folds", unit="year", ncols=115)
    for held_year in progress:
        started = time.time()
        c_train = concentration_rows["Year"].ne(held_year) & concentration_rows["y_logC"].notna()
        c_test = concentration_rows["Year"].eq(held_year) & concentration_rows["y_logC"].notna()
        v_train = events["Year"].ne(held_year) & events["y_logV"].notna()
        v_test = events["Year"].eq(held_year) & events["y_logV"].notna()
        legacy_base = ["SampleID", "Date", "Year", "Treatment", "Analyte", "Irrigation", "Rep"]
        c_load = pd.DataFrame(columns=["EventVolumeID", "Year", "Treatment", "Analyte", "loC", "hiC"])
        v_load = pd.DataFrame(columns=["EventVolumeID", "loV", "hiV"])

        if c_test.any():
            Xc_train = X_c.loc[c_train].reset_index(drop=True)
            yc_train = concentration_rows.loc[c_train, "y_logC"].reset_index(drop=True)
            gc_train = concentration_rows.loc[c_train, "Year"].reset_index(drop=True)
            Xc_test = X_c.loc[c_test]
            yc_test = concentration_rows.loc[c_test, "y_logC"].to_numpy(dtype=float)
            model_c, q_c, n_cal_c = fit_conformal_model(
                Xc_train, yc_train, cat_c, gc_train, args.alpha, args.calibration,
                args.calib_size, args.seed + held_year, params,
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
            c_base["EventVolumeID"] = concentration_rows.loc[c_test, "EventVolumeID"].to_numpy()
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
                "EventVolumeID", "Year", "Treatment", "Analyte"
            ]].copy()
            c_load["loC"] = lo_c
            c_load["hiC"] = hi_c
        else:
            warn(f"Held-out year {held_year} has no observed concentration targets; logC metrics omitted.")

        if v_test.any():
            Xv_train = X_v.loc[v_train].reset_index(drop=True)
            yv_train = events.loc[v_train, "y_logV"].reset_index(drop=True)
            gv_train = events.loc[v_train, "Year"].reset_index(drop=True)
            Xv_test = X_v.loc[v_test]
            yv_test = events.loc[v_test, "y_logV"].to_numpy(dtype=float)
            model_v, q_v, n_cal_v = fit_conformal_model(
                Xv_train, yv_train, cat_v, gv_train, args.alpha, args.calibration,
                args.calib_size, args.seed + 1000 + held_year, params,
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
            v_base["EventVolumeID"] = events.loc[v_test, "EventVolumeID"].to_numpy()
            v_base["Target"] = "Volume_L"
            v_base["y_true"] = inverse_log1p(yv_test)
            v_base["y_pred"] = inverse_log1p(pred_v)
            v_base["pi_low"] = inverse_log1p(lo_v)
            v_base["pi_high"] = inverse_log1p(hi_v)
            v_base["UnitOfAnalysis"] = "event_volume"
            v_base["mode"] = args.mode
            v_base["calibration"] = args.calibration
            prediction_records.append(v_base)
            v_load = events.loc[v_test, ["EventVolumeID"]].copy()
            v_load["loV"] = lo_v
            v_load["hiV"] = hi_v
        else:
            warn(f"Held-out year {held_year} has no observed event-volume targets; logV metrics omitted.")

        # Annual propagation requires both held-out components for the event.
        paired = c_load.merge(v_load, on="EventVolumeID", how="inner", validate="many_to_one")
        if not paired.empty and int(args.draws) > 0:
            c_draw = inverse_log1p(uniform_log_draws(
                paired["loC"].to_numpy(), paired["hiC"].to_numpy(), args.draws, rng
            ))
            # Share one volume draw vector across all analyte rows from an event.
            event_v_draws: dict[str, np.ndarray] = {}
            load_draws = np.empty_like(c_draw)
            for row_position, row in paired.reset_index(drop=True).iterrows():
                event_id = str(row["EventVolumeID"])
                if event_id not in event_v_draws:
                    event_v_draws[event_id] = inverse_log1p(uniform_log_draws(
                        np.array([row["loV"]]), np.array([row["hiV"]]), args.draws, rng
                    )[0])
                load_draws[row_position, :] = c_draw[row_position, :] * event_v_draws[event_id]

            for (year, treatment, analyte), positions in paired.groupby(
                ["Year", "Treatment", "Analyte"], dropna=False
            ).groups.items():
                total = load_draws[paired.index.get_indexer(list(positions)), :].sum(axis=0)
                annual_draw_records.append(pd.DataFrame({
                    "Year": int(year), "Treatment": treatment, "Analyte": analyte,
                    "Draw": np.arange(int(args.draws), dtype=int), "AnnualLoad_mg": total,
                }))

        progress.set_postfix_str(
            f"year={held_year} C={int(c_test.sum())} Vevents={int(v_test.sum())} "
            f"paired={len(paired)} s={time.time() - started:.1f}"
        )

    metrics = pd.DataFrame(metrics_records).sort_values(["Target", "Year_Test"])
    legacy_prediction_columns = legacy_base + [
        "Target", "y_true", "y_pred", "pi_low", "pi_high", "EventVolumeID",
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
    metrics.to_csv(output_dir / "cv_metrics_by_year.csv", index=False)
    predictions.to_csv(output_dir / "cv_predictions_samplelevel.csv", index=False)
    annual_draws.to_csv(output_dir / "annual_load_draws.csv", index=False)
    summarize_annual_draws(annual_draws, args.alpha).to_csv(
        output_dir / "annual_load_summary.csv", index=False
    )

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
    plot_legacy_cv_rmse(metrics, figure_dir)

    if args.impute_missing:
        refit_and_impute(
            df, concentration_rows, events, X_c, X_v, cat_c, cat_v,
            features_c, features_v, event_key, output_dir, args, params,
        )
    else:
        warn("Imputation disabled; compatibility files derived from full-data models were not regenerated.")

    print(f"[DONE] Revised event-level workflow outputs written to {output_dir}")


if __name__ == "__main__":
    main()
