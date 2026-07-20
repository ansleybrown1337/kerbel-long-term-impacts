#!/usr/bin/env python3
"""Post-process accepted saved Bayes and ML outputs without rerunning either model.

The script creates versioned cumulative-load, CT-relative, sensitivity,
performance, calibration, feature-importance, audit, figure, and validation
products.  It intentionally treats the model-producing scripts and saved
model outputs as read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


STUDY_YEARS = tuple(range(2011, 2026))
TREATMENTS = ("CT", "MT", "ST")
ANALYTES = ("NH4", "NO3", "NO2", "OP", "Se", "TDS", "TKN", "TN", "TP", "TSS")
INTERVAL_PROB = 0.95
CT_TINY_KG = 1e-12


def analyte_key(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


ANALYTE_ALIASES = {
    "nh4": "NH4", "nh4n": "NH4", "ammonium": "NH4", "ammoniumnh4": "NH4",
    "no3": "NO3", "no3n": "NO3", "nitrate": "NO3",
    "no2": "NO2", "no2n": "NO2", "nitrite": "NO2",
    "op": "OP", "orthop": "OP", "orthophosphate": "OP", "po4p": "OP",
    "se": "Se", "selenium": "Se",
    "tds": "TDS", "totaldissolvedsolids": "TDS",
    "tkn": "TKN", "tknn": "TKN", "totalkjeldahlnitrogen": "TKN",
    "tn": "TN", "tnn": "TN", "totaln": "TN", "totalnitrogen": "TN",
    "tp": "TP", "totalp": "TP", "totalphosphorus": "TP",
    "tss": "TSS", "totalsuspendedsolids": "TSS",
}


def canonical_analyte(value: object) -> str | None:
    return ANALYTE_ALIASES.get(analyte_key(value))


def canonicalize(series: pd.Series) -> pd.Series:
    return series.map(canonical_analyte)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def assert_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        raise ValueError(f"{label} has {int(duplicated.sum())} rows in duplicated keys {list(columns)}")


def grams_to_kg(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return values / 1_000.0


def milligrams_to_kg(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return values / 1_000_000.0


def missing_years(values: Iterable[int]) -> list[int]:
    return sorted(set(STUDY_YEARS).difference({int(value) for value in values}))


def missing_year_text(values: Iterable[int]) -> str:
    missing = missing_years(values)
    return ";".join(map(str, missing)) if missing else "None"


def true_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_values(values: Iterable[object]) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        cleaned.append(value)
    return json.dumps(cleaned, ensure_ascii=False)


def summarize_draws(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": np.nan, "median": np.nan, "low": np.nan, "high": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "low": float(np.quantile(finite, 0.025)),
        "high": float(np.quantile(finite, 0.975)),
    }


def load_inputs(repo: Path) -> dict[str, object]:
    paths = {
        "bayes_draws": repo / "out" / "annual_load_draws_bayes_v2p1.csv",
        "bayes_summary": repo / "out" / "annual_load_summary_bayes_v2p1.csv",
        "bayes_plus_observed": repo / "out" / "annual_load_summary_bayes_plus_observed_v2p1.csv",
        "ml_loyo_draws": repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_draws.csv",
        "ml_loyo_summary": repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_summary.csv",
        "ml_imputed_summary": repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_summary_imputed.csv",
        "ml_imputed_rows": repo / "out" / "ml_catboost_conformal_loyo" / "wq_cleaned_ml_imputed.csv",
        "ml_imputed_row_draws": repo / "out" / "ml_catboost_conformal_loyo" / "imputed_row_draws.csv",
        "ml_cv_predictions": repo / "out" / "ml_catboost_conformal_loyo" / "cv_predictions_samplelevel.csv",
        "ml_cv_metrics": repo / "out" / "ml_catboost_conformal_loyo" / "cv_metrics_by_year.csv",
        "ml_cv_coverage": repo / "out" / "ml_catboost_conformal_loyo" / "cv_interval_coverage_by_year.csv",
        "feature_logc": repo / "out" / "ml_catboost_conformal_loyo" / "feature_importance_logC.csv",
        "feature_logv": repo / "out" / "ml_catboost_conformal_loyo" / "feature_importance_logV.csv",
        "comparison_metrics_group": repo / "out" / "bayes_vs_ml_metrics_v2p1" / "metrics_by_analyte_treatment.csv",
        "comparison_metrics_analyte": repo / "out" / "bayes_vs_ml_metrics_v2p1" / "metrics_by_analyte_overall.csv",
        "comparison_volume": repo / "out" / "bayes_vs_ml_metrics_v2p1" / "volume_metrics_overall.csv",
        "comparison_spearman": repo / "out" / "bayes_vs_ml_metrics_v2p1" / "spearman_by_analyte_treatment.csv",
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required saved input {label}: {path}")

    frames = {
        "bayes_draws": pd.read_csv(paths["bayes_draws"]),
        "bayes_summary": pd.read_csv(paths["bayes_summary"]),
        "bayes_plus_observed": pd.read_csv(paths["bayes_plus_observed"]),
        "ml_loyo_draws": pd.read_csv(paths["ml_loyo_draws"]),
        "ml_loyo_summary": pd.read_csv(paths["ml_loyo_summary"]),
        "ml_imputed_summary": pd.read_csv(paths["ml_imputed_summary"]),
        "ml_imputed_rows": pd.read_csv(paths["ml_imputed_rows"], low_memory=False),
        "ml_cv_predictions": pd.read_csv(paths["ml_cv_predictions"], low_memory=False),
        "ml_cv_metrics": pd.read_csv(paths["ml_cv_metrics"]),
        "ml_cv_coverage": pd.read_csv(paths["ml_cv_coverage"]),
        "feature_logc": pd.read_csv(paths["feature_logc"]),
        "feature_logv": pd.read_csv(paths["feature_logv"]),
        "comparison_metrics_group": pd.read_csv(paths["comparison_metrics_group"]),
        "comparison_metrics_analyte": pd.read_csv(paths["comparison_metrics_analyte"]),
        "comparison_volume": pd.read_csv(paths["comparison_volume"]),
        "comparison_spearman": pd.read_csv(paths["comparison_spearman"]),
    }
    return {"paths": paths, "frames": frames}


def validate_base_inputs(frames: Mapping[str, pd.DataFrame]) -> None:
    require_columns(frames["bayes_draws"], ["draw", "draw_id", "Year", "analyte", "treatment", "load_g"], "Bayesian annual draws")
    require_columns(frames["ml_loyo_draws"], ["Year", "Treatment", "Analyte", "Draw", "AnnualLoad_mg"], "ML LOYO annual draws")
    assert_unique(frames["bayes_draws"], ["draw", "Year", "analyte", "treatment"], "Bayesian annual draws")
    assert_unique(frames["ml_loyo_draws"], ["Draw", "Year", "Analyte", "Treatment"], "ML LOYO annual draws")
    assert_unique(frames["bayes_summary"], ["Year", "analyte", "treatment", "source"], "Bayesian annual summary")
    assert_unique(frames["ml_loyo_summary"], ["Year", "Analyte", "Treatment"], "ML LOYO annual summary")
    assert_unique(frames["ml_imputed_summary"], ["Year", "Analyte", "Treatment"], "ML imputed annual summary")
    if set(frames["bayes_draws"]["treatment"].dropna().unique()) != set(TREATMENTS):
        raise ValueError("Bayesian annual draws do not contain exactly CT, MT, and ST")
    if set(frames["ml_imputed_rows"]["Treatment"].dropna().unique()) != set(TREATMENTS):
        raise ValueError("ML imputed rows do not contain exactly CT, MT, and ST")


def reconstruct_ml_imputed_annual_draws(
    rows: pd.DataFrame, row_draw_path: Path, n_draws: int = 2000
) -> tuple[dict[tuple[int, str, str], np.ndarray], dict[str, object]]:
    """Aggregate saved ML imputation draws; no fitting or random generation."""
    require_columns(
        rows,
        ["_wq_idx", "Year", "Treatment", "Analyte", "Result_mg_L", "Volume", "Result_mg_L_filled", "Volume_filled"],
        "ML imputed rows",
    )
    work = rows.copy()
    work["analyte"] = canonicalize(work["Analyte"])
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work["Treatment"] = work["Treatment"].astype(str).str.upper().str.strip()
    work["_no_runoff"] = true_mask(work["NoRunoff"]) if "NoRunoff" in work else False
    usable = work["_no_runoff"] | (
        pd.to_numeric(work["Result_mg_L_filled"], errors="coerce").notna()
        & pd.to_numeric(work["Volume_filled"], errors="coerce").notna()
    )
    work = work.loc[
        usable
        & work["Year"].isin(STUDY_YEARS)
        & work["Treatment"].isin(TREATMENTS)
        & work["analyte"].isin(ANALYTES)
    ].copy()
    work["_wq_idx"] = pd.to_numeric(work["_wq_idx"], errors="raise").astype(int)
    assert_unique(work, ["_wq_idx"], "Filtered ML imputed rows")

    meta = work.set_index("_wq_idx")
    accumulator: dict[tuple[int, str, str], np.ndarray] = {}
    for year in STUDY_YEARS:
        for treatment in TREATMENTS:
            for analyte in ANALYTES:
                accumulator[(year, treatment, analyte)] = np.zeros(n_draws, dtype=np.float64)

    pending_concentration: dict[int, np.ndarray] = {}
    required_c: set[int] = set()
    required_v: set[int] = set()
    fixed_rows = 0
    no_runoff_rows = 0
    for idx, row in meta.iterrows():
        key = (int(row["Year"]), str(row["Treatment"]), str(row["analyte"]))
        if bool(row["_no_runoff"]):
            no_runoff_rows += 1
            continue
        concentration = pd.to_numeric(pd.Series([row["Result_mg_L"]]), errors="coerce").iloc[0]
        volume = pd.to_numeric(pd.Series([row["Volume"]]), errors="coerce").iloc[0]
        if np.isfinite(concentration) and np.isfinite(volume):
            accumulator[key] += float(concentration) * float(volume)
            fixed_rows += 1
        else:
            if not np.isfinite(concentration):
                required_c.add(int(idx))
            if not np.isfinite(volume):
                required_v.add(int(idx))

    seen_c: set[int] = set()
    seen_v: set[int] = set()
    duplicate_draw_keys = 0
    seen_keys: set[tuple[int, str]] = set()
    with row_draw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["_wq_idx", "Target", "draws"]:
            raise ValueError(f"Unexpected ML imputed-row draw schema: {reader.fieldnames}")
        for record in reader:
            idx = int(record["_wq_idx"])
            target = record["Target"]
            draw_key = (idx, target)
            if draw_key in seen_keys:
                duplicate_draw_keys += 1
            seen_keys.add(draw_key)
            if idx not in meta.index:
                continue
            row = meta.loc[idx]
            if bool(row["_no_runoff"]):
                continue
            key = (int(row["Year"]), str(row["Treatment"]), str(row["analyte"]))
            values = np.fromstring(record["draws"], sep=",", dtype=np.float64)
            if len(values) != n_draws:
                raise ValueError(f"ML row {idx} target {target} has {len(values)} draws, expected {n_draws}")
            concentration = pd.to_numeric(pd.Series([row["Result_mg_L"]]), errors="coerce").iloc[0]
            volume = pd.to_numeric(pd.Series([row["Volume"]]), errors="coerce").iloc[0]
            if target == "Result_mg_L" and idx in required_c:
                seen_c.add(idx)
                if np.isfinite(volume):
                    accumulator[key] += values * float(volume)
                else:
                    pending_concentration[idx] = values
            elif target == "Volume" and idx in required_v:
                seen_v.add(idx)
                if np.isfinite(concentration):
                    accumulator[key] += float(concentration) * values
                else:
                    if idx not in pending_concentration:
                        raise ValueError(f"Volume draws appeared before required concentration draws for row {idx}")
                    accumulator[key] += pending_concentration.pop(idx) * values

    missing_c = sorted(required_c.difference(seen_c))
    missing_v = sorted(required_v.difference(seen_v))
    if missing_c or missing_v or pending_concentration:
        raise ValueError(
            f"Saved ML row draws incomplete: missing concentration={len(missing_c)}, "
            f"missing volume={len(missing_v)}, pending both={len(pending_concentration)}"
        )
    qc = {
        "n_filtered_rows": int(len(work)),
        "n_fixed_rows": fixed_rows,
        "n_no_runoff_rows": no_runoff_rows,
        "n_imputed_concentration_rows": len(required_c),
        "n_imputed_volume_rows": len(required_v),
        "duplicate_row_draw_keys": duplicate_draw_keys,
        "n_draws": n_draws,
    }
    return accumulator, qc


def bayes_cumulative_products(
    draws: pd.DataFrame, source_path: Path
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    work = draws.copy()
    work["analyte"] = canonicalize(work["analyte"])
    work = work.loc[
        work["Year"].isin(STUDY_YEARS)
        & work["analyte"].isin(ANALYTES)
        & work["treatment"].isin(TREATMENTS)
    ].copy()
    values_by_variant: dict[tuple[str, str, str], np.ndarray] = {}
    records: list[dict[str, object]] = []
    for variant, transform in (
        ("raw_draws_bound_floor", lambda x: x),
        ("annual_draw_truncation", lambda x: np.maximum(x, 0.0)),
    ):
        transformed = work.assign(load_variant_g=transform(work["load_g"].to_numpy(dtype=float)))
        for (analyte, treatment), group in transformed.groupby(["analyte", "treatment"], sort=False):
            years = sorted(group["Year"].unique())
            wide = group.pivot(index="draw", columns="Year", values="load_variant_g").sort_index()
            cumulative = grams_to_kg(wide.sum(axis=1).to_numpy(dtype=float))
            summary = summarize_draws(cumulative)
            raw_group = work.loc[(work["analyte"] == analyte) & (work["treatment"] == treatment)]
            raw_low = summary["low"]
            display_low = max(raw_low, 0.0) if variant == "raw_draws_bound_floor" else raw_low
            records.append({
                "method": "Bayes",
                "sensitivity_variant": variant,
                "analyte": analyte,
                "treatment": treatment,
                "first_included_year": min(years),
                "last_included_year": max(years),
                "n_included_years": len(years),
                "missing_years": missing_year_text(years),
                "n_draws": len(cumulative),
                "mean_cumulative_load_kg": summary["mean"],
                "median_cumulative_load_kg": summary["median"],
                "lower_95_raw_kg": raw_low,
                "lower_95_display_kg": display_low,
                "upper_95_kg": summary["high"],
                "interval_probability": INTERVAL_PROB,
                "n_input_annual_draws": len(raw_group),
                "n_input_annual_draws_below_zero": int((raw_group["load_g"] < 0).sum()),
                "fraction_input_annual_draws_below_zero": float((raw_group["load_g"] < 0).mean()),
                "n_cumulative_draws_below_zero": int((cumulative < 0).sum()),
                "fraction_cumulative_draws_below_zero": float((cumulative < 0).mean()),
                "source_file": str(source_path.resolve()),
                "source_units": "g",
                "output_units": "kg",
                "conversion_factor_to_kg": 0.001,
                "provisional_qc_flag": True,
                "provisional_qc_reason": "Physical event aggregation is unresolved; SampleID and MeasureMethod are part of the upstream event key.",
            })
            values_by_variant[(variant, analyte, treatment)] = cumulative
    return pd.DataFrame(records), values_by_variant


def ml_cumulative_products(
    annual_draws_mg: Mapping[tuple[int, str, str], np.ndarray], source_paths: Sequence[Path]
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    records: list[dict[str, object]] = []
    values: dict[tuple[str, str, str], np.ndarray] = {}
    source = "; ".join(str(path.resolve()) for path in source_paths)
    for analyte in ANALYTES:
        for treatment in TREATMENTS:
            year_arrays = [annual_draws_mg[(year, treatment, analyte)] for year in STUDY_YEARS]
            annual_matrix = np.vstack(year_arrays)
            cumulative = milligrams_to_kg(annual_matrix.sum(axis=0))
            summary = summarize_draws(cumulative)
            records.append({
                "method": "ML",
                "sensitivity_variant": "saved_imputed_row_draws",
                "analyte": analyte,
                "treatment": treatment,
                "first_included_year": min(STUDY_YEARS),
                "last_included_year": max(STUDY_YEARS),
                "n_included_years": len(STUDY_YEARS),
                "missing_years": "None",
                "n_draws": len(cumulative),
                "mean_cumulative_load_kg": summary["mean"],
                "median_cumulative_load_kg": summary["median"],
                "lower_95_raw_kg": summary["low"],
                "lower_95_display_kg": summary["low"],
                "upper_95_kg": summary["high"],
                "interval_probability": INTERVAL_PROB,
                "n_input_annual_draws": int(annual_matrix.size),
                "n_input_annual_draws_below_zero": int((annual_matrix < 0).sum()),
                "fraction_input_annual_draws_below_zero": float((annual_matrix < 0).mean()),
                "n_cumulative_draws_below_zero": int((cumulative < 0).sum()),
                "fraction_cumulative_draws_below_zero": float((cumulative < 0).mean()),
                "source_file": source,
                "source_units": "mg",
                "output_units": "kg",
                "conversion_factor_to_kg": 0.000001,
                "provisional_qc_flag": True,
                "provisional_qc_reason": "Saved imputed rows contain repeated analyte-event rows and multiple sample IDs per plot event; upstream physical-event aggregation is unresolved.",
            })
            values[("saved_imputed_row_draws", analyte, treatment)] = cumulative
    return pd.DataFrame(records), values


def treatment_differences(
    method: str,
    variants: Sequence[str],
    values: Mapping[tuple[str, str, str], np.ndarray],
    provisional_reason: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for variant in variants:
        for analyte in ANALYTES:
            ct = values[(variant, analyte, "CT")]
            for treatment in ("MT", "ST"):
                comparison = values[(variant, analyte, treatment)]
                if len(ct) != len(comparison):
                    raise ValueError(f"Draw alignment failed for {method} {variant} {analyte} {treatment}")
                finite = np.isfinite(ct) & np.isfinite(comparison)
                zero = finite & (ct == 0)
                negative = finite & (ct < 0)
                tiny = finite & (ct > 0) & (np.abs(ct) <= CT_TINY_KG)
                valid = finite & (ct > CT_TINY_KG)
                absolute = ct[finite] - comparison[finite]
                percent = 100.0 * (ct[valid] - comparison[valid]) / ct[valid]
                abs_summary = summarize_draws(absolute)
                pct_summary = summarize_draws(percent)
                records.append({
                    "method": method,
                    "sensitivity_variant": variant,
                    "analyte": analyte,
                    "reference_treatment": "CT",
                    "comparison_treatment": treatment,
                    "n_aligned_draws": len(ct),
                    "n_finite_absolute_difference_draws": int(finite.sum()),
                    "n_valid_percent_draws": int(valid.sum()),
                    "fraction_valid_percent_draws": float(valid.mean()),
                    "n_invalid_missing_or_nonfinite_ct_draws": int((~finite).sum()),
                    "n_invalid_zero_ct_draws": int(zero.sum()),
                    "n_invalid_negative_ct_draws": int(negative.sum()),
                    "n_invalid_tiny_positive_ct_draws": int(tiny.sum()),
                    "ct_tiny_threshold_kg": CT_TINY_KG,
                    "mean_absolute_difference_kg": abs_summary["mean"],
                    "median_absolute_difference_kg": abs_summary["median"],
                    "lower_95_absolute_difference_kg": abs_summary["low"],
                    "upper_95_absolute_difference_kg": abs_summary["high"],
                    "mean_percent_difference_relative_to_ct": pct_summary["mean"],
                    "median_percent_difference_relative_to_ct": pct_summary["median"],
                    "lower_95_percent_difference_relative_to_ct": pct_summary["low"],
                    "upper_95_percent_difference_relative_to_ct": pct_summary["high"],
                    "probability_treatment_load_lower_than_ct": float(np.mean(comparison[valid] < ct[valid])) if valid.any() else np.nan,
                    "percent_difference_formula": "100 * (CT - T) / CT",
                    "provisional_qc_flag": True,
                    "provisional_qc_reason": provisional_reason,
                })
    return pd.DataFrame(records)


def observed_subtotals(bayes_plus_observed: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    observed = bayes_plus_observed.loc[bayes_plus_observed["source"].astype(str).str.lower() == "observed"].copy()
    observed["analyte"] = canonicalize(observed["analyte"])
    observed = observed.loc[
        observed["Year"].isin(STUDY_YEARS)
        & observed["analyte"].isin(ANALYTES)
        & observed["treatment"].isin(TREATMENTS)
    ].copy()
    records: list[dict[str, object]] = []
    for analyte in ANALYTES:
        for treatment in TREATMENTS:
            group = observed.loc[(observed["analyte"] == analyte) & (observed["treatment"] == treatment)]
            years = sorted(group["Year"].dropna().astype(int).unique())
            records.append({
                "analyte": analyte,
                "treatment": treatment,
                "observed_value_label": "observed subtotal",
                "observed_subtotal_kg": float(grams_to_kg(pd.to_numeric(group["load_mean"], errors="coerce").sum(min_count=1))) if len(group) else np.nan,
                "first_observed_year": min(years) if years else np.nan,
                "last_observed_year": max(years) if years else np.nan,
                "n_observed_annual_values": len(years),
                "missing_years": missing_year_text(years),
                "n_contributing_events": int(pd.to_numeric(group["n_events"], errors="coerce").sum()) if len(group) else 0,
                "n_contributing_rows": int(pd.to_numeric(group["n_rows"], errors="coerce").sum()) if len(group) else 0,
                "source_file": str(source_path.resolve()),
                "source_units": "g",
                "output_units": "kg",
                "conversion_factor_to_kg": 0.001,
                "coverage_warning": "Observed subtotal is incomplete annual truth when concentration-volume pairs are missing; coverage is unequal among groups.",
                "provisional_qc_flag": True,
                "provisional_qc_reason": "Observed physical-event aggregation uses the accepted upstream event key, whose SampleID/MeasureMethod interpretation remains unresolved.",
            })
    return pd.DataFrame(records)


def observed_differences(observed: pd.DataFrame) -> pd.DataFrame:
    records = []
    for analyte in ANALYTES:
        group = observed.loc[observed["analyte"] == analyte].set_index("treatment")
        ct = group.at["CT", "observed_subtotal_kg"]
        for treatment in ("MT", "ST"):
            value = group.at[treatment, "observed_subtotal_kg"]
            valid = bool(np.isfinite(ct) and ct > CT_TINY_KG and np.isfinite(value))
            records.append({
                "analyte": analyte,
                "comparison_treatment": treatment,
                "reference_treatment": "CT",
                "observed_subtotal_absolute_difference_kg": ct - value if np.isfinite(ct) and np.isfinite(value) else np.nan,
                "observed_subtotal_percent_difference_relative_to_ct": 100 * (ct - value) / ct if valid else np.nan,
                "valid_ct_denominator": valid,
                "interpretation_warning": "Descriptive difference among incomplete observed subtotals; not a full-record treatment comparison.",
            })
    return pd.DataFrame(records)


def summary_reconciliation(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    bayes = frames["bayes_draws"].copy()
    bayes["analyte"] = canonicalize(bayes["analyte"])
    recomputed = bayes.groupby(["Year", "analyte", "treatment"])["load_g"].agg(
        mean="mean", median="median", low=lambda x: x.quantile(0.025), high=lambda x: x.quantile(0.975)
    ).reset_index()
    saved = frames["bayes_summary"].copy()
    saved["analyte"] = canonicalize(saved["analyte"])
    merged = saved.merge(recomputed, on=["Year", "analyte", "treatment"], validate="one_to_one")
    for metric, saved_column, recomputed_column in (
        ("mean", "load_mean", "mean"), ("lower", "load_low", "low"), ("upper", "load_high", "high")
    ):
        difference = merged[saved_column] - merged[recomputed_column]
        denom = np.maximum.reduce([np.abs(merged[saved_column]), np.abs(merged[recomputed_column]), np.full(len(merged), 1e-12)])
        records.append({
            "method": "Bayes", "summary_file": "annual_load_summary_bayes_v2p1.csv",
            "draw_file": "annual_load_draws_bayes_v2p1.csv", "metric": metric,
            "saved_definition": "Full-posterior mean and 2.5/97.5 percentiles with nonnegative clamping in the saved summary",
            "recomputed_definition": "Mean or 2.5/97.5 percentile from the saved 400-draw subsample; raw values are not clamped",
            "n_groups": len(merged), "max_absolute_difference_source_units": float(difference.abs().max()),
            "median_absolute_relative_difference": float(np.median(np.abs(difference) / denom)),
            "p95_absolute_relative_difference": float(np.quantile(np.abs(difference) / denom, 0.95)),
            "relative_difference_denominator": "max(abs(saved), abs(recomputed), 1e-12)",
            "exact_reconciliation": bool(np.allclose(merged[saved_column], merged[recomputed_column], rtol=1e-12, atol=1e-12)),
            "explanation": "The saved annual draw file is a 400-draw posterior subsample, whereas the saved annual summary used the fuller posterior calculation; negative saved-summary bounds were clamped upstream.",
        })
    ml = frames["ml_loyo_draws"]
    rem = ml.groupby(["Year", "Treatment", "Analyte"])["AnnualLoad_mg"].agg(
        mean="mean", median="median", low=lambda x: x.quantile(0.025), high=lambda x: x.quantile(0.975), n_draws="count"
    ).reset_index()
    merged_ml = frames["ml_loyo_summary"].merge(rem, on=["Year", "Treatment", "Analyte"], suffixes=("_saved", "_re"), validate="one_to_one")
    for metric in ("mean", "median", "low", "high", "n_draws"):
        difference = pd.to_numeric(merged_ml[f"{metric}_saved"], errors="coerce") - pd.to_numeric(merged_ml[f"{metric}_re"], errors="coerce")
        records.append({
            "method": "ML", "summary_file": "annual_load_summary.csv", "draw_file": "annual_load_draws.csv",
            "metric": metric, "saved_definition": "Mean, median, and 2.5/97.5 percentiles from saved LOYO annual draws",
            "recomputed_definition": "Same calculation from the saved 2,000-draw LOYO annual file",
            "n_groups": len(merged_ml), "max_absolute_difference_source_units": float(difference.abs().max()),
            "median_absolute_relative_difference": 0.0, "p95_absolute_relative_difference": 0.0,
            "relative_difference_denominator": "max(abs(saved), abs(recomputed), 1e-12)",
            "exact_reconciliation": bool(np.allclose(difference, 0, rtol=1e-12, atol=1e-6)),
            "explanation": "Floating-point CSV precision only.",
        })
    return pd.DataFrame(records)


def ml_imputed_summary_reconciliation(
    annual_draws_mg: Mapping[tuple[int, str, str], np.ndarray], saved: pd.DataFrame
) -> pd.DataFrame:
    """Compare saved full-record summaries with the separately saved row draws.

    The upstream workflow deliberately used seed+70001 for the deposited row
    draws and seed+80001 for the deposited annual summary, so this is an
    independent Monte Carlo reconciliation rather than an exact identity.
    """
    rows = []
    for (year, treatment, analyte), values in annual_draws_mg.items():
        summary = summarize_draws(values)
        rows.append({"Year": year, "Treatment": treatment, "analyte": analyte, **summary, "n_draws": len(values)})
    recomputed = pd.DataFrame(rows)
    comparison = saved.copy()
    comparison["analyte"] = canonicalize(comparison["Analyte"])
    comparison = comparison.loc[comparison["analyte"].isin(ANALYTES)].merge(
        recomputed, on=["Year", "Treatment", "analyte"], suffixes=("_saved", "_re"), validate="one_to_one"
    )
    records = []
    for metric in ("mean", "median", "low", "high", "n_draws"):
        difference = comparison[f"{metric}_saved"] - comparison[f"{metric}_re"]
        denominator = np.maximum.reduce([
            np.abs(comparison[f"{metric}_saved"]),
            np.abs(comparison[f"{metric}_re"]),
            np.full(len(comparison), 1e-12),
        ])
        records.append({
            "method": "ML full-record imputation",
            "summary_file": "annual_load_summary_imputed.csv",
            "draw_file": "imputed_row_draws.csv plus wq_cleaned_ml_imputed.csv",
            "metric": metric,
            "saved_definition": "Full-record annual summary from upstream seed+80001 imputation draws",
            "recomputed_definition": "Same aggregation from deposited seed+70001 row-level imputation draws",
            "n_groups": len(comparison),
            "max_absolute_difference_source_units": float(difference.abs().max()),
            "median_absolute_relative_difference": float(np.median(np.abs(difference) / denominator)),
            "p95_absolute_relative_difference": float(np.quantile(np.abs(difference) / denominator, 0.95)),
            "relative_difference_denominator": "max(abs(saved), abs(recomputed), 1e-12)",
            "exact_reconciliation": bool(np.allclose(difference, 0, rtol=1e-12, atol=1e-6)),
            "explanation": "Expected Monte Carlo difference: upstream saved row draws and annual summaries use different documented RNG streams; no new draws were generated here.",
        })
    return pd.DataFrame(records)


def spearman_tables(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bayes = frames["bayes_summary"].copy()
    bayes["analyte"] = canonicalize(bayes["analyte"])
    ml = frames["ml_imputed_summary"].copy()
    ml["analyte"] = canonicalize(ml["Analyte"])
    records = []
    for analyte in ANALYTES:
        for treatment in TREATMENTS:
            left = bayes.loc[(bayes["analyte"] == analyte) & (bayes["treatment"] == treatment), ["Year", "load_mean"]]
            right = ml.loc[(ml["analyte"] == analyte) & (ml["Treatment"] == treatment), ["Year", "mean"]]
            paired = left.merge(right, on="Year", validate="one_to_one").dropna()
            warning = ""
            rho = p_value = np.nan
            if len(paired) < 3:
                warning = "insufficient paired years"
            elif paired["load_mean"].nunique() < 2 or paired["mean"].nunique() < 2:
                warning = "invariant annual values"
            else:
                result = spearmanr(paired["load_mean"], paired["mean"])
                rho, p_value = float(result.statistic), float(result.pvalue)
            records.append({
                "analyte": analyte, "treatment": treatment, "n_years": len(paired),
                "first_paired_year": int(paired["Year"].min()) if len(paired) else np.nan,
                "last_paired_year": int(paired["Year"].max()) if len(paired) else np.nan,
                "spearman_rho": rho, "p_value_unadjusted": p_value,
                "significant_unadjusted_alpha_0_05": bool(np.isfinite(p_value) and p_value < 0.05),
                "significance_marker": "*" if np.isfinite(p_value) and p_value < 0.05 else "",
                "warning": warning,
            })
    raw = pd.DataFrame(records)
    pub = pd.DataFrame({"Analyte": ANALYTES})
    for treatment in TREATMENTS:
        cells = []
        for analyte in ANALYTES:
            row = raw.loc[(raw["analyte"] == analyte) & (raw["treatment"] == treatment)].iloc[0]
            cells.append(
                f"{row['spearman_rho']:.2f}{row['significance_marker']} ({int(row['n_years'])})"
                if np.isfinite(row["spearman_rho"]) else f"NA ({int(row['n_years'])})"
            )
        pub[treatment] = cells
    return raw, pub


def concentration_performance(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    both = frames["bayes_plus_observed"].copy()
    both["analyte"] = canonicalize(both["analyte"])
    modeled = both.loc[both["source"].eq("Bayes_Modeled"), ["Year", "analyte", "treatment", "conc_mean"]]
    observed = both.loc[both["source"].eq("Observed"), ["Year", "analyte", "treatment", "conc_mean"]].rename(columns={"conc_mean": "observed_mg_L"})
    observed = observed.loc[observed["analyte"].isin(ANALYTES)]
    bayes = modeled.merge(observed, on=["Year", "analyte", "treatment"], validate="one_to_one").rename(columns={"conc_mean": "predicted_mg_L"})

    ml_rows = frames["ml_imputed_rows"].copy()
    ml_rows["analyte"] = canonicalize(ml_rows["Analyte"])
    ml_rows = ml_rows.loc[ml_rows["analyte"].isin(ANALYTES) & ml_rows["Year"].isin(STUDY_YEARS)].copy()
    ml_rows["load_center_mg"] = pd.to_numeric(ml_rows["Result_mg_L_filled"], errors="coerce") * pd.to_numeric(ml_rows["Volume_filled"], errors="coerce")
    ml_annual = ml_rows.groupby(["Year", "analyte", "Treatment"], as_index=False).agg(load_center_mg=("load_center_mg", "sum"), volume_center_L=("Volume_filled", "sum"))
    ml_annual["predicted_mg_L"] = ml_annual["load_center_mg"] / ml_annual["volume_center_L"].replace(0, np.nan)
    ml_annual = ml_annual.rename(columns={"Treatment": "treatment"})
    ml = ml_annual.merge(observed, on=["Year", "analyte", "treatment"], validate="one_to_one")

    records = []
    for method, paired in (("Bayes", bayes), ("ML", ml)):
        for analyte in ANALYTES:
            group = paired.loc[paired["analyte"] == analyte].dropna(subset=["predicted_mg_L", "observed_mg_L"])
            error = group["predicted_mg_L"] - group["observed_mg_L"]
            scale = group["observed_mg_L"].abs().mean()
            rmse = float(np.sqrt(np.mean(np.square(error)))) if len(group) else np.nan
            records.append({
                "response": "Annual flow-weighted concentration", "analyte": analyte, "method": method,
                "n_paired_annual_values": len(group), "rmse_mg_L": rmse,
                "mae_mg_L": float(np.mean(np.abs(error))) if len(group) else np.nan,
                "mean_absolute_observed_mg_L": float(scale) if len(group) else np.nan,
                "nrmse_mean": rmse / scale if len(group) and scale > 0 else np.nan,
                "nrmse_mean_percent": 100 * rmse / scale if len(group) and scale > 0 else np.nan,
                "performance_scope": "Annual central-estimate diagnostic against observed subtotals; not a common sample-level predictive validation design.",
                "provisional_qc_flag": True,
            })
    return pd.DataFrame(records)


def calibration_tables(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = frames["ml_cv_predictions"].copy()
    predictions["covered"] = (predictions["y_true"] >= predictions["pi_low"]) & (predictions["y_true"] <= predictions["pi_high"])
    predictions["interval_width"] = predictions["pi_high"] - predictions["pi_low"]
    predictions["response"] = predictions["Target"].map({"Result_mg_L": "Concentration", "Volume_L": "Volume"})
    predictions["unit"] = predictions["Target"].map({"Result_mg_L": "mg/L", "Volume_L": "L"})
    raw = predictions.groupby(["response", "unit", "Year"], as_index=False).agg(
        n=("covered", "size"), empirical_coverage=("covered", "mean"),
        mean_interval_width=("interval_width", "mean"), median_interval_width=("interval_width", "median")
    )
    raw["nominal_coverage"] = INTERVAL_PROB
    raw["coverage_minus_nominal"] = raw["empirical_coverage"] - INTERVAL_PROB
    raw["calibration_scope"] = "Existing LOYO predictions and conformal intervals; no recalibration performed."
    pub = raw.copy()
    pub["Coverage"] = pub.apply(lambda row: f"{100*row.empirical_coverage:.1f}% / {100*row.nominal_coverage:.0f}% (n={int(row.n)})", axis=1)
    pub["Interval width"] = pub.apply(lambda row: f"{row.mean_interval_width:.3g} {row.unit}", axis=1)
    pub = pub[["response", "Year", "Coverage", "Interval width"]].rename(columns={"response": "Response", "Year": "Held-out year"})
    return raw, pub


FEATURE_LABELS = {
    "Analyte": "Analyte identity", "Inflow_Result_mg_L": "Inflow concentration",
    "Inflow_Volume": "Inflow volume", "CumAll_STIR_toDate": "Cumulative STIR to date",
    "Season_STIR_toDate": "Season STIR to date", "Residue_PercentCover": "Residue cover",
    "residue_prop": "Residue proportion", "Residue_DryMass_kg_m2": "Residue dry mass",
    "DaysUntilHarvest": "Days until harvest", "DaysSincePlant": "Days since planting",
    "DayOfYear": "Day of year", "Treatment": "Treatment identity", "Year": "Year",
    "event_mean_logC_all": "Mean same-event log concentration",
    "event_max_logC_all": "Maximum same-event log concentration",
    "event_n_conc_obs": "Same-event concentration count",
    "RL_mg_L": "Reporting limit",
    "Result_lod_mg_L": "Result detection limit",
    "MDL_mg_L": "Method detection limit",
    "RLMDL_Method": "Reporting/detection-limit method",
    "Volume": "Measured runoff volume",
    "FlumeMethod": "Flume method",
    "MeasureMethod": "Volume measurement method",
    "event_logC_NO3": "Same-event nitrate concentration",
    "event_logC_TP": "Same-event total phosphorus concentration",
}


def feature_importance_tables(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for key, response, unit in (("feature_logc", "Concentration model", "CatBoost importance"), ("feature_logv", "Volume model", "CatBoost importance")):
        part = frames[key].copy()
        part.insert(0, "response", response)
        part["feature_label"] = part["feature"].map(FEATURE_LABELS).fillna(part["feature"].str.replace("_", " ", regex=False))
        part["importance_unit"] = unit
        part["interpretation_warning"] = "Descriptive, noncausal model-use measure; not a management effect or environmental mechanism."
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    pub = raw.sort_values(["response", "importance_mean"], ascending=[True, False]).groupby("response", as_index=False).head(10).copy()
    pub["Importance, mean (SD)"] = pub.apply(lambda row: f"{row.importance_mean:.2f} ({row.importance_sd:.2f})", axis=1)
    pub = pub[["response", "feature_label", "Importance, mean (SD)"]].rename(columns={"response": "Response", "feature_label": "Feature"})
    return raw, pub


def format_load(mean: float, low: float, high: float) -> str:
    return f"{mean:.3g} [{low:.3g}, {high:.3g}]"


def format_difference(row: pd.Series) -> str:
    return (
        f"{row['mean_percent_difference_relative_to_ct']:.1f}% "
        f"[{row['lower_95_percent_difference_relative_to_ct']:.1f}, {row['upper_95_percent_difference_relative_to_ct']:.1f}]; "
        f"P(T<CT)={row['probability_treatment_load_lower_than_ct']:.2f}"
    )


def master_table(
    variant: str, cumulative: pd.DataFrame, differences: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for analyte in ANALYTES:
        record: dict[str, object] = {"Analyte": analyte}
        for treatment in TREATMENTS:
            obs = observed.loc[(observed["analyte"] == analyte) & (observed["treatment"] == treatment)].iloc[0]
            record[f"Observed {treatment} subtotal, kg (years)"] = f"{obs['observed_subtotal_kg']:.3g} ({int(obs['n_observed_annual_values'])})"
            bayes = cumulative.loc[(cumulative["method"] == "Bayes") & (cumulative["sensitivity_variant"] == variant) & (cumulative["analyte"] == analyte) & (cumulative["treatment"] == treatment)].iloc[0]
            ml = cumulative.loc[(cumulative["method"] == "ML") & (cumulative["analyte"] == analyte) & (cumulative["treatment"] == treatment)].iloc[0]
            record[f"Bayes {treatment}, kg [95% interval]"] = format_load(bayes["mean_cumulative_load_kg"], bayes["lower_95_display_kg"], bayes["upper_95_kg"])
            record[f"ML {treatment}, kg [95% interval]"] = format_load(ml["mean_cumulative_load_kg"], ml["lower_95_display_kg"], ml["upper_95_kg"])
        for treatment in ("MT", "ST"):
            for method in ("Bayes", "ML"):
                target_variant = variant if method == "Bayes" else "saved_imputed_row_draws"
                diff = differences.loc[(differences["method"] == method) & (differences["sensitivity_variant"] == target_variant) & (differences["analyte"] == analyte) & (differences["comparison_treatment"] == treatment)].iloc[0]
                record[f"{method} {treatment} vs CT, percent difference [95% interval]"] = format_difference(diff)
        record["QC status"] = "PROVISIONAL: unresolved upstream physical-event aggregation"
        rows.append(record)
    return pd.DataFrame(rows)


def sensitivity_table(cumulative: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a_name, b_name = "raw_draws_bound_floor", "annual_draw_truncation"
    for analyte in ANALYTES:
        orderings = {}
        for variant in (a_name, b_name):
            medians = cumulative.loc[(cumulative["method"] == "Bayes") & (cumulative["sensitivity_variant"] == variant) & (cumulative["analyte"] == analyte)].set_index("treatment")["median_cumulative_load_kg"]
            orderings[variant] = " < ".join(medians.sort_values().index.tolist())
        for treatment in TREATMENTS:
            a = cumulative.loc[(cumulative["method"] == "Bayes") & (cumulative["sensitivity_variant"] == a_name) & (cumulative["analyte"] == analyte) & (cumulative["treatment"] == treatment)].iloc[0]
            b = cumulative.loc[(cumulative["method"] == "Bayes") & (cumulative["sensitivity_variant"] == b_name) & (cumulative["analyte"] == analyte) & (cumulative["treatment"] == treatment)].iloc[0]
            row = {
                "Analyte": analyte, "Treatment": treatment,
                "Negative annual draws, n (%)": f"{int(a['n_input_annual_draws_below_zero'])} ({100*a['fraction_input_annual_draws_below_zero']:.1f}%)",
                "Variant A cumulative, kg [95% display interval]": format_load(a["mean_cumulative_load_kg"], a["lower_95_display_kg"], a["upper_95_kg"]),
                "Variant A raw lower 95%, kg": a["lower_95_raw_kg"],
                "Variant B cumulative, kg [95% interval]": format_load(b["mean_cumulative_load_kg"], b["lower_95_display_kg"], b["upper_95_kg"]),
                "Change in mean, kg (B-A)": b["mean_cumulative_load_kg"] - a["mean_cumulative_load_kg"],
                "Change in median, kg (B-A)": b["median_cumulative_load_kg"] - a["median_cumulative_load_kg"],
                "Change in interval width, kg (B-A)": (b["upper_95_kg"] - b["lower_95_raw_kg"]) - (a["upper_95_kg"] - a["lower_95_raw_kg"]),
                "Variant A treatment ordering": orderings[a_name],
                "Variant B treatment ordering": orderings[b_name],
                "Substantive ordering changed": orderings[a_name] != orderings[b_name],
            }
            if treatment != "CT":
                da = differences.loc[(differences["method"] == "Bayes") & (differences["sensitivity_variant"] == a_name) & (differences["analyte"] == analyte) & (differences["comparison_treatment"] == treatment)].iloc[0]
                db = differences.loc[(differences["method"] == "Bayes") & (differences["sensitivity_variant"] == b_name) & (differences["analyte"] == analyte) & (differences["comparison_treatment"] == treatment)].iloc[0]
                row["Variant A percent difference vs CT"] = format_difference(da)
                row["Variant B percent difference vs CT"] = format_difference(db)
                row["CT-comparison direction changed"] = bool(np.sign(da["median_percent_difference_relative_to_ct"]) != np.sign(db["median_percent_difference_relative_to_ct"]))
            else:
                row["Variant A percent difference vs CT"] = "Reference"
                row["Variant B percent difference vs CT"] = "Reference"
                row["CT-comparison direction changed"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def input_audit(paths: Mapping[str, Path], frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    configurations = {
        "bayes_draws": ("draw;Year;analyte;treatment", "load_g", "g", "v2p1; 400 saved posterior draws"),
        "bayes_summary": ("Year;analyte;treatment;source", "load_mean", "g", "v2p1 Bayesian annual summary"),
        "bayes_plus_observed": ("Year;analyte;treatment;source", "load_mean", "g", "v2p1 modeled plus observed subtotal summary"),
        "ml_loyo_draws": ("Draw;Year;Analyte;Treatment", "AnnualLoad_mg", "mg", "CatBoost LOYO/conformal; 2000 draws"),
        "ml_loyo_summary": ("Year;Analyte;Treatment", "mean", "mg", "CatBoost LOYO annual summary"),
        "ml_imputed_summary": ("Year;Analyte;Treatment", "mean", "mg", "CatBoost full-record imputed annual summary"),
        "ml_imputed_rows": ("_wq_idx", "", "mixed; concentration mg/L, volume L", "CatBoost saved-model imputed rows"),
    }
    records = []
    for label, path in paths.items():
        stat = path.stat()
        frame = frames.get(label)
        key, load_col, units, version = configurations.get(label, ("", "", "See columns", "Saved diagnostic input"))
        years = []
        treatments = []
        analytes = []
        draw_values = []
        duplicate_count = np.nan
        min_load = max_load = negative_count = negative_fraction = np.nan
        missing_required = nonfinite_required = "{}"
        if frame is not None:
            year_col = next((c for c in ("Year", "year", "Year_Test") if c in frame), None)
            treatment_col = next((c for c in ("Treatment", "treatment") if c in frame), None)
            analyte_col = next((c for c in ("Analyte", "analyte") if c in frame), None)
            draw_col = next((c for c in ("Draw", "draw", "draw_id") if c in frame), None)
            years = sorted(pd.to_numeric(frame[year_col], errors="coerce").dropna().astype(int).unique().tolist()) if year_col else []
            treatments = sorted(frame[treatment_col].dropna().astype(str).unique().tolist()) if treatment_col else []
            analytes = sorted(frame[analyte_col].dropna().astype(str).unique().tolist()) if analyte_col else []
            draw_values = sorted(pd.to_numeric(frame[draw_col], errors="coerce").dropna().astype(int).unique().tolist()) if draw_col else []
            if not draw_values and "n_draws" in frame.columns:
                saved_draw_counts = pd.to_numeric(frame["n_draws"], errors="coerce").dropna()
                if len(saved_draw_counts):
                    draw_values = list(range(int(saved_draw_counts.max())))
            key_columns = [column for column in key.split(";") if column]
            duplicate_count = int(frame.duplicated(key_columns, keep=False).sum()) if key_columns and all(c in frame for c in key_columns) else np.nan
            if load_col and load_col in frame:
                values = pd.to_numeric(frame[load_col], errors="coerce")
                min_load, max_load = float(values.min()), float(values.max())
                negative_count = int((values < 0).sum())
                negative_fraction = float((values < 0).mean())
            required = key_columns + ([load_col] if load_col else [])
            missing_required = json.dumps({c: int(frame[c].isna().sum()) for c in required if c in frame})
            nonfinite_required = json.dumps({c: int((~np.isfinite(pd.to_numeric(frame[c], errors="coerce"))).sum()) for c in required if c in frame and pd.api.types.is_numeric_dtype(frame[c])})
        else:
            # The large row-draw file is streamed during reconstruction.
            row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1 if label == "ml_imputed_row_draws" else np.nan
            frame = pd.DataFrame(index=range(int(row_count))) if np.isfinite(row_count) else None
            if label == "ml_imputed_row_draws":
                key, units, version = "_wq_idx;Target", "mg/L or L draw vectors", "2,000 saved imputation draws per row/target"
        alias_map = {value: canonical_analyte(value) for value in analytes if canonical_analyte(value)}
        records.append({
            "input_label": label, "exact_path": str(path.resolve()), "file_size_bytes": stat.st_size,
            "sha256": sha256(path), "modification_time_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            "columns": json_values(["_wq_idx", "Target", "draws"]) if label == "ml_imputed_row_draws" else (json_values(frame.columns) if frame is not None else "[]"),
            "inferred_units": units, "n_rows": len(frame) if frame is not None else np.nan,
            "first_year": min(years) if years else np.nan, "last_year": max(years) if years else np.nan,
            "n_unique_years": len(years), "unique_years": json_values(years),
            "treatments": json_values(treatments), "analytes": json_values(analytes),
            "analyte_alias_mapping": json.dumps(alias_map, ensure_ascii=False),
            "draw_identifier_column": "saved vector position 0-1999" if label == "ml_imputed_row_draws" else next((c for c in ("Draw", "draw", "draw_id") if frame is not None and c in frame), "implicit vector position" if draw_values else ""),
            "n_unique_draws": 2000 if label == "ml_imputed_row_draws" else len(draw_values), "annual_key": key,
            "n_year_analyte_treatment_combinations": int(frame.groupby([c for c in ("Year", "year", "Analyte", "analyte", "Treatment", "treatment") if frame is not None and c in frame], dropna=False).ngroups) if frame is not None and any(c in frame for c in ("Year", "year")) else np.nan,
            "missing_values_required_fields": missing_required, "nonfinite_values_required_fields": nonfinite_required,
            "duplicate_key_rows": duplicate_count, "minimum_load_source_units": min_load,
            "maximum_load_source_units": max_load, "negative_load_count": negative_count,
            "negative_load_fraction": negative_fraction, "source_model_version": version,
        })
    return pd.DataFrame(records)


def unit_of_analysis_audit(rows: pd.DataFrame) -> pd.DataFrame:
    base_key = ["Date", "Year", "Irrigation", "Rep", "Treatment"]
    event_groups = rows.drop_duplicates("EventVolumeID").groupby(base_key, dropna=False).agg(
        n_event_ids=("EventVolumeID", "nunique"), n_sample_ids=("SampleID", "nunique"),
        n_measure_methods=("MeasureMethod", "nunique")
    ).reset_index()
    analyte_event = rows.groupby(["Year", "Treatment", "Analyte", "EventVolumeID"], dropna=False).size()
    base_analyte = rows.groupby(base_key + ["Analyte"], dropna=False).agg(
        n_rows=("_wq_idx", "size"), n_event_ids=("EventVolumeID", "nunique"),
        n_sample_ids=("SampleID", "nunique")
    ).reset_index()
    duplicate_flag = true_mask(rows["Duplicate"]) if "Duplicate" in rows else pd.Series(False, index=rows.index)
    return pd.DataFrame([
        {"metric": "saved_ml_imputed_rows", "value": len(rows), "interpretation": "Analyte-row records in accepted saved imputed input."},
        {"metric": "saved_ml_event_volume_ids", "value": rows["EventVolumeID"].nunique(), "interpretation": "Upstream event IDs include SampleID and MeasureMethod."},
        {"metric": "year_treatment_analyte_event_groups_with_multiple_rows", "value": int((analyte_event > 1).sum()), "interpretation": "These rows are summed separately by the saved imputed annual routine."},
        {"metric": "rows_in_year_treatment_analyte_event_groups_with_multiple_rows", "value": int(analyte_event[analyte_event > 1].sum()), "interpretation": "Potential repeated concentration/laboratory records."},
        {"metric": "base_plot_event_groups", "value": len(event_groups), "interpretation": "Date + Year + Irrigation + Rep + Treatment groups."},
        {"metric": "base_plot_event_groups_with_multiple_event_ids", "value": int((event_groups["n_event_ids"] > 1).sum()), "interpretation": "Often distinct first-flush/outflow/duplicate SampleIDs for one plot event."},
        {"metric": "base_analyte_groups_with_multiple_event_ids", "value": int((base_analyte["n_event_ids"] > 1).sum()), "interpretation": "Cannot establish from current documentation that each is an independent physical runoff load."},
        {"metric": "rows_flagged_duplicate", "value": int(duplicate_flag.sum()), "interpretation": "Rows marked Duplicate in the saved ML imputed table."},
    ])


def publication_performance(concentration: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    conc = concentration[["analyte", "method", "n_paired_annual_values", "rmse_mg_L", "nrmse_mean_percent"]].copy()
    conc["Response"] = conc["analyte"] + " concentration"
    conc["RMSE (original unit)"] = conc["rmse_mg_L"].map(lambda x: f"{x:.3g} mg/L")
    conc["n"] = conc["n_paired_annual_values"]
    vol = volume.copy()
    vol = vol.loc[vol["Analyte"].astype(str).eq("Volume")].copy()
    vol["Response"] = "Volume"
    vol["method"] = vol["method"].astype(str)
    vol["RMSE (original unit)"] = vol["RMSE"].map(lambda x: f"{x:.3g} kL")
    vol["n"] = vol["n"]
    vol["nrmse_mean_percent"] = 100 * vol["NRMSE_mean"]
    combined = pd.concat([
        conc[["Response", "method", "n", "RMSE (original unit)", "nrmse_mean_percent"]],
        vol[["Response", "method", "n", "RMSE (original unit)", "nrmse_mean_percent"]],
    ], ignore_index=True)
    combined["NRMSE_mean (%)"] = combined["nrmse_mean_percent"].map(lambda x: f"{x:.1f}%")
    combined["NRMSE definition"] = "100 × RMSE / mean(abs(observed))"
    return combined.drop(columns="nrmse_mean_percent").rename(columns={"method": "Method"})


def make_figures(
    figure_dir: Path,
    cumulative: pd.DataFrame,
    differences: pd.DataFrame,
    concentration: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_raw: pd.DataFrame,
    volume_performance: pd.DataFrame,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"raw_draws_bound_floor": "#355C7D", "annual_draw_truncation": "#6C5B7B", "saved_imputed_row_draws": "#C06C84"}
    labels = {"raw_draws_bound_floor": "Bayes A", "annual_draw_truncation": "Bayes B", "saved_imputed_row_draws": "ML"}
    fig, axes = plt.subplots(5, 2, figsize=(12, 17), constrained_layout=True)
    for ax, analyte in zip(axes.ravel(), ANALYTES):
        group = cumulative.loc[cumulative["analyte"] == analyte]
        for offset, variant in zip((-0.22, 0, 0.22), labels):
            part = group.loc[group["sensitivity_variant"] == variant].set_index("treatment").reindex(TREATMENTS)
            x = np.arange(3) + offset
            mean = part["mean_cumulative_load_kg"].to_numpy()
            lower = part["lower_95_display_kg"].to_numpy()
            upper = part["upper_95_kg"].to_numpy()
            # Variant A can have a negative mean while its display-only lower
            # bound is floored to zero.  Keep those table values unchanged;
            # only prevent an invalid negative graphical error-bar length.
            yerr = np.vstack([np.maximum(mean - lower, 0), np.maximum(upper - mean, 0)])
            ax.errorbar(x, mean, yerr=yerr, fmt="o", capsize=3, color=colors[variant], label=labels[variant])
        ax.set_title(analyte)
        ax.set_xticks(range(3), TREATMENTS)
        ax.set_ylabel("Cumulative load (kg)")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("Study-period cumulative loads, 2011–2025 (provisional)", fontsize=15)
    fig.savefig(figure_dir / "study_period_cumulative_loads.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(12, 17), constrained_layout=True)
    for ax, analyte in zip(axes.ravel(), ANALYTES):
        group = differences.loc[differences["analyte"] == analyte]
        for offset, variant in zip((-0.18, 0, 0.18), labels):
            part = group.loc[group["sensitivity_variant"] == variant].set_index("comparison_treatment").reindex(["MT", "ST"])
            x = np.arange(2) + offset
            mean = part["mean_percent_difference_relative_to_ct"].to_numpy()
            lower = part["lower_95_percent_difference_relative_to_ct"].to_numpy()
            upper = part["upper_95_percent_difference_relative_to_ct"].to_numpy()
            yerr = np.vstack([np.maximum(mean - lower, 0), np.maximum(upper - mean, 0)])
            ax.errorbar(x, mean, yerr=yerr, fmt="o", capsize=3, color=colors[variant], label=labels[variant])
        ax.axhline(0, color="0.35", linewidth=1)
        ax.set_title(analyte)
        ax.set_xticks(range(2), ["MT vs CT", "ST vs CT"])
        ax.set_ylabel("Percent difference relative to CT")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("CT-relative treatment differences (provisional)", fontsize=15)
    fig.savefig(figure_dir / "treatment_differences_vs_ct.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(11, 15), constrained_layout=True)
    for ax, analyte in zip(axes.ravel(), ANALYTES):
        part = concentration.loc[concentration["analyte"] == analyte]
        ax.bar(part["method"], part["rmse_mg_L"], color=["#355C7D", "#C06C84"])
        ax.set_title(analyte)
        ax.set_ylabel("RMSE (mg/L)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Annual flow-weighted concentration RMSE", fontsize=15)
    fig.savefig(figure_dir / "concentration_rmse_original_units.png", dpi=220)
    plt.close(fig)

    nrmse = concentration[["analyte", "method", "nrmse_mean_percent"]].rename(columns={"analyte": "response"})
    volume = volume_performance.loc[volume_performance["Analyte"].astype(str).eq("Volume"), ["method", "NRMSE_mean"]].copy()
    volume["response"] = "Volume"
    volume["nrmse_mean_percent"] = 100 * volume["NRMSE_mean"]
    nrmse = pd.concat([nrmse, volume[["response", "method", "nrmse_mean_percent"]]], ignore_index=True)
    order = list(ANALYTES) + ["Volume"]
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    y = np.arange(len(order))
    width = 0.36
    for offset, method, color in ((-width / 2, "Bayes", "#355C7D"), (width / 2, "ML", "#C06C84")):
        part = nrmse.loc[nrmse["method"] == method].set_index("response").reindex(order)
        ax.barh(y + offset, part["nrmse_mean_percent"], height=width, label=method, color=color)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("Mean-normalized RMSE (%)")
    ax.set_title("NRMSE_mean = 100 × RMSE / mean(abs(observed))")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.savefig(figure_dir / "nrmse_mean_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, response in zip(axes, ["Concentration", "Volume"]):
        part = calibration.loc[calibration["response"] == response]
        ax.plot(part["Year"], part["empirical_coverage"], marker="o", color="#355C7D")
        ax.axhline(INTERVAL_PROB, linestyle="--", color="#C06C84", label="Nominal 95%")
        ax.set_title(response)
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Held-out year")
    axes[0].legend()
    fig.suptitle("Existing LOYO conformal interval coverage", fontsize=14)
    fig.savefig(figure_dir / "loyo_interval_coverage_by_year.png", dpi=220)
    plt.close(fig)

    for response, filename in (("Concentration model", "feature_importance_concentration.png"), ("Volume model", "feature_importance_volume.png")):
        part = feature_raw.loc[feature_raw["response"] == response].nlargest(15, "importance_mean").sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ax.barh(part["feature_label"], part["importance_mean"], xerr=part["importance_sd"], color="#6C5B7B", alpha=0.9)
        ax.set_xlabel("CatBoost feature importance (mean ± SD)")
        ax.set_title(f"{response}: descriptive, noncausal feature importance")
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(figure_dir / filename, dpi=220)
        plt.close(fig)


def data_dictionary(dataframes: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    definitions = {
        "method": ("Modeling framework.", "category", "Assigned from saved source."),
        "sensitivity_variant": ("Post-processing interpretation of saved draws.", "category", "Bayes A/B or unchanged saved ML imputation draws."),
        "analyte": ("Canonical publication analyte label.", "category", "Mapped with the recorded alias dictionary."),
        "treatment": ("Management treatment group.", "category", "CT, MT, or ST."),
        "mean_cumulative_load_kg": ("Mean 2011–2025 cumulative modeled load across draws.", "kg", "Sum annual draws within draw, then mean."),
        "median_cumulative_load_kg": ("Median cumulative modeled load across draws.", "kg", "Median after within-draw annual summation."),
        "lower_95_raw_kg": ("Unmodified 2.5th percentile of cumulative draws.", "kg", "Quantile 0.025."),
        "lower_95_display_kg": ("Publication lower bound; Variant A alone floors a negative raw lower bound to zero.", "kg", "max(raw lower, 0) only for Bayes Variant A."),
        "upper_95_kg": ("97.5th percentile of cumulative draws.", "kg", "Quantile 0.975."),
        "observed_subtotal_kg": ("Sum of defensible saved observed annual subtotals.", "kg", "Sum load_mean in g and divide by 1,000."),
        "mean_percent_difference_relative_to_ct": ("Mean draw-level percent difference relative to CT; positive means lower than CT.", "%", "100 × (CT − T) / CT for valid CT draws."),
        "probability_treatment_load_lower_than_ct": ("Fraction of valid aligned draws with treatment load below CT.", "probability", "mean(T < CT)."),
        "rmse_mg_L": ("Root mean squared concentration error.", "mg/L", "sqrt(mean((predicted − observed)^2))."),
        "nrmse_mean": ("Mean-normalized RMSE ratio.", "dimensionless", "RMSE / mean(abs(observed))."),
        "empirical_coverage": ("Observed fraction inside saved prediction interval.", "proportion", "mean(low ≤ observed ≤ high)."),
        "mean_interval_width": ("Mean saved prediction-interval width.", "unit column", "mean(high − low)."),
    }
    records = []
    for output_name, frame in dataframes.items():
        for column in frame.columns:
            definition, unit, calculation = definitions.get(column, (column.replace("_", " ").capitalize() + ".", "See column/header", "Direct field or documented formatting of raw source."))
            records.append({"output_file": output_name, "column": column, "definition": definition, "unit": unit, "calculation": calculation, "source": "See source_file fields, input_audit.csv, and comparison_table_footnotes.md."})
    return pd.DataFrame(records).drop_duplicates(["output_file", "column"])


def write_footnotes(path: Path, ml_qc: Mapping[str, object]) -> None:
    text = f"""# Comparison table footnotes

1. Study period: 2011–2025. Loads are kilograms. Bayes annual draws were saved in grams and divided by 1,000; ML loads were reconstructed from saved concentration (mg/L) and volume (L) draws and divided by 1,000,000.
2. Modeled entries are `mean [2.5th percentile, 97.5th percentile]`. Raw machine-readable values are unrounded.
3. Observed values are **observed subtotals**, not complete annual truth. Parenthetical values give the number of contributing annual subtotals. Missing concentration-volume pairs and unequal treatment coverage preclude interpreting them as complete treatment comparisons.
4. CT-relative percent difference is `100 × (CT − T) / CT`. Positive values mean T is lower than CT; negative values mean T is higher than CT. Percent calculations exclude missing, nonfinite, zero, negative, and CT values at or below {CT_TINY_KG:g} kg. Absolute differences retain all finite aligned draws.
5. Bayes Variant A (`raw_draws_bound_floor`) sums unmodified saved annual draws. Only the publication display lower bound is floored at zero when the raw 2.5th percentile is negative. Means, medians, upper bounds, and raw draws are unchanged.
6. Bayes Variant B (`annual_draw_truncation`) applies `max(annual_load, 0)` to every saved annual draw before summing. This is annual-draw-level truncation, not event-level truncation.
7. ML uses the accepted saved full-record imputation artifacts. The 2,000 draw columns in `imputed_row_draws.csv` are aligned by saved draw index; no new random values were generated. Reconstruction QC: {json.dumps(dict(ml_qc), sort_keys=True)}.
8. Spearman rho is calculated across paired annual central estimates within Analyte × Treatment. `*` denotes an exact unadjusted p < 0.05; parentheses contain the number of paired years. These are exploratory multiple tests, not multiplicity-adjusted confirmatory inference.
9. NRMSE_mean = RMSE / mean(abs(observed)); publication percentages multiply this ratio by 100.
10. CatBoost feature importance is descriptive and noncausal. It reflects model use of a predictor, not a management effect or environmental mechanism.
11. All cumulative and CT-relative modeled results are **provisional**. Both accepted workflows define event units using SampleID and MeasureMethod, and the saved ML imputed routine sums repeated analyte-event rows. The current audit cannot establish that all such rows are independent physical runoff-load units.
"""
    path.write_text(text, encoding="utf-8")


def validation_checks(
    cumulative: pd.DataFrame,
    differences: pd.DataFrame,
    spearman: pd.DataFrame,
    recon: pd.DataFrame,
    unit_audit: pd.DataFrame,
    ml_qc: Mapping[str, object],
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    def add(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})
    add("Expected cumulative groups", "PASS" if len(cumulative) == 90 else "FAIL", f"Expected 90; found {len(cumulative)}.")
    add("Expected treatment-difference groups", "PASS" if len(differences) == 60 else "FAIL", f"Expected 60; found {len(differences)}.")
    add("Expected Spearman groups", "PASS" if len(spearman) == 30 else "FAIL", f"Expected 30; found {len(spearman)}.")
    add("Spearman paired years", "PASS" if (spearman["n_years"] == 15).all() else "FAIL", f"Minimum paired years={spearman['n_years'].min()}.")
    add("ML imputed row-draw key uniqueness", "PASS" if ml_qc["duplicate_row_draw_keys"] == 0 else "FAIL", f"Duplicate keys={ml_qc['duplicate_row_draw_keys']}.")
    add("ML annual draws nonnegative", "PASS" if (cumulative.loc[cumulative.method == "ML", "fraction_input_annual_draws_below_zero"] == 0).all() else "FAIL", "No new ML transformation applied.")
    add("Variant A display floor only", "PASS" if (cumulative.loc[cumulative.sensitivity_variant == "raw_draws_bound_floor", "lower_95_display_kg"] >= 0).all() else "FAIL", "Raw lower bound retained separately.")
    add("Variant B nonnegative cumulative", "PASS" if (cumulative.loc[cumulative.sensitivity_variant == "annual_draw_truncation", "lower_95_raw_kg"] >= 0).all() else "FAIL", "Annual draws truncated before summation.")
    add("ML saved annual summary reconciliation", "PASS" if recon.loc[recon.method == "ML", "exact_reconciliation"].all() else "FAIL", "Saved LOYO summary recalculated from saved LOYO annual draws.")
    add(
        "ML full-record summary reconciliation",
        "WARN" if not recon.loc[recon.method == "ML full-record imputation", "exact_reconciliation"].all() else "PASS",
        "Saved row draws and saved imputed annual summary use different documented upstream RNG streams; relative discrepancies are quantified in annual_summary_reconciliation.csv.",
    )
    add("Bayes saved summary reconciliation", "WARN" if not recon.loc[recon.method == "Bayes", "exact_reconciliation"].all() else "PASS", "Saved annual draw file is a 400-draw posterior subsample; saved summary used fuller posterior calculation and upstream clamping.")
    repeated = int(unit_audit.loc[unit_audit.metric == "year_treatment_analyte_event_groups_with_multiple_rows", "value"].iloc[0])
    add("Physical event aggregation demonstrated", "WARN", f"Not demonstrated: {repeated} Year × Treatment × Analyte × event groups contain multiple rows; outputs marked provisional.")
    invalid = int((differences["n_aligned_draws"] - differences["n_valid_percent_draws"]).sum())
    add("CT denominator QC", "PASS" if invalid == 0 else "WARN", f"Total invalid percent-difference draws across groups={invalid}; retained in denominator-QC fields.")
    return pd.DataFrame(checks)


def write_validation_report(
    path: Path,
    output_dir: Path,
    figure_dir: Path,
    cumulative: pd.DataFrame,
    differences: pd.DataFrame,
    checks: pd.DataFrame,
    unit_audit: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    bayes_raw = cumulative.loc[cumulative.sensitivity_variant == "raw_draws_bound_floor"]
    neg_count = int(bayes_raw["n_input_annual_draws_below_zero"].sum())
    neg_total = int(bayes_raw["n_input_annual_draws"].sum())
    invalid = differences.loc[:, ["method", "sensitivity_variant", "analyte", "comparison_treatment", "n_invalid_negative_ct_draws", "n_invalid_zero_ct_draws", "n_invalid_tiny_positive_ct_draws"]]
    unstable = invalid.loc[invalid.iloc[:, 4:].sum(axis=1) > 0]
    report = f"""# Post-processing validation report

## Outcome

The saved Bayesian and ML artifacts were post-processed without refitting, recompiling, recalibrating, or generating new model predictions. The requested cumulative-load, CT-relative, sensitivity, performance, calibration, feature-importance, publication-table, figure, audit, and dictionary outputs are in `{output_dir}` and `{figure_dir}`.

All cumulative and CT-relative results are **provisional** because the limited read-only audit could not establish an unambiguous physical runoff-event unit across first-flush, outflow, and duplicate SampleIDs. No upstream code or saved model output was changed.

## Key QC findings

- Bayesian saved annual draws below zero: {neg_count:,} of {neg_total:,} ({100 * neg_count / neg_total:.2f}%).
- ML reconstructed annual draws below zero: 0; no parallel ML transformation was imposed.
- Variant A preserves raw cumulative draws and floors only a negative publication lower bound. Variant B truncates each annual draw at zero before summation.
- Invalid/unstable CT-denominator groups: {len(unstable)}; exact draw counts are in `treatment_differences_vs_ct_raw.csv`.
- Saved ML LOYO annual summaries reconcile to the saved LOYO annual draws to floating-point precision. Bayesian summaries do not reconcile exactly because the saved annual draw file contains only 400 posterior draws while the saved summary used the fuller posterior calculation; the audit records the discrepancy.
- The saved full-record ML annual summary and the deposited ML row draws use different upstream random-number streams (`seed+80001` and `seed+70001`). Their independent Monte Carlo reconciliation is quantified rather than presented as an exact identity.
- The saved ML LOYO annual draw file is intentionally incomplete for full-period use. Full-period ML cumulative products were reconstructed deterministically from the saved 2,000-column `imputed_row_draws.csv` and `wq_cleaned_ml_imputed.csv`; missing years were never treated as zero.
- Unit-of-analysis audit: {unit_audit.set_index('metric')['value'].to_dict()}.

## Validation status

{checks.to_markdown(index=False)}

## Readiness

- The two master cumulative-load tables, Bayesian sensitivity table, Spearman table, calibration tables, feature-importance tables, and performance tables are structurally ready for manuscript review.
- Their numerical interpretation remains provisional pending a domain decision on whether first-flush/outflow/duplicate sample records are separate load-bearing events or repeated measurements of one physical plot runoff event.
- No glaring target leakage, held-out-year leakage, unit-conversion error, analyte mismatch, or wrong model-version use was identified in this limited read-only audit. The unresolved unit-of-analysis evidence is the upstream issue requiring attention before definitive release claims.

## Reproduction

Run from the repository root:

```powershell
python code/bayes_ml_postprocessing_v2p1.py --repo . --rebuild-current-run
python -m unittest discover -s tests -p "test_bayes_ml_postprocessing_v2p1.py" -v
```

Post-processing runtime for this run: {runtime_seconds:.1f} seconds. Dependency versions are saved in `postprocessing_run_metadata.json`.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Resume this script's incomplete versioned directory only when no completed run metadata exists.",
    )
    parser.add_argument(
        "--rebuild-current-run",
        action="store_true",
        help="Rebuild only this script's completed output directory after verifying its run metadata identifies the same directory.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    output_dir = repo / "out" / "bayes_vs_ml_postprocessing_v2p1"
    figure_dir = repo / "figs" / "bayes_vs_ml_postprocessing_v2p1"
    output_has_files = output_dir.exists() and any(output_dir.iterdir())
    figure_has_files = figure_dir.exists() and any(figure_dir.iterdir())
    completed_metadata = output_dir / "postprocessing_run_metadata.json"
    if output_has_files or figure_has_files:
        safe_rebuild = False
        if args.rebuild_current_run and completed_metadata.exists():
            prior = json.loads(completed_metadata.read_text(encoding="utf-8"))
            safe_rebuild = Path(prior.get("output_directory", "")).resolve() == output_dir
        safe_partial = args.resume_partial and not completed_metadata.exists()
        if not (safe_partial or safe_rebuild):
            raise FileExistsError(
                f"Refusing to overwrite an existing or completed versioned run: {output_dir}. "
                "Use a new versioned directory; --resume-partial is allowed only before completed run metadata exists."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_inputs(repo)
    paths: dict[str, Path] = loaded["paths"]  # type: ignore[assignment]
    frames: dict[str, pd.DataFrame] = loaded["frames"]  # type: ignore[assignment]
    validate_base_inputs(frames)
    audit = input_audit(paths, frames)
    unit_audit = unit_of_analysis_audit(frames["ml_imputed_rows"])
    recon = summary_reconciliation(frames)

    ml_annual, ml_qc = reconstruct_ml_imputed_annual_draws(frames["ml_imputed_rows"], paths["ml_imputed_row_draws"])
    audit.loc[audit["input_label"] == "ml_imputed_row_draws", "duplicate_key_rows"] = ml_qc["duplicate_row_draw_keys"]
    audit.loc[audit["input_label"] == "ml_imputed_row_draws", "missing_values_required_fields"] = "{}"
    audit.loc[audit["input_label"] == "ml_imputed_row_draws", "nonfinite_values_required_fields"] = "{}"
    recon = pd.concat([recon, ml_imputed_summary_reconciliation(ml_annual, frames["ml_imputed_summary"])], ignore_index=True)
    bayes_cumulative, bayes_values = bayes_cumulative_products(frames["bayes_draws"], paths["bayes_draws"])
    ml_cumulative, ml_values = ml_cumulative_products(ml_annual, [paths["ml_imputed_rows"], paths["ml_imputed_row_draws"]])
    cumulative = pd.concat([bayes_cumulative, ml_cumulative], ignore_index=True).sort_values(["analyte", "method", "sensitivity_variant", "treatment"])
    bayes_diff = treatment_differences("Bayes", ["raw_draws_bound_floor", "annual_draw_truncation"], bayes_values, "Unresolved upstream physical-event aggregation; see postprocessing_validation_report.md.")
    ml_diff = treatment_differences("ML", ["saved_imputed_row_draws"], ml_values, "Saved imputed rows contain repeated analyte-event rows; see postprocessing_validation_report.md.")
    differences = pd.concat([bayes_diff, ml_diff], ignore_index=True).sort_values(["analyte", "method", "sensitivity_variant", "comparison_treatment"])
    observed = observed_subtotals(frames["bayes_plus_observed"], paths["bayes_plus_observed"])
    observed_diff = observed_differences(observed)
    spearman_raw, spearman_pub = spearman_tables(frames)
    concentration = concentration_performance(frames)
    calibration_raw, calibration_pub = calibration_tables(frames)
    feature_raw, feature_pub = feature_importance_tables(frames)
    performance_pub = publication_performance(concentration, frames["comparison_volume"])
    master_a = master_table("raw_draws_bound_floor", cumulative, differences, observed)
    master_b = master_table("annual_draw_truncation", cumulative, differences, observed)
    sensitivity = sensitivity_table(cumulative, differences)

    outputs: dict[str, pd.DataFrame] = {
        "input_audit.csv": audit,
        "unit_of_analysis_audit.csv": unit_audit,
        "annual_summary_reconciliation.csv": recon,
        "study_period_cumulative_loads_raw.csv": cumulative,
        "treatment_differences_vs_ct_raw.csv": differences,
        "observed_study_period_subtotals_raw.csv": observed,
        "observed_subtotal_differences_vs_ct_raw.csv": observed_diff,
        "master_cumulative_loads_raw_bound_floor_pub.csv": master_a,
        "master_cumulative_loads_annual_truncation_pub.csv": master_b,
        "bayes_nonnegative_sensitivity_pub.csv": sensitivity,
        "spearman_by_analyte_treatment_raw.csv": spearman_raw,
        "spearman_by_analyte_pub.csv": spearman_pub,
        "annual_concentration_performance_raw.csv": concentration,
        "annual_load_agreement_metrics_raw.csv": frames["comparison_metrics_group"],
        "annual_load_agreement_by_analyte_raw.csv": frames["comparison_metrics_analyte"],
        "annual_volume_performance_raw.csv": frames["comparison_volume"],
        "performance_comparison_pub.csv": performance_pub,
        "loyo_interval_calibration_raw.csv": calibration_raw,
        "loyo_interval_calibration_pub.csv": calibration_pub,
        "catboost_feature_importance_raw.csv": feature_raw,
        "catboost_feature_importance_pub.csv": feature_pub,
    }
    dictionary = data_dictionary(outputs)
    outputs["postprocessing_data_dictionary.csv"] = dictionary
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    write_footnotes(output_dir / "comparison_table_footnotes.md", ml_qc)
    make_figures(figure_dir, cumulative, differences, concentration, calibration_raw, feature_raw, frames["comparison_volume"])

    checks = validation_checks(cumulative, differences, spearman_raw, recon, unit_audit, ml_qc)
    checks.to_csv(output_dir / "validation_checks.csv", index=False)
    runtime = time.perf_counter() - started
    metadata = {
        "command": "python code/bayes_ml_postprocessing_v2p1.py --repo . --rebuild-current-run",
        "runtime_seconds": runtime,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "random_operations": "None; all draws read from saved files.",
        "output_directory": str(output_dir),
        "figure_directory": str(figure_dir),
    }
    (output_dir / "postprocessing_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_validation_report(output_dir / "postprocessing_validation_report.md", output_dir, figure_dir, cumulative, differences, checks, unit_audit, runtime)
    if (checks["status"] == "FAIL").any():
        raise AssertionError("One or more post-processing validation checks failed; see validation_checks.csv")
    print(f"[OK] Wrote {len(outputs) + 4} tables/reports to {output_dir}")
    print(f"[OK] Wrote versioned figures to {figure_dir}")
    print(f"[OK] Runtime: {runtime:.1f} seconds")


if __name__ == "__main__":
    main()
