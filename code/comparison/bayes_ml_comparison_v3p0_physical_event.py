#!/usr/bin/env python3
"""Compare only completed Bayesian and ML v3p0 physical-event outputs.

This workflow has no legacy fallbacks. It refuses absent manifests, wrong
versions, incomplete study-year coverage, and non-unique event-analyte draws.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.physical_event import (  # noqa: E402
    CORRECTED_VERSION,
    PHYSICAL_EVENT_CONFIG,
    validate_corrected_artifact_metadata,
)


STUDY_YEARS = list(range(2011, 2026))
TREATMENTS = ["CT", "MT", "ST"]
PUBLICATION_ANALYTES = ["NH4", "NO3", "NO2", "OP", "Se", "TDS", "TKN", "TN", "TP", "TSS"]


def require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing column(s): {', '.join(missing)}")


def read_required(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required corrected {label} is absent: {path}")
    return pd.read_csv(path, low_memory=False)


def assert_study_years(frame: pd.DataFrame, label: str) -> None:
    require_columns(frame, ["Year"], label)
    found = sorted(pd.to_numeric(frame["Year"], errors="coerce").dropna().astype(int).unique())
    if found != STUDY_YEARS:
        raise ValueError(f"{label} has years {found}; required exactly {STUDY_YEARS}. No zero fill is allowed.")


def restrict_publication_analytes(
    frame: pd.DataFrame,
    label: str,
    *,
    require_all: bool,
) -> tuple[pd.DataFrame, list[str]]:
    found = set(frame["Analyte"].dropna().astype(str))
    missing = sorted(set(PUBLICATION_ANALYTES) - found)
    if require_all and missing:
        raise ValueError(f"{label} is missing publication analytes: {missing}.")
    extras = sorted(found - set(PUBLICATION_ANALYTES))
    restricted = frame.loc[frame["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)].copy()
    if restricted.empty:
        raise ValueError(f"{label} contains no publication analytes after alias normalization.")
    return restricted, extras


def assert_complete_analyte_treatment_years(frame: pd.DataFrame, label: str) -> None:
    for analyte in PUBLICATION_ANALYTES:
        for treatment in TREATMENTS:
            part = frame.loc[
                frame["Analyte"].astype(str).eq(analyte)
                & frame["Treatment"].astype(str).eq(treatment)
            ]
            found = sorted(part["Year"].unique().astype(int).tolist())
            if found != STUDY_YEARS:
                raise ValueError(
                    f"{label} {analyte}/{treatment} has years {found}; required exactly "
                    f"{STUDY_YEARS}. Missing years are never zero-filled."
                )


def normalize_ledger(
    frame: pd.DataFrame,
    method: str,
    scenario: str,
    *,
    require_complete_years: bool = True,
) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        raise ValueError(f"{method} {scenario} ledger is empty.")
    rename = {}
    if "analyte" in output and "Analyte" not in output:
        rename["analyte"] = "Analyte"
    if "treatment" in output and "Treatment" not in output:
        rename["treatment"] = "Treatment"
    if "draw" in output and "Draw" not in output:
        rename["draw"] = "Draw"
    output = output.rename(columns=rename)
    require_columns(
        output, ["PhysicalEventID", "Analyte", "Year", "Treatment", "Draw"],
        f"{method} {scenario} ledger",
    )
    if "Load_kg" not in output:
        if "Load_g" in output:
            output["Load_kg"] = pd.to_numeric(output["Load_g"], errors="raise") / 1000.0
        elif "AnnualLoad_mg" in output:
            output["Load_kg"] = pd.to_numeric(output["AnnualLoad_mg"], errors="raise") / 1_000_000.0
        else:
            raise ValueError(f"{method} {scenario} ledger has no supported load column.")
    key = ["PhysicalEventID", "Analyte", "Draw"]
    if output.duplicated(key).any():
        raise ValueError(f"{method} {scenario} has duplicate PhysicalEventID x Analyte x Draw rows.")
    output["Year"] = pd.to_numeric(output["Year"], errors="raise").astype(int)
    output["Method"] = method
    output["Scenario"] = scenario
    if require_complete_years:
        assert_study_years(output, f"{method} {scenario} ledger")
    else:
        unexpected = sorted(set(output["Year"].unique()) - set(STUDY_YEARS))
        if unexpected:
            raise ValueError(
                f"{method} {scenario} ledger contains years outside the study period: {unexpected}."
            )
    return output


def normalize_point_ledger(
    frame: pd.DataFrame,
    method: str,
    scenario: str,
    *,
    require_complete_years: bool = True,
) -> pd.DataFrame:
    """Validate deterministic physical-event point loads separately from draws."""

    output = frame.copy()
    if output.empty:
        raise ValueError(f"{method} {scenario} point ledger is empty.")
    rename = {}
    if "analyte" in output and "Analyte" not in output:
        rename["analyte"] = "Analyte"
    if "treatment" in output and "Treatment" not in output:
        rename["treatment"] = "Treatment"
    output = output.rename(columns=rename)
    require_columns(
        output,
        ["PhysicalEventID", "Analyte", "Year", "Treatment"],
        f"{method} {scenario} point ledger",
    )
    if "Load_kg" not in output:
        if "AnnualLoad_mg" in output:
            output["Load_kg"] = (
                pd.to_numeric(output["AnnualLoad_mg"], errors="raise") / 1_000_000.0
            )
        else:
            raise ValueError(f"{method} {scenario} point ledger has no supported load column.")
    key = ["PhysicalEventID", "Analyte"]
    if output.duplicated(key).any():
        raise ValueError(
            f"{method} {scenario} has duplicate PhysicalEventID x Analyte point rows."
        )
    output["Year"] = pd.to_numeric(output["Year"], errors="raise").astype(int)
    output["Method"] = method
    output["Scenario"] = scenario
    if require_complete_years:
        assert_study_years(output, f"{method} {scenario} point ledger")
    return output


def summarize_draws(frame: pd.DataFrame, groups: Sequence[str], value: str) -> pd.DataFrame:
    return (
        frame.groupby(list(groups), dropna=False)[value]
        .agg(
            mean="mean",
            median="median",
            lower_95=lambda values: values.quantile(0.025),
            upper_95=lambda values: values.quantile(0.975),
            n_draws="size",
        )
        .reset_index()
    )


def annual_products(ledgers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    draws = (
        ledgers.groupby(
            ["Method", "Scenario", "Year", "Analyte", "Treatment", "Draw"], as_index=False
        ).agg(Load_kg=("Load_kg", "sum"))
    )
    assert_study_years(draws, "Combined annual draws")
    summary = summarize_draws(
        draws, ["Method", "Scenario", "Year", "Analyte", "Treatment"], "Load_kg"
    )
    return draws, summary


def annual_point_products(point_ledgers: pd.DataFrame) -> pd.DataFrame:
    points = (
        point_ledgers.groupby(
            ["Method", "Scenario", "Year", "Analyte", "Treatment"],
            as_index=False,
        )
        .agg(
            PointTotal_kg=("Load_kg", "sum"),
            n_physical_events=("PhysicalEventID", "nunique"),
        )
    )
    assert_study_years(points, "Combined annual point totals")
    return points


def attach_primary_centers(
    summary: pd.DataFrame,
    point_totals: pd.DataFrame,
    *,
    keys: Sequence[str],
    point_column: str,
) -> pd.DataFrame:
    """Use ML point totals as centers while leaving draw intervals unchanged."""

    required_point_columns = [*keys, point_column]
    require_columns(point_totals, required_point_columns, "ML point totals")
    merge_points = point_totals[required_point_columns].copy()
    if merge_points.duplicated(list(keys)).any():
        raise ValueError("ML point totals are not unique by their comparison key.")
    merge_points["_point_row_present"] = True
    output = summary.merge(
        merge_points,
        on=list(keys),
        how="left",
        validate="one_to_one",
    )
    output["primary_center"] = pd.to_numeric(output["median"], errors="raise")
    output["primary_center_type"] = np.where(
        output["Method"].eq("Bayes"),
        "posterior_median",
        "monte_carlo_draw_median",
    )
    ml = output["Method"].eq("ML")
    missing_points = ml & output["_point_row_present"].isna()
    if missing_points.any():
        missing = output.loc[missing_points, list(keys)].head(10).to_dict("records")
        raise ValueError(f"ML point totals are missing for comparison rows: {missing}")
    output.loc[ml, "primary_center"] = pd.to_numeric(
        output.loc[ml, point_column], errors="raise"
    )
    output.loc[ml, "primary_center_type"] = "physical_event_point_total"
    output["primary_center_within_draw_interval"] = (
        pd.to_numeric(output["primary_center"], errors="coerce")
        .ge(pd.to_numeric(output["lower_95"], errors="coerce"))
        & pd.to_numeric(output["primary_center"], errors="coerce")
        .le(pd.to_numeric(output["upper_95"], errors="coerce"))
    )
    return output.drop(columns="_point_row_present")


def cumulative_point_products(annual_points: pd.DataFrame) -> pd.DataFrame:
    return (
        annual_points.groupby(
            ["Method", "Scenario", "Analyte", "Treatment"], as_index=False
        )
        .agg(PointCumulativeLoad_kg=("PointTotal_kg", "sum"))
    )


def ct_relative_point_products(cumulative_points: pd.DataFrame) -> pd.DataFrame:
    wide = cumulative_points.pivot(
        index=["Method", "Scenario", "Analyte"],
        columns="Treatment",
        values="PointCumulativeLoad_kg",
    ).reset_index()
    missing = [treatment for treatment in TREATMENTS if treatment not in wide]
    if missing:
        raise ValueError(f"Point CT-relative calculation is missing treatment(s): {missing}")
    records: list[pd.DataFrame] = []
    for treatment in ["MT", "ST"]:
        part = wide[["Method", "Scenario", "Analyte", "CT", treatment]].copy()
        part["ComparisonTreatment"] = treatment
        valid = np.isfinite(part["CT"]) & np.isfinite(part[treatment]) & part["CT"].gt(1e-12)
        part["PointPercentDifference"] = np.where(
            valid, 100.0 * (part["CT"] - part[treatment]) / part["CT"], np.nan
        )
        records.append(
            part[[
                "Method", "Scenario", "Analyte", "ComparisonTreatment",
                "PointPercentDifference",
            ]]
        )
    return pd.concat(records, ignore_index=True)


def cumulative_products(annual_draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    draws = (
        annual_draws.groupby(
            ["Method", "Scenario", "Analyte", "Treatment", "Draw"], as_index=False
        ).agg(CumulativeLoad_kg=("Load_kg", "sum"))
    )
    return draws, summarize_draws(
        draws, ["Method", "Scenario", "Analyte", "Treatment"], "CumulativeLoad_kg"
    )


def ct_relative(draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = draws.pivot(
        index=["Method", "Scenario", "Analyte", "Draw"],
        columns="Treatment", values="CumulativeLoad_kg",
    ).reset_index()
    missing = [treatment for treatment in TREATMENTS if treatment not in wide]
    if missing:
        raise ValueError(f"CT-relative calculation is missing treatment(s): {missing}")
    records = []
    for treatment in ["MT", "ST"]:
        part = wide[["Method", "Scenario", "Analyte", "Draw", "CT", treatment]].copy()
        part["ComparisonTreatment"] = treatment
        finite_pair = np.isfinite(part["CT"]) & np.isfinite(part[treatment])
        valid = finite_pair & part["CT"].gt(1e-12)
        part["valid_percent_denominator"] = valid
        part["AbsoluteDifference_CT_minus_T_kg"] = np.where(
            finite_pair, part["CT"] - part[treatment], np.nan
        )
        part["PercentDifferenceRelativeToCT"] = np.where(
            valid, 100.0 * (part["CT"] - part[treatment]) / part["CT"], np.nan
        )
        part["TreatmentLowerThanCT"] = np.where(
            valid, part[treatment] < part["CT"], np.nan
        )
        records.append(part)
    raw = pd.concat(records, ignore_index=True)
    group_columns = ["Method", "Scenario", "Analyte", "ComparisonTreatment"]
    summary = summarize_draws(
        raw,
        group_columns,
        "PercentDifferenceRelativeToCT",
    )
    absolute = summarize_draws(
        raw, group_columns, "AbsoluteDifference_CT_minus_T_kg"
    ).rename(columns={
        "mean": "absolute_mean_kg", "median": "absolute_median_kg",
        "lower_95": "absolute_lower_95_kg", "upper_95": "absolute_upper_95_kg",
        "n_draws": "n_absolute_draws",
    })
    qc = (
        raw.groupby(group_columns, as_index=False)
        .agg(
            n_total_draws=("Draw", "size"),
            n_valid_percent_draws=("valid_percent_denominator", "sum"),
            probability_treatment_lower_than_ct=("TreatmentLowerThanCT", "mean"),
        )
    )
    qc["n_invalid_percent_draws"] = qc["n_total_draws"] - qc["n_valid_percent_draws"]
    qc["fraction_invalid_percent_draws"] = (
        qc["n_invalid_percent_draws"] / qc["n_total_draws"]
    )
    summary = summary.merge(absolute, on=group_columns, validate="one_to_one").merge(
        qc, on=group_columns, validate="one_to_one"
    )
    summary["definition"] = (
        "100 * (CT - treatment) / CT within draw; positive means lower than CT; "
        "missing, nonpositive, or <=1e-12 kg CT denominators are invalid and retained in QC"
    )
    return raw, summary


def spearman_tables(annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    center_column = "primary_center" if "primary_center" in annual else "median"
    bayes = annual.loc[
        (annual["Method"] == "Bayes") & (annual["Scenario"] == "model_only"),
        ["Year", "Analyte", "Treatment", center_column],
    ].rename(columns={center_column: "BayesMedian_kg"})
    ml = annual.loc[
        (annual["Method"] == "ML") & (annual["Scenario"] == "full_record_model_only"),
        ["Year", "Analyte", "Treatment", center_column],
    ].rename(columns={center_column: "MLPointTotal_kg"})
    paired = bayes.merge(ml, on=["Year", "Analyte", "Treatment"], validate="one_to_one")
    records = []
    for (analyte, treatment), group in paired.groupby(["Analyte", "Treatment"], dropna=False):
        complete = group.dropna(subset=["BayesMedian_kg", "MLPointTotal_kg"])
        warning = ""
        if len(complete) < 3:
            rho, p_value = np.nan, np.nan
            warning = "fewer than three paired years"
        elif complete[["BayesMedian_kg", "MLPointTotal_kg"]].nunique().min() < 2:
            rho, p_value = np.nan, np.nan
            warning = "invariant annual values"
        else:
            rho, p_value = spearmanr(
                complete["BayesMedian_kg"], complete["MLPointTotal_kg"]
            )
        records.append({
            "Analyte": analyte, "Treatment": treatment, "n_paired_years": len(complete),
            "first_paired_year": int(complete["Year"].min()) if len(complete) else np.nan,
            "last_paired_year": int(complete["Year"].max()) if len(complete) else np.nan,
            "rho": rho, "p_value_unadjusted": p_value,
            "significant_unadjusted_p_lt_0_05": bool(pd.notna(p_value) and p_value < 0.05),
            "warning": warning,
        })
    raw = pd.DataFrame(records)
    publication = raw.copy()
    publication["cell"] = publication.apply(
        lambda row: (
            f"NA ({int(row['n_paired_years'])})"
            if pd.isna(row["rho"])
            else f"{row['rho']:.2f}{'*' if row['p_value_unadjusted'] < 0.05 else ''} "
            f"({int(row['n_paired_years'])})"
        ),
        axis=1,
    )
    publication = (
        publication.pivot(index="Analyte", columns="Treatment", values="cell")
        .reindex(index=PUBLICATION_ANALYTES, columns=TREATMENTS)
        .reset_index()
    )
    return raw, publication


def performance_table(frame: pd.DataFrame, method: str, target: str) -> pd.DataFrame:
    if method == "ML":
        subset = frame.loc[frame["Target"].eq(target)].copy()
        observed, predicted, low, high = "y_true", "y_pred", "pi_low", "pi_high"
    else:
        subset = frame.copy()
        observed, predicted, low, high = "Observed", "Predicted", "Lower95", "Upper95"
    required = [observed, predicted, low, high, "Treatment"]
    if target == "Result_mg_L":
        required.append("Analyte")
    require_columns(subset, required, f"{method} {target} diagnostics")
    subset["error"] = pd.to_numeric(subset[predicted], errors="coerce") - pd.to_numeric(subset[observed], errors="coerce")
    subset["covered"] = (
        pd.to_numeric(subset[observed], errors="coerce").ge(pd.to_numeric(subset[low], errors="coerce"))
        & pd.to_numeric(subset[observed], errors="coerce").le(pd.to_numeric(subset[high], errors="coerce"))
    )
    if target == "Result_mg_L":
        groups = [
            ("analyte_overall", str(analyte), "all", group)
            for analyte, group in subset.groupby("Analyte", dropna=False)
        ] + [
            ("analyte_treatment", str(analyte), str(treatment), group)
            for (analyte, treatment), group in subset.groupby(
                ["Analyte", "Treatment"], dropna=False
            )
        ]
    else:
        groups = [("overall", "all", "all", subset)] + [
            ("treatment", "all", str(treatment), group)
            for treatment, group in subset.groupby("Treatment", dropna=False)
        ]
    records = []
    for grouping, analyte, treatment, group in groups:
        valid = group.dropna(subset=[observed, predicted])
        rmse = float(np.sqrt(np.mean(np.square(valid["error"])))) if len(valid) else np.nan
        denominator = pd.to_numeric(valid[observed], errors="coerce").mean()
        interval_valid = group.dropna(subset=[observed, low, high])
        records.append({
            "Method": method, "Target": target, "Grouping": grouping,
            "Analyte": analyte, "Treatment": treatment,
            "n": len(valid),
            "RMSE_original_units": rmse,
            "NRMSE_mean_observed": rmse / denominator if pd.notna(denominator) and denominator > 0 else np.nan,
            "IntervalCoverage": float(interval_valid["covered"].mean()) if len(interval_valid) else np.nan,
        })
    return pd.DataFrame(records)


def coverage_by_year_target(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ML":
        require_columns(frame, ["Target", "Year", "y_true", "pi_low", "pi_high"], "ML diagnostics")
        parts = [(str(target), group, "y_true", "pi_low", "pi_high") for target, group in frame.groupby("Target")]
    else:
        require_columns(frame, ["Year", "Observed", "Lower95", "Upper95"], f"{method} diagnostics")
        target = "Result_mg_L" if "Analyte" in frame else "Volume_L"
        parts = [(target, frame, "Observed", "Lower95", "Upper95")]
    records = []
    for target, part, observed, low, high in parts:
        by_year = {int(year): group for year, group in part.groupby("Year", dropna=False)}
        for year in STUDY_YEARS:
            group = by_year.get(year, part.iloc[0:0])
            complete = group.dropna(subset=[observed, low, high])
            covered = (
                pd.to_numeric(complete[observed], errors="coerce")
                .ge(pd.to_numeric(complete[low], errors="coerce"))
                & pd.to_numeric(complete[observed], errors="coerce")
                .le(pd.to_numeric(complete[high], errors="coerce"))
            )
            records.append({
                "Method": method,
                "Target": target,
                "Year": year,
                "n": int(len(complete)),
                "IntervalCoverage": float(covered.mean()) if len(complete) else np.nan,
            })
    return pd.DataFrame(records)


def publication_performance_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["RMSE"] = output["RMSE_original_units"].map(
        lambda value: "NA" if pd.isna(value) else f"{value:.3g}"
    )
    output["NRMSE"] = output["NRMSE_mean_observed"].map(
        lambda value: "NA" if pd.isna(value) else f"{value:.3f}"
    )
    output["Coverage"] = output["IntervalCoverage"].map(
        lambda value: "NA" if pd.isna(value) else f"{100 * value:.1f}%"
    )
    return output[[
        "Method", "Target", "Grouping", "Analyte", "Treatment",
        "n", "RMSE", "NRMSE", "Coverage",
    ]]


def overall_nrmse_table(performance: pd.DataFrame) -> pd.DataFrame:
    concentration = performance.loc[
        performance["Target"].eq("Result_mg_L")
        & performance["Grouping"].eq("analyte_overall")
    ].copy()
    concentration["DisplayTarget"] = concentration["Analyte"].astype(str)
    volume = performance.loc[
        performance["Target"].eq("Volume_L")
        & performance["Grouping"].eq("overall")
    ].copy()
    volume["DisplayTarget"] = "Volume"
    output = pd.concat([concentration, volume], ignore_index=True)
    order = {value: position for position, value in enumerate([*PUBLICATION_ANALYTES, "Volume"])}
    output["_order"] = output["DisplayTarget"].map(order)
    return output.sort_values(["_order", "Method"]).drop(columns="_order").reset_index(drop=True)


def publication_overall_nrmse_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["NRMSE_percent"] = output["NRMSE_mean_observed"].map(
        lambda value: "NA" if pd.isna(value) else f"{100 * value:.1f}%"
    )
    output["IntervalCoverage"] = output["IntervalCoverage"].map(
        lambda value: "NA" if pd.isna(value) else f"{100 * value:.1f}%"
    )
    output["diagnostic_basis"] = np.where(
        output["Method"].eq("ML"),
        "LOYO held-out predictions",
        "Bayesian posterior-predictive diagnostics",
    )
    return output[[
        "DisplayTarget", "Method", "n", "NRMSE_percent",
        "IntervalCoverage", "diagnostic_basis",
    ]]


def publication_coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Coverage"] = output["IntervalCoverage"].map(
        lambda value: "NA" if pd.isna(value) else f"{100 * value:.1f}%"
    )
    return output[["Method", "Target", "Year", "n", "Coverage"]]


def publication_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame, ["Target", "feature", "importance_mean", "importance_sd"],
        "ML feature importance",
    )
    output = frame.copy()
    output["Importance_mean_sd"] = output.apply(
        lambda row: (
            f"{row['importance_mean']:.3g}"
            if pd.isna(row["importance_sd"])
            else f"{row['importance_mean']:.3g} ({row['importance_sd']:.3g})"
        ),
        axis=1,
    )
    return output[["Target", "feature", "Importance_mean_sd", "interpretation"]]


def bayes_negative_sensitivity(
    bayes_annual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative_counts = (
        bayes_annual.groupby(["Analyte", "Treatment"], as_index=False)
        .agg(
            n_annual_draws=("Load_kg", "size"),
            n_negative_annual_draws=("Load_kg", lambda values: int(values.lt(0).sum())),
        )
    )
    negative_counts["percent_negative_annual_draws"] = (
        100.0
        * negative_counts["n_negative_annual_draws"]
        / negative_counts["n_annual_draws"]
    )
    variants = []
    raw = bayes_annual.copy()
    raw["Scenario"] = "raw_annual_draws_display_floor_only"
    variants.append(raw)
    truncated = bayes_annual.copy()
    truncated["Load_kg"] = truncated["Load_kg"].clip(lower=0)
    truncated["Scenario"] = "annual_draw_truncation_at_zero"
    variants.append(truncated)
    combined = pd.concat(variants, ignore_index=True)
    cumulative = (
        combined.groupby(["Scenario", "Analyte", "Treatment", "Draw"], as_index=False)
        .agg(CumulativeLoad_kg=("Load_kg", "sum"))
    )
    cumulative["Method"] = "BayesSensitivity"
    summary = summarize_draws(
        cumulative, ["Method", "Scenario", "Analyte", "Treatment"], "CumulativeLoad_kg"
    ).merge(
        negative_counts, on=["Analyte", "Treatment"], how="left", validate="many_to_one"
    )
    summary["lower_95_display"] = summary["lower_95"].clip(lower=0)
    return cumulative, summary


def publication_bayes_negative_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Median_95_display_interval_kg"] = output.apply(
        lambda row: (
            f"{row['median']:.4g} ({row['lower_95_display']:.4g}, {row['upper_95']:.4g})"
        ),
        axis=1,
    )
    return output[[
        "Scenario", "Analyte", "Treatment", "n_annual_draws",
        "n_negative_annual_draws", "percent_negative_annual_draws",
        "Median_95_display_interval_kg", "lower_95",
    ]].rename(columns={"lower_95": "raw_lower_95_kg"})


def disagreement_table(bayes: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        bayes, ["ConcentrationObservationID", "PhysicalEventID", "Predicted"],
        "Bayesian row diagnostics",
    )
    ml = ml.loc[ml["Target"].eq("Result_mg_L")].copy()
    require_columns(ml, ["ConcentrationObservationID", "PhysicalEventID", "y_pred"], "ML row diagnostics")
    joined = bayes.merge(
        ml[["ConcentrationObservationID", "PhysicalEventID", "y_true", "y_pred", "standardized_discrepancy"]],
        on=["ConcentrationObservationID", "PhysicalEventID"], how="inner", validate="one_to_one",
        suffixes=("_Bayes", "_ML"),
    )
    joined["BayesMLAbsolutePredictionDifference_mg_L"] = (
        pd.to_numeric(joined["Predicted"], errors="coerce") - pd.to_numeric(joined["y_pred"], errors="coerce")
    ).abs()
    joined["review_note"] = "Descriptive disagreement only; no automatic error label or exclusion."
    return joined.sort_values("BayesMLAbsolutePredictionDifference_mg_L", ascending=False)


def publication_load_table(summary: pd.DataFrame, value_label: str) -> pd.DataFrame:
    output = summary.copy()
    center_column = "primary_center" if "primary_center" in output else "median"
    output[value_label] = output.apply(
        lambda row: (
            f"{row[center_column]:.4g} "
            f"({row['lower_95']:.4g}, {row['upper_95']:.4g})"
        ),
        axis=1,
    )
    return output[[
        column for column in output
        if column not in {
            "mean", "median", "primary_center", "lower_95", "upper_95", "n_draws"
        }
    ]]


def figure_slug(value: object) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def save_figure(
    figure: plt.Figure,
    *stems: Path,
    dpi: int = 220,
    layout_top: float = 0.96,
) -> None:
    figure.tight_layout(rect=(0, 0, 1, layout_top))
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".jpg"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def read_observed_annual_summary(bayes_dir: Path) -> pd.DataFrame:
    """Read the event-bootstrap observed annual summaries used by Bayes v3p0."""
    source = read_required(
        bayes_dir / "annual_load_summary_bayes_plus_observed_v3p0_physical_event.csv",
        "Bayesian modeled-plus-observed annual summary",
    )
    require_columns(
        source,
        ["source", "Year", "analyte", "treatment", "load_mean", "load_low", "load_high"],
        "Bayesian modeled-plus-observed annual summary",
    )
    observed = source.loc[source["source"].astype(str).eq("Observed")].copy()
    observed = observed.rename(columns={"analyte": "Analyte", "treatment": "Treatment"})
    observed = observed.loc[observed["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)].copy()
    observed["Year"] = pd.to_numeric(observed["Year"], errors="raise").astype(int)
    # The Bayes annual display table stores load summaries in grams; comparison
    # ledgers and tables use kilograms.
    observed["center_kg"] = pd.to_numeric(observed["load_mean"], errors="raise") / 1000.0
    observed["lower_95_kg"] = pd.to_numeric(observed["load_low"], errors="raise") / 1000.0
    observed["upper_95_kg"] = pd.to_numeric(observed["load_high"], errors="raise") / 1000.0
    key = ["Year", "Analyte", "Treatment"]
    if observed.duplicated(key).any():
        raise ValueError("Observed annual summary is not unique by Year x Analyte x Treatment.")
    return observed[key + ["center_kg", "lower_95_kg", "upper_95_kg"]]


def plot_annual_comparison(
    annual: pd.DataFrame,
    observed: pd.DataFrame,
    figure_dir: Path,
) -> None:
    primary = annual.loc[
        ((annual["Method"] == "Bayes") & (annual["Scenario"] == "model_only"))
        | ((annual["Method"] == "ML") & (annual["Scenario"] == "full_record_model_only"))
    ].copy()
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    line_styles = {"Bayes": "-", "ML": "--"}
    legend_handles = [
        Line2D(
            [0], [0], color=colors["Bayes"], marker="o", linewidth=2,
            label="Bayes posterior median + 95% credible interval",
        ),
        Line2D(
            [0], [0], color=colors["ML"], marker="o", linestyle="--", linewidth=2,
            label="ML physical-event point total + 95% calibration-residual PI",
        ),
        Line2D(
            [0], [0], color="black", marker="o", markerfacecolor="none",
            linestyle="none", markersize=8, label="Observed (event-bootstrap 95% CI)",
        ),
    ]
    for analyte, group in primary.groupby("Analyte", sort=False):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
        for axis, treatment in zip(axes, TREATMENTS):
            part = group.loc[group["Treatment"].eq(treatment)]
            for method in ["Bayes", "ML"]:
                method_rows = part.loc[part["Method"].eq(method)].sort_values("Year")
                if method_rows.empty:
                    continue
                years = pd.to_numeric(method_rows["Year"], errors="raise").to_numpy(dtype=float)
                center_column = (
                    "primary_center" if "primary_center" in method_rows else "median"
                )
                center = pd.to_numeric(
                    method_rows[center_column], errors="raise"
                ).to_numpy(dtype=float)
                lower = np.maximum(
                    pd.to_numeric(method_rows["lower_95"], errors="raise").to_numpy(dtype=float), 0.0
                )
                upper = pd.to_numeric(method_rows["upper_95"], errors="raise").to_numpy(dtype=float)
                axis.fill_between(years, lower, upper, color=colors[method], alpha=0.17, linewidth=0)
                axis.plot(
                    years, center, color=colors[method], linestyle=line_styles[method],
                    marker="o", markersize=4.5, linewidth=2,
                )
            observed_rows = observed.loc[
                observed["Analyte"].astype(str).eq(str(analyte))
                & observed["Treatment"].astype(str).eq(treatment)
            ].sort_values("Year")
            if not observed_rows.empty:
                center = observed_rows["center_kg"].to_numpy(dtype=float)
                lower = observed_rows["lower_95_kg"].to_numpy(dtype=float)
                upper = observed_rows["upper_95_kg"].to_numpy(dtype=float)
                axis.errorbar(
                    observed_rows["Year"].to_numpy(dtype=float), center,
                    yerr=np.vstack([np.maximum(center - lower, 0), np.maximum(upper - center, 0)]),
                    fmt="o", markerfacecolor="none", markeredgecolor="black",
                    markeredgewidth=1.5, color="black", ecolor="black",
                    elinewidth=1.25, capsize=3, markersize=7, zorder=10,
                )
            axis.set_title(treatment)
            axis.set_xticks(STUDY_YEARS[::2])
            axis.grid(True, alpha=0.22)
            axis.set_ylim(bottom=0)
        axes[0].set_ylabel("Annual load (kg)")
        axes[1].set_xlabel("Year")
        figure.suptitle(
            f"{analyte}: observed vs Bayes vs ML annual load (corrected physical-event)",
            fontsize=14,
        )
        figure.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
        safe = figure_slug(analyte)
        save_figure(
            figure,
            figure_dir / f"annual_load_{analyte}",
            figure_dir / "annual_obs_vs_modeled" / f"annual_load_{safe}_obs_vs_modeled_v3p0",
            layout_top=0.88,
        )


def plot_cumulative_comparison(cumulative: pd.DataFrame, figure_dir: Path) -> None:
    primary = cumulative.loc[
        ((cumulative["Method"] == "Bayes") & (cumulative["Scenario"] == "model_only"))
        | ((cumulative["Method"] == "ML") & (cumulative["Scenario"] == "full_record_model_only"))
    ].copy()
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    offsets = {"Bayes": -0.09, "ML": 0.09}
    figure, axes = plt.subplots(5, 2, figsize=(14, 24))
    for axis, analyte in zip(axes.flat, PUBLICATION_ANALYTES):
        group = primary.loc[primary["Analyte"].eq(analyte)]
        for method in ["Bayes", "ML"]:
            rows = group.loc[group["Method"].eq(method)].set_index("Treatment").reindex(TREATMENTS)
            center_column = "primary_center" if "primary_center" in rows else "median"
            center = pd.to_numeric(rows[center_column], errors="coerce").to_numpy(dtype=float)
            lower = np.maximum(pd.to_numeric(rows["lower_95"], errors="coerce").to_numpy(dtype=float), 0)
            upper = pd.to_numeric(rows["upper_95"], errors="coerce").to_numpy(dtype=float)
            x = np.arange(len(TREATMENTS), dtype=float) + offsets[method]
            axis.errorbar(
                x, center, yerr=np.vstack([np.maximum(center - lower, 0), np.maximum(upper - center, 0)]),
                fmt="o", color=colors[method], capsize=3, elinewidth=1.4, label=method,
            )
        axis.set_title(analyte)
        axis.set_xticks(np.arange(len(TREATMENTS)), TREATMENTS)
        axis.set_ylabel("Cumulative load (kg)")
        axis.grid(True, axis="y", alpha=0.22)
        axis.set_ylim(bottom=0)
    figure.suptitle(
        "Study-period cumulative loads, 2011-2025\n"
        "Bayes posterior medians; ML physical-event point totals; 95% intervals",
        fontsize=15,
    )
    axes.flat[0].legend(loc="best")
    save_figure(
        figure,
        figure_dir / "cumulative_loads_2011_2025",
        figure_dir / "postprocessing" / "study_period_cumulative_loads",
        layout_top=0.975,
    )


def plot_ct_relative(ct_summary: pd.DataFrame, figure_dir: Path) -> None:
    primary = ct_summary.loc[
        ((ct_summary["Method"] == "Bayes") & (ct_summary["Scenario"] == "model_only"))
        | ((ct_summary["Method"] == "ML") & (ct_summary["Scenario"] == "full_record_model_only"))
    ].copy()
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    offsets = {"Bayes": -0.09, "ML": 0.09}
    comparisons = ["MT", "ST"]
    figure, axes = plt.subplots(5, 2, figsize=(14, 24))
    for axis, analyte in zip(axes.flat, PUBLICATION_ANALYTES):
        group = primary.loc[primary["Analyte"].eq(analyte)]
        for method in ["Bayes", "ML"]:
            rows = group.loc[group["Method"].eq(method)].set_index("ComparisonTreatment").reindex(comparisons)
            center_column = "primary_center" if "primary_center" in rows else "median"
            center = pd.to_numeric(rows[center_column], errors="coerce").to_numpy(dtype=float)
            lower = pd.to_numeric(rows["lower_95"], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(rows["upper_95"], errors="coerce").to_numpy(dtype=float)
            x = np.arange(len(comparisons), dtype=float) + offsets[method]
            axis.errorbar(
                x, center, yerr=np.vstack([np.maximum(center - lower, 0), np.maximum(upper - center, 0)]),
                fmt="o", color=colors[method], capsize=3, elinewidth=1.4, label=method,
            )
        axis.axhline(0, color="0.35", linewidth=1)
        axis.set_title(analyte)
        axis.set_xticks(np.arange(len(comparisons)), ["MT vs CT", "ST vs CT"])
        axis.set_ylabel("Percent difference relative to CT")
        axis.grid(True, axis="y", alpha=0.22)
    figure.suptitle(
        "CT-relative treatment differences\n"
        "Bayes posterior medians; ML physical-event point totals; 95% intervals",
        fontsize=15,
    )
    axes.flat[0].legend(loc="best")
    save_figure(
        figure, figure_dir / "postprocessing" / "treatment_differences_vs_ct",
        layout_top=0.975,
    )


def plot_performance_comparison(performance: pd.DataFrame, figure_dir: Path) -> None:
    concentration = performance.loc[
        performance["Target"].eq("Result_mg_L")
        & performance["Grouping"].eq("analyte_treatment")
    ].copy()
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    offsets = {"Bayes": -0.18, "ML": 0.18}
    for column, ylabel, stem, multiplier in [
        ("RMSE_original_units", "RMSE (mg/L)", "concentration_rmse_original_units", 1.0),
        ("NRMSE_mean_observed", "Mean-normalized RMSE (%)", "nrmse_mean_comparison", 100.0),
    ]:
        figure, axes = plt.subplots(5, 2, figsize=(14, 23))
        for axis, analyte in zip(axes.flat, PUBLICATION_ANALYTES):
            group = concentration.loc[concentration["Analyte"].eq(analyte)]
            for method in ["Bayes", "ML"]:
                rows = group.loc[group["Method"].eq(method)].set_index("Treatment").reindex(TREATMENTS)
                values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float) * multiplier
                axis.bar(
                    np.arange(len(TREATMENTS), dtype=float) + offsets[method], values,
                    width=0.34, color=colors[method], alpha=0.85, label=method,
                )
            axis.set_title(analyte)
            axis.set_xticks(np.arange(len(TREATMENTS)), TREATMENTS)
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", alpha=0.22)
        figure.suptitle(
            f"Concentration-model {ylabel} by analyte and treatment\n"
            "Bayes posterior-predictive diagnostics; ML LOYO diagnostics",
            fontsize=15,
        )
        axes.flat[0].legend(loc="best")
        save_figure(figure, figure_dir / "postprocessing" / stem, layout_top=0.965)

    overall = overall_nrmse_table(performance)
    display_order = [*PUBLICATION_ANALYTES, "Volume"]
    figure, axis = plt.subplots(figsize=(15, 6.5))
    positions = np.arange(len(display_order), dtype=float)
    for method in ["Bayes", "ML"]:
        rows = (
            overall.loc[overall["Method"].eq(method)]
            .set_index("DisplayTarget")
            .reindex(display_order)
        )
        values = (
            pd.to_numeric(rows["NRMSE_mean_observed"], errors="coerce")
            .to_numpy(dtype=float)
            * 100.0
        )
        axis.bar(
            positions + offsets[method],
            values,
            width=0.34,
            color=colors[method],
            alpha=0.85,
            label=method,
        )
    axis.set_xticks(positions, display_order, rotation=35, ha="right")
    axis.set_ylabel("Mean-normalized RMSE (%)")
    axis.set_title(
        "Overall normalized RMSE by analyte and event volume\n"
        "Pooled across treatments; Bayes posterior-predictive diagnostics; ML LOYO diagnostics"
    )
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(loc="best")
    save_figure(
        figure,
        figure_dir / "gof_nrmse_mean_by_analyte",
        figure_dir / "postprocessing" / "nrmse_mean_overall_analytes_and_volume",
        layout_top=0.90,
    )


def plot_coverage_comparison(
    loyo_coverage: pd.DataFrame,
    bayes_coverage: pd.DataFrame,
    figure_dir: Path,
) -> None:
    coverage = pd.concat([bayes_coverage, loyo_coverage], ignore_index=True)
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, target in zip(axes, ["Result_mg_L", "Volume_L"]):
        group = coverage.loc[coverage["Target"].eq(target)]
        for method in ["Bayes", "ML"]:
            rows = group.loc[group["Method"].eq(method)].sort_values("Year")
            axis.plot(
                rows["Year"], rows["IntervalCoverage"], marker="o", linewidth=2,
                color=colors[method], label=method,
            )
        axis.axhline(0.95, color="0.35", linestyle="--", linewidth=1.2, label="Nominal 95%")
        axis.set_title("Concentration" if target == "Result_mg_L" else "Event volume")
        axis.set_xlabel("Year")
        axis.set_xticks(STUDY_YEARS[::2])
        axis.set_ylim(0, 1.03)
        axis.grid(True, axis="y", alpha=0.22)
    axes[0].set_ylabel("Interval coverage")
    axes[-1].legend(loc="best")
    figure.suptitle(
        "95% prediction-interval coverage by year "
        "(Bayes posterior predictive; ML LOYO split conformal)"
    )
    save_figure(
        figure, figure_dir / "postprocessing" / "loyo_interval_coverage_by_year",
        layout_top=0.90,
    )


def plot_feature_importance_comparison(feature_importance: pd.DataFrame, figure_dir: Path) -> None:
    for target, stem in [
        ("Concentration", "feature_importance_concentration"),
        ("Volume", "feature_importance_volume"),
    ]:
        rows = feature_importance.loc[feature_importance["Target"].eq(target)].copy()
        rows["importance_mean"] = pd.to_numeric(rows["importance_mean"], errors="coerce")
        rows["importance_sd"] = pd.to_numeric(rows["importance_sd"], errors="coerce").fillna(0)
        rows = rows.dropna(subset=["importance_mean"]).nlargest(20, "importance_mean").sort_values("importance_mean")
        figure, axis = plt.subplots(figsize=(11, max(6, 0.38 * len(rows) + 1.5)))
        axis.barh(
            rows["feature"].astype(str), rows["importance_mean"],
            xerr=rows["importance_sd"], color="#7b6d8d", alpha=0.9,
            error_kw={"ecolor": "black", "elinewidth": 1.2, "capsize": 2},
        )
        axis.set_title(f"{target} model: descriptive, noncausal feature importance")
        axis.set_xlabel("CatBoost feature importance (mean ± SD)")
        axis.grid(True, axis="x", alpha=0.22)
        save_figure(figure, figure_dir / "postprocessing" / stem)


def make_figures(
    annual: pd.DataFrame,
    cumulative: pd.DataFrame,
    ct_summary: pd.DataFrame,
    performance: pd.DataFrame,
    loyo_coverage: pd.DataFrame,
    bayes_coverage: pd.DataFrame,
    feature_importance: pd.DataFrame,
    observed_annual: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_annual_comparison(annual, observed_annual, figure_dir)
    plot_cumulative_comparison(cumulative, figure_dir)
    plot_ct_relative(ct_summary, figure_dir)
    plot_performance_comparison(performance, figure_dir)
    plot_coverage_comparison(loyo_coverage, bayes_coverage, figure_dir)
    plot_feature_importance_comparison(feature_importance, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Regenerate figures from saved comparison tables without re-reading draw ledgers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    output_roots = PHYSICAL_EVENT_CONFIG["output_roots"]
    bayes_dir = repo / output_roots["bayesian_results"]
    ml_dir = repo / output_roots["ml_results"]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir else repo / output_roots["comparison_results"]
    )
    figure_dir = (
        Path(args.figure_dir).resolve()
        if args.figure_dir else repo / output_roots["comparison_figures"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [
        bayes_dir / "run_manifest_bayes_v3p0_physical_event.json",
        ml_dir / "run_manifest_ml_v3p0_physical_event.json",
    ]
    validate_corrected_artifact_metadata(manifests, expected_years=STUDY_YEARS)

    if args.figures_only:
        make_figures(
            read_required(output_dir / "annual_load_summary_raw.csv", "saved annual-load summary"),
            read_required(
                output_dir / "cumulative_load_2011_2025_raw.csv",
                "saved cumulative-load summary",
            ),
            read_required(output_dir / "ct_relative_summary_raw.csv", "saved CT-relative summary"),
            read_required(
                output_dir / "performance_and_calibration_raw.csv",
                "saved performance and calibration table",
            ),
            read_required(
                output_dir / "loyo_interval_coverage_by_year_target_raw.csv",
                "saved ML coverage table",
            ),
            read_required(
                output_dir / "bayes_interval_coverage_by_year_target_raw.csv",
                "saved Bayesian coverage table",
            ),
            read_required(
                output_dir / "feature_importance_descriptive_noncausal_raw.csv",
                "saved feature-importance table",
            ),
            read_observed_annual_summary(bayes_dir),
            figure_dir,
        )
        print(f"[DONE] Corrected comparison figures regenerated from saved tables in {figure_dir}")
        return

    bayes_ledger = normalize_ledger(
        read_required(
            bayes_dir / "event_analyte_draw_ledger_bayes_v3p0_physical_event.csv",
            "Bayesian event ledger",
        ), "Bayes", "model_only",
    )
    ml_loyo = normalize_ledger(
        read_required(ml_dir / "event_analyte_draw_ledger_ml_v3p0.csv", "ML LOYO event ledger"),
        "ML", "loyo_model_only", require_complete_years=False,
    )
    ml_full = normalize_ledger(
        read_required(
            ml_dir / "event_analyte_draw_ledger_full_record_model_only.csv",
            "ML full-record model-only event ledger",
        ), "ML", "full_record_model_only",
    )
    ml_sensitivity = normalize_ledger(
        read_required(
            ml_dir / "event_analyte_draw_ledger_observed_plus_imputed_sensitivity.csv",
            "ML observed-plus-imputed event ledger",
        ), "ML", "observed_plus_imputed_sensitivity",
    )
    ml_full_points = normalize_point_ledger(
        read_required(
            ml_dir / "event_analyte_point_ledger_full_record_model_only.csv",
            "ML full-record model-only point ledger",
        ),
        "ML",
        "full_record_model_only",
    )
    ml_sensitivity_points = normalize_point_ledger(
        read_required(
            ml_dir / "event_analyte_point_ledger_observed_plus_imputed_sensitivity.csv",
            "ML observed-plus-imputed point ledger",
        ),
        "ML",
        "observed_plus_imputed_sensitivity",
    )
    bayes_ledger, bayes_extra_analytes = restrict_publication_analytes(
        bayes_ledger, "Bayesian ledger", require_all=True
    )
    if bayes_extra_analytes:
        raise ValueError(
            f"Bayesian ledger unexpectedly contains non-publication analytes: {bayes_extra_analytes}."
        )
    ml_loyo, ml_loyo_extra_analytes = restrict_publication_analytes(
        ml_loyo, "ML LOYO ledger", require_all=False
    )
    ml_full, ml_full_extra_analytes = restrict_publication_analytes(
        ml_full, "ML full-record ledger", require_all=True
    )
    ml_sensitivity, ml_sensitivity_extra_analytes = restrict_publication_analytes(
        ml_sensitivity, "ML observed-plus-imputed ledger", require_all=True
    )
    ml_full_points, ml_full_point_extras = restrict_publication_analytes(
        ml_full_points, "ML full-record point ledger", require_all=True
    )
    ml_sensitivity_points, ml_sensitivity_point_extras = restrict_publication_analytes(
        ml_sensitivity_points,
        "ML observed-plus-imputed point ledger",
        require_all=True,
    )
    assert_complete_analyte_treatment_years(bayes_ledger, "Bayesian ledger")
    assert_complete_analyte_treatment_years(ml_full, "ML full-record ledger")
    assert_complete_analyte_treatment_years(
        ml_sensitivity, "ML observed-plus-imputed ledger"
    )
    assert_complete_analyte_treatment_years(
        ml_full_points, "ML full-record point ledger"
    )
    assert_complete_analyte_treatment_years(
        ml_sensitivity_points, "ML observed-plus-imputed point ledger"
    )
    # LOYO ledgers only exist where held-out volume observations are available.
    # Keep them as validation diagnostics; complete-year annual products use the
    # corrected full-record ledgers and must never synthesize missing years.
    all_ledgers = pd.concat([bayes_ledger, ml_full, ml_sensitivity], ignore_index=True)
    annual_draws, annual_summary = annual_products(all_ledgers)
    all_point_ledgers = pd.concat(
        [ml_full_points, ml_sensitivity_points], ignore_index=True
    )
    annual_points = annual_point_products(all_point_ledgers)
    annual_summary = attach_primary_centers(
        annual_summary,
        annual_points,
        keys=["Method", "Scenario", "Year", "Analyte", "Treatment"],
        point_column="PointTotal_kg",
    )
    cumulative_draws, cumulative_summary = cumulative_products(annual_draws)
    cumulative_points = cumulative_point_products(annual_points)
    cumulative_summary = attach_primary_centers(
        cumulative_summary,
        cumulative_points,
        keys=["Method", "Scenario", "Analyte", "Treatment"],
        point_column="PointCumulativeLoad_kg",
    )
    ct_raw, ct_summary = ct_relative(cumulative_draws)
    ct_points = ct_relative_point_products(cumulative_points)
    ct_summary = attach_primary_centers(
        ct_summary,
        ct_points,
        keys=["Method", "Scenario", "Analyte", "ComparisonTreatment"],
        point_column="PointPercentDifference",
    )
    spearman_raw, spearman_publication = spearman_tables(annual_summary)

    bayes_rows = read_required(
        bayes_dir / "row_prediction_diagnostics_bayes_v3p0_physical_event.csv",
        "Bayesian row predictions",
    )
    bayes_volume = read_required(
        bayes_dir / "volume_prediction_diagnostics_bayes_v3p0_physical_event.csv",
        "Bayesian volume predictions",
    )
    ml_rows = read_required(ml_dir / "row_level_residual_diagnostics.csv", "ML row diagnostics")
    require_columns(ml_rows, ["Target", "Analyte"], "ML row diagnostics")
    ml_rows = ml_rows.loc[
        ml_rows["Target"].eq("Volume_L")
        | ml_rows["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)
    ].copy()
    performance = pd.concat([
        performance_table(bayes_rows, "Bayes", "Result_mg_L"),
        performance_table(bayes_volume, "Bayes", "Volume_L"),
        performance_table(ml_rows, "ML", "Result_mg_L"),
        performance_table(ml_rows, "ML", "Volume_L"),
    ], ignore_index=True)
    loyo_coverage = coverage_by_year_target(ml_rows, "ML")
    bayes_coverage = pd.concat([
        coverage_by_year_target(bayes_rows, "Bayes"),
        coverage_by_year_target(bayes_volume, "Bayes"),
    ], ignore_index=True)
    disagreement = disagreement_table(bayes_rows, ml_rows)
    negative_sensitivity_draws, negative_sensitivity = bayes_negative_sensitivity(
        annual_draws.loc[
            (annual_draws["Method"] == "Bayes") & (annual_draws["Scenario"] == "model_only")
        ]
    )
    negative_ct_raw, negative_ct_summary = ct_relative(negative_sensitivity_draws)

    observed_ledger = read_required(
        bayes_dir / "observed_event_analyte_ledger_v3p0_physical_event.csv",
        "corrected observed event ledger",
    )
    require_columns(observed_ledger, ["PhysicalEventID", "Analyte", "Year", "Treatment", "Load_kg"], "Observed ledger")
    observed_ledger = observed_ledger.loc[
        observed_ledger["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)
    ].copy()
    if observed_ledger.duplicated(["PhysicalEventID", "Analyte"]).any():
        raise ValueError("Observed ledger is not unique by PhysicalEventID x Analyte.")
    observed_subtotals = (
        observed_ledger.groupby(["Analyte", "Treatment"], as_index=False)
        .agg(ObservedSubtotal_kg=("Load_kg", "sum"), n_observed_events=("PhysicalEventID", "nunique"))
    )
    observed_subtotals_publication = observed_subtotals.copy()
    observed_subtotals_publication["ObservedSubtotal_kg"] = (
        observed_subtotals_publication["ObservedSubtotal_kg"].map(lambda value: f"{value:.4g}")
    )

    feature_frames = []
    for target, filename in [
        ("Concentration", "feature_importance_logC.csv"),
        ("Volume", "feature_importance_logV.csv"),
    ]:
        feature = read_required(ml_dir / filename, f"ML {target} feature importance")
        feature["Target"] = target
        feature["interpretation"] = "Descriptive CatBoost importance; noncausal."
        feature_frames.append(feature)
    feature_importance = pd.concat(feature_frames, ignore_index=True)

    annual_draws.to_csv(output_dir / "annual_load_draws_raw.csv", index=False)
    annual_points.to_csv(output_dir / "annual_load_point_totals_raw.csv", index=False)
    annual_summary.to_csv(output_dir / "annual_load_summary_raw.csv", index=False)
    publication_load_table(annual_summary, "Central_95_interval_kg").to_csv(
        output_dir / "annual_load_summary_publication.csv", index=False
    )
    cumulative_points.to_csv(output_dir / "cumulative_load_point_totals_raw.csv", index=False)
    cumulative_summary.to_csv(output_dir / "cumulative_load_2011_2025_raw.csv", index=False)
    publication_load_table(cumulative_summary, "Central_95_interval_kg").to_csv(
        output_dir / "cumulative_load_2011_2025_publication.csv", index=False
    )
    ct_raw.to_csv(output_dir / "ct_relative_draws_raw.csv", index=False)
    ct_points.to_csv(output_dir / "ct_relative_point_totals_raw.csv", index=False)
    ct_summary.to_csv(output_dir / "ct_relative_summary_raw.csv", index=False)
    publication_load_table(ct_summary, "Central_95_interval_percent").to_csv(
        output_dir / "ct_relative_summary_publication.csv", index=False
    )
    alignment_audit = pd.concat([
        annual_summary.assign(product="annual_load").rename(
            columns={"Year": "time_or_comparison"}
        ),
        cumulative_summary.assign(
            product="cumulative_load", time_or_comparison="2011-2025"
        ),
        ct_summary.assign(product="ct_relative").rename(
            columns={"ComparisonTreatment": "time_or_comparison"}
        ),
    ], ignore_index=True, sort=False)
    alignment_audit.loc[
        alignment_audit["Method"].eq("ML"),
        [
            "product", "Method", "Scenario", "Analyte", "Treatment",
            "time_or_comparison", "primary_center", "lower_95", "upper_95",
            "primary_center_within_draw_interval", "primary_center_type",
        ],
    ].to_csv(output_dir / "ml_point_center_interval_alignment_audit.csv", index=False)
    spearman_raw.to_csv(output_dir / "temporal_spearman_raw.csv", index=False)
    spearman_publication.to_csv(output_dir / "temporal_spearman_publication.csv", index=False)
    (output_dir / "temporal_spearman_footnote.md").write_text(
        "Spearman rho compares paired Bayesian and ML annual central estimates within "
        "analyte and treatment. Parentheses give paired-year n. * indicates unadjusted "
        "p < 0.05; these exploratory tests are not multiplicity-adjusted.\n",
        encoding="utf-8",
    )
    performance.to_csv(output_dir / "performance_and_calibration_raw.csv", index=False)
    publication_performance_table(performance).to_csv(
        output_dir / "performance_and_calibration_publication.csv", index=False
    )
    nrmse_overall = overall_nrmse_table(performance)
    nrmse_overall.to_csv(
        output_dir / "nrmse_overall_analytes_and_volume_raw.csv", index=False
    )
    publication_overall_nrmse_table(nrmse_overall).to_csv(
        output_dir / "nrmse_overall_analytes_and_volume_publication.csv", index=False
    )
    loyo_coverage.to_csv(output_dir / "loyo_interval_coverage_by_year_target_raw.csv", index=False)
    publication_coverage_table(loyo_coverage).to_csv(
        output_dir / "loyo_interval_coverage_by_year_target_publication.csv", index=False
    )
    bayes_coverage.to_csv(output_dir / "bayes_interval_coverage_by_year_target_raw.csv", index=False)
    publication_coverage_table(bayes_coverage).to_csv(
        output_dir / "bayes_interval_coverage_by_year_target_publication.csv", index=False
    )
    feature_importance.to_csv(
        output_dir / "feature_importance_descriptive_noncausal_raw.csv", index=False
    )
    publication_feature_importance(feature_importance).to_csv(
        output_dir / "feature_importance_descriptive_noncausal_publication.csv", index=False
    )
    observed_subtotals.to_csv(output_dir / "observed_subtotals_raw.csv", index=False)
    observed_subtotals_publication.to_csv(
        output_dir / "observed_subtotals_publication.csv", index=False
    )
    negative_sensitivity_draws.to_csv(
        output_dir / "bayes_negative_draw_sensitivity_draws_raw.csv", index=False
    )
    negative_sensitivity.to_csv(
        output_dir / "bayes_negative_draw_sensitivity_summary_raw.csv", index=False
    )
    publication_bayes_negative_sensitivity(negative_sensitivity).to_csv(
        output_dir / "bayes_negative_draw_sensitivity_publication.csv", index=False
    )
    negative_ct_raw.to_csv(
        output_dir / "bayes_negative_draw_sensitivity_ct_relative_draws_raw.csv", index=False
    )
    negative_ct_summary.to_csv(
        output_dir / "bayes_negative_draw_sensitivity_ct_relative_summary_raw.csv", index=False
    )
    publication_load_table(
        negative_ct_summary, "Median_95_interval_percent"
    ).to_csv(
        output_dir / "bayes_negative_draw_sensitivity_ct_relative_publication.csv", index=False
    )
    disagreement.to_csv(output_dir / "cross_model_observation_disagreement.csv", index=False)
    (output_dir / "central_estimate_and_interval_definitions.md").write_text(
        "# Central estimates and uncertainty intervals\n\n"
        "- Bayes line: posterior median of annual physical-event load draws; band: "
        "95% posterior credible interval.\n"
        "- ML line: deterministic sum of physical-event point loads; band: 95% "
        "Monte Carlo empirical calibration-residual prediction interval. Signed "
        "log-scale residuals are resampled from the physical-event-grouped "
        "split-conformal calibration set, with performance evaluated by outer LOYO. "
        "The ML line is not "
        "the median of the propagated draws.\n"
        "- Observed markers: corrected physical-event observed annual summaries, "
        "kept separate from both model-only products; error bars are the existing "
        "event-bootstrap 95% intervals.\n\n"
        "A prediction interval quantifies uncertainty for predicted outcomes. It is "
        "not a confidence interval for a fitted parameter. The event-level split-"
        "conformal coverage guarantee does not automatically become a 95% frequentist "
        "coverage guarantee for the summed annual Monte Carlo band. The saved "
        "ml_point_center_interval_alignment_audit.csv reports whether each point "
        "total lies within its propagated draw interval.\n",
        encoding="utf-8",
    )
    if not args.skip_figures:
        make_figures(
            annual_summary,
            cumulative_summary,
            ct_summary,
            performance,
            loyo_coverage,
            bayes_coverage,
            feature_importance,
            read_observed_annual_summary(bayes_dir),
            figure_dir,
        )

    manifest = {
        "workflow_version": CORRECTED_VERSION,
        "event_unit": "PhysicalEventID",
        "years": STUDY_YEARS,
        "legacy_fallbacks": False,
        "missing_year_zero_fill": False,
        "event_analyte_draw_uniqueness_asserted": True,
        "event_analyte_point_uniqueness_asserted": True,
        "primary_bayes_central_estimate": "posterior_median",
        "primary_ml_central_estimate": "physical_event_point_total",
        "primary_observed_role": "separate_reference_markers_only",
        "ml_interval_type": "95% Monte Carlo empirical calibration-residual prediction interval",
        "ml_interval_evaluation": "outer leave-one-year-out",
        "ml_monte_carlo_propagation": "weighted_resampling_of_signed_log_scale_split_conformal_calibration_residuals",
        "ml_interval_is_parameter_confidence_interval": False,
        "loyo_ledger_role": "diagnostics_only_because_years_without_observed_volume_are_absent",
        "loyo_ledger_years_present": sorted(ml_loyo["Year"].unique().astype(int).tolist()),
        "publication_analytes": PUBLICATION_ANALYTES,
        "excluded_ml_only_analytes": sorted(set(
            ml_loyo_extra_analytes
            + ml_full_extra_analytes
            + ml_sensitivity_extra_analytes
            + ml_full_point_extras
            + ml_sensitivity_point_extras
        )),
        "inputs": [str(path) for path in manifests],
    }
    (output_dir / "run_manifest_comparison_v3p0_physical_event.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Corrected comparison outputs written to {output_dir}")


if __name__ == "__main__":
    main()
