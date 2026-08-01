#!/usr/bin/env python3
"""Compare completed Bayesian v3p3 and ML v3p4 physical-event outputs.

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
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.physical_event import (  # noqa: E402
    aggregate_replicate_mean,
    observed_annual_plot_summary,
    resolve_prediction_draws,
    validate_corrected_artifact_metadata,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "physical_event_v3p4.json"
)
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Comparison configuration is absent: {CONFIG_PATH}")
PHYSICAL_EVENT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
WORKFLOW_VERSION = str(PHYSICAL_EVENT_CONFIG["workflow_version"])
BAYES_VERSION = str(PHYSICAL_EVENT_CONFIG["versions"]["bayesian"])
ML_VERSION = str(PHYSICAL_EVENT_CONFIG["versions"]["ml"])
COMPARISON_VERSION = str(PHYSICAL_EVENT_CONFIG["versions"]["comparison"])
BAYES_TAG = BAYES_VERSION.removesuffix("_physical_event")
ML_TAG = ML_VERSION.removesuffix("_physical_event")
COMPARISON_TAG = COMPARISON_VERSION.removesuffix("_physical_event")


STUDY_YEARS = list(range(2011, 2026))
TREATMENTS = ["CT", "MT", "ST"]
PUBLICATION_ANALYTES = ["NH4", "NO3", "NO2", "OP", "Se", "TDS", "TKN", "TN", "TP", "TSS"]
PRIMARY_MANUSCRIPT_ANALYTES = ["TSS", "TP", "TN"]
BAYES_POSTERIOR_FIT = "posterior_predictive_fit"
ML_FULL_RECORD_RECONSTRUCTION = "full_record_physical_event_reconstruction"
ML_OUTER_LOYO_VALIDATION = "outer_loyo_physical_event_validation"


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
        output, ["PhysicalEventID", "Analyte", "Year", "Treatment", "Rep", "Draw"],
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
        ["PhysicalEventID", "Analyte", "Year", "Treatment", "Rep"],
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
    plot_totals, draws = aggregate_replicate_mean(
        ledgers,
        value_column="Load_kg",
        group_columns=["Method", "Scenario", "Year", "Analyte", "Treatment"],
        draw_column="Draw",
        plot_total_column="PlotAnnualLoad_kg",
        treatment_mean_column="Load_kg",
    )
    draws["annual_reporting_unit"] = "mean_per_treatment_plot"
    assert_study_years(draws, "Combined annual draws")
    summary = summarize_draws(
        draws, ["Method", "Scenario", "Year", "Analyte", "Treatment"], "Load_kg"
    )
    return draws, summary


def annual_point_products(point_ledgers: pd.DataFrame) -> pd.DataFrame:
    _, points = aggregate_replicate_mean(
        point_ledgers,
        value_column="Load_kg",
        group_columns=["Method", "Scenario", "Year", "Analyte", "Treatment"],
        plot_total_column="PlotAnnualPointTotal_kg",
        treatment_mean_column="PointTotal_kg",
    )
    event_counts = (
        point_ledgers.groupby(
            ["Method", "Scenario", "Year", "Analyte", "Treatment"],
            as_index=False,
        )
        .agg(n_physical_events=("PhysicalEventID", "nunique"))
    )
    points = points.merge(
        event_counts,
        on=["Method", "Scenario", "Year", "Analyte", "Treatment"],
        validate="one_to_one",
    )
    points["annual_reporting_unit"] = "mean_per_treatment_plot"
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
    output.loc[ml, "primary_center_type"] = (
        "mean_of_replicate_annual_plot_totals"
    )
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


def performance_table(
    frame: pd.DataFrame,
    method: str,
    target: str,
    evaluation: str,
) -> pd.DataFrame:
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
            "Method": method, "Evaluation": evaluation,
            "Target": target, "Grouping": grouping,
            "Analyte": analyte, "Treatment": treatment,
            "n": len(valid),
            "RMSE_original_units": rmse,
            "NRMSE_mean_observed": rmse / denominator if pd.notna(denominator) and denominator > 0 else np.nan,
            "IntervalCoverage": float(interval_valid["covered"].mean()) if len(interval_valid) else np.nan,
        })
    return pd.DataFrame(records)


def assert_constant_within_keys(
    frame: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str],
    label: str,
) -> None:
    """Reject physical-event identifiers that map to conflicting metadata."""

    for column in columns:
        counts = frame.groupby(list(keys), dropna=False)[column].nunique(dropna=False)
        if counts.gt(1).any():
            examples = counts.loc[counts.gt(1)].head(5).index.tolist()
            raise ValueError(
                f"{label} has conflicting {column} values within {list(keys)}; "
                f"examples: {examples}."
            )


def resolve_ml_loyo_physical_event_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve LOYO predictions to the physical-event units used for load points."""

    require_columns(
        frame,
        [
            "Target", "PhysicalEventID", "Year", "Treatment", "Analyte",
            "y_true", "y_pred", "pi_low", "pi_high",
        ],
        "ML LOYO diagnostics",
    )
    records = []
    for target in ["Result_mg_L", "Volume_L"]:
        subset = frame.loc[frame["Target"].eq(target)].copy()
        if target == "Result_mg_L":
            subset = subset.loc[
                subset["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)
            ].copy()
            keys = ["PhysicalEventID", "Analyte"]
            unit = "PhysicalEventID x Analyte"
        else:
            keys = ["PhysicalEventID"]
            unit = "PhysicalEventID"
        assert_constant_within_keys(
            subset, keys, ["Year", "Treatment"], f"ML LOYO {target} diagnostics"
        )
        for column in ["y_true", "y_pred", "pi_low", "pi_high"]:
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
        aggregations = {
            "Year": ("Year", "first"),
            "Treatment": ("Treatment", "first"),
            "y_true": ("y_true", "median"),
            "y_pred": ("y_pred", "median"),
            "pi_low": ("pi_low", "median"),
            "pi_high": ("pi_high", "median"),
            "n_source_observations": ("PhysicalEventID", "size"),
        }
        resolved = subset.groupby(keys, as_index=False, dropna=False).agg(**aggregations)
        if target == "Volume_L":
            resolved["Analyte"] = "all"
        resolved["Target"] = target
        resolved["UnitOfAnalysis"] = unit
        records.append(resolved)
    output = pd.concat(records, ignore_index=True, sort=False)
    if output.duplicated(["Target", "PhysicalEventID", "Analyte"]).any():
        raise ValueError("Resolved ML LOYO diagnostics are not unique by physical-event unit.")
    return output


def ml_full_record_point_diagnostics(ml_dir: Path) -> pd.DataFrame:
    """Match full-record ML point predictions to observed physical-event units.

    These are in-sample reconstruction diagnostics for the deterministic point
    predictions that underlie the annual comparison sums. They are intentionally
    distinct from outer-LOYO validation diagnostics.
    """

    points = normalize_point_ledger(
        read_required(
            ml_dir / "event_analyte_point_ledger_full_record_model_only.csv",
            "ML full-record model-only point ledger",
        ),
        "ML",
        "full_record_model_only",
    )
    points, _ = restrict_publication_analytes(
        points, "ML full-record point ledger for diagnostics", require_all=True
    )

    concentration = read_required(
        ml_dir / f"concentration_observation_training_table_{ML_TAG}.csv",
        "ML concentration observation training table",
    )
    if "analyte_abbr" in concentration:
        concentration["Analyte"] = concentration["analyte_abbr"].astype(str)
    require_columns(
        concentration,
        ["PhysicalEventID", "Analyte", "Year", "Treatment", "Result_mg_L"],
        "ML concentration observation training table",
    )
    concentration = concentration.loc[
        concentration["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)
    ].copy()
    concentration["Result_mg_L"] = pd.to_numeric(
        concentration["Result_mg_L"], errors="coerce"
    )
    concentration = concentration.dropna(subset=["Result_mg_L"])
    concentration_keys = ["PhysicalEventID", "Analyte"]
    assert_constant_within_keys(
        concentration,
        concentration_keys,
        ["Year", "Treatment"],
        "ML concentration observations",
    )
    concentration_observed = (
        concentration.groupby(concentration_keys, as_index=False, dropna=False)
        .agg(
            Year=("Year", "first"),
            Treatment=("Treatment", "first"),
            y_true=("Result_mg_L", "median"),
            n_source_observations=("Result_mg_L", "size"),
        )
    )
    concentration_predicted = points[[
        "PhysicalEventID", "Analyte", "Year", "Treatment", "Concentration_mg_L"
    ]].rename(
        columns={
            "Year": "Year_predicted",
            "Treatment": "Treatment_predicted",
            "Concentration_mg_L": "y_pred",
        }
    )
    concentration_diagnostics = concentration_observed.merge(
        concentration_predicted,
        on=concentration_keys,
        how="inner",
        validate="one_to_one",
    )
    if not concentration_diagnostics["Year"].eq(
        concentration_diagnostics["Year_predicted"]
    ).all() or not concentration_diagnostics["Treatment"].astype(str).eq(
        concentration_diagnostics["Treatment_predicted"].astype(str)
    ).all():
        raise ValueError("ML concentration observations and point predictions disagree on event metadata.")
    concentration_diagnostics = concentration_diagnostics.drop(
        columns=["Year_predicted", "Treatment_predicted"]
    )
    concentration_diagnostics["Target"] = "Result_mg_L"
    concentration_diagnostics["UnitOfAnalysis"] = "PhysicalEventID x Analyte"

    volume = read_required(
        ml_dir / f"volume_observation_training_table_{ML_TAG}.csv",
        "ML volume observation training table",
    )
    require_columns(
        volume,
        ["PhysicalEventID", "Year", "Treatment", "Volume"],
        "ML volume observation training table",
    )
    volume["Volume"] = pd.to_numeric(volume["Volume"], errors="coerce")
    volume = volume.dropna(subset=["Volume"])
    assert_constant_within_keys(
        volume, ["PhysicalEventID"], ["Year", "Treatment"], "ML volume observations"
    )
    volume_observed = (
        volume.groupby("PhysicalEventID", as_index=False, dropna=False)
        .agg(
            Year=("Year", "first"),
            Treatment=("Treatment", "first"),
            y_true=("Volume", "median"),
            n_source_observations=("Volume", "size"),
        )
    )
    assert_constant_within_keys(
        points,
        ["PhysicalEventID"],
        ["Year", "Treatment", "Volume_L"],
        "ML full-record point-volume predictions",
    )
    volume_predicted = (
        points.groupby("PhysicalEventID", as_index=False, dropna=False)
        .agg(
            Year_predicted=("Year", "first"),
            Treatment_predicted=("Treatment", "first"),
            y_pred=("Volume_L", "first"),
        )
    )
    volume_diagnostics = volume_observed.merge(
        volume_predicted,
        on="PhysicalEventID",
        how="inner",
        validate="one_to_one",
    )
    if not volume_diagnostics["Year"].eq(
        volume_diagnostics["Year_predicted"]
    ).all() or not volume_diagnostics["Treatment"].astype(str).eq(
        volume_diagnostics["Treatment_predicted"].astype(str)
    ).all():
        raise ValueError("ML volume observations and point predictions disagree on event metadata.")
    volume_diagnostics = volume_diagnostics.drop(
        columns=["Year_predicted", "Treatment_predicted"]
    )
    volume_diagnostics["Analyte"] = "all"
    volume_diagnostics["Target"] = "Volume_L"
    volume_diagnostics["UnitOfAnalysis"] = "PhysicalEventID"

    output = pd.concat(
        [concentration_diagnostics, volume_diagnostics], ignore_index=True, sort=False
    )
    output["pi_low"] = np.nan
    output["pi_high"] = np.nan
    if output.duplicated(["Target", "PhysicalEventID", "Analyte"]).any():
        raise ValueError("ML full-record point diagnostics are not unique by physical-event unit.")
    return output


def build_performance_products(
    bayes_dir: Path,
    ml_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build primary Bayes-vs-ML and separate ML validation metric products."""

    bayes_rows = read_required(
        bayes_dir / f"row_prediction_diagnostics_bayes_{BAYES_VERSION}.csv",
        "Bayesian row predictions",
    )
    bayes_volume = read_required(
        bayes_dir / f"volume_prediction_diagnostics_bayes_{BAYES_VERSION}.csv",
        "Bayesian volume predictions",
    )
    ml_rows = read_required(
        ml_dir / "row_level_residual_diagnostics.csv", "ML row diagnostics"
    )
    require_columns(ml_rows, ["Target", "Analyte"], "ML row diagnostics")
    ml_rows = ml_rows.loc[
        ml_rows["Target"].eq("Volume_L")
        | ml_rows["Analyte"].astype(str).isin(PUBLICATION_ANALYTES)
    ].copy()
    ml_loyo_resolved = resolve_ml_loyo_physical_event_diagnostics(ml_rows)
    ml_reconstruction = ml_full_record_point_diagnostics(ml_dir)

    bayes_performance = pd.concat([
        performance_table(
            bayes_rows, "Bayes", "Result_mg_L", BAYES_POSTERIOR_FIT
        ),
        performance_table(bayes_volume, "Bayes", "Volume_L", BAYES_POSTERIOR_FIT),
    ], ignore_index=True)
    ml_reconstruction_performance = pd.concat([
        performance_table(
            ml_reconstruction,
            "ML",
            "Result_mg_L",
            ML_FULL_RECORD_RECONSTRUCTION,
        ),
        performance_table(
            ml_reconstruction,
            "ML",
            "Volume_L",
            ML_FULL_RECORD_RECONSTRUCTION,
        ),
    ], ignore_index=True)
    ml_loyo_performance = pd.concat([
        performance_table(
            ml_loyo_resolved,
            "ML",
            "Result_mg_L",
            ML_OUTER_LOYO_VALIDATION,
        ),
        performance_table(
            ml_loyo_resolved,
            "ML",
            "Volume_L",
            ML_OUTER_LOYO_VALIDATION,
        ),
    ], ignore_index=True)
    primary = pd.concat(
        [bayes_performance, ml_reconstruction_performance], ignore_index=True
    )
    ml_evaluation_tracks = pd.concat(
        [ml_reconstruction_performance, ml_loyo_performance], ignore_index=True
    )
    return primary, ml_evaluation_tracks, bayes_rows, bayes_volume, ml_rows


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
        "Method", "Evaluation", "Target", "Grouping", "Analyte", "Treatment",
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
        np.where(
            output["Evaluation"].eq(ML_FULL_RECORD_RECONSTRUCTION),
            "Full-record physical-event reconstruction points",
            "Outer-LOYO physical-event validation predictions",
        ),
        "Bayesian posterior-predictive diagnostics",
    )
    return output[[
        "DisplayTarget", "Method", "Evaluation", "n", "NRMSE_percent",
        "IntervalCoverage", "diagnostic_basis",
    ]]


def publication_ml_evaluation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format paired ML reconstruction and LOYO RMSE/NRMSE metrics."""

    output = frame.copy()
    output["RMSE"] = output["RMSE_original_units"].map(
        lambda value: "NA" if pd.isna(value) else f"{value:.4g}"
    )
    output["RMSE_units"] = np.where(
        output["Target"].eq("Volume_L"), "L", "mg/L"
    )
    output["NRMSE_percent"] = output["NRMSE_mean_observed"].map(
        lambda value: "NA" if pd.isna(value) else f"{100 * value:.1f}%"
    )
    output["Coverage"] = output["IntervalCoverage"].map(
        lambda value: "NA" if pd.isna(value) else f"{100 * value:.1f}%"
    )
    return output[[
        "DisplayTarget", "Evaluation", "n", "RMSE", "RMSE_units",
        "NRMSE_percent", "Coverage",
    ]]


def three_track_performance_table(
    primary: pd.DataFrame,
    ml_evaluation_tracks: pd.DataFrame,
) -> pd.DataFrame:
    """Combine Bayes fit, ML reconstruction, and ML LOYO without duplicates."""

    bayes = primary.loc[
        primary["Method"].eq("Bayes")
        & primary["Evaluation"].eq(BAYES_POSTERIOR_FIT)
    ].copy()
    ml = ml_evaluation_tracks.loc[
        ml_evaluation_tracks["Evaluation"].isin([
            ML_FULL_RECORD_RECONSTRUCTION,
            ML_OUTER_LOYO_VALIDATION,
        ])
    ].copy()
    output = pd.concat([bayes, ml], ignore_index=True)
    output["ComparisonSeries"] = output["Evaluation"].map({
        BAYES_POSTERIOR_FIT: "Bayes posterior-predictive fit",
        ML_FULL_RECORD_RECONSTRUCTION: "ML full-record reconstruction",
        ML_OUTER_LOYO_VALIDATION: "ML outer-LOYO validation",
    })
    keys = [
        "ComparisonSeries", "Target", "Grouping", "Analyte", "Treatment"
    ]
    if output["ComparisonSeries"].isna().any():
        raise ValueError("Three-track performance table contains an unknown evaluation.")
    if output.duplicated(keys).any():
        raise ValueError("Three-track performance table contains duplicate metric rows.")
    return output


def write_performance_products(
    output_dir: Path,
    primary: pd.DataFrame,
    ml_evaluation_tracks: pd.DataFrame,
) -> None:
    """Write primary comparison and explicitly separated ML evaluation tracks."""

    primary.to_csv(output_dir / "performance_and_calibration_raw.csv", index=False)
    publication_performance_table(primary).to_csv(
        output_dir / "performance_and_calibration_publication.csv", index=False
    )
    primary_overall = overall_nrmse_table(primary)
    primary_overall.to_csv(
        output_dir / "nrmse_overall_analytes_and_volume_raw.csv", index=False
    )
    publication_overall_nrmse_table(primary_overall).to_csv(
        output_dir / "nrmse_overall_analytes_and_volume_publication.csv", index=False
    )

    ml_evaluation_tracks.to_csv(
        output_dir / "ml_rmse_nrmse_reconstruction_and_loyo_raw.csv", index=False
    )
    publication_performance_table(ml_evaluation_tracks).to_csv(
        output_dir / "ml_rmse_nrmse_reconstruction_and_loyo_publication.csv",
        index=False,
    )
    ml_overall = overall_nrmse_table(ml_evaluation_tracks)
    ml_overall.to_csv(
        output_dir / "ml_rmse_nrmse_overall_reconstruction_and_loyo_raw.csv",
        index=False,
    )
    publication_ml_evaluation_table(ml_overall).to_csv(
        output_dir / "ml_rmse_nrmse_overall_reconstruction_and_loyo_publication.csv",
        index=False,
    )
    three_track = three_track_performance_table(primary, ml_evaluation_tracks)
    three_track.to_csv(
        output_dir / "rmse_nrmse_bayes_ml_reconstruction_loyo_raw.csv",
        index=False,
    )
    publication_performance_table(three_track).to_csv(
        output_dir / "rmse_nrmse_bayes_ml_reconstruction_loyo_publication.csv",
        index=False,
    )
    (output_dir / "rmse_nrmse_evaluation_definitions.md").write_text(
        "# RMSE and NRMSE evaluation tracks\n\n"
        "- Bayesian RMSE and NRMSE are unchanged posterior-predictive fit diagnostics. "
        "No Bayesian LOYO calculation is introduced.\n"
        "- The primary Bayes-versus-ML table uses ML full-record physical-event "
        "reconstruction points: the deterministic concentration and volume predictions "
        "that underlie the annual point-load sums, matched only where observations exist.\n"
        "- ML outer-LOYO RMSE and NRMSE are retained in separate validation tables and "
        "figures. They quantify held-out-year prediction performance and are not used as "
        "the ML center in the annual reconstruction figures.\n"
        "- Additive three-track figures place the unchanged Bayesian fit diagnostic, ML "
        "full-record reconstruction diagnostic, and ML outer-LOYO validation diagnostic "
        "side by side. LOYO is held-out prediction error, not an uncertainty interval.\n"
        "- To prevent measurement-method or sampler-method copies from being counted as "
        "separate load-producing events, ML concentration diagnostics are resolved by "
        "PhysicalEventID x Analyte and volume diagnostics by PhysicalEventID using the "
        "same median resolution as the point-load workflow. Legitimate source rows remain "
        "unchanged in the model inputs.\n"
        "- RMSE is reported in mg/L for concentration and L for event volume. NRMSE is "
        "RMSE divided by the mean observed value within the displayed group. Metrics are "
        "not calculated against partial observed annual subtotals.\n",
        encoding="utf-8",
    )


def metric_manifest_fields() -> dict[str, object]:
    return {
        "primary_rmse_nrmse_bayes_evaluation": BAYES_POSTERIOR_FIT,
        "primary_rmse_nrmse_ml_evaluation": ML_FULL_RECORD_RECONSTRUCTION,
        "separate_ml_validation_evaluation": ML_OUTER_LOYO_VALIDATION,
        "bayesian_loyo_metrics_introduced": False,
        "ml_concentration_metric_unit": "PhysicalEventID x Analyte; median resolution",
        "ml_volume_metric_unit": "PhysicalEventID; median resolution",
        "nrmse_definition": "RMSE divided by mean observed value within group",
        "rmse_nrmse_against_partial_observed_annual_subtotals": False,
        "three_track_rmse_nrmse_figures": True,
        "three_track_metric_series": [
            "Bayes posterior-predictive fit",
            "ML full-record reconstruction",
            "ML outer-LOYO validation",
        ],
        "three_track_nrmse_primary_layout": "single panel with shared linear y-axis capped at 500 percent",
        "three_track_nrmse_axis_cap_percent": 500,
        "three_track_nrmse_capped_bars": "hatched and labeled with actual values",
        "three_track_nrmse_faceted_supplement": True,
        "ml_loyo_rmse_nrmse_interpretation": "held-out prediction error, not interval uncertainty",
    }


def update_saved_comparison_manifest(output_dir: Path) -> None:
    """Add metric provenance when --figures-only refreshes saved products."""

    path = output_dir / f"run_manifest_comparison_{COMPARISON_VERSION}.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(metric_manifest_fields())
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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
    if "Analyte" in output:
        output = output.loc[
            output["Analyte"].astype(str).isin(PRIMARY_MANUSCRIPT_ANALYTES)
        ].copy()
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
        for suffix in (".png", ".jpg"):
            target = stem.with_suffix(suffix)
            temporary = target.with_name(
                f".{target.stem}.comparison-write{target.suffix}"
            )
            try:
                figure.savefig(
                    temporary,
                    dpi=dpi,
                    bbox_inches="tight",
                )
                if target.exists():
                    target.unlink()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    plt.close(figure)


def observed_annual_load_completeness(
    expected_event_analytes: pd.DataFrame,
    observed_event_analytes: pd.DataFrame,
    observed_annual_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Audit completeness independently for each annual replicate plot."""

    required = [
        "Year", "Irrigation", "Treatment", "Rep", "Analyte", "PhysicalEventID"
    ]
    require_columns(
        expected_event_analytes, required, "expected event-analyte provenance"
    )
    require_columns(
        observed_event_analytes,
        [*required, "Load_kg"],
        "observed event-analyte ledger",
    )
    expected = expected_event_analytes.loc[
        expected_event_analytes["Analyte"].astype(str).isin(PUBLICATION_ANALYTES),
        required,
    ].drop_duplicates()
    observed = observed_event_analytes.loc[
        observed_event_analytes["Analyte"].astype(str).isin(PUBLICATION_ANALYTES),
        [*required, "Load_kg"],
    ].copy()
    audit = observed_annual_plot_summary(
        expected,
        observed,
        value_column="Load_kg",
        analysis_columns=["Analyte"],
    )
    audit = audit.rename(
        columns={
            "ExpectedEventCount": "n_expected_physical_events",
            "ObservedEventCount": "n_observed_event_loads",
            "IncompleteObservedSubtotal": "ObservedIncompleteSubtotal_kg",
            "ReplicateAnnualValue": "ObservedReplicateAnnualLoad_kg",
            "TreatmentMean": "ObservedAnnualLoad_kg",
            "RangeLow": "ObservedAnnualLoadRangeLow_kg",
            "RangeHigh": "ObservedAnnualLoadRangeHigh_kg",
            "SampleSD": "ObservedSampleSD_kg",
            "SE": "ObservedSE_kg",
            "Minimum": "ObservedMinimum_kg",
            "Maximum": "ObservedMaximum_kg",
        }
    )
    audit["observed_event_fraction"] = (
        audit["n_observed_event_loads"] / audit["n_expected_physical_events"]
    )
    audit["ObservedAnnualLoadComplete"] = audit["n_complete_plots"].gt(0)
    audit["ObservedIntervalType"] = audit["IntervalType"]
    return audit


def observed_annual_volume_completeness(
    expected_events: pd.DataFrame,
    volume_event_rows: pd.DataFrame,
    annual_volume: pd.DataFrame,
) -> pd.DataFrame:
    """Audit annual observed volume independently for each replicate plot."""

    event_key = [
        "Year", "Irrigation", "Treatment", "Rep", "PhysicalEventID"
    ]
    require_columns(expected_events, event_key, "expected physical events")
    require_columns(
        volume_event_rows,
        event_key + ["Volume"],
        "volume observation and event-prediction table",
    )
    expected = expected_events[event_key].drop_duplicates().copy()
    observed_source = volume_event_rows.loc[
        pd.to_numeric(volume_event_rows["Volume"], errors="coerce").notna()
    ].copy()
    observed_source["ObservedVolume_L"] = pd.to_numeric(
        observed_source["Volume"], errors="raise"
    )
    observed = resolve_prediction_draws(
        observed_source,
        group_columns=event_key,
        value_column="ObservedVolume_L",
        method=PHYSICAL_EVENT_CONFIG["volume_resolution"],
        method_column=(
            "MeasureMethod" if "MeasureMethod" in observed_source else None
        ),
        method_priority=PHYSICAL_EVENT_CONFIG["method_priority"],
    )
    audit = observed_annual_plot_summary(
        expected,
        observed,
        value_column="ObservedVolume_L",
    )
    audit = audit.rename(
        columns={
            "ExpectedEventCount": "n_expected_physical_events",
            "ObservedEventCount": "n_observed_event_volumes",
            "IncompleteObservedSubtotal": "ObservedIncompleteSubtotal_L",
            "ReplicateAnnualValue": "ObservedReplicateAnnualVolume_L",
            "TreatmentMean": "ObservedAnnualVolume_L",
            "RangeLow": "ObservedAnnualVolumeRangeLow_L",
            "RangeHigh": "ObservedAnnualVolumeRangeHigh_L",
            "SampleSD": "ObservedSampleSD_L",
            "SE": "ObservedSE_L",
            "Minimum": "ObservedMinimum_L",
            "Maximum": "ObservedMaximum_L",
        }
    )
    audit["observed_event_fraction"] = (
        audit["n_observed_event_volumes"] / audit["n_expected_physical_events"]
    )
    audit["ObservedAnnualVolumeComplete"] = audit["n_complete_plots"].gt(0)
    audit["ObservedAnnualVolume_kL"] = audit["ObservedAnnualVolume_L"] / 1000.0
    audit["ObservedAnnualVolumeRangeLow_kL"] = (
        audit["ObservedAnnualVolumeRangeLow_L"] / 1000.0
    )
    audit["ObservedAnnualVolumeRangeHigh_kL"] = (
        audit["ObservedAnnualVolumeRangeHigh_L"] / 1000.0
    )
    audit["ObservedIntervalType"] = audit["IntervalType"]
    return audit


def complete_observed_load_rows(audit: pd.DataFrame) -> pd.DataFrame:
    rows = (
        audit.loc[audit["n_complete_plots"].gt(0)]
        .drop_duplicates(["Year", "Analyte", "Treatment"])
        .copy()
    )
    return rows.rename(
        columns={
            "ObservedAnnualLoad_kg": "center_kg",
            "ObservedAnnualLoadRangeLow_kg": "lower_95_kg",
            "ObservedAnnualLoadRangeHigh_kg": "upper_95_kg",
        }
    )[
        [
            "Year", "Analyte", "Treatment", "center_kg", "lower_95_kg",
            "upper_95_kg", "n_complete_plots", "ObservedIntervalType",
        ]
    ]


def complete_observed_volume_rows(
    annual_volume: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    modeled = annual_volume.loc[~annual_volume["Method"].eq("Observed")].copy()
    observed = (
        audit.loc[audit["n_complete_plots"].gt(0)]
        .drop_duplicates(["Year", "Treatment"])
        .copy()
    )
    observed = observed.rename(
        columns={
            "ObservedAnnualVolume_kL": "center_kL",
            "ObservedAnnualVolumeRangeLow_kL": "lower_95_kL",
            "ObservedAnnualVolumeRangeHigh_kL": "upper_95_kL",
        }
    )
    observed["Method"] = "Observed"
    observed["Scenario"] = "complete_observed_reference"
    observed["central_estimate_type"] = (
        "mean_of_complete_replicate_plot_totals_or_single_available_plot"
    )
    observed["interval_type"] = observed["ObservedIntervalType"]
    observed = observed[
        [
            "Method",
            "Scenario",
            "Year",
            "Treatment",
            "center_kL",
            "lower_95_kL",
            "upper_95_kL",
            "central_estimate_type",
            "interval_type",
            "n_complete_plots",
        ]
    ]
    return pd.concat([modeled, observed], ignore_index=True)


def annual_load_complete_observed_comparison(
    annual_summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    key = ["Year", "Analyte", "Treatment"]
    primary = annual_summary.loc[
        (
            annual_summary["Method"].eq("Bayes")
            & annual_summary["Scenario"].eq("model_only")
        )
        | (
            annual_summary["Method"].eq("ML")
            & annual_summary["Scenario"].eq("full_record_model_only")
        )
    ].copy()
    keep = key + ["primary_center", "lower_95", "upper_95"]
    bayes = primary.loc[primary["Method"].eq("Bayes"), keep].rename(
        columns={
            "primary_center": "BayesPosteriorMedian_kg",
            "lower_95": "BayesLower95_kg",
            "upper_95": "BayesUpper95_kg",
        }
    )
    ml = primary.loc[primary["Method"].eq("ML"), keep].rename(
        columns={
            "primary_center": "MLPointTotal_kg",
            "lower_95": "MLLower95_kg",
            "upper_95": "MLUpper95_kg",
        }
    )
    output = bayes.merge(ml, on=key, how="outer", validate="one_to_one")
    treatment_audit = audit.drop_duplicates(key)
    audit_columns = key + [
        "n_expected_physical_events",
        "n_observed_event_loads",
        "observed_event_fraction",
        "ObservedAnnualLoadComplete",
        "n_complete_plots",
        "ObservedAnnualLoad_kg",
        "ObservedAnnualLoadRangeLow_kg",
        "ObservedAnnualLoadRangeHigh_kg",
        "ObservedIntervalType",
    ]
    output = output.merge(
        treatment_audit[audit_columns],
        on=key,
        how="left",
        validate="one_to_one",
    )
    if output["ObservedAnnualLoadComplete"].isna().any():
        raise ValueError(
            "Annual model comparison contains groups absent from the observed "
            "annual-load completeness audit."
        )
    return output.sort_values(key).reset_index(drop=True)


def annual_volume_complete_observed_comparison(
    annual_volume: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    key = ["Year", "Treatment"]
    modeled = annual_volume.loc[
        annual_volume["Method"].isin(["Bayes", "ML"])
    ].copy()
    keep = key + ["center_kL", "lower_95_kL", "upper_95_kL"]
    bayes = modeled.loc[modeled["Method"].eq("Bayes"), keep].rename(
        columns={
            "center_kL": "BayesPosteriorMean_kL",
            "lower_95_kL": "BayesLower95_kL",
            "upper_95_kL": "BayesUpper95_kL",
        }
    )
    ml = modeled.loc[modeled["Method"].eq("ML"), keep].rename(
        columns={
            "center_kL": "MLPointTotal_kL",
            "lower_95_kL": "MLLower95_kL",
            "upper_95_kL": "MLUpper95_kL",
        }
    )
    output = bayes.merge(ml, on=key, how="outer", validate="one_to_one")
    treatment_audit = audit.drop_duplicates(key)
    audit_columns = key + [
        "n_expected_physical_events",
        "n_observed_event_volumes",
        "observed_event_fraction",
        "ObservedAnnualVolumeComplete",
        "n_complete_plots",
        "ObservedAnnualVolume_kL",
        "ObservedAnnualVolumeRangeLow_kL",
        "ObservedAnnualVolumeRangeHigh_kL",
        "ObservedIntervalType",
    ]
    output = output.merge(
        treatment_audit[audit_columns],
        on=key,
        how="left",
        validate="one_to_one",
    )
    if output["ObservedAnnualVolumeComplete"].isna().any():
        raise ValueError(
            "Annual volume model comparison contains groups absent from the "
            "observed annual-volume completeness audit."
        )
    return output.sort_values(key).reset_index(drop=True)


def publication_complete_observed_table(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    complete_column: str,
) -> pd.DataFrame:
    output = frame.copy()
    for column in value_columns:
        output[column] = output[column].map(
            lambda value: "-" if pd.isna(value) else f"{float(value):.4g}"
        )
    output[complete_column] = output[complete_column].map(
        {True: "Complete", False: "Incomplete"}
    )
    return output


def ml_annual_volume_products(
    point_ledger: pd.DataFrame,
    draw_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate ML runoff volume by Rep before treatment means and draw summaries."""

    event_columns = ["PhysicalEventID", "Year", "Treatment", "Rep", "Volume_L"]
    require_columns(point_ledger, event_columns, "ML point ledger for annual volume")
    require_columns(
        draw_ledger,
        [*event_columns, "Draw"],
        "ML draw ledger for annual volume",
    )
    point_ledger = point_ledger.copy()
    draw_ledger = draw_ledger.copy()
    point_ledger["Volume_L"] = pd.to_numeric(
        point_ledger["Volume_L"], errors="raise"
    )
    draw_ledger["Volume_L"] = pd.to_numeric(
        draw_ledger["Volume_L"], errors="raise"
    )

    point_conflicts = (
        point_ledger.groupby("PhysicalEventID", dropna=False)[
            ["Year", "Treatment", "Rep", "Volume_L"]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if point_conflicts.any():
        raise ValueError(
            "ML point ledger has conflicting Year, Treatment, Rep, or Volume_L "
            "values within PhysicalEventID."
        )
    point_events = point_ledger[event_columns].drop_duplicates("PhysicalEventID")
    _, annual_points = aggregate_replicate_mean(
        point_events,
        value_column="Volume_L",
        group_columns=["Year", "Treatment"],
        plot_total_column="PlotAnnualVolume_L",
        treatment_mean_column="PointTreatmentMeanVolume_L",
    )

    draw_key = ["PhysicalEventID", "Draw"]
    draw_conflicts = (
        draw_ledger.groupby(draw_key, dropna=False)[
            ["Year", "Treatment", "Rep", "Volume_L"]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if draw_conflicts.any():
        raise ValueError(
            "ML draw ledger has conflicting Year, Treatment, Rep, or Volume_L "
            "values within PhysicalEventID x Draw."
        )
    draw_events = draw_ledger[
        [*event_columns, "Draw"]
    ].drop_duplicates(draw_key)
    _, annual_draws = aggregate_replicate_mean(
        draw_events,
        value_column="Volume_L",
        group_columns=["Year", "Treatment"],
        draw_column="Draw",
        plot_total_column="PlotAnnualVolume_L",
        treatment_mean_column="TreatmentMeanVolume_L",
    )
    summary = summarize_draws(
        annual_draws,
        ["Year", "Treatment"],
        "TreatmentMeanVolume_L",
    ).rename(
        columns={
            "mean": "draw_mean_L",
            "median": "draw_median_L",
            "lower_95": "draw_lower_95_L",
            "upper_95": "draw_upper_95_L",
        }
    )
    output = annual_points.merge(
        summary,
        on=["Year", "Treatment"],
        validate="one_to_one",
    )
    output["annual_reporting_unit"] = "mean_per_treatment_plot"
    return output


def annual_volume_comparison(bayes_dir: Path, ml_dir: Path) -> pd.DataFrame:
    """Combine corrected Bayes, ML, and observed annual runoff volumes in kL."""

    records: list[dict[str, object]] = []
    bayes_sources = [
        (
            "Bayes",
            "model_only",
            read_required(
                bayes_dir / f"annual_volume_kL_wide_modeled_{BAYES_VERSION}.csv",
                "Bayesian modeled annual-volume summary",
            ),
            "mod",
            "posterior_mean",
            "95% posterior credible interval",
        ),
    ]
    for method, scenario, wide, source_tag, center_type, interval_type in bayes_sources:
        require_columns(wide, ["Year"], f"{method} annual-volume summary")
        years = pd.to_numeric(wide["Year"], errors="coerce")
        for treatment in TREATMENTS:
            center_column = f"{treatment}_{source_tag}_volume_kL_mean"
            lower_column = f"{treatment}_{source_tag}_volume_kL_low"
            upper_column = f"{treatment}_{source_tag}_volume_kL_high"
            require_columns(
                wide,
                [center_column, lower_column, upper_column],
                f"{method} annual-volume summary",
            )
            for position in np.flatnonzero(years.notna().to_numpy()):
                center = pd.to_numeric(
                    pd.Series([wide.iloc[position][center_column]]), errors="coerce"
                ).iloc[0]
                if not np.isfinite(center):
                    continue
                records.append({
                    "Method": method,
                    "Scenario": scenario,
                    "Year": int(years.iloc[position]),
                    "Treatment": treatment,
                    "center_kL": float(center),
                    "lower_95_kL": float(wide.iloc[position][lower_column]),
                    "upper_95_kL": float(wide.iloc[position][upper_column]),
                    "central_estimate_type": center_type,
                    "interval_type": interval_type,
                })

    ml_points = read_required(
        ml_dir / "event_analyte_point_ledger_full_record_model_only.csv",
        "ML full-record model-only point ledger for annual volume",
    )
    ml_draws = read_required(
        ml_dir / "event_analyte_draw_ledger_full_record_model_only.csv",
        "ML full-record model-only draw ledger for annual volume",
    )
    ml_annual = ml_annual_volume_products(ml_points, ml_draws)
    ml_annual["Year"] = pd.to_numeric(ml_annual["Year"], errors="raise").astype(int)
    ml_annual["center_kL"] = ml_annual["PointTreatmentMeanVolume_L"] / 1000.0
    for row in ml_annual.itertuples(index=False):
        records.append({
            "Method": "ML",
            "Scenario": "full_record_model_only",
            "Year": int(row.Year),
            "Treatment": str(row.Treatment),
            "center_kL": float(row.center_kL),
            "lower_95_kL": float(row.draw_lower_95_L) / 1000.0,
            "upper_95_kL": float(row.draw_upper_95_L) / 1000.0,
            "central_estimate_type": "mean_of_replicate_annual_plot_volumes",
            "interval_type": (
                "95% Monte Carlo empirical calibration-residual prediction interval"
            ),
        })

    output = pd.DataFrame.from_records(records)
    key = ["Method", "Year", "Treatment"]
    if output.duplicated(key).any():
        raise ValueError("Annual volume comparison is not unique by Method x Year x Treatment.")
    for method in ["Bayes", "ML"]:
        expected = pd.MultiIndex.from_product(
            [STUDY_YEARS, TREATMENTS], names=["Year", "Treatment"]
        )
        found = pd.MultiIndex.from_frame(
            output.loc[output["Method"].eq(method), ["Year", "Treatment"]]
        )
        missing = expected.difference(found)
        if len(missing):
            raise ValueError(f"{method} annual volume summary is missing {list(missing)}.")
    return output.sort_values(["Method", "Treatment", "Year"]).reset_index(drop=True)


def plot_observed_replicate_points(
    axis: plt.Axes,
    rows: pd.DataFrame,
    *,
    center_column: str,
    lower_column: str,
    upper_column: str,
) -> None:
    """Plot paired replicate ranges and single plots with distinct open markers."""

    paired = rows.loc[rows["n_complete_plots"].ge(2)]
    if not paired.empty:
        center = paired[center_column].to_numpy(dtype=float)
        lower = paired[lower_column].to_numpy(dtype=float)
        upper = paired[upper_column].to_numpy(dtype=float)
        axis.errorbar(
            paired["Year"].to_numpy(dtype=float),
            center,
            yerr=np.vstack(
                [
                    np.maximum(center - lower, 0),
                    np.maximum(upper - center, 0),
                ]
            ),
            fmt="o", markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=1.5, color="black", ecolor="black",
            elinewidth=1.25, capsize=3, markersize=7, zorder=10,
        )
    single = rows.loc[rows["n_complete_plots"].eq(1)]
    if not single.empty:
        axis.scatter(
            single["Year"].to_numpy(dtype=float),
            single[center_column].to_numpy(dtype=float),
            marker="s",
            facecolors="white",
            edgecolors="black",
            s=52,
            linewidths=1.5,
            zorder=10,
        )


def bayes_referenced_axis_upper(
    bayes_upper: pd.Series,
    ml_center: pd.Series,
    observed_upper: pd.Series,
    observed_center: pd.Series,
    *,
    padding: float = 1.08,
) -> float:
    """Set a readable shared axis without allowing ML PI bounds to expand it.

    Bayesian upper credible bounds define the uncertainty scale. ML centers
    and observed references remain visible, but ML prediction-interval bounds
    are deliberately excluded and may therefore be clipped by the axes.
    """

    values = pd.concat(
        [
            pd.to_numeric(series, errors="coerce")
            for series in (
                bayes_upper,
                ml_center,
                observed_upper,
                observed_center,
            )
        ],
        ignore_index=True,
    ).to_numpy(dtype=float)
    finite = values[np.isfinite(values) & (values >= 0)]
    if not len(finite) or float(np.max(finite)) <= 0:
        return 1.0
    return float(np.max(finite)) * float(padding)


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
            label=(
                "ML mean + 95% calibration-residual PI "
                "(ribbon may be clipped)"
            ),
        ),
        Line2D(
            [0], [0], color="black", marker="o", markerfacecolor="none",
            linestyle="none", markersize=8,
            label="Observed annual load (two complete plots; replicate range)",
        ),
        Line2D(
            [0], [0], color="black", marker="s", markerfacecolor="white",
            linestyle="none", markersize=7,
            label="Observed annual load (one complete plot)",
        ),
    ]
    for analyte, group in primary.groupby("Analyte", sort=False):
        observed_analyte = observed.loc[
            observed["Analyte"].astype(str).eq(str(analyte))
        ].copy()
        bayes_group = group.loc[group["Method"].eq("Bayes")]
        ml_group = group.loc[group["Method"].eq("ML")]
        ml_center_column = (
            "primary_center" if "primary_center" in ml_group else "median"
        )
        axis_upper = bayes_referenced_axis_upper(
            bayes_group["upper_95"],
            ml_group[ml_center_column],
            observed_analyte["upper_95_kg"],
            observed_analyte["center_kg"],
        )
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
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
                    pd.to_numeric(
                        method_rows["lower_95"], errors="raise"
                    ).to_numpy(dtype=float),
                    0.0,
                )
                upper = pd.to_numeric(
                    method_rows["upper_95"], errors="raise"
                ).to_numpy(dtype=float)
                axis.fill_between(
                    years,
                    lower,
                    upper,
                    color=colors[method],
                    alpha=0.17 if method == "Bayes" else 0.13,
                    linewidth=0,
                )
                axis.plot(
                    years, center, color=colors[method], linestyle=line_styles[method],
                    marker="o", markersize=4.5, linewidth=2,
                )
            observed_rows = observed_analyte.loc[
                observed_analyte["Treatment"].astype(str).eq(treatment)
            ].sort_values("Year")
            if not observed_rows.empty:
                plot_observed_replicate_points(
                    axis,
                    observed_rows,
                    center_column="center_kg",
                    lower_column="lower_95_kg",
                    upper_column="upper_95_kg",
                )
            axis.set_title(treatment)
            axis.set_xticks(STUDY_YEARS[::2])
            axis.grid(True, alpha=0.22)
            axis.set_ylim(0, axis_upper)
        axes[0].set_ylabel("Annual load (kg)")
        axes[1].set_xlabel("Year")
        figure.suptitle(
            f"{analyte}: complete observed annual loads vs Bayes vs ML\n"
            "Bayes-referenced y-axis; ML prediction ribbon may be clipped",
            fontsize=14,
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.90),
        )
        safe = figure_slug(analyte)
        save_figure(
            figure,
            figure_dir / f"annual_load_{analyte}",
            figure_dir
            / "annual_obs_vs_modeled"
            / f"annual_load_{safe}_obs_vs_modeled_{COMPARISON_TAG}",
            figure_dir / "annual_complete_observed"
            / f"annual_load_{safe}_complete_observed_{COMPARISON_TAG}",
            layout_top=0.82,
        )


def plot_annual_volume_comparison(volume: pd.DataFrame, figure_dir: Path) -> None:
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    legend_handles = [
        Line2D(
            [0], [0], color=colors["Bayes"], marker="o", linewidth=2,
            label="Bayes posterior mean + 95% credible interval",
        ),
        Line2D(
            [0], [0], color=colors["ML"], marker="o", linestyle="--", linewidth=2,
            label=(
                "ML mean + 95% calibration-residual PI "
                "(ribbon may be clipped)"
            ),
        ),
        Line2D(
            [0], [0], color="black", marker="o", markerfacecolor="none",
            linestyle="none", markersize=8,
            label="Observed annual volume (two complete plots; replicate range)",
        ),
        Line2D(
            [0], [0], color="black", marker="s", markerfacecolor="white",
            linestyle="none", markersize=7,
            label="Observed annual volume (one complete plot)",
        ),
    ]
    bayes_rows_all = volume.loc[volume["Method"].eq("Bayes")]
    ml_rows_all = volume.loc[volume["Method"].eq("ML")]
    observed_rows_all = volume.loc[volume["Method"].eq("Observed")]
    axis_upper = bayes_referenced_axis_upper(
        bayes_rows_all["upper_95_kL"],
        ml_rows_all["center_kL"],
        observed_rows_all["upper_95_kL"],
        observed_rows_all["center_kL"],
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    for axis, treatment in zip(axes, TREATMENTS):
        for method in ["Bayes", "ML"]:
            rows = volume.loc[
                volume["Method"].eq(method) & volume["Treatment"].eq(treatment)
            ].sort_values("Year")
            years = rows["Year"].to_numpy(dtype=float)
            center = rows["center_kL"].to_numpy(dtype=float)
            lower = np.maximum(rows["lower_95_kL"].to_numpy(dtype=float), 0.0)
            upper = rows["upper_95_kL"].to_numpy(dtype=float)
            axis.fill_between(
                years,
                lower,
                upper,
                color=colors[method],
                alpha=0.17 if method == "Bayes" else 0.13,
                linewidth=0,
            )
            axis.plot(
                years, center, color=colors[method],
                linestyle="-" if method == "Bayes" else "--",
                marker="o", markersize=4.5, linewidth=2,
            )
        observed_rows = volume.loc[
            volume["Method"].eq("Observed")
            & volume["Treatment"].eq(treatment)
        ].sort_values("Year")
        if not observed_rows.empty:
            plot_observed_replicate_points(
                axis,
                observed_rows,
                center_column="center_kL",
                lower_column="lower_95_kL",
                upper_column="upper_95_kL",
            )
        axis.set_title(treatment)
        axis.set_xticks(STUDY_YEARS[::2])
        axis.set_ylim(0, axis_upper)
        axis.grid(True, alpha=0.22)
    axes[0].set_ylabel("Annual runoff volume (kL)")
    axes[1].set_xlabel("Year")
    figure.suptitle(
        "Complete observed annual runoff volume vs Bayes vs ML\n"
        "Bayes-referenced y-axis; ML prediction ribbon may be clipped",
        fontsize=14,
    )
    figure.legend(
        handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90)
    )
    save_figure(
        figure,
        figure_dir / f"annual_runoff_volume_bayes_vs_ml_{COMPARISON_TAG}",
        figure_dir
        / "annual_obs_vs_modeled"
        / f"annual_runoff_volume_obs_vs_modeled_{COMPARISON_TAG}",
        figure_dir
        / "annual_complete_observed"
        / f"annual_runoff_volume_complete_observed_{COMPARISON_TAG}",
        layout_top=0.82,
    )


def plot_annual_uncertainty_supplement(
    annual: pd.DataFrame,
    observed: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Retain complete model intervals with readable, method-specific y scales."""

    primary = annual.loc[
        ((annual["Method"] == "Bayes") & (annual["Scenario"] == "model_only"))
        | ((annual["Method"] == "ML") & (annual["Scenario"] == "full_record_model_only"))
    ].copy()
    colors = {"Bayes": "#1f77b4", "ML": "#ff7f0e"}
    line_styles = {"Bayes": "-", "ML": "--"}
    method_labels = {
        "Bayes": "Bayes posterior median + 95% credible interval",
        "ML": "ML mean per plot + 95% calibration-residual PI",
    }
    legend_handles = [
        Line2D(
            [0], [0], color=colors["Bayes"], marker="o", linewidth=2,
            label=method_labels["Bayes"],
        ),
        Line2D(
            [0], [0], color=colors["ML"], marker="o", linestyle="--", linewidth=2,
            label=method_labels["ML"],
        ),
        Line2D(
            [0], [0], color="black", marker="o", markerfacecolor="none",
            linestyle="none", markersize=8,
            label="Observed annual load (two complete plots; replicate range)",
        ),
        Line2D(
            [0], [0], color="black", marker="s", markerfacecolor="white",
            linestyle="none", markersize=7,
            label="Observed annual load (one complete plot)",
        ),
    ]
    supplemental_dir = figure_dir / "supplemental_uncertainty"
    for analyte, group in primary.groupby("Analyte", sort=False):
        figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        observed_analyte = observed.loc[
            observed["Analyte"].astype(str).eq(str(analyte))
        ].copy()
        for row_index, method in enumerate(["Bayes", "ML"]):
            method_group = group.loc[group["Method"].eq(method)]
            method_upper = pd.to_numeric(
                method_group["upper_95"], errors="coerce"
            ).to_numpy(dtype=float)
            observed_upper = pd.to_numeric(
                observed_analyte["upper_95_kg"], errors="coerce"
            ).to_numpy(dtype=float)
            finite_upper = np.concatenate([
                method_upper[np.isfinite(method_upper)],
                observed_upper[np.isfinite(observed_upper)],
            ])
            row_upper = float(np.max(finite_upper)) * 1.05 if len(finite_upper) else 1.0
            row_upper = max(row_upper, np.finfo(float).eps)
            for column_index, treatment in enumerate(TREATMENTS):
                axis = axes[row_index, column_index]
                method_rows = method_group.loc[
                    method_group["Treatment"].eq(treatment)
                ].sort_values("Year")
                if not method_rows.empty:
                    years = pd.to_numeric(
                        method_rows["Year"], errors="raise"
                    ).to_numpy(dtype=float)
                    center_column = (
                        "primary_center" if "primary_center" in method_rows else "median"
                    )
                    center = pd.to_numeric(
                        method_rows[center_column], errors="raise"
                    ).to_numpy(dtype=float)
                    lower = np.maximum(
                        pd.to_numeric(
                            method_rows["lower_95"], errors="raise"
                        ).to_numpy(dtype=float),
                        0.0,
                    )
                    upper = pd.to_numeric(
                        method_rows["upper_95"], errors="raise"
                    ).to_numpy(dtype=float)
                    axis.fill_between(
                        years, lower, upper, color=colors[method], alpha=0.17, linewidth=0
                    )
                    axis.plot(
                        years, center, color=colors[method],
                        linestyle=line_styles[method], marker="o",
                        markersize=4.5, linewidth=2,
                    )
                observed_rows = observed_analyte.loc[
                    observed_analyte["Treatment"].astype(str).eq(treatment)
                ].sort_values("Year")
                if not observed_rows.empty:
                    plot_observed_replicate_points(
                        axis,
                        observed_rows,
                        center_column="center_kg",
                        lower_column="lower_95_kg",
                        upper_column="upper_95_kg",
                    )
                if row_index == 0:
                    axis.set_title(treatment)
                axis.set_xticks(STUDY_YEARS[::2])
                axis.set_ylim(0, row_upper)
                axis.grid(True, alpha=0.22)
            axes[row_index, 0].set_ylabel(
                f"{method} annual load (kg)\n(method-specific scale)"
            )
        axes[1, 1].set_xlabel("Year")
        figure.suptitle(
            f"{analyte}: full model uncertainty by method (corrected physical-event)",
            fontsize=14,
        )
        figure.legend(
            handles=legend_handles, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.95),
        )
        safe = figure_slug(analyte)
        save_figure(
            figure,
            supplemental_dir
            / f"annual_load_{safe}_full_uncertainty_{COMPARISON_TAG}",
            layout_top=0.89,
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
        "Bayes posterior medians; ML mean-per-plot totals; 95% intervals",
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
        "Bayes posterior medians; ML mean-per-plot totals; 95% intervals",
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
            "Bayes posterior-predictive fit; ML full-record physical-event reconstruction",
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
        "Pooled across treatments; Bayes posterior-predictive fit; "
        "ML full-record reconstruction"
    )
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(loc="best")
    save_figure(
        figure,
        figure_dir / "gof_nrmse_mean_by_analyte",
        figure_dir / "postprocessing" / "nrmse_mean_overall_analytes_and_volume",
        layout_top=0.90,
    )

    volume = performance.loc[
        performance["Target"].eq("Volume_L")
        & performance["Grouping"].eq("overall")
    ].copy()
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    methods = ["Bayes", "ML"]
    values = [
        pd.to_numeric(
            volume.loc[volume["Method"].eq(method), "RMSE_original_units"],
            errors="coerce",
        ).iloc[0]
        for method in methods
    ]
    axis.bar(methods, values, color=[colors[method] for method in methods], alpha=0.85)
    axis.set_ylabel("RMSE (L)")
    axis.set_title(
        "Event-volume RMSE\n"
        "Bayes posterior-predictive fit; ML full-record physical-event reconstruction"
    )
    axis.grid(True, axis="y", alpha=0.22)
    save_figure(
        figure,
        figure_dir / "postprocessing" / "volume_rmse_original_units",
        layout_top=0.90,
    )


def plot_ml_reconstruction_vs_loyo(
    performance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Keep full-record reconstruction and held-out LOYO diagnostics distinct."""

    labels = {
        ML_FULL_RECORD_RECONSTRUCTION: "Full-record reconstruction",
        ML_OUTER_LOYO_VALIDATION: "Outer LOYO validation",
    }
    colors = {
        ML_FULL_RECORD_RECONSTRUCTION: "#ff7f0e",
        ML_OUTER_LOYO_VALIDATION: "#6b7280",
    }
    offsets = {
        ML_FULL_RECORD_RECONSTRUCTION: -0.18,
        ML_OUTER_LOYO_VALIDATION: 0.18,
    }
    evaluations = [ML_FULL_RECORD_RECONSTRUCTION, ML_OUTER_LOYO_VALIDATION]
    overall = overall_nrmse_table(performance)
    display_order = [*PUBLICATION_ANALYTES, "Volume"]

    figure, axis = plt.subplots(figsize=(15, 6.5))
    positions = np.arange(len(display_order), dtype=float)
    for evaluation in evaluations:
        rows = (
            overall.loc[overall["Evaluation"].eq(evaluation)]
            .set_index("DisplayTarget")
            .reindex(display_order)
        )
        values = (
            pd.to_numeric(rows["NRMSE_mean_observed"], errors="coerce")
            .to_numpy(dtype=float)
            * 100.0
        )
        axis.bar(
            positions + offsets[evaluation],
            values,
            width=0.34,
            color=colors[evaluation],
            alpha=0.88,
            label=labels[evaluation],
        )
    axis.set_xticks(positions, display_order, rotation=35, ha="right")
    axis.set_ylabel("Mean-normalized RMSE (%)")
    axis.set_title(
        "ML normalized RMSE: reconstruction points versus held-out validation\n"
        "Resolved to PhysicalEventID x Analyte (concentration) and PhysicalEventID (volume)"
    )
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(loc="best")
    save_figure(
        figure,
        figure_dir / "postprocessing" / "ml_reconstruction_vs_loyo_nrmse",
        layout_top=0.90,
    )

    concentration = overall.loc[overall["Target"].eq("Result_mg_L")].copy()
    figure, axis = plt.subplots(figsize=(14, 6.5))
    positions = np.arange(len(PUBLICATION_ANALYTES), dtype=float)
    for evaluation in evaluations:
        rows = (
            concentration.loc[concentration["Evaluation"].eq(evaluation)]
            .set_index("DisplayTarget")
            .reindex(PUBLICATION_ANALYTES)
        )
        values = pd.to_numeric(
            rows["RMSE_original_units"], errors="coerce"
        ).to_numpy(dtype=float)
        axis.bar(
            positions + offsets[evaluation],
            values,
            width=0.34,
            color=colors[evaluation],
            alpha=0.88,
            label=labels[evaluation],
        )
    axis.set_xticks(positions, PUBLICATION_ANALYTES, rotation=35, ha="right")
    axis.set_ylabel("RMSE (mg/L)")
    axis.set_title(
        "ML concentration RMSE: reconstruction points versus held-out validation\n"
        "Pooled across treatments at PhysicalEventID x Analyte resolution"
    )
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(loc="best")
    save_figure(
        figure,
        figure_dir / "postprocessing" / "ml_reconstruction_vs_loyo_concentration_rmse",
        layout_top=0.90,
    )

    volume = overall.loc[overall["Target"].eq("Volume_L")].copy()
    volume_values = [
        pd.to_numeric(
            volume.loc[volume["Evaluation"].eq(evaluation), "RMSE_original_units"],
            errors="coerce",
        ).iloc[0]
        for evaluation in evaluations
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.bar(
        [labels[evaluation] for evaluation in evaluations],
        volume_values,
        color=[colors[evaluation] for evaluation in evaluations],
        alpha=0.88,
    )
    axis.set_ylabel("RMSE (L)")
    axis.set_title("ML event-volume RMSE: reconstruction versus outer LOYO validation")
    axis.grid(True, axis="y", alpha=0.22)
    save_figure(
        figure,
        figure_dir / "postprocessing" / "ml_reconstruction_vs_loyo_volume_rmse",
        layout_top=0.90,
    )


def plot_three_track_performance_comparison(
    primary: pd.DataFrame,
    ml_evaluation_tracks: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Plot Bayes fit, ML reconstruction, and ML LOYO as distinct diagnostics."""

    performance = three_track_performance_table(primary, ml_evaluation_tracks)
    series_order = [
        "Bayes posterior-predictive fit",
        "ML full-record reconstruction",
        "ML outer-LOYO validation",
    ]
    short_labels = ["Bayes", "ML\nrecon.", "ML\nLOYO"]
    short_series_labels = {
        "Bayes posterior-predictive fit": "Bayes",
        "ML full-record reconstruction": "ML reconstruction",
        "ML outer-LOYO validation": "ML LOYO",
    }
    colors = {
        "Bayes posterior-predictive fit": "#1f77b4",
        "ML full-record reconstruction": "#ff7f0e",
        "ML outer-LOYO validation": "#6b7280",
    }
    offsets = {
        "Bayes posterior-predictive fit": -0.24,
        "ML full-record reconstruction": 0.0,
        "ML outer-LOYO validation": 0.24,
    }
    legend_handles = [
        Patch(facecolor=colors[series], alpha=0.88, label=series)
        for series in series_order
    ]

    concentration = performance.loc[
        performance["Target"].eq("Result_mg_L")
        & performance["Grouping"].eq("analyte_treatment")
    ].copy()
    for column, ylabel, stem, multiplier in [
        (
            "RMSE_original_units",
            "RMSE (mg/L)",
            "concentration_rmse_bayes_ml_reconstruction_loyo_by_treatment",
            1.0,
        ),
        (
            "NRMSE_mean_observed",
            "Mean-normalized RMSE (%)",
            "nrmse_bayes_ml_reconstruction_loyo_by_treatment",
            100.0,
        ),
    ]:
        figure, axes = plt.subplots(5, 2, figsize=(14, 23))
        for axis, analyte in zip(axes.flat, PUBLICATION_ANALYTES):
            group = concentration.loc[concentration["Analyte"].eq(analyte)]
            for series in series_order:
                rows = (
                    group.loc[group["ComparisonSeries"].eq(series)]
                    .set_index("Treatment")
                    .reindex(TREATMENTS)
                )
                values = (
                    pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
                    * multiplier
                )
                axis.bar(
                    np.arange(len(TREATMENTS), dtype=float) + offsets[series],
                    values,
                    width=0.22,
                    color=colors[series],
                    alpha=0.88,
                )
            axis.set_title(analyte)
            axis.set_xticks(np.arange(len(TREATMENTS)), TREATMENTS)
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", alpha=0.22)
        figure.suptitle(
            f"Concentration {ylabel} by analyte and treatment\n"
            "Bayes fit and ML reconstruction are full-record diagnostics; "
            "ML LOYO is held-out validation",
            fontsize=15,
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=3,
        )
        save_figure(
            figure,
            figure_dir / "postprocessing" / stem,
            layout_top=0.935,
        )

    overall = overall_nrmse_table(performance)
    display_order = [*PUBLICATION_ANALYTES, "Volume"]
    for column, ylabel, stem, multiplier in [
        (
            "RMSE_original_units",
            "RMSE",
            "rmse_bayes_ml_reconstruction_loyo_overall",
            1.0,
        ),
        (
            "NRMSE_mean_observed",
            "Mean-normalized RMSE (%)",
            "nrmse_bayes_ml_reconstruction_loyo_overall",
            100.0,
        ),
    ]:
        figure, axes = plt.subplots(4, 3, figsize=(14, 16))
        for axis, target in zip(axes.flat, display_order):
            group = overall.loc[overall["DisplayTarget"].eq(target)]
            values = []
            for series in series_order:
                value = pd.to_numeric(
                    group.loc[group["ComparisonSeries"].eq(series), column],
                    errors="coerce",
                )
                values.append(float(value.iloc[0]) * multiplier)
            bars = axis.bar(
                np.arange(len(series_order)),
                values,
                color=[colors[series] for series in series_order],
                alpha=0.88,
                width=0.72,
            )
            if column == "NRMSE_mean_observed":
                labels = [
                    f"{value:.0f}%" if abs(value) >= 10 else f"{value:.1f}%"
                    for value in values
                ]
                panel_unit = "%"
            else:
                labels = [f"{value:.3g}" for value in values]
                panel_unit = "L" if target == "Volume" else "mg/L"
            axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
            axis.set_title(
                target if column == "NRMSE_mean_observed" else f"{target} ({panel_unit})"
            )
            axis.set_xticks(np.arange(len(series_order)), short_labels, fontsize=8)
            axis.set_ylabel(ylabel if column == "NRMSE_mean_observed" else f"RMSE ({panel_unit})")
            finite_values = [value for value in values if np.isfinite(value)]
            upper = max(finite_values) * 1.25 if finite_values else 1.0
            axis.set_ylim(0, upper if upper > 0 else 1.0)
            axis.grid(True, axis="y", alpha=0.22)
        for axis in axes.flat[len(display_order):]:
            axis.set_visible(False)
        metric_title = (
            "Mean-normalized RMSE (NRMSE)"
            if column == "NRMSE_mean_observed"
            else "RMSE"
        )
        figure.suptitle(
            f"Overall {metric_title}: three-track comparison\n"
            "Bayes fit, ML reconstruction, and ML outer-LOYO; target-specific axes; "
            "LOYO measures held-out prediction error, "
            "not interval uncertainty",
            fontsize=15,
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=3,
        )
        if column == "NRMSE_mean_observed":
            save_figure(
                figure,
                figure_dir / "postprocessing" / f"{stem}_faceted",
                layout_top=0.875,
            )
        else:
            save_figure(
                figure,
                figure_dir / f"gof_{stem}",
                figure_dir / "postprocessing" / stem,
                layout_top=0.875,
            )

    figure, axis = plt.subplots(figsize=(16, 8.5))
    positions = np.arange(len(display_order), dtype=float)
    # Let Bayes and the full-record ML reconstruction determine the shared
    # scale, rounded to the next 100%. Outer-LOYO is intentionally excluded so
    # an extreme held-out error can be hatched rather than flattening the plot.
    fit_scale_values = (
        pd.to_numeric(
            overall.loc[
                overall["ComparisonSeries"].isin(series_order[:2]),
                "NRMSE_mean_observed",
            ],
            errors="coerce",
        ).to_numpy(dtype=float)
        * 100.0
    )
    finite_fit_scale_values = fit_scale_values[np.isfinite(fit_scale_values)]
    largest_fit_value = (
        float(np.max(finite_fit_scale_values))
        if finite_fit_scale_values.size
        else 500.0
    )
    nrmse_axis_cap = max(500.0, float(np.ceil(largest_fit_value / 100.0) * 100.0))
    capped_bars = []
    for series in series_order:
        rows = (
            overall.loc[overall["ComparisonSeries"].eq(series)]
            .set_index("DisplayTarget")
            .reindex(display_order)
        )
        values = (
            pd.to_numeric(rows["NRMSE_mean_observed"], errors="coerce")
            .to_numpy(dtype=float)
            * 100.0
        )
        display_values = np.minimum(values, nrmse_axis_cap)
        bar_positions = positions + offsets[series]
        bars = axis.bar(
            bar_positions,
            display_values,
            width=0.22,
            color=colors[series],
            alpha=0.88,
            label=series,
        )
        for bar, x_position, target, actual_value in zip(
            bars, bar_positions, display_order, values
        ):
            if np.isfinite(actual_value) and actual_value > nrmse_axis_cap:
                bar.set_hatch("///")
                bar.set_edgecolor("0.25")
                capped_bars.append((x_position, target, series, actual_value))
    axis.set_xticks(
        positions, display_order, rotation=35, ha="right", fontsize=13
    )
    axis.tick_params(axis="y", labelsize=12)
    axis.set_ylabel("Mean-normalized RMSE (%)", fontsize=14)
    figure.suptitle(
        "Overall NRMSE by analyte and event volume: three-track comparison\n"
        f"Shared linear scale capped at {nrmse_axis_cap:,.0f}%; "
        "higher bars are hatched and labeled with actual values",
        fontsize=17,
        y=0.985,
    )
    axis.set_ylim(0, nrmse_axis_cap)
    axis.grid(True, axis="y", alpha=0.22)
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=3,
        fontsize=12,
    )
    for x_position, _target, _series, actual_value in capped_bars:
        axis.text(
            x_position,
            nrmse_axis_cap - 10,
            f"{actual_value:,.0f}%",
            ha="center",
            va="top",
            rotation=90,
            fontsize=10.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "0.45",
                "alpha": 0.92,
            },
        )
    save_figure(
        figure,
        figure_dir / "gof_nrmse_bayes_ml_reconstruction_loyo_overall",
        figure_dir / "postprocessing" / "nrmse_bayes_ml_reconstruction_loyo_overall",
        layout_top=0.80,
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
    top_k = int(
        PHYSICAL_EVENT_CONFIG["ml_feature_contract"][
            "feature_importance_top_k"
        ]
    )
    for target, stem in [
        ("Concentration", "feature_importance_concentration"),
        ("Volume", "feature_importance_volume"),
    ]:
        rows = feature_importance.loc[feature_importance["Target"].eq(target)].copy()
        rows["importance_mean"] = pd.to_numeric(rows["importance_mean"], errors="coerce")
        rows["importance_sd"] = pd.to_numeric(rows["importance_sd"], errors="coerce").fillna(0)
        rows = rows.dropna(subset=["importance_mean"])
        total_features = len(rows)
        rows = rows.nlargest(top_k, "importance_mean").sort_values("importance_mean")
        figure, axis = plt.subplots(
            figsize=(12, max(9, 0.55 * len(rows) + 2.5))
        )
        axis.barh(
            rows["feature"].astype(str), rows["importance_mean"],
            xerr=rows["importance_sd"], color="#7b6d8d", alpha=0.9,
            error_kw={"ecolor": "black", "elinewidth": 1.2, "capsize": 2},
        )
        display_scope = (
            f"all {total_features} inputs"
            if total_features <= top_k
            else f"top {top_k} of {total_features} inputs"
        )
        axis.set_title(
            f"{target} model: descriptive, noncausal feature importance\n"
            f"({display_scope}, ranked by mean across LOYO folds)",
            fontsize=15,
            pad=12,
        )
        axis.set_xlabel(
            "CatBoost feature importance (mean ± SD)",
            fontsize=13,
        )
        axis.tick_params(axis="y", labelsize=12)
        axis.tick_params(axis="x", labelsize=11)
        axis.grid(True, axis="x", alpha=0.22)
        save_figure(figure, figure_dir / "postprocessing" / stem)


def make_figures(
    annual: pd.DataFrame,
    cumulative: pd.DataFrame,
    ct_summary: pd.DataFrame,
    performance: pd.DataFrame,
    ml_evaluation_tracks: pd.DataFrame,
    loyo_coverage: pd.DataFrame,
    bayes_coverage: pd.DataFrame,
    feature_importance: pd.DataFrame,
    observed_annual: pd.DataFrame,
    annual_volume: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_annual_comparison(annual, observed_annual, figure_dir)
    plot_annual_volume_comparison(annual_volume, figure_dir)
    plot_annual_uncertainty_supplement(annual, observed_annual, figure_dir)
    plot_cumulative_comparison(cumulative, figure_dir)
    plot_ct_relative(ct_summary, figure_dir)
    plot_performance_comparison(performance, figure_dir)
    plot_ml_reconstruction_vs_loyo(ml_evaluation_tracks, figure_dir)
    plot_three_track_performance_comparison(
        performance, ml_evaluation_tracks, figure_dir
    )
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
        bayes_dir / f"run_manifest_bayes_{BAYES_VERSION}.json",
        ml_dir / f"run_manifest_ml_{ML_VERSION}.json",
    ]
    validate_corrected_artifact_metadata(
        manifests,
        expected_years=STUDY_YEARS,
        expected_versions=[BAYES_VERSION, ML_VERSION],
    )
    annual_volume = annual_volume_comparison(bayes_dir, ml_dir)
    annual_volume.to_csv(
        output_dir / "annual_runoff_volume_summary_raw.csv", index=False
    )
    expected_event_analytes = read_required(
        bayes_dir / f"event_analyte_provenance_bayes_{BAYES_VERSION}.csv",
        "Bayesian event-analyte provenance",
    )
    observed_event_analytes = read_required(
        bayes_dir / f"observed_event_analyte_ledger_{BAYES_VERSION}.csv",
        "corrected observed event ledger",
    )
    observed_load_audit = observed_annual_load_completeness(
        expected_event_analytes,
        observed_event_analytes,
    )
    volume_event_rows = read_required(
        ml_dir / "volume_observation_and_event_predictions.csv",
        "volume observation and event-prediction table",
    )
    expected_events = expected_event_analytes[
        ["Year", "Irrigation", "Treatment", "Rep", "PhysicalEventID"]
    ].drop_duplicates()
    observed_volume_audit = observed_annual_volume_completeness(
        expected_events,
        volume_event_rows,
        annual_volume,
    )
    complete_observed_loads = complete_observed_load_rows(observed_load_audit)
    annual_volume_for_figures = complete_observed_volume_rows(
        annual_volume,
        observed_volume_audit,
    )
    annual_volume_complete = annual_volume_complete_observed_comparison(
        annual_volume,
        observed_volume_audit,
    )

    observed_load_audit.to_csv(
        output_dir / "observed_annual_load_completeness_audit_raw.csv",
        index=False,
    )
    observed_load_audit_publication = observed_load_audit[
        [
            "Year",
            "Analyte",
            "Treatment",
            "Rep",
            "ExpectedIrrigationLabels",
            "n_expected_physical_events",
            "n_observed_event_loads",
            "MissingIrrigationLabels",
            "observed_event_fraction",
            "PlotComplete",
            "ObservedReplicateAnnualLoad_kg",
            "n_complete_plots",
            "ObservedAnnualLoad_kg",
            "ObservedSampleSD_kg",
            "ObservedSE_kg",
            "ObservedMinimum_kg",
            "ObservedMaximum_kg",
            "ObservedAnnualLoadRangeLow_kg",
            "ObservedAnnualLoadRangeHigh_kg",
            "ObservedIntervalType",
            "CompletenessDefinition",
        ]
    ].loc[
        lambda frame: frame["Analyte"].astype(str).isin(
            PRIMARY_MANUSCRIPT_ANALYTES
        )
    ]
    publication_complete_observed_table(
        observed_load_audit_publication,
        value_columns=[
            "ObservedReplicateAnnualLoad_kg",
            "ObservedAnnualLoad_kg",
            "ObservedSampleSD_kg",
            "ObservedSE_kg",
            "ObservedMinimum_kg",
            "ObservedMaximum_kg",
            "ObservedAnnualLoadRangeLow_kg",
            "ObservedAnnualLoadRangeHigh_kg",
        ],
        complete_column="PlotComplete",
    ).to_csv(
        output_dir / "observed_annual_load_completeness_publication.csv",
        index=False,
    )
    observed_volume_audit.to_csv(
        output_dir / "observed_annual_volume_completeness_audit_raw.csv",
        index=False,
    )
    observed_volume_audit_publication = observed_volume_audit[
        [
            "Year",
            "Treatment",
            "Rep",
            "ExpectedIrrigationLabels",
            "n_expected_physical_events",
            "n_observed_event_volumes",
            "MissingIrrigationLabels",
            "observed_event_fraction",
            "PlotComplete",
            "ObservedReplicateAnnualVolume_L",
            "n_complete_plots",
            "ObservedAnnualVolume_kL",
            "ObservedSampleSD_L",
            "ObservedSE_L",
            "ObservedMinimum_L",
            "ObservedMaximum_L",
            "ObservedAnnualVolumeRangeLow_kL",
            "ObservedAnnualVolumeRangeHigh_kL",
            "ObservedIntervalType",
            "CompletenessDefinition",
        ]
    ]
    publication_complete_observed_table(
        observed_volume_audit_publication,
        value_columns=[
            "ObservedReplicateAnnualVolume_L",
            "ObservedAnnualVolume_kL",
            "ObservedSampleSD_L",
            "ObservedSE_L",
            "ObservedMinimum_L",
            "ObservedMaximum_L",
            "ObservedAnnualVolumeRangeLow_kL",
            "ObservedAnnualVolumeRangeHigh_kL",
        ],
        complete_column="PlotComplete",
    ).to_csv(
        output_dir / "observed_annual_volume_completeness_publication.csv",
        index=False,
    )
    annual_volume_complete.to_csv(
        output_dir / "annual_runoff_volume_complete_observed_raw.csv",
        index=False,
    )
    publication_complete_observed_table(
        annual_volume_complete,
        value_columns=[
            "BayesPosteriorMean_kL",
            "BayesLower95_kL",
            "BayesUpper95_kL",
            "MLPointTotal_kL",
            "MLLower95_kL",
            "MLUpper95_kL",
            "ObservedAnnualVolume_kL",
            "ObservedAnnualVolumeRangeLow_kL",
            "ObservedAnnualVolumeRangeHigh_kL",
        ],
        complete_column="ObservedAnnualVolumeComplete",
    ).to_csv(
        output_dir / "annual_runoff_volume_complete_observed_publication.csv",
        index=False,
    )
    (
        performance,
        ml_evaluation_tracks,
        bayes_rows,
        bayes_volume,
        ml_rows,
    ) = build_performance_products(bayes_dir, ml_dir)
    write_performance_products(output_dir, performance, ml_evaluation_tracks)

    if args.figures_only:
        saved_annual = read_required(
            output_dir / "annual_load_summary_raw.csv",
            "saved annual-load summary",
        )
        annual_load_complete = annual_load_complete_observed_comparison(
            saved_annual,
            observed_load_audit,
        )
        annual_load_complete.to_csv(
            output_dir / "annual_load_complete_observed_raw.csv",
            index=False,
        )
        publication_complete_observed_table(
            annual_load_complete,
            value_columns=[
                "BayesPosteriorMedian_kg",
                "BayesLower95_kg",
                "BayesUpper95_kg",
                "MLPointTotal_kg",
                "MLLower95_kg",
                "MLUpper95_kg",
                "ObservedAnnualLoad_kg",
                "ObservedAnnualLoadRangeLow_kg",
                "ObservedAnnualLoadRangeHigh_kg",
            ],
            complete_column="ObservedAnnualLoadComplete",
        ).to_csv(
            output_dir / "annual_load_complete_observed_publication.csv",
            index=False,
        )
        make_figures(
            saved_annual,
            read_required(
                output_dir / "cumulative_load_2011_2025_raw.csv",
                "saved cumulative-load summary",
            ),
            read_required(output_dir / "ct_relative_summary_raw.csv", "saved CT-relative summary"),
            performance,
            ml_evaluation_tracks,
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
            complete_observed_loads,
            annual_volume_for_figures,
            figure_dir,
        )
        update_saved_comparison_manifest(output_dir)
        print(f"[DONE] Corrected comparison figures regenerated from saved tables in {figure_dir}")
        return

    bayes_ledger = normalize_ledger(
        read_required(
            bayes_dir / f"event_analyte_draw_ledger_bayes_{BAYES_VERSION}.csv",
            "Bayesian event ledger",
        ), "Bayes", "model_only",
    )
    ml_loyo = normalize_ledger(
        read_required(
            ml_dir / f"event_analyte_draw_ledger_ml_{ML_TAG}.csv",
            "ML LOYO event ledger",
        ),
        "ML", "loyo_model_only", require_complete_years=False,
    )
    ml_full = normalize_ledger(
        read_required(
            ml_dir / "event_analyte_draw_ledger_full_record_model_only.csv",
            "ML full-record model-only event ledger",
        ), "ML", "full_record_model_only",
    )
    ml_full_points = normalize_point_ledger(
        read_required(
            ml_dir / "event_analyte_point_ledger_full_record_model_only.csv",
            "ML full-record model-only point ledger",
        ),
        "ML",
        "full_record_model_only",
    )
    bayes_ledger, bayes_extra_analytes = restrict_publication_analytes(
        bayes_ledger, "Bayesian ledger", require_all=True
    )
    ml_loyo, ml_loyo_extra_analytes = restrict_publication_analytes(
        ml_loyo, "ML LOYO ledger", require_all=False
    )
    ml_full, ml_full_extra_analytes = restrict_publication_analytes(
        ml_full, "ML full-record ledger", require_all=True
    )
    ml_full_points, ml_full_point_extras = restrict_publication_analytes(
        ml_full_points, "ML full-record point ledger", require_all=True
    )
    assert_complete_analyte_treatment_years(bayes_ledger, "Bayesian ledger")
    assert_complete_analyte_treatment_years(ml_full, "ML full-record ledger")
    assert_complete_analyte_treatment_years(
        ml_full_points, "ML full-record point ledger"
    )
    # LOYO ledgers only exist where held-out volume observations are available.
    # Keep them as validation diagnostics; complete-year annual products use the
    # corrected full-record ledgers and must never synthesize missing years.
    all_ledgers = pd.concat([bayes_ledger, ml_full], ignore_index=True)
    annual_draws, annual_summary = annual_products(all_ledgers)
    annual_points = annual_point_products(ml_full_points)
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
    annual_load_complete = annual_load_complete_observed_comparison(
        annual_summary,
        observed_load_audit,
    )
    spearman_raw, spearman_publication = spearman_tables(annual_summary)

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

    observed_subtotals = observed_load_audit[
        [
            "Year", "Analyte", "Treatment", "Rep",
            "ExpectedIrrigationLabels", "MissingIrrigationLabels",
            "n_expected_physical_events", "n_observed_event_loads",
            "ObservedIncompleteSubtotal_kg", "PlotComplete",
            "ObservedReplicateAnnualLoad_kg",
        ]
    ].copy()
    observed_subtotals_publication = observed_subtotals.copy()
    observed_subtotals_publication["ObservedIncompleteSubtotal_kg"] = (
        observed_subtotals_publication["ObservedIncompleteSubtotal_kg"].map(
            lambda value: "-" if pd.isna(value) else f"{value:.4g}"
        )
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
    annual_load_complete.to_csv(
        output_dir / "annual_load_complete_observed_raw.csv",
        index=False,
    )
    publication_complete_observed_table(
        annual_load_complete,
        value_columns=[
            "BayesPosteriorMedian_kg",
            "BayesLower95_kg",
            "BayesUpper95_kg",
            "MLPointTotal_kg",
            "MLLower95_kg",
            "MLUpper95_kg",
            "ObservedAnnualLoad_kg",
            "ObservedAnnualLoadRangeLow_kg",
            "ObservedAnnualLoadRangeHigh_kg",
        ],
        complete_column="ObservedAnnualLoadComplete",
    ).to_csv(
        output_dir / "annual_load_complete_observed_publication.csv",
        index=False,
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
        "- ML line: deterministic mean of replicate-specific annual plot totals; "
        "band: 95% "
        "Monte Carlo empirical calibration-residual prediction interval. Signed "
        "log-scale residuals are resampled from the physical-event-grouped "
        "split-conformal calibration set, with performance evaluated by outer LOYO. "
        "The ML line is not "
        "the median of the propagated draws.\n"
        "- Observed markers: completeness is evaluated independently within Year x "
        "Treatment x Rep x Analyte. Two complete plots are averaged and their "
        "minimum-to-maximum range is shown descriptively; one complete plot is shown "
        "with an x marker and no interval; no complete plots produce no point. "
        "Incomplete subtotals remain audit-only and never enter the primary mean. "
        "The replicate range is not a confidence interval.\n\n"
        "- Annual runoff-volume figure: Bayesian posterior mean with 95% credible "
        "interval, deterministic ML mean of replicate annual plot volumes without an "
        "ML ribbon, and the observed complete-plot mean/range; all volumes are shown "
        "in kL. Observed volume completeness is assessed within each replicate plot.\n\n"
        "The primary annual comparison figures retain the Bayesian 95% credible-interval "
        "ribbon and observed replicate ranges, while omitting the ML prediction ribbon "
        "so that Bayes and ML centers remain readable on a common linear scale. Full model "
        "intervals are retained in the "
        "supplemental_uncertainty figure folder, using separate Bayes and ML rows with "
        "method-specific linear y-axis scales.\n\n"
        "RMSE/NRMSE evaluation is deliberately split into two ML tracks. The primary "
        "Bayes-versus-ML performance figures retain the unchanged Bayesian posterior-"
        "predictive fit metrics and use ML full-record physical-event reconstruction "
        "points. Separate tables and figures retain ML outer-LOYO held-out validation "
        "metrics. ML concentration metrics are resolved by PhysicalEventID x Analyte; "
        "volume metrics are resolved by PhysicalEventID. Both use median resolution, "
        "and neither is calculated against partial observed annual subtotals.\n\n"
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
            ml_evaluation_tracks,
            loyo_coverage,
            bayes_coverage,
            feature_importance,
            complete_observed_loads,
            annual_volume_for_figures,
            figure_dir,
        )

    manifest = {
        "workflow_version": COMPARISON_VERSION,
        "bayesian_workflow_version": BAYES_VERSION,
        "ml_workflow_version": ML_VERSION,
        "data_contract_version": PHYSICAL_EVENT_CONFIG["data_contract"][
            "workflow_version"
        ],
        "event_unit": "PhysicalEventID",
        "physical_event_key": PHYSICAL_EVENT_CONFIG["physical_event_key"],
        "years": STUDY_YEARS,
        "legacy_fallbacks": False,
        "missing_year_zero_fill": False,
        "event_analyte_draw_uniqueness_asserted": True,
        "event_analyte_point_uniqueness_asserted": True,
        "primary_bayes_central_estimate": "posterior_median",
        "primary_ml_central_estimate": "mean_of_replicate_annual_plot_totals",
        "primary_bayes_prediction_scope": "all modeled rows for the 10 prespecified study analytes and all physical events, observed and missing",
        "primary_ml_prediction_scope": "all eligible concentration rows and physical events, observed and missing",
        "primary_modeled_products_use_observed_value_substitution": False,
        "annual_reporting_unit": "mean_per_treatment_plot",
        "annual_aggregation_hierarchy": PHYSICAL_EVENT_CONFIG[
            "annual_reporting"
        ]["hierarchy"],
        "primary_observed_role": "complete_annual_reference_markers_only",
        "observed_annual_load_completeness_unit": "Year x Treatment x Rep x Analyte",
        "observed_annual_volume_completeness_unit": "Year x Treatment x Rep",
        "partial_observed_annual_subtotals_plotted": False,
        "observed_annual_load_complete_groups": int(
            observed_load_audit.loc[
                observed_load_audit["n_complete_plots"].gt(0),
                ["Year", "Analyte", "Treatment"],
            ].drop_duplicates().shape[0]
        ),
        "observed_annual_load_total_groups": int(
            observed_load_audit[
                ["Year", "Analyte", "Treatment"]
            ].drop_duplicates().shape[0]
        ),
        "observed_annual_volume_complete_groups": int(
            observed_volume_audit.loc[
                observed_volume_audit["n_complete_plots"].gt(0),
                ["Year", "Treatment"],
            ].drop_duplicates().shape[0]
        ),
        "observed_annual_volume_total_groups": int(
            observed_volume_audit[
                ["Year", "Treatment"]
            ].drop_duplicates().shape[0]
        ),
        "ml_interval_type": "95% Monte Carlo empirical calibration-residual prediction interval",
        "ml_interval_evaluation": "outer leave-one-year-out",
        "ml_monte_carlo_propagation": "weighted_resampling_of_signed_log_scale_split_conformal_calibration_residuals",
        "ml_interval_is_parameter_confidence_interval": False,
        "primary_annual_figure_bayes_credible_ribbon_shown": True,
        "primary_annual_figure_ml_prediction_ribbon_shown": True,
        "primary_annual_figure_y_scale_reference": (
            "Bayesian upper 95% credible bounds plus observed references and "
            "ML centers; ML prediction-interval bounds excluded"
        ),
        "primary_annual_figure_ml_prediction_ribbon_may_be_clipped": True,
        "primary_annual_figure_observed_replicate_ranges_shown": True,
        "supplemental_annual_uncertainty_figures": True,
        "supplemental_uncertainty_y_scales": "method-specific linear scales",
        "annual_runoff_volume_comparison_figure": True,
        "annual_runoff_volume_units": "kL",
        "annual_runoff_volume_bayes_center": "posterior_mean",
        "annual_runoff_volume_ml_center": "mean_of_replicate_annual_plot_volumes",
        "annual_runoff_volume_ml_interval": (
            "95% Monte Carlo empirical calibration-residual prediction interval "
            "of mean-per-plot volume draws"
        ),
        "annual_runoff_volume_ml_prediction_ribbon_shown": True,
        "annual_runoff_volume_y_scale_reference": (
            "Bayesian upper 95% credible bounds plus observed references and "
            "ML centers; ML prediction-interval bounds excluded"
        ),
        "annual_runoff_volume_ml_prediction_ribbon_may_be_clipped": True,
        "annual_runoff_volume_observed_center": (
            "mean_of_complete_replicate_plot_totals_or_single_available_plot"
        ),
        "observed_interval_type": "descriptive_replicate_minimum_to_maximum_range",
        "storm_handling": PHYSICAL_EVENT_CONFIG["storm_handling"],
        "ml_feature_contract": PHYSICAL_EVENT_CONFIG["ml_feature_contract"],
        "feature_importance_top_k": int(
            PHYSICAL_EVENT_CONFIG["ml_feature_contract"][
                "feature_importance_top_k"
            ]
        ),
        **metric_manifest_fields(),
        "loyo_ledger_role": "diagnostics_only_because_years_without_observed_volume_are_absent",
        "loyo_ledger_years_present": sorted(ml_loyo["Year"].unique().astype(int).tolist()),
        "primary_manuscript_analytes": PRIMARY_MANUSCRIPT_ANALYTES,
        "technical_export_analytes": PUBLICATION_ANALYTES,
        "bayes_modeled_analytes": PUBLICATION_ANALYTES,
        "bayes_unexpected_analytes": sorted(set(bayes_extra_analytes)),
        "excluded_ml_only_analytes": sorted(set(
            ml_loyo_extra_analytes
            + ml_full_extra_analytes
            + ml_full_point_extras
        )),
        "inputs": [str(path) for path in manifests],
    }
    (
        output_dir / f"run_manifest_comparison_{COMPARISON_VERSION}.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Corrected comparison outputs written to {output_dir}")


if __name__ == "__main__":
    main()
