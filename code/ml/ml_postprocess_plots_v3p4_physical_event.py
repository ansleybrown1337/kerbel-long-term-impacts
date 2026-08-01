#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Post-process the Kerbel v3p4 physical-event ML workflow.

Reads corrected outputs in ``results/ml`` and writes figures to ``figures/ml``
by default. Both paths can be
overridden explicitly for strict-prediction sensitivity runs or smoke tests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.physical_event import (  # noqa: E402
    add_concentration_observation_id,
    add_physical_event_id,
    build_volume_observation_table,
    observed_annual_plot_summary,
    resolve_prediction_draws,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "physical_event_v3p4.json"
)
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"ML v3p4 configuration is absent: {CONFIG_PATH}")
PHYSICAL_EVENT_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
    raise FileNotFoundError("Could not locate the repository root; pass --repo.")


def resolve_path(repo: Path, value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_both(figure: plt.Figure, stem: Path, dpi: int = 220) -> None:
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".jpg"), dpi=dpi)
    figure.savefig(stem.with_suffix(".png"), dpi=dpi)
    plt.close(figure)


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required ML output is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def plot_cv_diagnostics(metrics: pd.DataFrame, figure_dir: Path) -> None:
    required = {"Target", "Year_Test", "RMSE", "R2"}
    if not required.issubset(metrics.columns):
        warn(f"CV metrics are missing {sorted(required - set(metrics.columns))}; skipping CV diagnostics.")
        return
    for metric, ylabel, stem_name in [
        ("RMSE", "RMSE (log1p scale)", "cv_rmse_by_year"),
        ("R2", "R-squared", "cv_r2_by_year"),
    ]:
        figure, axis = plt.subplots(figsize=(10, 5))
        for target in ["logC", "logV"]:
            subset = metrics.loc[metrics["Target"] == target].sort_values("Year_Test")
            if not subset.empty:
                axis.plot(numeric(subset["Year_Test"]), numeric(subset[metric]), marker="o", label=target)
        if metric == "R2":
            axis.axhline(0, color="0.4", linestyle="--", linewidth=1)
        axis.set_title(f"LOYO full-record prediction {metric}")
        axis.set_xlabel("Held-out year")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        save_both(figure, figure_dir / stem_name)


def plot_feature_importance(path: Path, figure_dir: Path, target: str, top_k: int) -> None:
    if not path.exists():
        warn(f"Missing {path.name}; skipping {target} feature importance.")
        return
    data = pd.read_csv(path)
    if data.empty or "feature" not in data.columns:
        warn(f"{path.name} is empty or malformed.")
        return
    value_column = "importance_mean" if "importance_mean" in data.columns else "importance"
    if value_column not in data.columns:
        warn(f"{path.name} has no importance column.")
        return
    data[value_column] = numeric(data[value_column])
    data = data.dropna(subset=[value_column]).nlargest(int(top_k), value_column).sort_values(value_column)
    xerr = None
    if "importance_sd" in data.columns:
        xerr = numeric(data["importance_sd"]).fillna(0).to_numpy(dtype=float)
    figure, axis = plt.subplots(
        figsize=(12, max(8, 0.55 * len(data) + 2.0))
    )
    axis.barh(
        data["feature"].astype(str), data[value_column], xerr=xerr,
        error_kw={"ecolor": "black", "elinewidth": 1.1, "capsize": 2},
    )
    unit = "analyte rows" if target == "logC" else "unique volume events"
    axis.set_title(
        f"Feature importance: {target} model ({unit})",
        fontsize=15,
        pad=12,
    )
    axis.set_xlabel(
        "CatBoost importance (mean ± SD across LOYO folds)",
        fontsize=13,
    )
    axis.tick_params(axis="y", labelsize=12)
    axis.tick_params(axis="x", labelsize=11)
    axis.grid(True, axis="x", alpha=0.2)
    save_both(figure, figure_dir / f"feature_importance_{target}")


def plot_parity(predictions: pd.DataFrame, figure_dir: Path) -> None:
    required = {"Target", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        warn("CV predictions lack columns required for parity plots.")
        return
    specifications = [
        ("Result_mg_L", "logC", "parity_logC"),
        ("Volume_L", "logV (unique events)", "parity_logV"),
    ]
    for target_value, title_target, stem_name in specifications:
        subset = predictions.loc[predictions["Target"] == target_value].copy()
        x = np.log1p(numeric(subset["y_true"]).clip(lower=0))
        y = np.log1p(numeric(subset["y_pred"]).clip(lower=0))
        valid = x.notna() & y.notna()
        if not valid.any():
            continue
        x_values, y_values = x.loc[valid].to_numpy(), y.loc[valid].to_numpy()
        limit_low = float(min(x_values.min(), y_values.min()))
        limit_high = float(max(x_values.max(), y_values.max()))
        figure, axis = plt.subplots(figsize=(6.5, 6.5))
        axis.scatter(x_values, y_values, s=13, alpha=0.28)
        axis.plot([limit_low, limit_high], [limit_low, limit_high], color="black", linewidth=1.5)
        axis.set_title(f"LOYO parity: {title_target}")
        axis.set_xlabel("Observed log1p")
        axis.set_ylabel("Predicted log1p")
        axis.grid(True, alpha=0.2)
        save_both(figure, figure_dir / stem_name)


def plot_coverage(predictions: pd.DataFrame, figure_dir: Path, nominal: float) -> None:
    required = {"Target", "y_true", "pi_low", "pi_high"}
    if not required.issubset(predictions.columns):
        warn("CV predictions lack columns required for interval-coverage diagnostics.")
        return
    data = predictions.copy()
    for column in ["y_true", "pi_low", "pi_high"]:
        data[column] = numeric(data[column])
    data = data.dropna(subset=["Target", "y_true", "pi_low", "pi_high"])
    data["covered"] = data["y_true"].between(data["pi_low"], data["pi_high"])

    overall = data.groupby("Target", as_index=False).agg(n=("covered", "size"), coverage=("covered", "mean"))
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(overall["Target"], overall["coverage"])
    axis.axhline(nominal, linestyle="--", color="0.35", label=f"Nominal {nominal:.0%}")
    axis.set_ylim(0, 1)
    axis.set_title("LOYO prediction-interval coverage")
    axis.set_ylabel("Coverage fraction")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.2)
    save_both(figure, figure_dir / "pi_coverage_overall")

    if "Year" in data.columns:
        data["Year"] = numeric(data["Year"])
        by_year = data.dropna(subset=["Year"]).groupby(["Target", "Year"], as_index=False).agg(
            n=("covered", "size"), coverage=("covered", "mean")
        )
        figure, axis = plt.subplots(figsize=(10, 5))
        for target in ["Result_mg_L", "Volume_L"]:
            subset = by_year.loc[by_year["Target"] == target].sort_values("Year")
            if not subset.empty:
                axis.plot(subset["Year"], subset["coverage"], marker="o", label=target)
        axis.axhline(nominal, linestyle="--", color="0.35", label=f"Nominal {nominal:.0%}")
        axis.set_ylim(0, 1)
        axis.set_title("LOYO prediction-interval coverage by held-out year")
        axis.set_xlabel("Held-out year")
        axis.set_ylabel("Coverage fraction")
        axis.legend()
        axis.grid(True, axis="y", alpha=0.2)
        save_both(figure, figure_dir / "pi_coverage_by_year")


def observed_annual_loads(source_path: Path) -> pd.DataFrame:
    data = pd.read_csv(source_path, low_memory=False)
    required = {
        "Year", "Irrigation", "Rep", "Treatment", "Analyte",
        "Result_mg_L", "Volume",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"Source data is missing {sorted(required - set(data.columns))}")
    rows = add_concentration_observation_id(add_physical_event_id(data))
    volume_observations, _ = build_volume_observation_table(rows, strict=True)
    analyte_column = "analyte_abbr" if "analyte_abbr" in rows else "Analyte"
    concentration = rows.loc[rows["Result_mg_L"].notna()].copy()
    concentration["Draw"] = 0
    concentration["Concentration_mg_L"] = numeric(concentration["Result_mg_L"])
    concentration_event = resolve_prediction_draws(
        concentration,
        group_columns=[
            "PhysicalEventID", "Year", "Treatment", "Rep", analyte_column, "Draw"
        ],
        value_column="Concentration_mg_L",
        method=PHYSICAL_EVENT_CONFIG["concentration_resolution"],
        method_column="SampleMethod" if "SampleMethod" in concentration else None,
        method_priority=PHYSICAL_EVENT_CONFIG["method_priority"],
    ).rename(columns={analyte_column: "Analyte"})
    concentration_counts = (
        concentration.groupby(["PhysicalEventID", analyte_column], as_index=False)
        .agg(observed_rows=("ConcentrationObservationID", "size"))
        .rename(columns={analyte_column: "Analyte"})
    )
    volume_observations["Draw"] = 0
    volume_observations["Volume_L"] = numeric(volume_observations["Volume"])
    volume_event = resolve_prediction_draws(
        volume_observations,
        group_columns=["PhysicalEventID", "Draw"],
        value_column="Volume_L",
        method=PHYSICAL_EVENT_CONFIG["volume_resolution"],
        method_column="MeasureMethod" if "MeasureMethod" in volume_observations else None,
        method_priority=PHYSICAL_EVENT_CONFIG["method_priority"],
    )
    event_roster = rows[
        ["PhysicalEventID", "Year", "Irrigation", "Rep", "Treatment"]
    ].drop_duplicates()
    analytes = pd.DataFrame(
        {"Analyte": sorted(rows[analyte_column].dropna().astype(str).unique())}
    )
    expected = event_roster.assign(_join=1).merge(
        analytes.assign(_join=1), on="_join"
    ).drop(columns="_join")
    ledger = expected.merge(
        concentration_event[
            ["PhysicalEventID", "Analyte", "Concentration_mg_L"]
        ],
        on=["PhysicalEventID", "Analyte"],
        how="left",
        validate="one_to_one",
    ).merge(
        concentration_counts,
        on=["PhysicalEventID", "Analyte"],
        how="left",
        validate="one_to_one",
    ).merge(
        volume_event[["PhysicalEventID", "Volume_L"]],
        on="PhysicalEventID",
        how="left",
        validate="many_to_one",
    )
    ledger["event_load_mg"] = np.where(
        numeric(ledger["Volume_L"]).eq(0),
        0.0,
        numeric(ledger["Concentration_mg_L"]) * numeric(ledger["Volume_L"]),
    )
    audit = observed_annual_plot_summary(
        expected,
        ledger.loc[ledger["event_load_mg"].notna()],
        value_column="event_load_mg",
        analysis_columns=["Analyte"],
    )
    return (
        audit[
            [
                "Year", "Treatment", "Analyte", "TreatmentMean",
                "RangeLow", "RangeHigh", "n_complete_plots", "IntervalType",
            ]
        ]
        .drop_duplicates(["Year", "Treatment", "Analyte"])
        .rename(
            columns={
                "TreatmentMean": "observed_load_mg",
                "RangeLow": "observed_range_low_mg",
                "RangeHigh": "observed_range_high_mg",
            }
        )
        .dropna(subset=["observed_load_mg"])
    )


def normalize_ml_summary(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Year", "Treatment", "Analyte"}
    if not required.issubset(data.columns):
        # Accommodate Bayes-aligned lowercase aliases without changing the v2 writer.
        rename = {"year": "Year", "treatment": "Treatment", "analyte": "Analyte"}
        data = data.rename(columns={key: value for key, value in rename.items() if key in data.columns})
    if not required.issubset(data.columns):
        raise ValueError(f"Annual summary is missing {sorted(required - set(data.columns))}")
    center = next(
        (c for c in ["point_total_mg", "median", "mean", "load_mean"] if c in data.columns),
        None,
    )
    low = next((c for c in ["low", "load_low"] if c in data.columns), None)
    high = next((c for c in ["high", "load_high"] if c in data.columns), None)
    if center is None:
        raise ValueError("Annual summary has no recognized load-center column.")
    result = data[["Year", "Treatment", "Analyte"]].copy()
    result["center_mg"] = numeric(data[center])
    result["low_mg"] = numeric(data[low]) if low else result["center_mg"]
    result["high_mg"] = numeric(data[high]) if high else result["center_mg"]
    result["Year"] = numeric(result["Year"])
    return result.dropna(subset=["Year", "Treatment", "Analyte", "center_mg"])


def slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "analyte"


def plot_annual_loads(
    summary: pd.DataFrame,
    observed: pd.DataFrame,
    figure_dir: Path,
    analytes: Sequence[str] | None,
    suffix: str,
    units: str,
    scope_label: str,
) -> None:
    modeled = normalize_ml_summary(summary)
    available = sorted(modeled["Analyte"].astype(str).unique().tolist())
    selected = available if analytes is None else [value for value in analytes if value in available]
    factor, ylabel = {
        "mg": (1.0, "Annual load (mg)"),
        "g": (1e-3, "Annual load (g)"),
        "kg": (1e-6, "Annual load (kg)"),
    }[units]
    colors = {"CT": "C0", "MT": "C1", "ST": "C2"}

    for analyte in selected:
        figure, axis = plt.subplots(figsize=(11, 5.8))
        for treatment in sorted(modeled["Treatment"].dropna().astype(str).unique()):
            subset = modeled.loc[
                (modeled["Analyte"].astype(str) == analyte)
                & (modeled["Treatment"].astype(str) == treatment)
            ].sort_values("Year")
            if subset.empty:
                continue
            color = colors.get(treatment)
            axis.plot(
                subset["Year"], subset["center_mg"] * factor,
                marker="o", linewidth=2, label=f"{treatment} ML", color=color,
            )
            axis.fill_between(
                numeric(subset["Year"]), subset["low_mg"] * factor, subset["high_mg"] * factor,
                alpha=0.18, color=color,
            )
            observed_subset = observed.loc[
                (observed["Analyte"].astype(str) == analyte)
                & (observed["Treatment"].astype(str) == treatment)
            ].sort_values("Year")
            if not observed_subset.empty:
                complete_pair = observed_subset["n_complete_plots"].ge(2)
                pair = observed_subset.loc[complete_pair]
                single = observed_subset.loc[~complete_pair]
                if not pair.empty:
                    axis.errorbar(
                        pair["Year"],
                        pair["observed_load_mg"] * factor,
                        yerr=np.vstack(
                            [
                                (pair["observed_load_mg"] - pair["observed_range_low_mg"])
                                * factor,
                                (pair["observed_range_high_mg"] - pair["observed_load_mg"])
                                * factor,
                            ]
                        ),
                        fmt="o", mfc="none", mec=color, ecolor=color,
                        markersize=8, capsize=3, linewidth=1.2,
                        label=f"{treatment} observed (n=2; replicate range)",
                    )
                if not single.empty:
                    axis.scatter(
                        single["Year"], single["observed_load_mg"] * factor,
                        marker="x", color=color, s=70, linewidths=1.8,
                        label=f"{treatment} observed (n=1 plot)",
                    )
        axis.set_title(
            f"{analyte}: observed vs ML annual load ({scope_label})\n"
            "ML line: mean of replicate annual plot totals; band: 95% "
            "calibration-residual PI"
        )
        axis.set_xlabel("Year")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(ncol=2)
        save_both(figure, figure_dir / f"annual_load_{slug(analyte)}_obs_vs_ml{suffix}")


def plot_event_unit_audit(output_dir: Path, figure_dir: Path) -> None:
    summary_path = output_dir / "event_volume_audit_summary.csv"
    training_path = output_dir / "volume_observation_training_table_v3p4.csv"
    if not summary_path.exists():
        warn("event_volume_audit_summary.csv is missing; skipping event-unit audit figure.")
        return
    summary = pd.read_csv(summary_path, dtype=str)
    values = dict(zip(summary["metric"], summary["value"]))
    labels_and_metrics = [
        ("Concentration observations", "concentration_observations"),
        ("Physical events", "physical_events"),
        ("Raw volume rows", "volume_candidate_rows_before_dedup"),
        ("Genuine volume observations", "genuine_volume_observations"),
    ]
    labels, counts = [], []
    for label, metric in labels_and_metrics:
        try:
            labels.append(label)
            counts.append(float(values[metric]))
        except (KeyError, TypeError, ValueError):
            pass
    if counts:
        figure, axis = plt.subplots(figsize=(9, 5))
        bars = axis.bar(labels, counts)
        for bar, count in zip(bars, counts):
            axis.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,.0f}", ha="center", va="bottom")
        axis.set_title("Volume-model unit correction: analyte rows to physical events")
        axis.set_ylabel("Row/event count")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(True, axis="y", alpha=0.2)
        save_both(figure, figure_dir / "event_volume_unit_audit")

    if training_path.exists():
        events = pd.read_csv(training_path, low_memory=False)
        if "source_row_count" in events.columns:
            rows_per_event = numeric(events["source_row_count"]).dropna()
            figure, axis = plt.subplots(figsize=(8, 4.8))
            axis.hist(rows_per_event, bins=min(30, max(5, int(rows_per_event.nunique()))), edgecolor="white")
            axis.set_title("Copied-row multiplicity before volume-observation deduplication")
            axis.set_xlabel("Source rows per genuine volume observation")
            axis.set_ylabel("Volume observations")
            axis.grid(True, axis="y", alpha=0.2)
            save_both(figure, figure_dir / "event_volume_rows_per_event")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--fig_dir", default=None)
    parser.add_argument("--source_data", default=None)
    parser.add_argument("--analytes", default=None, help="Comma-separated exact Analyte labels; default: all.")
    parser.add_argument("--units", choices=["mg", "g", "kg"], default="g")
    parser.add_argument("--alpha", type=float, default=0.05)
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
        "--use_imputed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also plot the non-primary observed-plus-imputed sensitivity. "
            "Primary figures always use model predictions at every point."
        ),
    )
    return parser.parse_args()


def generate_figures(
    *,
    repo: Path,
    output_dir: Path,
    figure_dir: Path,
    source_path: Path,
    analytes: Sequence[str] | None = None,
    units: str = "g",
    alpha: float = 0.05,
    fi_topk: int = 20,
    use_imputed: bool = False,
) -> None:
    """Generate the complete ML figure suite from already-saved outputs."""
    repo = Path(repo).resolve()
    output_dir = Path(output_dir).resolve()
    figure_dir = Path(figure_dir).resolve()
    source_path = Path(source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Required source data not found: {source_path}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading ML outputs from {output_dir}")
    print(f"[INFO] Writing figures to {figure_dir}")
    metrics = read_required(output_dir / "cv_metrics_by_year.csv")
    predictions = read_required(output_dir / "loyo_predictions_row_level.csv")
    observed = observed_annual_loads(source_path)

    plot_cv_diagnostics(metrics, figure_dir)
    plot_parity(predictions, figure_dir)
    plot_coverage(predictions, figure_dir, nominal=1.0 - alpha)
    plot_feature_importance(output_dir / "feature_importance_logC.csv", figure_dir, "logC", fi_topk)
    plot_feature_importance(output_dir / "feature_importance_logV.csv", figure_dir, "logV", fi_topk)

    full_record_path = output_dir / "annual_load_summary_full_record_model_only.csv"
    if full_record_path.exists():
        plot_annual_loads(
            pd.read_csv(full_record_path), observed, figure_dir, analytes, "", units,
            "full-record model-only",
        )
    else:
        warn("Full-record model-only annual summary is missing; skipping primary annual-load plots.")

    loyo_path = output_dir / "annual_load_summary_model_only_loyo.csv"
    if loyo_path.exists():
        plot_annual_loads(
            pd.read_csv(loyo_path), observed, figure_dir, analytes, "_loyo_supported", units,
            "LOYO-supported years only",
        )
    else:
        warn("LOYO annual summary is missing; skipping validation-only annual-load plots.")

    if use_imputed:
        imputed_path = output_dir / "annual_load_summary_observed_plus_imputed_sensitivity.csv"
        if imputed_path.exists():
            plot_annual_loads(
                pd.read_csv(imputed_path), observed, figure_dir, analytes, "_imputed", units,
                "observed plus imputed sensitivity",
            )
        else:
            warn("Observed-plus-imputed sensitivity summary is missing; skipping sensitivity plots.")
    plot_event_unit_audit(output_dir, figure_dir)
    print("[DONE] v3p4 physical-event post-processing figures created.")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())
    output_dir = resolve_path(
        repo, args.output_dir, repo / PHYSICAL_EVENT_CONFIG["output_roots"]["ml_results"]
    )
    figure_dir = resolve_path(
        repo, args.fig_dir, repo / PHYSICAL_EVENT_CONFIG["output_roots"]["ml_figures"]
    )
    source_path = resolve_path(repo, args.source_data, repo / "out" / "wq_cleaned.csv")
    analytes = [item.strip() for item in args.analytes.split(",") if item.strip()] if args.analytes else None
    generate_figures(
        repo=repo,
        output_dir=output_dir,
        figure_dir=figure_dir,
        source_path=source_path,
        analytes=analytes,
        units=args.units,
        alpha=args.alpha,
        fi_topk=args.fi_topk,
        use_imputed=args.use_imputed,
    )


if __name__ == "__main__":
    main()
