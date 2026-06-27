#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Post-process the Kerbel v2 event-level ML workflow.

Reads the compatible outputs in ``out/ml_catboost_conformal_loyo`` and writes
figures to ``figs/ml_catboost_conformal_loyo`` by default. Both paths can be
overridden explicitly for strict-prediction sensitivity runs or smoke tests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    raise FileNotFoundError("Could not locate the repository root; pass --repo.")


def resolve_path(repo: Path, value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def true_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


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
        axis.set_title(f"LOYO conditional-reconstruction {metric}")
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
    figure, axis = plt.subplots(figsize=(11, max(5, 0.32 * len(data) + 1.5)))
    axis.barh(data["feature"].astype(str), data[value_column])
    unit = "analyte rows" if target == "logC" else "unique volume events"
    axis.set_title(f"Feature importance: {target} model ({unit})")
    axis.set_xlabel("Mean CatBoost importance")
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
    required = {"Year", "Treatment", "Analyte", "Result_mg_L", "Volume"}
    if not required.issubset(data.columns):
        raise ValueError(f"Source data is missing {sorted(required - set(data.columns))}")
    data["Year"] = numeric(data["Year"])
    data["Result_mg_L"] = numeric(data["Result_mg_L"])
    data["Volume"] = numeric(data["Volume"])
    no_runoff = true_mask(data["NoRunoff"]) if "NoRunoff" in data.columns else pd.Series(False, index=data.index)
    data["event_load_mg"] = data["Result_mg_L"] * data["Volume"]
    data.loc[no_runoff, "event_load_mg"] = 0.0
    data = data.dropna(subset=["Year", "Treatment", "Analyte", "event_load_mg"])
    return data.groupby(["Year", "Treatment", "Analyte"], as_index=False).agg(
        observed_load_mg=("event_load_mg", "sum"), observed_rows=("event_load_mg", "size")
    )


def normalize_ml_summary(data: pd.DataFrame) -> pd.DataFrame:
    required = {"Year", "Treatment", "Analyte"}
    if not required.issubset(data.columns):
        # Accommodate Bayes-aligned lowercase aliases without changing the v2 writer.
        rename = {"year": "Year", "treatment": "Treatment", "analyte": "Analyte"}
        data = data.rename(columns={key: value for key, value in rename.items() if key in data.columns})
    if not required.issubset(data.columns):
        raise ValueError(f"Annual summary is missing {sorted(required - set(data.columns))}")
    center = next((c for c in ["median", "mean", "load_mean"] if c in data.columns), None)
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
                axis.scatter(
                    observed_subset["Year"], observed_subset["observed_load_mg"] * factor,
                    facecolors="none", edgecolors=color, s=80, linewidths=1.8,
                    label=f"{treatment} observed",
                )
        scope = "imputed-inclusive" if suffix else "LOYO supported"
        axis.set_title(f"{analyte}: observed vs ML annual load ({scope})")
        axis.set_xlabel("Year")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(ncol=2)
        save_both(figure, figure_dir / f"annual_load_{slug(analyte)}_obs_vs_ml{suffix}")


def plot_event_unit_audit(output_dir: Path, figure_dir: Path) -> None:
    summary_path = output_dir / "event_volume_audit_summary.csv"
    training_path = output_dir / "event_volume_training_table.csv"
    if not summary_path.exists():
        warn("event_volume_audit_summary.csv is missing; skipping event-unit audit figure.")
        return
    summary = pd.read_csv(summary_path, dtype=str)
    values = dict(zip(summary["metric"], summary["value"]))
    labels_and_metrics = [
        ("All analyte rows", "original_analyte_rows"),
        ("Rows with volume", "rows_with_nonmissing_volume"),
        ("Observed volume events", "unique_event_volume_groups_with_nonmissing_volume"),
        ("Duplicate rows removed", "duplicated_volume_rows_removed"),
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
        if "_event_n_analyte_rows" in events.columns:
            rows_per_event = numeric(events["_event_n_analyte_rows"]).dropna()
            figure, axis = plt.subplots(figsize=(8, 4.8))
            axis.hist(rows_per_event, bins=min(30, max(5, int(rows_per_event.nunique()))), edgecolor="white")
            axis.set_title("Analyte-row multiplicity within each unique volume event")
            axis.set_xlabel("Analyte rows per event")
            axis.set_ylabel("Unique volume events")
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
    parser.add_argument("--fi_topk", type=int, default=25)
    parser.add_argument("--use_imputed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())
    output_dir = resolve_path(repo, args.output_dir, repo / "out" / "ml_catboost_conformal_loyo")
    figure_dir = resolve_path(repo, args.fig_dir, repo / "figs" / "ml_catboost_conformal_loyo")
    source_path = resolve_path(repo, args.source_data, repo / "out" / "wq_cleaned.csv")
    if not source_path.exists():
        raise FileNotFoundError(f"Required source data not found: {source_path}")
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading ML outputs from {output_dir}")
    print(f"[INFO] Writing figures to {figure_dir}")
    metrics = read_required(output_dir / "cv_metrics_by_year.csv")
    predictions = read_required(output_dir / "cv_predictions_samplelevel.csv")
    annual = read_required(output_dir / "annual_load_summary.csv")
    observed = observed_annual_loads(source_path)
    analytes = [item.strip() for item in args.analytes.split(",") if item.strip()] if args.analytes else None

    plot_cv_diagnostics(metrics, figure_dir)
    plot_parity(predictions, figure_dir)
    plot_coverage(predictions, figure_dir, nominal=1.0 - args.alpha)
    plot_feature_importance(output_dir / "feature_importance_logC.csv", figure_dir, "logC", args.fi_topk)
    plot_feature_importance(output_dir / "feature_importance_logV.csv", figure_dir, "logV", args.fi_topk)
    plot_annual_loads(annual, observed, figure_dir, analytes, "", args.units)
    if args.use_imputed:
        imputed_path = output_dir / "annual_load_summary_imputed.csv"
        if imputed_path.exists():
            plot_annual_loads(pd.read_csv(imputed_path), observed, figure_dir, analytes, "_imputed", args.units)
        else:
            warn("annual_load_summary_imputed.csv is missing; skipping imputed-inclusive plots.")
    plot_event_unit_audit(output_dir, figure_dir)
    print("[DONE] v2 event-level post-processing figures created.")


if __name__ == "__main__":
    main()
