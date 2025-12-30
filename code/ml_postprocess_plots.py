#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_postprocess_plots.py

Post-processing and plotting for the LOYO ML pipeline outputs:
  - Saves CV plot as JPG
  - Creates annual load plots (Observed vs ML) with uncertainty intervals
  - Creates parity (1:1) plots for concentration and volume (log scale)
  - Creates prediction-interval coverage diagnostics

Observed annual load uncertainty:
  - Nonparametric bootstrap (within Year × Treatment × Analyte) over event loads
    where event load = Result_mg_L × Volume_L
  - Produces 95% bootstrap intervals by default (configurable)

Expected inputs (produced by ml_catboost_conformal_loyo.py):
  out/ml_catboost_conformal_loyo/cv_metrics_by_year.csv
  out/ml_catboost_conformal_loyo/cv_predictions_samplelevel.csv
  out/ml_catboost_conformal_loyo/annual_load_summary.csv

Optional (if imputation was run):
  out/ml_catboost_conformal_loyo/wq_cleaned_ml_imputed.csv

Also requires original data:
  out/wq_with_stir_by_season.csv

Outputs are written to:
  figs/ml_catboost_conformal_loyo/

Run:
  conda activate wq_ml
  python code/ml_postprocess_plots.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "README.md").exists() and (cur / "out").exists() and (cur / "figs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find repo root containing README.md, out/, figs/. "
        "Run from within the repo or pass --repo explicitly."
    )


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("Date", "PlantDate", "HarvestDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_treatment(df: pd.DataFrame) -> pd.DataFrame:
    if "Treatment" not in df.columns and "System" in df.columns:
        df["Treatment"] = df["System"]
    return df


def coerce_bool(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        s = df[col].astype(str).str.strip().str.lower()
        df[col] = s.isin(["true", "t", "1", "yes", "y"])
    return df


def bootstrap_interval(values: np.ndarray,
                       n_boot: int = 2000,
                       alpha: float = 0.05,
                       rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """
    Bootstrap the sum of 'values' (event loads) using resampling with replacement.

    Returns:
      (point_estimate, low, high)

    If len(values) == 0: returns (nan, nan, nan)
    If len(values) == 1: returns (v, v, v)
    """
    if rng is None:
        rng = np.random.default_rng(123)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    if values.size == 1:
        v = float(values[0])
        return (v, v, v)

    point = float(values.sum())

    # Bootstrap sums
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_sums = values[idx].sum(axis=1)

    lo = float(np.quantile(boot_sums, alpha / 2))
    hi = float(np.quantile(boot_sums, 1 - alpha / 2))
    return (point, lo, hi)


def compute_observed_annual_loads(data_csv: Path,
                                  n_boot: int = 2000,
                                  alpha: float = 0.05,
                                  seed: int = 123) -> pd.DataFrame:
    """
    Observed annual load in mg:
      event load = Result_mg_L * Volume_L
      annual load = sum(event load) within Year x Treatment x Analyte

    Adds bootstrap uncertainty intervals by resampling events within each group.

    Notes:
    - Uses outflow concentration (Result_mg_L) and outflow volume (Volume).
    - Excludes NoRunoff==TRUE if present (consistent with ML training).
    """
    df = pd.read_csv(data_csv)
    df = parse_dates(df)
    df = ensure_treatment(df)
    df = coerce_bool(df, "NoRunoff")

    if "Year" not in df.columns:
        if "Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
        else:
            raise ValueError("Observed loads require Year or Date in the source data.")

    df["Result_mg_L"] = safe_numeric(df.get("Result_mg_L"))
    df["Volume"] = safe_numeric(df.get("Volume"))

    if "NoRunoff" in df.columns:
        df = df.loc[~df["NoRunoff"].fillna(False)].copy()

    df = df.dropna(subset=["Year", "Treatment", "Analyte", "Result_mg_L", "Volume"]).copy()

    df["Load_mg"] = df["Result_mg_L"] * df["Volume"]

    rng = np.random.default_rng(seed)

    records = []
    for (year, trt, an), g in df.groupby(["Year", "Treatment", "Analyte"]):
        vals = g["Load_mg"].to_numpy()
        point, lo, hi = bootstrap_interval(vals, n_boot=n_boot, alpha=alpha, rng=rng)
        records.append({
            "Year": int(year),
            "Treatment": trt,
            "Analyte": an,
            "obs_annual_load_mg": point,
            "obs_low_mg": lo,
            "obs_high_mg": hi,
            "n_events": int(len(vals)),
        })

    return pd.DataFrame.from_records(records)


def _sample_log_uniform(lo_log: np.ndarray,
                        hi_log: np.ndarray,
                        draws: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly within [lo_log, hi_log] in log space. Returns shape (n, draws)."""
    lo = np.asarray(lo_log, dtype=float)
    hi = np.asarray(hi_log, dtype=float)
    if lo.ndim != 1 or hi.ndim != 1:
        raise ValueError("lo_log/hi_log must be 1D arrays.")
    if lo.shape != hi.shape:
        raise ValueError("lo_log and hi_log must have same length.")
    u = rng.random((lo.size, draws))
    return lo[:, None] + u * (hi[:, None] - lo[:, None])


def compute_ml_imputed_annual_loads(imputed_csv: Path,
                                   draws: int = 2000,
                                   alpha: float = 0.05,
                                   seed: int = 123) -> pd.DataFrame:
    """
    Compute ML annual loads including imputed rows (mg), using the imputed dataset
    written by ml_catboost_conformal_loyo.py.

    Point estimate:
      event load = Result_mg_L_filled * Volume_filled
      annual load = sum(event load) within Year x Treatment x Analyte

    Uncertainty propagation (optional):
      If draws > 0, draws are generated per row using prediction-interval bounds
      for imputed values. Observed (non-imputed) values are treated as fixed (degenerate).

    Returns:
      DataFrame with columns:
        Year, Treatment, Analyte, mean, median, low, high, n_events, n_draws
    """
    df = pd.read_csv(imputed_csv, na_values=["NA", "NaN", "nan", ""], keep_default_na=True)
    df = parse_dates(df)
    df = ensure_treatment(df)
    df = coerce_bool(df, "NoRunoff")

    if "Year" not in df.columns:
        if "Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
        else:
            raise ValueError("Imputed-loads require Year or Date in the imputed dataset.")

    for c in ["Result_mg_L", "Volume", "Result_mg_L_filled", "Volume_filled"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column in imputed dataset: {c}")
        df[c] = safe_numeric(df[c])

    if "NoRunoff" in df.columns:
        df = df.loc[~df["NoRunoff"].fillna(False)].copy()

    df = df.dropna(subset=["Year", "Treatment", "Analyte", "Result_mg_L_filled", "Volume_filled"]).copy()
    df["Year"] = safe_numeric(df["Year"]).astype(int)

    missC = df["Result_mg_L"].isna()
    missV = df["Volume"].isna()

    df["Load_point_mg"] = df["Result_mg_L_filled"] * df["Volume_filled"]

    if int(draws) <= 0:
        out = (df.groupby(["Year", "Treatment", "Analyte"], as_index=False)
                 .agg(mean=("Load_point_mg", "sum"),
                      median=("Load_point_mg", "sum"),
                      low=("Load_point_mg", "sum"),
                      high=("Load_point_mg", "sum"),
                      n_events=("Load_point_mg", "size")))
        out["n_draws"] = 0
        return out

    rng = np.random.default_rng(seed)
    D = int(draws)
    group_sums: dict[tuple[int, str, str], np.ndarray] = {}
    group_counts: dict[tuple[int, str, str], int] = {}

    def _get(key):
        if key not in group_sums:
            group_sums[key] = np.zeros(D, dtype=float)
            group_counts[key] = 0
        return group_sums[key]

    # Observed-only rows: add constant to every draw
    obs_mask = ~(missC | missV)
    if obs_mask.any():
        for (year, trt, an), g in df.loc[obs_mask].groupby(["Year", "Treatment", "Analyte"]):
            key = (int(year), str(trt), str(an))
            _get(key)[:] += float(g["Load_point_mg"].sum())
            group_counts[key] += int(len(g))

    # Rows with at least one imputed component
    need_mask = (missC | missV)
    if need_mask.any():
        sub = df.loc[need_mask].copy()

        # PI columns may be absent; create as NaN
        for c in ["Result_mg_L_pi_low", "Result_mg_L_pi_high", "Volume_pi_low", "Volume_pi_high"]:
            if c not in sub.columns:
                sub[c] = np.nan
            sub[c] = safe_numeric(sub[c])

        for (year, trt, an), g in sub.groupby(["Year", "Treatment", "Analyte"]):
            key = (int(year), str(trt), str(an))
            vec = _get(key)
            group_counts[key] += int(len(g))

            C_filled = g["Result_mg_L_filled"].to_numpy(dtype=float)
            V_filled = g["Volume_filled"].to_numpy(dtype=float)

            # Default: degenerate draws at filled value
            C_draw = np.repeat(C_filled[:, None], D, axis=1)
            V_draw = np.repeat(V_filled[:, None], D, axis=1)

            # If concentration missing, sample from PI when bounds exist
            if g["Result_mg_L"].isna().any():
                loC = g["Result_mg_L_pi_low"].to_numpy(dtype=float)
                hiC = g["Result_mg_L_pi_high"].to_numpy(dtype=float)
                ok = np.isfinite(loC) & np.isfinite(hiC) & (loC >= 0) & (hiC >= loC)
                if ok.any():
                    C_draw_log = _sample_log_uniform(np.log1p(loC[ok]), np.log1p(hiC[ok]), D, rng)
                    C_draw[ok, :] = np.expm1(C_draw_log)

            # If volume missing, sample from PI when bounds exist
            if g["Volume"].isna().any():
                loV = g["Volume_pi_low"].to_numpy(dtype=float)
                hiV = g["Volume_pi_high"].to_numpy(dtype=float)
                ok = np.isfinite(loV) & np.isfinite(hiV) & (loV >= 0) & (hiV >= loV)
                if ok.any():
                    V_draw_log = _sample_log_uniform(np.log1p(loV[ok]), np.log1p(hiV[ok]), D, rng)
                    V_draw[ok, :] = np.expm1(V_draw_log)

            vec += (C_draw * V_draw).sum(axis=0)

    q_lo = float(alpha) / 2.0
    q_hi = 1.0 - float(alpha) / 2.0

    rows = []
    for (year, trt, an), vec in group_sums.items():
        rows.append({
            "Year": int(year),
            "Treatment": trt,
            "Analyte": an,
            "mean": float(np.mean(vec)),
            "median": float(np.median(vec)),
            "low": float(np.quantile(vec, q_lo)),
            "high": float(np.quantile(vec, q_hi)),
            "n_events": int(group_counts.get((year, trt, an), 0)),
            "n_draws": int(D),
        })
    return pd.DataFrame.from_records(rows)



def plot_feature_importance_csv(fi_csv: Path,
                                figpath: Path,
                                title: str,
                                top_k: int = 25) -> None:
    """
    Plot feature importance from the CSV written by ml_catboost_conformal_loyo.py.

    Expected columns:
      - feature
      - importance_mean
    Optional:
      - importance_sd
    """
    if not fi_csv.exists():
        print(f"[WARN] Missing {fi_csv}; skipping feature-importance plot.", file=sys.stderr)
        return

    fi = pd.read_csv(fi_csv)
    if fi.empty or "feature" not in fi.columns:
        print(f"[WARN] {fi_csv} is empty or malformed; skipping feature-importance plot.", file=sys.stderr)
        return

    if "importance_mean" in fi.columns:
        fi = fi.sort_values("importance_mean", ascending=False).head(int(top_k)).copy()
        vals = safe_numeric(fi["importance_mean"]).to_numpy()
        xlabel = "Mean importance"
    elif "importance" in fi.columns:
        fi = fi.sort_values("importance", ascending=False).head(int(top_k)).copy()
        vals = safe_numeric(fi["importance"]).to_numpy()
        xlabel = "Importance"
    else:
        print(f"[WARN] {fi_csv} missing importance column; skipping feature-importance plot.", file=sys.stderr)
        return

    features = fi["feature"].astype(str).to_numpy()[::-1]
    vals = vals[::-1]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(features) + 1)))
    ax.barh(features, vals)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figpath, dpi=220)
    plt.close(fig)


def save_cv_rmse_plots(outdir: Path, figdir: Path) -> None:
    metrics_path = outdir / "cv_metrics_by_year.csv"
    if not metrics_path.exists():
        print(f"[WARN] Missing {metrics_path}; skipping CV plot.", file=sys.stderr)
        return

    m = pd.read_csv(metrics_path)
    if m.empty:
        print("[WARN] cv_metrics_by_year.csv is empty; skipping CV plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for target in ["logC", "logV"]:
        d = m.loc[m["Target"] == target].sort_values("Year_Test")
        if d.empty:
            continue
        ax.plot(d["Year_Test"], d["RMSE"], marker="o", label=target)

    ax.set_title("LOYO CV RMSE by Held-out Year (log scale)")
    ax.set_xlabel("Held-out Year")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()

    figpath_jpg = figdir / "cv_rmse_by_year.jpg"
    fig.savefig(figpath_jpg, dpi=220)
    plt.close(fig)


def annual_load_plots(ml_summary: pd.DataFrame,
                      obs: pd.DataFrame,
                      figdir: Path,
                      analytes: list[str] | None = None,
                      units: str = "g",
                      out_suffix: str = "",
                      title_suffix: str = "") -> None:
    """
    One figure per analyte: Observed vs ML annual load by treatment.

    ML: line with point markers, plus vertical interval (low/high) per year.
    Obs: open circles with bootstrap interval error bars (low/high).

    ML summary expected columns:
      Year, Treatment, Analyte, mean, median, low, high

    Observed expected:
      Year, Treatment, Analyte, obs_annual_load_mg, obs_low_mg, obs_high_mg
    """
    df = ml_summary.copy()
    df["Year"] = safe_numeric(df["Year"]).astype(int)

    # Convert mg -> units
    factor = 1.0
    ylabel = "Annual load (mg)"
    if units.lower() == "g":
        factor = 1e-3
        ylabel = "Annual load (g)"
    elif units.lower() == "kg":
        factor = 1e-6
        ylabel = "Annual load (kg)"

    for c in ["mean", "median", "low", "high"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c]) * factor

    obs2 = obs.copy()
    obs2["Year"] = safe_numeric(obs2["Year"]).astype(int)
    for c in ["obs_annual_load_mg", "obs_low_mg", "obs_high_mg"]:
        if c in obs2.columns:
            obs2[c] = safe_numeric(obs2[c]) * factor

    # analytes to plot
    available = sorted(df["Analyte"].dropna().unique().tolist())
    use_analytes = available if analytes is None else [a for a in analytes if a in available]

    treatments = ["CT", "MT", "ST"]
    trts_avail = df["Treatment"].dropna().unique().tolist()
    if any(t not in trts_avail for t in treatments):
        treatments = sorted(trts_avail)

    for an in use_analytes:
        d_an = df.loc[df["Analyte"] == an].copy()
        if d_an.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.8))

        for trt in treatments:
            dd = d_an.loc[d_an["Treatment"] == trt].sort_values("Year")
            if dd.empty:
                continue

            center = dd["median"] if "median" in dd.columns else dd["mean"]

            ax.plot(dd["Year"], center, marker="o", linewidth=2, label=f"{trt} modeled")

            # ML uncertainty
            if "low" in dd.columns and "high" in dd.columns:
                ax.vlines(dd["Year"], dd["low"], dd["high"], linewidth=4, alpha=0.30)

            # observed point + bootstrap interval
            oo = obs2.loc[(obs2["Analyte"] == an) & (obs2["Treatment"] == trt)].sort_values("Year")
            if not oo.empty:
                # open circles
                ax.scatter(
                    oo["Year"], oo["obs_annual_load_mg"],
                    facecolors="none",
                    edgecolors=ax.lines[-1].get_color(),
                    s=120, linewidths=2, label=f"{trt} observed"
                )
                # error bars (vertical interval)
                if {"obs_low_mg", "obs_high_mg"}.issubset(oo.columns):
                    ax.vlines(
                        oo["Year"], oo["obs_low_mg"], oo["obs_high_mg"],
                        linewidth=3, alpha=0.45, colors=ax.lines[-1].get_color()
                    )

        ax.set_title(f"{an}: annual load by treatment (Observed vs ML{title_suffix})")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(ncol=2, frameon=True)

        fig.tight_layout()
        fig.savefig(figdir / f"annual_load_{an.lower()}_obs_vs_ml{out_suffix}.jpg", dpi=220)
        plt.close(fig)


def parity_plots(preds: pd.DataFrame, figdir: Path) -> None:
    required = {"Target", "y_true", "y_pred"}
    if not required.issubset(set(preds.columns)):
        print("[WARN] Missing required columns for parity plots; skipping.", file=sys.stderr)
        return

    for target, fname, xlabel in [
        ("Result_mg_L", "parity_logC.jpg", "Observed log1p(Result_mg_L)"),
        ("Volume_L", "parity_logV.jpg", "Observed log1p(Volume_L)"),
    ]:
        d = preds.loc[preds["Target"] == target].copy()
        if d.empty:
            continue

        y_true = safe_numeric(d["y_true"])
        y_pred = safe_numeric(d["y_pred"])
        mask = y_true.notna() & y_pred.notna() & (y_true >= 0) & (y_pred >= 0)
        y_true = np.log1p(y_true[mask].to_numpy())
        y_pred = np.log1p(y_pred[mask].to_numpy())

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(y_true, y_pred, s=12, alpha=0.25)

        mn = float(np.nanmin([y_true.min(), y_pred.min()]))
        mx = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax.plot([mn, mx], [mn, mx], linewidth=2)

        ax.set_title(f"Parity plot: {target} (log scale)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted log1p")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=220)
        plt.close(fig)


def coverage_plot(preds: pd.DataFrame, figdir: Path) -> None:
    required = {"Target", "y_true", "pi_low", "pi_high"}
    if not required.issubset(set(preds.columns)):
        print("[WARN] Missing PI columns for coverage plot; skipping.", file=sys.stderr)
        return

    df = preds.copy()
    df["y_true"] = safe_numeric(df["y_true"])
    df["pi_low"] = safe_numeric(df["pi_low"])
    df["pi_high"] = safe_numeric(df["pi_high"])

    df = df.dropna(subset=["y_true", "pi_low", "pi_high", "Target"]).copy()
    df["covered"] = (df["y_true"] >= df["pi_low"]) & (df["y_true"] <= df["pi_high"])

    targets = df["Target"].unique().tolist()

    overall = (df.groupby("Target", as_index=False)
                 .agg(n=("covered", "size"), coverage=("covered", "mean")))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(overall["Target"], overall["coverage"])
    ax.set_ylim(0, 1)
    ax.set_title("Empirical PI coverage (overall)")
    ax.set_ylabel("Coverage fraction")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "pi_coverage_overall.jpg", dpi=220)
    plt.close(fig)

    if "Year" in df.columns:
        df["Year"] = safe_numeric(df["Year"])
        per_year = (df.dropna(subset=["Year"])
                      .groupby(["Target", "Year"], as_index=False)
                      .agg(n=("covered", "size"), coverage=("covered", "mean")))
        if not per_year.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            for t in targets:
                dd = per_year.loc[per_year["Target"] == t].sort_values("Year")
                if dd.empty:
                    continue
                ax.plot(dd["Year"], dd["coverage"], marker="o", label=t)
            ax.set_ylim(0, 1)
            ax.set_title("Empirical PI coverage by held-out year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Coverage fraction")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figdir / "pi_coverage_by_year.jpg", dpi=220)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default=None, help="Repo root. If omitted, auto-detected.")
    ap.add_argument("--units", type=str, default="g", choices=["mg", "g", "kg"], help="Annual load plotting units.")
    ap.add_argument("--analytes", type=str, default=None,
                    help="Optional comma-separated analytes to plot (default: all in ML summary).")
    ap.add_argument("--obs_boot", type=int, default=2000, help="Bootstrap replicates for observed annual loads.")
    ap.add_argument("--obs_alpha", type=float, default=0.05, help="Alpha for observed bootstrap intervals (0.05 -> 95%).")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap.")

    ap.add_argument(
    "--use_imputed",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "If true, and an imputed dataset exists, also create a second set of annual-load plots "
        "that include imputed rows (suffix: _imputed). If the imputed file is missing, the script "
        "falls back to CV-only plots."
    )
    )
    ap.add_argument(
    "--imputed_csv",
    type=str,
    default=None,
    help=(
        "Path to the imputed dataset written by ml_catboost_conformal_loyo.py "
        "(default: <repo>/out/<out_subdir>/wq_cleaned_ml_imputed.csv)."
    )
    )
    ap.add_argument(
    "--imputed_draws",
    type=int,
    default=2000,
    help=(
        "Monte Carlo draws used to propagate PI uncertainty for the imputed-inclusive annual-load plots. "
        "Set to 0 to disable and plot deterministic filled annual sums."
    )
    )
    ap.add_argument(
    "--ml_alpha",
    type=float,
    default=0.05,
    help="Alpha used when summarizing the imputed-inclusive annual-load draws (0.05 -> 95%)."
    )

    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())

    outdir = repo / "out" / "ml_catboost_conformal_loyo"
    figdir = repo / "figs" / "ml_catboost_conformal_loyo"
    figdir.mkdir(parents=True, exist_ok=True)

    imputed_csv = Path(args.imputed_csv).resolve() if args.imputed_csv else (outdir / "wq_cleaned_ml_imputed.csv")


    data_csv = repo / "out" / "wq_with_stir_by_season.csv"
    ml_summary_csv = outdir / "annual_load_summary.csv"
    preds_csv = outdir / "cv_predictions_samplelevel.csv"

    if not data_csv.exists():
        raise FileNotFoundError(f"Missing source data: {data_csv}")
    if not ml_summary_csv.exists():
        raise FileNotFoundError(f"Missing ML annual load summary: {ml_summary_csv}")
    if not preds_csv.exists():
        raise FileNotFoundError(f"Missing ML CV predictions: {preds_csv}")

    print(f"[INFO] Repo: {repo}")
    print(f"[INFO] Reading ML outputs from: {outdir}")
    print(f"[INFO] Writing figures to: {figdir}")
    print(f"[INFO] Observed bootstrap: B={args.obs_boot}, alpha={args.obs_alpha}")

    save_cv_rmse_plots(outdir, figdir)

    obs = compute_observed_annual_loads(
        data_csv,
        n_boot=args.obs_boot,
        alpha=args.obs_alpha,
        seed=args.seed
    )

    ml = pd.read_csv(ml_summary_csv)

    analytes = None
    if args.analytes:
        analytes = [a.strip() for a in args.analytes.split(",") if a.strip()]

    annual_load_plots(ml, obs, figdir, analytes=analytes, units=args.units)

    # Parity + PI coverage diagnostics (CV predictions)
    preds = pd.read_csv(preds_csv)
    parity_plots(preds, figdir)
    coverage_plot(preds, figdir)

    # Feature importance plots (if CSVs exist)
    plot_feature_importance_csv(
        outdir / "feature_importance_logC.csv",
        figdir / "feature_importance_logC.jpg",
        title="Feature importance: logC model (mean across LOYO folds)",
        top_k=25,
    )
    plot_feature_importance_csv(
        outdir / "feature_importance_logV.csv",
        figdir / "feature_importance_logV.jpg",
        title="Feature importance: logV model (mean across LOYO folds)",
        top_k=25,
    )



   


    # Optional: imputed-inclusive annual-load plots (keeps CV-valid plots intact)
    if bool(args.use_imputed):
        if imputed_csv.exists():
            try:
                ml_imp = compute_ml_imputed_annual_loads(
                    imputed_csv,
                    draws=int(args.imputed_draws),
                    alpha=float(args.ml_alpha),
                    seed=int(args.seed) + 77,
                )
                annual_load_plots(
                    ml_imp,
                    obs,
                    figdir,
                    analytes=analytes,
                    units=args.units,
                    out_suffix="_imputed",
                    title_suffix="; imputed-inclusive",
                )
                ml_imp.to_csv(outdir / "annual_load_summary_imputed.csv", index=False)
                print(f"[INFO] Wrote imputed-inclusive annual summary: {outdir / 'annual_load_summary_imputed.csv'}")
            except Exception as e:
                print(f"[WARN] Failed to create imputed-inclusive plots: {e}", file=sys.stderr)
        else:
            print(f"[INFO] No imputed dataset found at: {imputed_csv}. Skipping imputed-inclusive plots.")

    print("[DONE] Post-processing figures created.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] KeyboardInterrupt received.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
