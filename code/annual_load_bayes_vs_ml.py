#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""annual_load_bayes_vs_ml.py

Creates annual load comparison plots faceted by Treatment (CT/MT/ST) and
computes quantitative metrics summarizing Bayes vs ML performance against
Observed annual loads.

Adds (optional) CRPS (Continuous Ranked Probability Score) using per-draw
annual load tables from Bayes and ML, if present.

Plots:
  - Bayes modeled: solid line + filled markers + colored CI band (if low/high exist)
  - ML modeled   : dashed line + filled markers + colored CI band (if low/high exist)
  - Observed     : hollow markers + vertical error bars (if low/high exist)

Key implementation rule:
  Observed values are ONLY plotted from rows explicitly labeled as observed
  in the Bayes summary file. We do NOT infer "observed" from missing CI columns.

Inputs (repo-root relative defaults):
  Bayes summary (contains modeled + observed rows):
    out/annual_load_summary_bayes_plus_observed_v1p6.csv
  ML summary (imputed-inclusive):
    out/ml_catboost_conformal_loyo/annual_load_summary_imputed.csv

Optional draws for CRPS:
  Bayes draws:
    out/annual_load_draws_bayes_v1p6.csv
  ML draws:
    out/ml_catboost_conformal_loyo/annual_load_draws.csv

Outputs:
  Figures:
    figs/annual_bayes_vs_ml_faceted_jpg/annual_load_<analyte>_bayes_vs_ml_faceted.jpg

  Metrics:
    out/bayes_vs_ml_metrics/
      metrics_by_analyte_treatment.csv
      metrics_by_analyte_overall.csv
      metrics_overall.csv

"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "README.md").exists() and (cur / "out").exists() and (cur / "figs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(
        "Could not find repo root containing README.md, out/, figs/. "
        "Run from within the repo or pass --repo explicitly."
    )


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Year" not in df.columns:
        for c in df.columns:
            if c.lower() == "year":
                df = df.rename(columns={c: "Year"})
                break
    if "Year" in df.columns:
        df["Year"] = safe_numeric(df["Year"]).astype("Int64")

    if "Treatment" not in df.columns:
        for c in df.columns:
            if c.lower() in {"treatment", "system"}:
                df = df.rename(columns={c: "Treatment"})
                break
    if "Treatment" in df.columns:
        df["Treatment"] = df["Treatment"].astype(str).str.upper().str.strip()

    if "Analyte" not in df.columns:
        for c in df.columns:
            if c.lower() == "analyte":
                df = df.rename(columns={c: "Analyte"})
                break
    if "Analyte" in df.columns:
        df["Analyte"] = df["Analyte"].astype(str).str.strip()

    return df


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def standardize_summary(
    df: pd.DataFrame,
    source_name: str,
    series_label: str,
    center_candidates: List[str],
    low_candidates: List[str],
    high_candidates: List[str],
) -> pd.DataFrame:
    df = normalize_cols(df)
    need = {"Year", "Treatment", "Analyte"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{source_name}: missing required columns: {sorted(missing)}")

    c_center = pick_first_existing(df, center_candidates)
    c_low = pick_first_existing(df, low_candidates)
    c_high = pick_first_existing(df, high_candidates)

    if c_center is None:
        raise ValueError(
            f"{source_name}: could not find a center column. Tried: {center_candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.loc[:, ["Year", "Treatment", "Analyte"]].copy()
    out["source"] = source_name
    out["series"] = series_label

    out["center_mg"] = safe_numeric(df[c_center])
    out["low_mg"] = safe_numeric(df[c_low]) if c_low is not None else np.nan
    out["high_mg"] = safe_numeric(df[c_high]) if c_high is not None else np.nan

    out = out.dropna(subset=["Year", "Treatment", "Analyte", "center_mg"]).copy()
    out["Year"] = out["Year"].astype(int)

    return out


def mg_to_units(x: pd.Series, units: str) -> pd.Series:
    u = units.lower().strip()
    if u == "mg":
        return x
    if u == "g":
        return x / 1e3
    if u == "kg":
        return x / 1e6
    raise ValueError("units must be one of: mg, g, kg")


def _analyte_key(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


ANALYTE_CANON: Dict[str, str] = {
    "op": "OP",
    "orthop": "OP",
    "orthophosphate": "OP",
    "po4p": "OP",
    "ortho": "OP",
    "nh4": "NH4",
    "ammonium": "NH4",
    "ammoniumnh4": "NH4",
    "no3": "NO3",
    "nitrate": "NO3",
    "no2": "NO2",
    "nitrite": "NO2",
    "nox": "NOx",
    "nitratenitrite": "NOx",
    "nitratenitriteasno3": "NOx",
    "nitrateplusnitrite": "NOx",
    "nitrateandnitrite": "NOx",
    "tn": "TN",
    "totaln": "TN",
    "totalnitrogen": "TN",
    "tp": "TP",
    "totalp": "TP",
    "totalphosphorus": "TP",
    "se": "Se",
    "selenium": "Se",
    "tss": "TSS",
    "tkn": "TKN",
    "tsp": "TSP",
    "tds": "TDS",
    "npoc": "NPOC",
    "icp": "ICP",
}


def canonicalize_analytes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Analyte"] = df["Analyte"].astype(str).str.strip()
    keys = df["Analyte"].map(_analyte_key)
    df["Analyte"] = keys.map(ANALYTE_CANON).fillna(df["Analyte"])
    return df


def detect_series_column(df: pd.DataFrame) -> str:
    candidates = ["series", "type", "kind", "source", "data_type", "model"]
    c = pick_first_existing(df, candidates)
    if c is None:
        raise ValueError(
            "Bayes file must contain an explicit series label column (e.g., 'series' or 'source') "
            "to distinguish observed vs modeled rows. None found. "
            f"Available columns: {list(df.columns)}"
        )
    return c


def split_bayes_observed_modeled(bayes_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_cols(bayes_raw)
    series_col = detect_series_column(df)

    s = df[series_col].astype(str).str.strip().str.lower()

    is_obs = s.str.contains(r"\bobs\b") | s.str.contains("observed")
    is_bayes = (
        s.str.contains("bayes")
        | s.str.contains("posterior")
        | s.str.contains("modeled")
        | s.str.contains("modelled")
        | s.str.contains("fit")
        | s.str.contains("pred")
        | s.str.contains("estimate")
    )

    if is_obs.sum() == 0:
        raise ValueError(
            f"Bayes file has series column '{series_col}' but no rows look like observed (obs/observed). "
            f"Unique labels: {sorted(df[series_col].astype(str).unique())}"
        )
    if is_bayes.sum() == 0:
        raise ValueError(
            f"Bayes file has series column '{series_col}' but no rows look like Bayes modeled. "
            f"Unique labels: {sorted(df[series_col].astype(str).unique())}"
        )

    df_obs = df.loc[is_obs].copy()
    df_bayes = df.loc[is_bayes & ~is_obs].copy()

    bayes_modeled = standardize_summary(
        df_bayes,
        source_name="Bayes",
        series_label="Bayes modeled",
        center_candidates=["load_mean", "median", "mean", "center", "estimate", "mu", "annual_load_mg"],
        low_candidates=["load_low", "low", "lo", "lower", "lwr", "p2_5", "q0.025", "hdi_low", "ci_low"],
        high_candidates=["load_high", "high", "hi", "upper", "upr", "p97_5", "q0.975", "hdi_high", "ci_high"],
    )

    observed = standardize_summary(
        df_obs,
        source_name="Observed",
        series_label="Observed",
        center_candidates=["load_mean", "mean", "median", "center"],
        low_candidates=["load_low", "low", "lo", "lower", "lwr"],
        high_candidates=["load_high", "high", "hi", "upper", "upr"],
    )

    return bayes_modeled, observed


def plot_analyte_faceted(
    analyte: str,
    bayes: pd.DataFrame,
    ml: pd.DataFrame,
    obs: pd.DataFrame,
    out_jpg: Path,
    units: str = "g",
    treatments: List[str] = ["CT", "MT", "ST"],
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15.5, 5.3), sharey=True)
    fig.suptitle(f"{analyte}: annual load comparison (Observed vs Bayes vs ML; imputed-inclusive)")

    bayes_color = "C0"
    ml_color = "C1"

    def add_band(ax, x, ylo, yhi, color, alpha):
        if len(x) == 0:
            return
        ax.fill_between(x, ylo, yhi, color=color, alpha=alpha, linewidth=0, label="_nolegend_")

    def plot_modeled(ax, d: pd.DataFrame, label: str, linestyle: str, color: str):
        if d.empty:
            return
        d = d.sort_values("Year")
        x = d["Year"].to_numpy()
        y = mg_to_units(d["center_mg"], units).to_numpy()

        if np.isfinite(d["low_mg"]).any() and np.isfinite(d["high_mg"]).any():
            ylo = mg_to_units(d["low_mg"], units).to_numpy()
            yhi = mg_to_units(d["high_mg"], units).to_numpy()
            add_band(ax, x, ylo, yhi, color=color, alpha=0.18)

        ax.plot(
            x, y,
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=color,
            label=label,
            zorder=6,
        )

    def plot_observed(ax, d: pd.DataFrame):
        if d.empty:
            return
        d = d.sort_values("Year")
        x = d["Year"].to_numpy()
        y = mg_to_units(d["center_mg"], units).to_numpy()

        if np.isfinite(d["low_mg"]).any() and np.isfinite(d["high_mg"]).any():
            yerr_lower = y - mg_to_units(d["low_mg"], units).to_numpy()
            yerr_upper = mg_to_units(d["high_mg"], units).to_numpy() - y
            yerr = np.vstack([yerr_lower, yerr_upper])

            ax.errorbar(
                x, y,
                yerr=yerr,
                fmt="none",
                ecolor="0.6",
                elinewidth=2,
                capsize=3,
                zorder=6,
                label="_nolegend_",
            )

        ax.scatter(
            x, y,
            facecolors="none",
            edgecolors="0.25",
            s=110,
            linewidths=2,
            label="Observed",
            zorder=7,
        )

    for i, trt in enumerate(treatments):
        ax = axes[i]
        ax.set_title(trt)
        ax.set_xlabel("Year")
        if i == 0:
            ax.set_ylabel(f"Annual load ({units})")

        b = bayes[(bayes["Analyte"] == analyte) & (bayes["Treatment"] == trt)]
        m = ml[(ml["Analyte"] == analyte) & (ml["Treatment"] == trt)]
        o = obs[(obs["Analyte"] == analyte) & (obs["Treatment"] == trt)]

        plot_observed(ax, o)
        plot_modeled(ax, b, "Bayes modeled", linestyle="-", color=bayes_color)
        plot_modeled(ax, m, "ML modeled", linestyle="--", color=ml_color)

        ax.grid(True, axis="y", alpha=0.30)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels and ll != "_nolegend_":
                handles.append(hh)
                labels.append(ll)

    fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.tight_layout(rect=[0, 0, 0.96, 0.93])
    fig.savefig(out_jpg, dpi=220, format="jpg")
    plt.close(fig)


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(err ** 2))) if err.size else float("nan")


def _mae(err: np.ndarray) -> float:
    return float(np.nanmean(np.abs(err))) if err.size else float("nan")


def _nrmse_mean(rmse: float, y: np.ndarray) -> float:
    denom = float(np.nanmean(np.abs(y)))
    if not np.isfinite(denom) or denom == 0.0 or not np.isfinite(rmse):
        return float("nan")
    return float(rmse / denom)


def _nrmse_range(rmse: float, y: np.ndarray) -> float:
    rng = float(np.nanmax(y) - np.nanmin(y)) if y.size else float("nan")
    if not np.isfinite(rng) or rng == 0.0 or not np.isfinite(rmse):
        return float("nan")
    return float(rmse / rng)


def compute_metrics_point_interval(
    observed: pd.DataFrame,
    pred: pd.DataFrame,
    method_label: str,
    interval_prob: float | None = None,
) -> pd.DataFrame:
    o = observed.loc[:, ["Year", "Treatment", "Analyte", "center_mg"]].rename(columns={"center_mg": "y_obs"})
    p = pred.loc[:, ["Year", "Treatment", "Analyte", "center_mg", "low_mg", "high_mg"]].rename(
        columns={"center_mg": "y_hat", "low_mg": "y_low", "high_mg": "y_high"}
    )

    m = o.merge(p, on=["Year", "Treatment", "Analyte"], how="inner")
    out_rows = []
    for (an, trt), g in m.groupby(["Analyte", "Treatment"], dropna=False):
        y = g["y_obs"].to_numpy(dtype=float)
        yhat = g["y_hat"].to_numpy(dtype=float)
        err = yhat - y

        rmse = _rmse(err)
        mae = _mae(err)

        has_int = np.isfinite(g["y_low"]).any() and np.isfinite(g["y_high"]).any()
        if has_int:
            low = g["y_low"].to_numpy(dtype=float)
            high = g["y_high"].to_numpy(dtype=float)
            cover = float(np.nanmean((y >= low) & (y <= high)))
            width = float(np.nanmean(high - low))
        else:
            cover = float("nan")
            width = float("nan")

        out_rows.append({
            "Analyte": an,
            "Treatment": trt,
            "method": method_label,
            "n": int(g.shape[0]),
            "MAE": mae,
            "RMSE": rmse,
            "NRMSE_mean": _nrmse_mean(rmse, y),
            "NRMSE_range": _nrmse_range(rmse, y),
            "Coverage": cover,
            "MeanWidth": width,
            "IntervalProb": interval_prob,
            "CRPS": float("nan"),
            "CRPS_norm_mean": float("nan"),
        })

    return pd.DataFrame(out_rows)


def _pairwise_abs_mean_fast(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    xs = np.sort(x)
    k = np.arange(1, n + 1, dtype=float)
    sum_ij = float(np.sum((2.0 * k - n - 1.0) * xs))
    return (2.0 * sum_ij) / (n * n)


def crps_from_draws(draws: np.ndarray, y_obs: float) -> float:
    x = draws[np.isfinite(draws)]
    if x.size == 0 or not np.isfinite(y_obs):
        return float("nan")
    term1 = float(np.mean(np.abs(x - y_obs)))
    term2 = 0.5 * _pairwise_abs_mean_fast(x)
    return term1 - term2


def standardize_draws(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, str]:
    df = normalize_cols(df)

    draw_id_col = pick_first_existing(df, ["draw", "Draw", "draw_id", "iter", "iteration", "sample", "s", "mcmc_draw"])
    if draw_id_col is None:
        draw_id_col = "__draw_id__"
        df[draw_id_col] = np.arange(len(df), dtype=int)

    value_candidates = [
        # mg
        "load_mg", "annual_load_mg", "annualLoad_mg", "AnnualLoad_mg", "annualload_mg",
        "load_draw_mg", "load_draw_mgg", "y_mg", "draw_value_mg",
        # g
        "load_g", "annual_load_g", "annualLoad_g", "AnnualLoad_g", "load_draw_g", "draw_value_g",
        # kg
        "load_kg", "annual_load_kg", "annualLoad_kg", "AnnualLoad_kg", "load_draw_kg", "draw_value_kg",
        # generic
        "y", "load", "annual_load", "annualLoad", "AnnualLoad", "load_draw", "y_draw", "pred", "prediction"
    ]
    value_col = pick_first_existing(df, value_candidates)
    if value_col is None:
        raise ValueError(
            f"{label} draws: could not find a draw value column. Tried: {value_candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.loc[:, ["Year", "Treatment", "Analyte", draw_id_col, value_col]].copy()
    out = out.rename(columns={draw_id_col: "draw_id", value_col: "draw_value_mg"})
    out["draw_value_mg"] = safe_numeric(out["draw_value_mg"])

    # Unit normalization: allow draw files that store load in g or kg.
    # We convert everything to mg internally for consistency with summaries.
    vcol_lower = str(value_col).lower()
    if vcol_lower.endswith("_g") or vcol_lower == "load_g" or vcol_lower == "annual_load_g":
        out["draw_value_mg"] = out["draw_value_mg"] * 1e3
    elif vcol_lower.endswith("_kg") or vcol_lower == "load_kg" or vcol_lower == "annual_load_kg":
        out["draw_value_mg"] = out["draw_value_mg"] * 1e6
    out = out.dropna(subset=["Year", "Treatment", "Analyte", "draw_value_mg"]).copy()
    out["Year"] = out["Year"].astype(int)
    out["Treatment"] = out["Treatment"].astype(str).str.upper().str.strip()
    out["Analyte"] = out["Analyte"].astype(str).str.strip()
    out = canonicalize_analytes(out)

    return out, value_col


def compute_crps_table(
    observed: pd.DataFrame,
    draws_df: pd.DataFrame,
    method_label: str,
    max_draws: Optional[int],
    seed: int,
    progress_prefix: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    obs_key = observed.loc[:, ["Year", "Treatment", "Analyte", "center_mg"]].rename(columns={"center_mg": "y_obs"})
    obs_key["Year"] = obs_key["Year"].astype(int)
    obs_key["Treatment"] = obs_key["Treatment"].astype(str).str.upper().str.strip()
    obs_key["Analyte"] = obs_key["Analyte"].astype(str).str.strip()
    obs_key = canonicalize_analytes(obs_key)

    d = draws_df.merge(obs_key, on=["Year", "Treatment", "Analyte"], how="inner")
    if d.empty:
        return pd.DataFrame(columns=["Analyte", "Treatment", "method", "n", "CRPS"])

    group_cols = ["Year", "Treatment", "Analyte"]
    groups = list(d.groupby(group_cols, sort=True))
    total = len(groups)
    print(f"{progress_prefix} CRPS: scoring {total} (Year×Treatment×Analyte) groups for {method_label}...")

    rows = []
    for idx, ((yr, trt, an), g) in enumerate(groups, start=1):
        x = g["draw_value_mg"].to_numpy(dtype=float)
        y = float(g["y_obs"].iloc[0])

        if max_draws is not None and x.size > max_draws:
            take = rng.choice(x.size, size=max_draws, replace=False)
            x = x[take]

        rows.append({"Year": yr, "Treatment": trt, "Analyte": an, "CRPS": crps_from_draws(x, y)})

        if idx == 1 or idx % 50 == 0 or idx == total:
            pct = 100.0 * idx / total
            print(f"{progress_prefix} CRPS {method_label}: {idx}/{total} groups ({pct:.1f}%)")

    per_year = pd.DataFrame(rows)

    out = (
        per_year.groupby(["Analyte", "Treatment"], dropna=False)
        .agg(n=("CRPS", "count"), CRPS=("CRPS", "mean"))
        .reset_index()
    )
    out["method"] = method_label
    return out.loc[:, ["Analyte", "Treatment", "method", "n", "CRPS"]]


def aggregate_overall(metrics_by_group: pd.DataFrame) -> pd.DataFrame:
    if metrics_by_group.empty:
        return pd.DataFrame()

    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not mask.any():
            return float("nan")
        return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))

    rows = []
    for (an, method), g in metrics_by_group.groupby(["Analyte", "method"], dropna=False):
        n = g["n"].to_numpy(dtype=float)
        rmse = g["RMSE"].to_numpy(dtype=float)
        rmse_overall = float(np.sqrt(np.sum(n * (rmse ** 2)) / np.sum(n))) if np.isfinite(rmse).any() else float("nan")

        rows.append({
            "Analyte": an,
            "Treatment": "ALL",
            "method": method,
            "n": int(np.sum(n)),
            "MAE": wmean(g["MAE"].to_numpy(dtype=float), n),
            "RMSE": rmse_overall,
            "NRMSE_mean": wmean(g["NRMSE_mean"].to_numpy(dtype=float), n),
            "NRMSE_range": wmean(g["NRMSE_range"].to_numpy(dtype=float), n),
            "Coverage": wmean(g["Coverage"].to_numpy(dtype=float), n),
            "MeanWidth": wmean(g["MeanWidth"].to_numpy(dtype=float), n),
            "IntervalProb": g["IntervalProb"].iloc[0] if "IntervalProb" in g.columns else np.nan,
            "CRPS": wmean(g.get("CRPS", np.nan).to_numpy(dtype=float), n),
            "CRPS_norm_mean": wmean(g.get("CRPS_norm_mean", np.nan).to_numpy(dtype=float), n),
        })

    for method, g in metrics_by_group.groupby(["method"], dropna=False):
        n = g["n"].to_numpy(dtype=float)
        rmse = g["RMSE"].to_numpy(dtype=float)
        rmse_overall = float(np.sqrt(np.sum(n * (rmse ** 2)) / np.sum(n))) if np.isfinite(rmse).any() else float("nan")

        rows.append({
            "Analyte": "ALL",
            "Treatment": "ALL",
            "method": method,
            "n": int(np.sum(n)),
            "MAE": wmean(g["MAE"].to_numpy(dtype=float), n),
            "RMSE": rmse_overall,
            "NRMSE_mean": wmean(g["NRMSE_mean"].to_numpy(dtype=float), n),
            "NRMSE_range": wmean(g["NRMSE_range"].to_numpy(dtype=float), n),
            "Coverage": wmean(g["Coverage"].to_numpy(dtype=float), n),
            "MeanWidth": wmean(g["MeanWidth"].to_numpy(dtype=float), n),
            "IntervalProb": g["IntervalProb"].iloc[0] if "IntervalProb" in g.columns else np.nan,
            "CRPS": wmean(g.get("CRPS", np.nan).to_numpy(dtype=float), n),
            "CRPS_norm_mean": wmean(g.get("CRPS_norm_mean", np.nan).to_numpy(dtype=float), n),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default=None, help="Repo root (optional). Auto-detected if omitted.")
    ap.add_argument("--bayes", type=str, default=None, help="Bayes annual load summary CSV (contains observed+modeled).")
    ap.add_argument("--ml", type=str, default=None, help="ML annual load summary CSV (imputed-inclusive).")
    ap.add_argument("--units", type=str, default="g", choices=["mg", "g", "kg"], help="Units for plotting.")
    ap.add_argument("--analytes", type=str, default=None, help="Comma-separated analytes to plot (default: all).")
    ap.add_argument("--shared_only", action="store_true", help="Only plot analytes present in BOTH Bayes and ML.")
    ap.add_argument("--skip_plots", action="store_true", help="Only compute metrics, do not render plots.")

    ap.add_argument("--skip_crps", action="store_true", help="Skip CRPS even if draws exist.")
    ap.add_argument("--bayes_draws", type=str, default=None, help="Bayes draws CSV (default: out/annual_load_draws_bayes_v1p6.csv)")
    ap.add_argument("--ml_draws", type=str, default=None, help="ML draws CSV (default: out/ml_catboost_conformal_loyo/annual_load_draws.csv)")
    ap.add_argument("--crps_max_draws", type=int, default=None, help="Subsample cap per group for CRPS.")
    ap.add_argument("--crps_seed", type=int, default=1, help="Seed for CRPS subsampling.")

    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())
    bayes_path = Path(args.bayes).resolve() if args.bayes else (repo / "out" / "annual_load_summary_bayes_plus_observed_v1p6.csv")
    ml_path = Path(args.ml).resolve() if args.ml else (repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_summary_imputed.csv")

    if not bayes_path.exists():
        raise FileNotFoundError(f"Bayes file not found: {bayes_path}")
    if not ml_path.exists():
        raise FileNotFoundError(f"ML file not found: {ml_path}")

    figs_outdir = repo / "figs" / "annual_bayes_vs_ml_faceted_jpg"
    figs_outdir.mkdir(parents=True, exist_ok=True)

    metrics_outdir = repo / "out" / "bayes_vs_ml_metrics"
    metrics_outdir.mkdir(parents=True, exist_ok=True)

    bayes_raw = pd.read_csv(bayes_path)
    ml_raw = pd.read_csv(ml_path)

    bayes_modeled, observed = split_bayes_observed_modeled(bayes_raw)

    ml_std = standardize_summary(
        ml_raw,
        source_name="ML",
        series_label="ML modeled",
        center_candidates=["load_mean", "mean", "median", "center"],
        low_candidates=["load_low", "low", "pi_low", "lower", "q_low", "p2_5", "lo"],
        high_candidates=["load_high", "high", "pi_high", "upper", "q_high", "p97_5", "hi"],
    )

    bayes_modeled = canonicalize_analytes(bayes_modeled)
    observed = canonicalize_analytes(observed)
    ml_std = canonicalize_analytes(ml_std)

    bayes_an = set(bayes_modeled["Analyte"].unique())
    ml_an = set(ml_std["Analyte"].unique())
    shared = sorted(bayes_an.intersection(ml_an))
    union = sorted(bayes_an.union(ml_an))

    if args.analytes:
        wanted = [a.strip() for a in args.analytes.split(",") if a.strip()]
        analytes = [a for a in wanted if a in union]
    else:
        analytes = shared if args.shared_only else union

    treatments = ["CT", "MT", "ST"]
    trts_avail = sorted(set(bayes_modeled["Treatment"]).union(set(ml_std["Treatment"])).union(set(observed["Treatment"])))
    use_trts = treatments if all(t in trts_avail for t in treatments) else [t for t in treatments if t in trts_avail] or trts_avail

    print(f"[INFO] Repo        : {repo}")
    print(f"[INFO] Bayes       : {bayes_path}")
    print(f"[INFO] ML          : {ml_path}")
    print(f"[INFO] Figs outdir  : {figs_outdir}")
    print(f"[INFO] Metrics dir  : {metrics_outdir}")
    print(f"[INFO] Units       : {args.units}")
    print(f"[INFO] Facets      : {use_trts}")
    print(f"[INFO] Bayes analytes: {len(bayes_an)} | ML analytes: {len(ml_an)} | Shared: {len(shared)}")
    print(f"[INFO] Analytes to plot: {len(analytes)} {'(shared_only)' if args.shared_only else '(union/default)'}")

    bayes_interval_prob = None
    ml_interval_prob = None
    if "interval_prob" in bayes_raw.columns:
        try:
            bayes_interval_prob = float(pd.to_numeric(bayes_raw["interval_prob"], errors="coerce").dropna().unique()[0])
        except Exception:
            bayes_interval_prob = None
    if "interval_prob" in ml_raw.columns:
        try:
            ml_interval_prob = float(pd.to_numeric(ml_raw["interval_prob"], errors="coerce").dropna().unique()[0])
        except Exception:
            ml_interval_prob = None

    metrics_bayes = compute_metrics_point_interval(observed=observed, pred=bayes_modeled, method_label="Bayes", interval_prob=bayes_interval_prob)
    metrics_ml = compute_metrics_point_interval(observed=observed, pred=ml_std, method_label="ML", interval_prob=ml_interval_prob)
    metrics_by_group = pd.concat([metrics_bayes, metrics_ml], ignore_index=True)

    if not args.skip_crps:
        bayes_draws_path = Path(args.bayes_draws).resolve() if args.bayes_draws else (repo / "out" / "annual_load_draws_bayes_v1p6.csv")
        ml_draws_path = Path(args.ml_draws).resolve() if args.ml_draws else (repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_draws.csv")

        if bayes_draws_path.exists() and ml_draws_path.exists():
            print(f"[INFO] CRPS draws (Bayes): {bayes_draws_path}")
            print(f"[INFO] CRPS draws (ML)   : {ml_draws_path}")

            print("[INFO] Reading Bayes draws...")
            bayes_draws_raw = pd.read_csv(bayes_draws_path)
            bayes_draws, _ = standardize_draws(bayes_draws_raw, label="Bayes")

            print("[INFO] Reading ML draws...")
            ml_draws_raw = pd.read_csv(ml_draws_path)
            ml_draws, _ = standardize_draws(ml_draws_raw, label="ML")

            crps_bayes = compute_crps_table(observed, bayes_draws, "Bayes", args.crps_max_draws, args.crps_seed, "[INFO]")
            crps_ml = compute_crps_table(observed, ml_draws, "ML", args.crps_max_draws, args.crps_seed, "[INFO]")
            crps = pd.concat([crps_bayes, crps_ml], ignore_index=True)

            metrics_by_group = metrics_by_group.merge(
                crps.loc[:, ["Analyte", "Treatment", "method", "CRPS"]],
                on=["Analyte", "Treatment", "method"],
                how="left",
                suffixes=("", "_crps"),
            )

            # If metrics_by_group already had a CRPS column (initialized as NaN),
            # the merge will create CRPS_crps. Coalesce to a single CRPS column.
            if "CRPS_crps" in metrics_by_group.columns:
                metrics_by_group["CRPS"] = metrics_by_group["CRPS_crps"].combine_first(metrics_by_group.get("CRPS"))
                metrics_by_group = metrics_by_group.drop(columns=["CRPS_crps"])
            print("[OK] CRPS computed and merged into metrics.")

            # Mean-normalized CRPS uses the mean absolute observed load for each (Analyte, Treatment)
            # as the scale factor. This enables scale-adjusted comparisons (e.g., across analytes).
            obs_scale = (
                observed.loc[:, ["Analyte", "Treatment", "center_mg"]]
                .assign(mean_abs_obs=lambda d: d["center_mg"].abs())
                .groupby(["Analyte", "Treatment"], dropna=False)["mean_abs_obs"]
                .mean()
                .reset_index()
            )

            metrics_by_group = metrics_by_group.merge(
                obs_scale,
                on=["Analyte", "Treatment"],
                how="left",
            )

            if "CRPS" in metrics_by_group.columns:
                metrics_by_group["CRPS_norm_mean"] = np.where(
                np.isfinite(metrics_by_group["CRPS"]) & np.isfinite(metrics_by_group["mean_abs_obs"]) & (metrics_by_group["mean_abs_obs"] > 0),
                metrics_by_group["CRPS"] / metrics_by_group["mean_abs_obs"],
                np.nan,
            )
            else:
                metrics_by_group["CRPS_norm_mean"] = np.nan
        else:
            print("[WARN] CRPS skipped because draws file(s) missing:")
            if not bayes_draws_path.exists():
                print(f"       - missing Bayes draws: {bayes_draws_path}")
            if not ml_draws_path.exists():
                print(f"       - missing ML draws   : {ml_draws_path}")
    else:
        print("[INFO] CRPS disabled via --skip_crps")
    # Ensure mean-normalized CRPS column exists even when CRPS was skipped/missing.
    if "mean_abs_obs" not in metrics_by_group.columns:
        obs_scale = (
            observed.loc[:, ["Analyte", "Treatment", "center_mg"]]
            .assign(mean_abs_obs=lambda d: d["center_mg"].abs())
            .groupby(["Analyte", "Treatment"], dropna=False)["mean_abs_obs"]
            .mean()
            .reset_index()
        )
        metrics_by_group = metrics_by_group.merge(
            obs_scale, on=["Analyte", "Treatment"], how="left"
        )

    if "CRPS_norm_mean" not in metrics_by_group.columns:
        if "CRPS" in metrics_by_group.columns:
            metrics_by_group["CRPS_norm_mean"] = np.where(
            np.isfinite(metrics_by_group["CRPS"]) & np.isfinite(metrics_by_group["mean_abs_obs"]) & (metrics_by_group["mean_abs_obs"] > 0),
            metrics_by_group["CRPS"] / metrics_by_group["mean_abs_obs"],
            np.nan,
        )
        else:
            metrics_by_group["CRPS_norm_mean"] = np.nan

    metrics_by_group = metrics_by_group.sort_values(["Analyte", "Treatment", "method"])
    metrics_by_analyte_overall = aggregate_overall(metrics_by_group)
    metrics_overall = metrics_by_analyte_overall.loc[metrics_by_analyte_overall["Analyte"].eq("ALL")].copy()
    metrics_by_analyte_overall = metrics_by_analyte_overall.loc[~metrics_by_analyte_overall["Analyte"].eq("ALL")].copy()

    metrics_by_group.to_csv(metrics_outdir / "metrics_by_analyte_treatment.csv", index=False)
    metrics_by_analyte_overall.to_csv(metrics_outdir / "metrics_by_analyte_overall.csv", index=False)
    metrics_overall.to_csv(metrics_outdir / "metrics_overall.csv", index=False)

    print("[OK] Metrics written:")
    print(f"     - {metrics_outdir / 'metrics_by_analyte_treatment.csv'}")
    print(f"     - {metrics_outdir / 'metrics_by_analyte_overall.csv'}")
    print(f"     - {metrics_outdir / 'metrics_overall.csv'}")

    if not args.skip_plots:
        for an in analytes:
            safe_an = re.sub(r"[^A-Za-z0-9]+", "_", an.strip().lower()).strip("_")
            out_jpg = figs_outdir / f"annual_load_{safe_an}_bayes_vs_ml_faceted.jpg"
            plot_analyte_faceted(an, bayes_modeled, ml_std, observed, out_jpg, units=args.units, treatments=use_trts[:3] if len(use_trts) >= 3 else use_trts)

        print(f"[OK] Wrote {len(analytes)} JPG figures to: {figs_outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
