#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annual_load_bayes_vs_ml.py

Creates annual load comparison plots faceted by Treatment (CT/MT/ST) and
computes quantitative goodness-of-fit (GoF) metrics comparing Bayes vs ML
against Observed annual loads.
Also creates an annual volume comparison plot and volume metrics when Bayes
annual volume CSVs are available.

New in this drop-in (v1p7p5 request):
  - Adds NRMSE_sd = RMSE / SD(Observed).
  - Writes 3 GoF figures by analyte (Treatment aggregated to ALL):
      1) Paired bars: CRPS_norm_mean by analyte (Bayes vs ML)
      2) Paired bars: NRMSE_mean by analyte (Bayes vs ML), ranked by Bayes (best→worst)
      3) Scatter: Coverage vs MeanWidth by analyte (Bayes vs ML), with nominal coverage line.
  - Saves GoF figures to: figs/annual_bayes_vs_ml_gof_jpg_<tag> (or without tag)
  - Saves metrics to: out/bayes_vs_ml_metrics_<tag> (or without tag)

Optional:
  - CRPS (Continuous Ranked Probability Score) is computed if per-draw annual load files exist.

Key rule:
  Observed values are ONLY taken from rows explicitly labeled as observed
  in the Bayes summary file. We do NOT infer "observed" from missing CI columns.

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


# ----------------------------
# Repo utilities
# ----------------------------
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


# ----------------------------
# Units
# ----------------------------
def mg_to_units(x: pd.Series, units: str) -> pd.Series:
    u = units.lower().strip()
    if u == "mg":
        return x
    if u == "g":
        return x / 1e3
    if u == "kg":
        return x / 1e6
    raise ValueError("units must be one of: mg, g, kg")


def units_to_mg(x: pd.Series, units: str) -> pd.Series:
    u = str(units).lower().strip()
    if u == "mg":
        return x
    if u == "g":
        return x * 1e3
    if u == "kg":
        return x * 1e6
    if u == "auto":
        return x
    raise ValueError("input units must be one of: mg, g, kg, auto")


def volume_to_kL(x: pd.Series, units: str) -> pd.Series:
    u = str(units).lower().strip()
    if u in {"kl", "kiloliter", "kiloliters", "kilolitre", "kilolitres"}:
        return x
    if u in {"l", "liter", "liters", "litre", "litres"}:
        return x / 1e3
    if u in {"m3", "m^3", "cubic_meter", "cubic_meters", "cubic_metre", "cubic_metres"}:
        return x
    raise ValueError("volume units must be one of: L, kL, m3")


def kL_to_units(x: pd.Series, units: str) -> pd.Series:
    u = str(units).lower().strip()
    if u in {"kl", "kiloliter", "kiloliters", "kilolitre", "kilolitres"}:
        return x
    if u in {"l", "liter", "liters", "litre", "litres"}:
        return x * 1e3
    if u in {"m3", "m^3", "cubic_meter", "cubic_meters", "cubic_metre", "cubic_metres"}:
        return x
    raise ValueError("volume units must be one of: L, kL, m3")


def infer_volume_units_from_col(col: str, default_units: str) -> str:
    c = str(col).lower()
    if "volume_kl" in c or "_kl_" in c or c.endswith("_kl"):
        return "kL"
    if "volume_l" in c or "_l_" in c or c.endswith("_l"):
        return "L"
    return default_units


# ----------------------------
# Analyte canonicalization
# ----------------------------
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


# ----------------------------
# Summary standardization
# ----------------------------
def standardize_summary(
    df: pd.DataFrame,
    source_name: str,
    series_label: str,
    center_candidates: List[str],
    low_candidates: List[str],
    high_candidates: List[str],
    input_units: str = "mg",
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

    out["center_mg"] = units_to_mg(safe_numeric(df[c_center]), input_units)
    out["low_mg"] = units_to_mg(safe_numeric(df[c_low]), input_units) if c_low is not None else np.nan
    out["high_mg"] = units_to_mg(safe_numeric(df[c_high]), input_units) if c_high is not None else np.nan

    out = out.dropna(subset=["Year", "Treatment", "Analyte", "center_mg"]).copy()
    out["Year"] = out["Year"].astype(int)

    return out


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


def split_bayes_observed_modeled(bayes_raw: pd.DataFrame, input_units: str = "mg") -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        input_units=input_units,
    )

    observed = standardize_summary(
        df_obs,
        source_name="Observed",
        series_label="Observed",
        center_candidates=["load_mean", "mean", "median", "center"],
        low_candidates=["load_low", "low", "lo", "lower", "lwr"],
        high_candidates=["load_high", "high", "hi", "upper", "upr"],
        input_units=input_units,
    )

    return bayes_modeled, observed


def standardize_volume_wide(
    df: pd.DataFrame,
    source_name: str,
    series_label: str,
    source_tag: str,
    input_units: str = "kL",
    treatments: List[str] = ["CT", "MT", "ST"],
) -> pd.DataFrame:
    """Convert Bayes annual volume wide CSVs to the long comparison schema.

    The internal numeric columns are named center_mg/low_mg/high_mg for reuse
    with the existing metrics functions, but values are annual volume in kL.
    """
    df = normalize_cols(df)
    if "Year" not in df.columns:
        raise ValueError(f"{source_name}: volume file missing Year column.")

    rows = []
    for trt in treatments:
        c_center = pick_first_existing(df, [
            f"{trt}_{source_tag}_volume_kL_mean",
            f"{trt}_{source_tag}_volume_mean",
            f"{trt}_{source_tag}_volume_L_mean",
            f"{trt}_volume_kL_mean",
            f"{trt}_volume_mean",
        ])
        c_low = pick_first_existing(df, [
            f"{trt}_{source_tag}_volume_kL_low",
            f"{trt}_{source_tag}_volume_low",
            f"{trt}_{source_tag}_volume_L_low",
            f"{trt}_volume_kL_low",
            f"{trt}_volume_low",
        ])
        c_high = pick_first_existing(df, [
            f"{trt}_{source_tag}_volume_kL_high",
            f"{trt}_{source_tag}_volume_high",
            f"{trt}_{source_tag}_volume_L_high",
            f"{trt}_volume_kL_high",
            f"{trt}_volume_high",
        ])

        if c_center is None:
            continue

        col_units = infer_volume_units_from_col(c_center, input_units)
        tmp = pd.DataFrame({
            "Year": df["Year"],
            "Treatment": trt,
            "Analyte": "Volume",
            "source": source_name,
            "series": series_label,
            "center_mg": volume_to_kL(safe_numeric(df[c_center]), col_units),
            "low_mg": volume_to_kL(safe_numeric(df[c_low]), col_units) if c_low is not None else np.nan,
            "high_mg": volume_to_kL(safe_numeric(df[c_high]), col_units) if c_high is not None else np.nan,
        })
        rows.append(tmp)

    if not rows:
        raise ValueError(
            f"{source_name}: no annual volume columns found for tag '{source_tag}'. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["Year", "Treatment", "center_mg"]).copy()
    out["Year"] = safe_numeric(out["Year"]).astype(int)
    return out


def standardize_ml_volume_summary(
    df: pd.DataFrame,
    source_name: str = "ML",
    series_label: str = "ML modeled",
    input_units: str = "L",
) -> pd.DataFrame:
    """Read ML annual volume fields if they are present in the annual summary."""
    df = normalize_cols(df)
    if "Year" not in df.columns or "Treatment" not in df.columns:
        return pd.DataFrame()

    c_center = pick_first_existing(df, ["volume_mean", "volume_kL_mean", "volume_L_mean"])
    c_low = pick_first_existing(df, ["volume_low", "volume_kL_low", "volume_L_low"])
    c_high = pick_first_existing(df, ["volume_high", "volume_kL_high", "volume_L_high"])

    if c_center is None:
        return pd.DataFrame()

    col_units = infer_volume_units_from_col(c_center, input_units)
    tmp = df.loc[:, ["Year", "Treatment"]].copy()
    tmp["center_mg"] = volume_to_kL(safe_numeric(df[c_center]), col_units)
    tmp["low_mg"] = volume_to_kL(safe_numeric(df[c_low]), col_units) if c_low is not None else np.nan
    tmp["high_mg"] = volume_to_kL(safe_numeric(df[c_high]), col_units) if c_high is not None else np.nan
    tmp = tmp.dropna(subset=["Year", "Treatment", "center_mg"]).copy()
    if tmp.empty:
        return pd.DataFrame()

    # Annual ML summaries are often one row per analyte. Volume is event-level,
    # so average repeated Year x Treatment estimates into one volume row.
    out = (
        tmp.groupby(["Year", "Treatment"], dropna=False)
        .agg(center_mg=("center_mg", "mean"), low_mg=("low_mg", "mean"), high_mg=("high_mg", "mean"))
        .reset_index()
    )
    out["Analyte"] = "Volume"
    out["source"] = source_name
    out["series"] = series_label
    out["Year"] = safe_numeric(out["Year"]).astype(int)
    return out.loc[:, ["Year", "Treatment", "Analyte", "source", "series", "center_mg", "low_mg", "high_mg"]]


def standardize_ml_volume_from_events(
    df: pd.DataFrame,
    source_name: str = "ML",
    series_label: str = "ML modeled",
    input_units: str = "L",
) -> pd.DataFrame:
    """Aggregate ML event-level filled volume to annual Treatment volume.

    This is a fallback for older ML annual summaries that only contain load
    columns. The ML event files repeat each event by analyte, so rows are first
    deduplicated to unique water-quality events.
    """
    df = normalize_cols(df)
    if "Year" not in df.columns or "Treatment" not in df.columns:
        raise ValueError("ML volume events file must contain Year and Treatment columns.")

    if "NoRunoff" in df.columns:
        no_runoff = df["NoRunoff"]
        if no_runoff.dtype == bool:
            no_runoff_mask = no_runoff.fillna(False)
        else:
            no_runoff_mask = no_runoff.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})
        df = df.loc[~no_runoff_mask].copy()

    c_center = pick_first_existing(df, ["Volume_filled", "Volume_pred", "Volume"])
    c_low = pick_first_existing(df, ["Volume_pi_low", "volume_low", "Volume_low"])
    c_high = pick_first_existing(df, ["Volume_pi_high", "volume_high", "Volume_high"])

    if c_center is None:
        raise ValueError(
            "ML volume events file does not contain a usable volume column. "
            "Tried Volume_filled, Volume_pred, Volume."
        )

    # v2 event-level ML outputs provide the exact physical volume-event key.
    # Prefer it over the legacy approximate key so mapped analyte rows do not
    # re-weight an event during annual volume aggregation.
    if "EventVolumeID" in df.columns:
        event_cols = ["EventVolumeID"]
    elif "orig_row" in df.columns:
        event_cols = ["orig_row"]
    else:
        event_candidates = [
            "Date", "Year", "Treatment", "Irrigation", "Rep", "SampleID",
            "InflowOutflow", "FF", "Composite", "Duplicate", "NoRunoff"
        ]
        event_cols = [c for c in event_candidates if c in df.columns]
        if not event_cols:
            event_cols = ["Year", "Treatment"]

    cols = list(dict.fromkeys(event_cols + ["Year", "Treatment", c_center] + ([c_low] if c_low else []) + ([c_high] if c_high else [])))
    tmp = df.loc[:, cols].copy()
    tmp[c_center] = safe_numeric(tmp[c_center])
    if c_low is not None:
        tmp[c_low] = safe_numeric(tmp[c_low])
    if c_high is not None:
        tmp[c_high] = safe_numeric(tmp[c_high])

    tmp = tmp.dropna(subset=["Year", "Treatment", c_center]).copy()
    tmp = tmp.groupby(event_cols, dropna=False, as_index=False).first()

    col_units = infer_volume_units_from_col(c_center, input_units)
    center = volume_to_kL(safe_numeric(tmp[c_center]), col_units)
    low = volume_to_kL(safe_numeric(tmp[c_low]), col_units) if c_low is not None else center.copy()
    high = volume_to_kL(safe_numeric(tmp[c_high]), col_units) if c_high is not None else center.copy()

    low = low.where(np.isfinite(low), center)
    high = high.where(np.isfinite(high), center)

    tmp2 = tmp.loc[:, ["Year", "Treatment"]].copy()
    tmp2["center_mg"] = center
    tmp2["low_mg"] = low
    tmp2["high_mg"] = high

    out = (
        tmp2.groupby(["Year", "Treatment"], dropna=False)
        .agg(center_mg=("center_mg", "sum"), low_mg=("low_mg", "sum"), high_mg=("high_mg", "sum"))
        .reset_index()
    )
    out["Analyte"] = "Volume"
    out["source"] = source_name
    out["series"] = series_label
    out["Year"] = safe_numeric(out["Year"]).astype(int)
    return out.loc[:, ["Year", "Treatment", "Analyte", "source", "series", "center_mg", "low_mg", "high_mg"]]


# ----------------------------
# Annual faceted plots
# ----------------------------
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


def plot_volume_faceted(
    bayes: pd.DataFrame,
    ml: pd.DataFrame,
    obs: pd.DataFrame,
    out_jpg: Path,
    units: str = "kL",
    treatments: List[str] = ["CT", "MT", "ST"],
) -> None:
    fig, axes = plt.subplots(nrows=1, ncols=len(treatments), figsize=(5.2 * len(treatments), 5.3), sharey=True)
    if len(treatments) == 1:
        axes = [axes]
    fig.suptitle("Annual volume comparison (Observed vs Bayes vs ML; imputed-inclusive)")

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
        y = kL_to_units(d["center_mg"], units).to_numpy()

        if np.isfinite(d["low_mg"]).any() and np.isfinite(d["high_mg"]).any():
            ylo = kL_to_units(d["low_mg"], units).to_numpy()
            yhi = kL_to_units(d["high_mg"], units).to_numpy()
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
        y = kL_to_units(d["center_mg"], units).to_numpy()

        if np.isfinite(d["low_mg"]).any() and np.isfinite(d["high_mg"]).any():
            yerr_lower = y - kL_to_units(d["low_mg"], units).to_numpy()
            yerr_upper = kL_to_units(d["high_mg"], units).to_numpy() - y
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
            ax.set_ylabel(f"Annual volume ({units})")

        b = bayes[bayes["Treatment"] == trt]
        m = ml[ml["Treatment"] == trt]
        o = obs[obs["Treatment"] == trt]

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


# ----------------------------
# GoF metrics
# ----------------------------
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


def _nrmse_sd(rmse: float, y: np.ndarray) -> float:
    sd = float(np.nanstd(y)) if y.size else float("nan")
    if not np.isfinite(sd) or sd == 0.0 or not np.isfinite(rmse):
        return float("nan")
    return float(rmse / sd)


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
            "NRMSE_sd": _nrmse_sd(rmse, y),
            "Coverage": cover,
            "MeanWidth": width,
            "IntervalProb": interval_prob,
            "CRPS": float("nan"),
            "CRPS_norm_mean": float("nan"),
        })

    return pd.DataFrame(out_rows)


# ----------------------------
# CRPS (from draws)
# ----------------------------
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


def standardize_draws(df: pd.DataFrame, label: str, input_units: str = "auto") -> Tuple[pd.DataFrame, str]:
    df = normalize_cols(df)

    draw_id_col = pick_first_existing(df, ["draw", "Draw", "draw_id", "iter", "iteration", "sample", "s", "mcmc_draw"])
    if draw_id_col is None:
        draw_id_col = "__draw_id__"
        df[draw_id_col] = np.arange(len(df), dtype=int)

    value_candidates = [
        # mg
        "load_mg", "annual_load_mg", "annualLoad_mg", "AnnualLoad_mg", "annualload_mg",
        "load_draw_mg", "y_mg", "draw_value_mg",
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

    vcol_lower = str(value_col).lower()
    if vcol_lower.endswith("_g") or vcol_lower in {"load_g", "annual_load_g"}:
        out["draw_value_mg"] = out["draw_value_mg"] * 1e3
    elif vcol_lower.endswith("_kg") or vcol_lower in {"load_kg", "annual_load_kg"}:
        out["draw_value_mg"] = out["draw_value_mg"] * 1e6
    elif vcol_lower.endswith("_mg") or vcol_lower in {"load_mg", "annual_load_mg"}:
        pass
    else:
        if input_units is not None and str(input_units).lower().strip() in {"mg", "g", "kg"}:
            out["draw_value_mg"] = units_to_mg(out["draw_value_mg"], str(input_units))

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


# ----------------------------
# Aggregation (Treatment -> ALL) + GoF plots
# ----------------------------
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
            "NRMSE_sd": wmean(g["NRMSE_sd"].to_numpy(dtype=float), n),
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
            "NRMSE_sd": wmean(g["NRMSE_sd"].to_numpy(dtype=float), n),
            "Coverage": wmean(g["Coverage"].to_numpy(dtype=float), n),
            "MeanWidth": wmean(g["MeanWidth"].to_numpy(dtype=float), n),
            "IntervalProb": g["IntervalProb"].iloc[0] if "IntervalProb" in g.columns else np.nan,
            "CRPS": wmean(g.get("CRPS", np.nan).to_numpy(dtype=float), n),
            "CRPS_norm_mean": wmean(g.get("CRPS_norm_mean", np.nan).to_numpy(dtype=float), n),
        })

    return pd.DataFrame(rows)


def _prep_by_analyte_all(metrics_by_analyte_overall: pd.DataFrame) -> pd.DataFrame:
    """Prepare Treatment-aggregated metrics for GoF plots.

    - Keeps ONLY analytes where BOTH Bayes and ML rows exist (Treatment == 'ALL').
    - Returns a DataFrame sorted by analyte then method (ordering may later be overridden per-plot).
    """
    d = metrics_by_analyte_overall.copy()
    d = d[(d["Treatment"] == "ALL") & (d["Analyte"] != "ALL")].copy()

    # Enforce "shared analytes only" for GoF plots (Bayes and ML must both be present)
    have_bayes = set(d.loc[d["method"] == "Bayes", "Analyte"].astype(str))
    have_ml = set(d.loc[d["method"] == "ML", "Analyte"].astype(str))
    shared = sorted(have_bayes.intersection(have_ml))
    dropped = sorted((have_bayes.union(have_ml)) - set(shared))

    if dropped:
        print(f"[INFO] GoF: dropping {len(dropped)} analyte(s) missing Bayes or ML: {', '.join(dropped)}")
    else:
        print("[INFO] GoF: all analytes have both Bayes and ML.")

    d = d[d["Analyte"].astype(str).isin(shared)].copy()

    # Default ordering (plots can re-order)
    order = sorted(d["Analyte"].unique())
    d["Analyte"] = pd.Categorical(d["Analyte"], categories=order, ordered=True)
    return d.sort_values(["Analyte", "method"])


def plot_paired_bars_by_analyte(d_all: pd.DataFrame, metric: str, outpath: Path, title: str, ylabel: str) -> None:
    """Paired bar chart (Bayes vs ML) by analyte.

    Rules:
      - Only analytes where BOTH Bayes and ML are present are eligible (should already be enforced upstream),
        but we also enforce it here for safety.
      - Drops analytes where either Bayes or ML is non-finite for the requested metric.
      - Orders analytes from best (lowest Bayes value) to worst (highest Bayes value).
    """
    d = d_all.copy()

    # Pivot to ensure paired values exist
    pvt = d.pivot_table(index="Analyte", columns="method", values=metric, aggfunc="first")
    if "Bayes" not in pvt.columns or "ML" not in pvt.columns:
        print(f"[WARN] GoF plot skipped (missing Bayes/ML columns after pivot): {metric}")
        return

    # Keep only analytes where BOTH methods have finite values for this metric
    keep_mask = np.isfinite(pvt["Bayes"].to_numpy(dtype=float)) & np.isfinite(pvt["ML"].to_numpy(dtype=float))
    pvt_keep = pvt.loc[keep_mask].copy()

    dropped = [str(a) for a in pvt.index if a not in pvt_keep.index]
    if dropped:
        print(f"[INFO] GoF: for metric '{metric}', dropping {len(dropped)} analyte(s) with missing/non-finite Bayes or ML values.")

    if pvt_keep.empty:
        print(f"[WARN] GoF plot skipped (no analytes with paired finite values): {metric}")
        return

    # Order by Bayes (ascending = best)
    pvt_keep = pvt_keep.sort_values("Bayes", ascending=True)

    analytes = [str(a) for a in pvt_keep.index.tolist()]
    bayes_vals = pvt_keep["Bayes"].to_numpy(dtype=float)
    ml_vals = pvt_keep["ML"].to_numpy(dtype=float)

    x = np.arange(len(analytes), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(analytes)), 5.6))
    ax.bar(x - width / 2, bayes_vals, width=width, label="Bayes")
    ax.bar(x + width / 2, ml_vals, width=width, label="ML")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Analyte (ranked by Bayes, best → worst)")
    ax.set_xticks(x)
    ax.set_xticklabels(analytes, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.30)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, format="jpg")
    plt.close(fig)


def plot_coverage_vs_width(d_all: pd.DataFrame, outpath: Path, nominal: float = 0.95) -> None:
    d = d_all.copy()
    d = d[np.isfinite(d["Coverage"].to_numpy(dtype=float)) & np.isfinite(d["MeanWidth"].to_numpy(dtype=float))].copy()
    if d.empty:
        print("[WARN] Coverage vs width plot skipped (Coverage/MeanWidth missing or non-finite).")
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    for method, g in d.groupby("method"):
        ax.scatter(g["Coverage"].to_numpy(dtype=float), g["MeanWidth"].to_numpy(dtype=float), label=str(method), s=70)
        for _, r in g.iterrows():
            ax.annotate(
                str(r["Analyte"]),
                (float(r["Coverage"]), float(r["MeanWidth"])),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=8,
                alpha=0.8,
            )

    ax.axvline(nominal, linestyle="--", linewidth=1.5, color="0.4")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean interval width (mg)")
    ax.set_title("Uncertainty calibration by analyte (Coverage vs MeanWidth)")
    ax.grid(True, alpha=0.30)
    ax.set_xlim(0, 1.0)
    try:
        ax.set_yscale("log")
    except Exception:
        pass

    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, format="jpg")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default=None, help="Repo root (optional). Auto-detected if omitted.")
    ap.add_argument("--bayes", type=str, default=None, help="Bayes annual load summary CSV (contains observed+modeled).")
    ap.add_argument("--ml", type=str, default=None, help="ML annual load summary CSV (imputed-inclusive).")
    ap.add_argument("--bayes_volume_observed", type=str, default=None, help="Bayes observed annual volume wide CSV.")
    ap.add_argument("--bayes_volume_modeled", type=str, default=None, help="Bayes modeled annual volume wide CSV.")
    ap.add_argument("--ml_volume_events", type=str, default=None, help="ML event-level imputed CSV for annual volume fallback.")

    ap.add_argument("--units", type=str, default="g", choices=["mg", "g", "kg"], help="Units for plotting.")
    ap.add_argument("--bayes_input_units", type=str, default="g", choices=["mg", "g", "kg"])
    ap.add_argument("--ml_input_units", type=str, default="mg", choices=["mg", "g", "kg"])
    ap.add_argument("--volume_units", type=str, default="kL", choices=["L", "kL", "m3"], help="Units for volume plotting.")
    ap.add_argument("--bayes_volume_input_units", type=str, default="kL", choices=["L", "kL", "m3"])
    ap.add_argument("--ml_volume_input_units", type=str, default="L", choices=["L", "kL", "m3"])

    ap.add_argument("--bayes_draws_units", type=str, default="auto", choices=["auto", "mg", "g", "kg"])
    ap.add_argument("--ml_draws_units", type=str, default="auto", choices=["auto", "mg", "g", "kg"])

    ap.add_argument("--analytes", type=str, default=None, help="Comma-separated analytes to plot (default: all).")
    ap.add_argument("--shared_only", action="store_true", help="Only plot analytes present in BOTH Bayes and ML.")
    ap.add_argument("--skip_plots", action="store_true", help="Only compute metrics, do not render annual plots.")
    ap.add_argument("--skip_gof_plots", action="store_true", help="Skip GoF plots by analyte.")
    ap.add_argument("--skip_volume", action="store_true", help="Skip annual volume comparison plot and metrics.")

    ap.add_argument("--tag", type=str, default=None, help="Optional label to version outputs (e.g., v1p7p5).")

    ap.add_argument("--skip_crps", action="store_true", help="Skip CRPS even if draws exist.")
    ap.add_argument("--bayes_draws", type=str, default=None, help="Bayes draws CSV (optional override).")
    ap.add_argument("--ml_draws", type=str, default=None, help="ML draws CSV (optional override).")
    ap.add_argument("--crps_max_draws", type=int, default=None, help="Subsample cap per group for CRPS.")
    ap.add_argument("--crps_seed", type=int, default=1, help="Seed for CRPS subsampling.")

    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())
    tag = None if args.tag is None else re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tag.strip())

    # --- inputs ---
    if args.bayes:
        bayes_path = Path(args.bayes).resolve()
    else:
        cand = repo / "out" / f"annual_load_summary_bayes_plus_observed_{tag}.csv" if tag else None
        bayes_path = cand if (cand is not None and cand.exists()) else (repo / "out" / "annual_load_summary_bayes_plus_observed_v1p6.csv")

    ml_path = Path(args.ml).resolve() if args.ml else (repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_summary_imputed.csv")
    if args.bayes_volume_observed:
        bayes_volume_observed_path = Path(args.bayes_volume_observed).resolve()
    else:
        cand = repo / "out" / f"annual_volume_kL_wide_observed_{tag}.csv" if tag else None
        bayes_volume_observed_path = cand if (cand is not None and cand.exists()) else None

    if args.bayes_volume_modeled:
        bayes_volume_modeled_path = Path(args.bayes_volume_modeled).resolve()
    else:
        cand = repo / "out" / f"annual_volume_kL_wide_modeled_{tag}.csv" if tag else None
        bayes_volume_modeled_path = cand if (cand is not None and cand.exists()) else None

    ml_volume_events_path = (
        Path(args.ml_volume_events).resolve()
        if args.ml_volume_events
        else (repo / "out" / "ml_catboost_conformal_loyo" / "wq_cleaned_ml_imputed.csv")
    )

    if not bayes_path.exists():
        raise FileNotFoundError(f"Bayes file not found: {bayes_path}")
    if not ml_path.exists():
        raise FileNotFoundError(f"ML file not found: {ml_path}")

    figs_dirname = "annual_bayes_vs_ml_faceted_jpg" + (f"_{tag}" if tag else "")
    gof_figs_dirname = "annual_bayes_vs_ml_faceted_jpg" + (f"_{tag}" if tag else "")
    metrics_dirname = "bayes_vs_ml_metrics" + (f"_{tag}" if tag else "")

    figs_outdir = repo / "figs" / figs_dirname
    figs_outdir.mkdir(parents=True, exist_ok=True)

    gof_outdir = repo / "figs" / gof_figs_dirname
    gof_outdir.mkdir(parents=True, exist_ok=True)

    metrics_outdir = repo / "out" / metrics_dirname
    metrics_outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo        : {repo}")
    print(f"[INFO] Bayes       : {bayes_path}")
    print(f"[INFO] ML          : {ml_path}")
    if not args.skip_volume:
        print(f"[INFO] Bayes volume observed: {bayes_volume_observed_path if bayes_volume_observed_path else 'not found'}")
        print(f"[INFO] Bayes volume modeled : {bayes_volume_modeled_path if bayes_volume_modeled_path else 'not found'}")
        print(f"[INFO] ML volume events     : {ml_volume_events_path}")
    print(f"[INFO] Figs outdir  : {figs_outdir}")
    print(f"[INFO] GoF outdir   : {gof_outdir}")
    print(f"[INFO] Metrics dir  : {metrics_outdir}")

    # --- read ---
    bayes_raw = pd.read_csv(bayes_path)
    ml_raw = pd.read_csv(ml_path)

    bayes_modeled, observed = split_bayes_observed_modeled(bayes_raw, input_units=args.bayes_input_units)

    ml_std = standardize_summary(
        ml_raw,
        source_name="ML",
        series_label="ML modeled",
        center_candidates=["load_mean", "mean", "median", "center"],
        low_candidates=["load_low", "low", "pi_low", "lower", "q_low", "p2_5", "lo"],
        high_candidates=["load_high", "high", "pi_high", "upper", "q_high", "p97_5", "hi"],
        input_units=args.ml_input_units,
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

    # interval prob (optional metadata)
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

    # --- point + interval metrics ---
    metrics_bayes = compute_metrics_point_interval(observed=observed, pred=bayes_modeled, method_label="Bayes", interval_prob=bayes_interval_prob)
    metrics_ml = compute_metrics_point_interval(observed=observed, pred=ml_std, method_label="ML", interval_prob=ml_interval_prob)
    metrics_by_group = pd.concat([metrics_bayes, metrics_ml], ignore_index=True)

    # --- CRPS (optional) ---
    have_crps = False
    if not args.skip_crps:
        if args.bayes_draws:
            bayes_draws_path = Path(args.bayes_draws).resolve()
        else:
            cand = repo / "out" / f"annual_load_draws_bayes_{tag}.csv" if tag else None
            bayes_draws_path = cand if (cand is not None and cand.exists()) else (repo / "out" / "annual_load_draws_bayes_v1p6.csv")

        ml_draws_path = Path(args.ml_draws).resolve() if args.ml_draws else (repo / "out" / "ml_catboost_conformal_loyo" / "annual_load_draws.csv")

        if bayes_draws_path.exists() and ml_draws_path.exists():
            bayes_draws_raw = pd.read_csv(bayes_draws_path)
            bayes_draws, _ = standardize_draws(bayes_draws_raw, label="Bayes", input_units=args.bayes_draws_units)

            ml_draws_raw = pd.read_csv(ml_draws_path)
            ml_draws, _ = standardize_draws(ml_draws_raw, label="ML", input_units=args.ml_draws_units)

            crps_bayes = compute_crps_table(observed, bayes_draws, "Bayes", args.crps_max_draws, args.crps_seed, "[INFO]")
            crps_ml = compute_crps_table(observed, ml_draws, "ML", args.crps_max_draws, args.crps_seed, "[INFO]")
            crps = pd.concat([crps_bayes, crps_ml], ignore_index=True)

            # Avoid suffixing problems: drop any placeholder CRPS columns before merge.
            for c in ["CRPS", "CRPS_norm_mean"]:
                if c in metrics_by_group.columns:
                    metrics_by_group = metrics_by_group.drop(columns=[c])

            metrics_by_group = metrics_by_group.merge(
                crps.loc[:, ["Analyte", "Treatment", "method", "CRPS"]],
                on=["Analyte", "Treatment", "method"],
                how="left",
            )
            have_crps = True
        else:
            print("[WARN] CRPS skipped because draws file(s) missing.")
            if not bayes_draws_path.exists():
                print(f"       - missing Bayes draws: {bayes_draws_path}")
            if not ml_draws_path.exists():
                print(f"       - missing ML draws   : {ml_draws_path}")

    # Scale for CRPS normalization (always compute, cheap)
    obs_scale = (
        observed.loc[:, ["Analyte", "Treatment", "center_mg"]]
        .assign(mean_abs_obs=lambda d: d["center_mg"].abs())
        .groupby(["Analyte", "Treatment"], dropna=False)["mean_abs_obs"]
        .mean()
        .reset_index()
    )
    metrics_by_group = metrics_by_group.merge(obs_scale, on=["Analyte", "Treatment"], how="left")

    if have_crps and "CRPS" in metrics_by_group.columns:
        metrics_by_group["CRPS_norm_mean"] = np.where(
            np.isfinite(metrics_by_group["CRPS"]) & np.isfinite(metrics_by_group["mean_abs_obs"]) & (metrics_by_group["mean_abs_obs"] > 0),
            metrics_by_group["CRPS"] / metrics_by_group["mean_abs_obs"],
            np.nan,
        )
    else:
        metrics_by_group["CRPS"] = np.nan
        metrics_by_group["CRPS_norm_mean"] = np.nan

    # --- aggregate + write metrics ---
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

    # --- annual volume plot + metrics ---
    if not args.skip_volume:
        volume_ready = (
            bayes_volume_observed_path is not None
            and bayes_volume_modeled_path is not None
            and bayes_volume_observed_path.exists()
            and bayes_volume_modeled_path.exists()
        )

        if not volume_ready:
            print("[WARN] Annual volume comparison skipped because Bayes volume CSV(s) were not found.")
            if bayes_volume_observed_path is None or not (bayes_volume_observed_path and bayes_volume_observed_path.exists()):
                print(f"       - missing observed volume: {bayes_volume_observed_path}")
            if bayes_volume_modeled_path is None or not (bayes_volume_modeled_path and bayes_volume_modeled_path.exists()):
                print(f"       - missing modeled volume : {bayes_volume_modeled_path}")
        else:
            volume_observed = standardize_volume_wide(
                pd.read_csv(bayes_volume_observed_path),
                source_name="Observed",
                series_label="Observed",
                source_tag="obs",
                input_units=args.bayes_volume_input_units,
                treatments=use_trts,
            )
            volume_bayes = standardize_volume_wide(
                pd.read_csv(bayes_volume_modeled_path),
                source_name="Bayes",
                series_label="Bayes modeled",
                source_tag="mod",
                input_units=args.bayes_volume_input_units,
                treatments=use_trts,
            )
            volume_ml = standardize_ml_volume_summary(
                ml_raw,
                source_name="ML",
                series_label="ML modeled",
                input_units=args.ml_volume_input_units,
            )

            if volume_ml.empty:
                if ml_volume_events_path.exists():
                    print("[INFO] ML annual summary has no volume_* fields; deriving volume from ML event-level imputed CSV.")
                    volume_ml = standardize_ml_volume_from_events(
                        pd.read_csv(ml_volume_events_path),
                        source_name="ML",
                        series_label="ML modeled",
                        input_units=args.ml_volume_input_units,
                    )
                else:
                    print("[WARN] Annual volume comparison skipped because ML volume data were not found.")
                    print(f"       - missing ML volume events: {ml_volume_events_path}")

            if not volume_ml.empty:
                volume_metrics_bayes = compute_metrics_point_interval(
                    observed=volume_observed,
                    pred=volume_bayes,
                    method_label="Bayes",
                    interval_prob=bayes_interval_prob,
                )
                volume_metrics_ml = compute_metrics_point_interval(
                    observed=volume_observed,
                    pred=volume_ml,
                    method_label="ML",
                    interval_prob=ml_interval_prob,
                )
                volume_metrics_by_treatment = pd.concat([volume_metrics_bayes, volume_metrics_ml], ignore_index=True)
                if volume_metrics_by_treatment.empty:
                    print("[WARN] Annual volume metrics skipped because no observed/modelled years overlapped.")
                else:
                    volume_metrics_by_treatment = volume_metrics_by_treatment.sort_values(["Treatment", "method"])
                    volume_metrics_overall_all = aggregate_overall(volume_metrics_by_treatment)
                    volume_metrics_overall = volume_metrics_overall_all.loc[
                        volume_metrics_overall_all["Analyte"].eq("Volume")
                        & volume_metrics_overall_all["Treatment"].eq("ALL")
                    ].copy()

                    volume_metrics_by_treatment.to_csv(metrics_outdir / "volume_metrics_by_treatment.csv", index=False)
                    volume_metrics_overall.to_csv(metrics_outdir / "volume_metrics_overall.csv", index=False)

                    print("[OK] Volume metrics written:")
                    print(f"     - {metrics_outdir / 'volume_metrics_by_treatment.csv'}")
                    print(f"     - {metrics_outdir / 'volume_metrics_overall.csv'}")

                    if not args.skip_plots:
                        volume_out_jpg = figs_outdir / "annual_volume_bayes_vs_ml_faceted.jpg"
                        plot_volume_faceted(
                            volume_bayes,
                            volume_ml,
                            volume_observed,
                            volume_out_jpg,
                            units=args.volume_units,
                            treatments=use_trts[:3] if len(use_trts) >= 3 else use_trts,
                        )
                        print(f"[OK] Volume JPG figure written: {volume_out_jpg}")

    # --- annual plots ---
    if not args.skip_plots:
        for an in analytes:
            safe_an = re.sub(r"[^A-Za-z0-9]+", "_", an.strip().lower()).strip("_")
            out_jpg = figs_outdir / f"annual_load_{safe_an}_bayes_vs_ml_faceted.jpg"
            plot_analyte_faceted(
                an,
                bayes_modeled,
                ml_std,
                observed,
                out_jpg,
                units=args.units,
                treatments=use_trts[:3] if len(use_trts) >= 3 else use_trts,
            )
        print(f"[OK] Wrote {len(analytes)} annual JPG figures to: {figs_outdir}")

    # --- GoF plots ---
    if not args.skip_gof_plots:
        d_all = _prep_by_analyte_all(metrics_by_analyte_overall)

        plot_paired_bars_by_analyte(
            d_all=d_all,
            metric="NRMSE_mean",
            outpath=gof_outdir / "gof_nrmse_mean_by_analyte.jpg",
            title="NRMSE (normalized by mean |Observed|) by analyte",
            ylabel="RMSE / mean(|Observed|)",
        )

        plot_paired_bars_by_analyte(
            d_all=d_all,
            metric="CRPS_norm_mean",
            outpath=gof_outdir / "gof_crps_norm_mean_by_analyte.jpg",
            title="CRPS (mean-normalized) by analyte",
            ylabel="CRPS / mean(|Observed|)",
        )

        plot_coverage_vs_width(
            d_all=d_all,
            outpath=gof_outdir / "gof_coverage_vs_width.jpg",
            nominal=0.95,
        )

        print(f"[OK] Wrote GoF JPG figures to: {gof_outdir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
