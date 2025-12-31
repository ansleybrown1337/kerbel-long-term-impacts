#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate ML outputs (row-level predictions + annual summaries) from *saved* CatBoost models
and *saved* conformal quantiles (q_conformal) stored in model meta JSON.

Goals:
  - No LOYO cross-validation
  - No recalibration (reuse q_conformal from meta JSON)
  - No refitting (load .cbm models)
  - Overwrite existing annual summaries with Bayes-aligned schema:
      out/ml_catboost_conformal_loyo/annual_load_summary.csv
      out/ml_catboost_conformal_loyo/annual_load_summary_imputed.csv

Assumptions (consistent with existing pipeline):
  - Targets are modeled on log1p scale ("logC" for concentration mg/L, "logV" for volume L).
  - Conformal interval is symmetric in log1p space using absolute-error quantile q_conformal:
        log1p(y) ∈ [pred - q, pred + q]
  - Uncertainty propagation for annual metrics uses log-uniform sampling between PI bounds
    in log1p space (mirrors existing imputed logic).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:  # pragma: no cover
    raise RuntimeError("catboost must be installed in your wq_ml environment.") from e


# -----------------------------
# Helpers
# -----------------------------
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _sample_log_uniform(lo_log: np.ndarray, hi_log: np.ndarray, D: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample U ~ Uniform(lo_log, hi_log) row-wise for D draws.
    lo_log, hi_log are 1D arrays length n.
    Returns array shape (n, D).
    """
    u = rng.random((lo_log.shape[0], D))
    return lo_log[:, None] + (hi_log - lo_log)[:, None] * u


def _load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_cb_model(model_path: Path) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(str(model_path))
    return m


def _prepare_X(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required feature columns: {missing}")

    X = df.loc[:, feature_cols].copy()

    # Coerce categoricals to string (CatBoost handles missing)
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")
    # Coerce numeric where possible (leave categoricals as string)
    for c in X.columns:
        if c not in cat_cols:
            X[c] = safe_numeric(X[c])
    return X



def _prepare_X_for_catboost(
    X: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, list[int]]:
    """
    Ensure CatBoost receives the same feature set/order as training, and that
    missing values won't be pandas.NA (which CatBoost can't parse).
    """
    Xp = X.reindex(columns=feature_cols).copy()

    # Convert pandas.NA -> np.nan everywhere first
    Xp = Xp.replace({pd.NA: np.nan})

    cat_set = set(cat_cols)
    cat_idx: list[int] = []
    for j, col in enumerate(feature_cols):
        if col in cat_set:
            cat_idx.append(j)
            # CatBoost expects categorical values as strings/ints; pandas.NA breaks it.
            # Use a sentinel string for missing.
            Xp[col] = Xp[col].astype("string").fillna("__NA__").astype(str)
        else:
            # Numeric: ensure float; keep np.nan (CatBoost handles it)
            Xp[col] = pd.to_numeric(Xp[col], errors="coerce")

    return Xp, cat_idx

def _predict_with_pi(
    model: CatBoostRegressor,
    X: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    q_conformal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict on log1p scale, then compute PI bounds in original space.

    Returns:
      y_hat (original space),
      y_lo (original space),
      y_hi (original space)
    """
    Xp, cat_idx = _prepare_X_for_catboost(X, feature_cols, cat_cols)
    pool = Pool(Xp, cat_features=cat_idx)
    pred_log = model.predict(pool)  # log1p scale

    lo_log = pred_log - float(q_conformal)
    hi_log = pred_log + float(q_conformal)

    y_hat = np.expm1(pred_log)
    y_lo = np.expm1(lo_log)
    y_hi = np.expm1(hi_log)

    # Safety: keep non-negative
    y_hat = np.clip(y_hat, 0.0, np.inf)
    y_lo = np.clip(y_lo, 0.0, np.inf)
    y_hi = np.clip(y_hi, 0.0, np.inf)

    return y_hat, y_lo, y_hi

def _annual_summaries_from_row_pis(df: pd.DataFrame,
                                  year_col: str = "Year",
                                  trt_col: str = "Treatment",
                                  an_col: str = "Analyte",
                                  C_hat_col: str = "Result_mg_L_filled",
                                  V_hat_col: str = "Volume_filled",
                                  C_lo_col: str = "Result_mg_L_pi_low",
                                  C_hi_col: str = "Result_mg_L_pi_high",
                                  V_lo_col: str = "Volume_pi_low",
                                  V_hi_col: str = "Volume_pi_high",
                                  draws: int = 2000,
                                  alpha: float = 0.05,
                                  seed: int = 123) -> pd.DataFrame:
    """
    Build annual distributions for volume (L), load (g), and flow-weighted concentration (mg/L)
    from row-level PI bounds by Monte Carlo sampling (log-uniform between PI bounds in log1p space).

    Returns Bayes-aligned schema (ML modeled-only), with analyte/treatment lowercase column names.
    """
    df = df.dropna(subset=[year_col, trt_col, an_col, C_hat_col, V_hat_col]).copy()
    df[year_col] = safe_numeric(df[year_col]).astype(int)

    # Ensure PI cols exist; if absent, degenerate at point
    for c in [C_lo_col, C_hi_col, V_lo_col, V_hi_col]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = safe_numeric(df[c])

    rng = np.random.default_rng(seed)
    D = int(draws)

    q_lo = float(alpha) / 2.0
    q_hi = 1.0 - float(alpha) / 2.0

    rows = []
    for (yr, trt, an), g in df.groupby([year_col, trt_col, an_col]):
        # point (deterministic)
        C_hat = g[C_hat_col].to_numpy(dtype=float)
        V_hat = g[V_hat_col].to_numpy(dtype=float)

        # Bounds (fallback to point if missing)
        loC = g[C_lo_col].to_numpy(dtype=float)
        hiC = g[C_hi_col].to_numpy(dtype=float)
        loV = g[V_lo_col].to_numpy(dtype=float)
        hiV = g[V_hi_col].to_numpy(dtype=float)

        # Replace invalid bounds with point
        okC = np.isfinite(loC) & np.isfinite(hiC) & (loC >= 0) & (hiC >= loC)
        okV = np.isfinite(loV) & np.isfinite(hiV) & (loV >= 0) & (hiV >= loV)

        # Build draws
        if D <= 0:
            V_sum = float(np.sum(V_hat))
            L_sum_mg = float(np.sum(C_hat * V_hat))
            conc_fw = (L_sum_mg / V_sum) if V_sum > 0 else np.nan

            rows.append({
                "Year": int(yr),
                "treatment": str(trt),
                "analyte": str(an),
                "n_events": int(len(g)),
                "n_rows": int(len(g)),
                "volume_mean": V_sum,
                "volume_low": V_sum,
                "volume_high": V_sum,
                "conc_mean": conc_fw,
                "conc_low": conc_fw,
                "conc_high": conc_fw,
                "load_mean": L_sum_mg * 1e-3,   # mg -> g
                "load_low":  L_sum_mg * 1e-3,
                "load_high": L_sum_mg * 1e-3,
                "n_draws": 0
            })
            continue

        # Start with degenerate draws at point
        C_draw = np.repeat(C_hat[:, None], D, axis=1)
        V_draw = np.repeat(V_hat[:, None], D, axis=1)

        if okC.any():
            C_draw_log = _sample_log_uniform(np.log1p(loC[okC]), np.log1p(hiC[okC]), D, rng)
            C_draw[okC, :] = np.expm1(C_draw_log)

        if okV.any():
            V_draw_log = _sample_log_uniform(np.log1p(loV[okV]), np.log1p(hiV[okV]), D, rng)
            V_draw[okV, :] = np.expm1(V_draw_log)

        V_sum = V_draw.sum(axis=0)  # (D,)
        L_sum_mg = (C_draw * V_draw).sum(axis=0)  # (D,)
        with np.errstate(divide="ignore", invalid="ignore"):
            conc_fw = np.where(V_sum > 0, L_sum_mg / V_sum, np.nan)

        rows.append({
            "Year": int(yr),
            "treatment": str(trt),
            "analyte": str(an),
            "n_events": int(len(g)),
            "n_rows": int(len(g)),
            "volume_mean": float(np.mean(V_sum)),
            "volume_low": float(np.quantile(V_sum, q_lo)),
            "volume_high": float(np.quantile(V_sum, q_hi)),
            "conc_mean": float(np.nanmean(conc_fw)),
            "conc_low": float(np.nanquantile(conc_fw, q_lo)),
            "conc_high": float(np.nanquantile(conc_fw, q_hi)),
            "load_mean": float(np.mean(L_sum_mg)) * 1e-3,   # mg -> g
            "load_low": float(np.quantile(L_sum_mg, q_lo)) * 1e-3,
            "load_high": float(np.quantile(L_sum_mg, q_hi)) * 1e-3,
            "n_draws": int(D),
        })

    out = pd.DataFrame(rows)
    return out


def _to_bayes_schema(annual: pd.DataFrame,
                     interval_prob: float,
                     model_version: str,
                     source: str) -> pd.DataFrame:
    # Ensure required columns exist
    required = ["Year","analyte","treatment","n_events","n_rows",
                "volume_mean","volume_low","volume_high",
                "conc_mean","conc_low","conc_high",
                "load_mean","load_low","load_high"]
    missing = [c for c in required if c not in annual.columns]
    if missing:
        raise ValueError(f"Annual summary missing expected columns: {missing}")

    out = annual.copy()
    out.insert(0, "interval_prob", float(interval_prob))
    out.insert(0, "source", str(source))
    out.insert(0, "model_version", str(model_version))

    # Order to match Bayes output
    col_order = [
        "model_version","source","interval_prob",
        "Year","analyte","treatment","n_events","n_rows",
        "volume_mean","volume_low","volume_high",
        "conc_mean","conc_low","conc_high",
        "load_mean","load_low","load_high"
    ]
    out = out.loc[:, col_order]
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="Repo root (default: .)")
    ap.add_argument("--out_subdir", default=r"out\ml_catboost_conformal_loyo", help="Relative outputs dir (default: out\\ml_catboost_conformal_loyo)")
    ap.add_argument("--models_subdir", default=r"out\ml_catboost_conformal_loyo\models", help="Relative models dir (default: out\\ml_catboost_conformal_loyo\\models)")
    ap.add_argument("--data_csv", default=r"out\ml_catboost_conformal_loyo\wq_cleaned_ml_imputed.csv",
                    help="Input data CSV to regenerate from (default: out\\ml_catboost_conformal_loyo\\wq_cleaned_ml_imputed.csv)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for intervals (default 0.05 -> 95%%)")
    ap.add_argument("--draws", type=int, default=2000, help="MC draws per annual group (default 2000)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for draws")
    ap.add_argument("--overwrite", action="store_true", help="Actually overwrite annual_load_summary*.csv (recommended).")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    outdir = (repo / Path(args.out_subdir)).resolve()
    models_dir = (repo / Path(args.models_subdir)).resolve()
    data_csv = (repo / Path(args.data_csv)).resolve()

    if not data_csv.exists():
        raise FileNotFoundError(f"Input data CSV not found: {data_csv}")

    # Load models + meta
    mC_path = models_dir / "model_logC.cbm"
    mV_path = models_dir / "model_logV.cbm"
    metaC_path = models_dir / "model_logC_meta.json"
    metaV_path = models_dir / "model_logV_meta.json"

    for p in [mC_path, mV_path, metaC_path, metaV_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required model artifact missing: {p}")

    metaC = _load_meta(metaC_path)
    metaV = _load_meta(metaV_path)

    modelC = _load_cb_model(mC_path)
    modelV = _load_cb_model(mV_path)

    feature_cols = metaC["feature_cols"]
    cat_cols = metaC.get("cat_cols", [])
    qC = float(metaC["q_conformal"])
    qV = float(metaV["q_conformal"])

    # Read data
    df = pd.read_csv(data_csv, na_values=["NA","NaN","nan",""], keep_default_na=True)

    # Exclude NoRunoff if present (consistent with other scripts)
    if "NoRunoff" in df.columns:
        df = df.loc[~df["NoRunoff"].fillna(False)].copy()

    # Create Year from Date if absent
    if "Year" not in df.columns:
        if "Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
        else:
            raise ValueError("Input data must contain Year or Date.")

    # Predict conc + vol with PI
    X = _prepare_X(df, feature_cols, cat_cols)

    C_hat, C_lo, C_hi = _predict_with_pi(modelC, X, feature_cols, cat_cols, qC)
    V_hat, V_lo, V_hi = _predict_with_pi(modelV, X, feature_cols, cat_cols, qV)

    # Attach predictions (filled)
    df["Result_mg_L_filled"] = C_hat
    df["Result_mg_L_pi_low"] = C_lo
    df["Result_mg_L_pi_high"] = C_hi

    df["Volume_filled"] = V_hat
    df["Volume_pi_low"] = V_lo
    df["Volume_pi_high"] = V_hi

    interval_prob = 1.0 - float(args.alpha)
    model_version = "ml_catboost_saved_models"
    source = "ML_Modeled"

    # Modeled-only: rows where BOTH observed C and V exist (matches your prior split logic)
    modeled_mask = df.get("Result_mg_L").notna() & df.get("Volume").notna() if ("Result_mg_L" in df.columns and "Volume" in df.columns) else pd.Series([True]*len(df))
    df_modeled = df.loc[modeled_mask].copy()
    annual_modeled = _annual_summaries_from_row_pis(df_modeled,
                                                    draws=args.draws,
                                                    alpha=args.alpha,
                                                    seed=args.seed)
    bayes_modeled = _to_bayes_schema(annual_modeled, interval_prob, model_version, source)

    # Imputed-inclusive: all rows that have filled values (df already filtered)
    annual_imp = _annual_summaries_from_row_pis(df,
                                                draws=args.draws,
                                                alpha=args.alpha,
                                                seed=args.seed)
    bayes_imp = _to_bayes_schema(annual_imp, interval_prob, model_version, source)

    outdir.mkdir(parents=True, exist_ok=True)
    out_modeled_csv = outdir / "annual_load_summary.csv"
    out_imputed_csv = outdir / "annual_load_summary_imputed.csv"

    if args.overwrite:
        bayes_modeled.to_csv(out_modeled_csv, index=False)
        bayes_imp.to_csv(out_imputed_csv, index=False)
        print(f"[INFO] Overwrote: {out_modeled_csv}")
        print(f"[INFO] Overwrote: {out_imputed_csv}")
    else:
        # Dry run safeguard
        print("[DRY RUN] Not overwriting files because --overwrite was not set.")
        print(f"[DRY RUN] Would write: {out_modeled_csv}")
        print(f"[DRY RUN] Would write: {out_imputed_csv}")

    # Also update row-level predictions if you want them refreshed for plotting
    pred_out = outdir / "predictions_from_saved_models.csv"
    df.to_csv(pred_out, index=False)
    print(f"[INFO] Wrote row-level predictions snapshot: {pred_out}")


if __name__ == "__main__":
    main()
