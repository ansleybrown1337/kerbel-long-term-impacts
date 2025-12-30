#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_catboost_conformal_loyo.py

Leave-One-Year-Out (LOYO) CV using CatBoost + *manual split conformal* prediction
to model:
  1) OUT concentration: Result_mg_L  (modeled as log1p)
  2) OUT runoff volume: Volume (L)  (modeled as log1p)

Then propagate uncertainty to annual loads via Monte Carlo:
  AnnualLoad_mg = sum_i (C_i * V_i), where C is mg/L and V is L.

Why "manual split conformal"?
-----------------------------
MAPIE's API has changed across versions, and your pinned installation differs in
SplitConformalRegressor.fit signatures. To avoid brittle imports and ensure
reproducibility, this script implements *split conformal regression* directly:

  - Split training fold into proper-train and calibration subsets.
  - Fit CatBoost on proper-train.
  - Compute absolute residuals on calibration.
  - Let q = (1 - alpha) quantile of residuals.
  - Prediction interval on test: [y_hat - q, y_hat + q] on the log scale.

This is the standard split-conformal interval for regression with symmetric
nonconformity score |y - y_hat|.

Repo layout assumed
-------------------
kerbel-long-term-impacts/
  code/
  out/wq_cleaned.csv
  figs/
  out/

Outputs
-------
out/ml_catboost_conformal_loyo/
  cv_metrics_by_year.csv
  cv_predictions_samplelevel.csv
  annual_load_draws.csv
  annual_load_summary.csv

figs/ml_catboost_conformal_loyo/
  cv_rmse_by_year.png
"""

from __future__ import annotations

import argparse
import json
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Repo utilities
# -----------------------------

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


# -----------------------------
# Helpers
# -----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_log1p_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x >= 0, np.nan)
    return np.log1p(x)

def safe_expm1(arr: np.ndarray) -> np.ndarray:
    return np.expm1(arr)

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("Date", "PlantDate", "HarvestDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["DayOfYear"] = df["Date"].dt.dayofyear
    else:
        df["DayOfYear"] = np.nan

    if "Date" in df.columns and "PlantDate" in df.columns:
        df["DaysSincePlant"] = (df["Date"] - df["PlantDate"]).dt.days
    else:
        df["DaysSincePlant"] = np.nan

    if "Date" in df.columns and "HarvestDate" in df.columns:
        df["DaysUntilHarvest"] = (df["HarvestDate"] - df["Date"]).dt.days
    else:
        df["DaysUntilHarvest"] = np.nan

    return df


def coerce_bool(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        s = df[col].astype(str).str.strip().str.lower()
        df[col] = s.isin(["true", "t", "1", "yes", "y"])
    return df

def ensure_treatment(df: pd.DataFrame) -> pd.DataFrame:
    if "Treatment" not in df.columns and "System" in df.columns:
        df["Treatment"] = df["System"]
    return df

def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    desired = [
        "Year", "Irrigation", "Rep",
        "Treatment", "Analyte",
        "Inflow_Result_mg_L",
        "Inflow_Volume",          # <-- ADD THIS
        "Flag", "Inflow_Flag",
        "FlumeMethod", "MeasureMethod", "IrrMethod", "TSSMethod",
        "Lab",
        "SeasonYear", "Crop",
        "CumAll_STIR_toDate", "Season_STIR_toDate",
        "DaysSincePlant", "DaysUntilHarvest",
        "DayOfYear",
    ]
    cols = [c for c in desired if c in df.columns]
    X = df[cols].copy()

    cat_cols = [c for c in [
        "Treatment", "Analyte", "Flag", "Inflow_Flag",
        "FlumeMethod", "MeasureMethod", "IrrMethod", "TSSMethod",
        "Lab", "Crop"
    ] if c in X.columns]

    if "SeasonYear" in X.columns:
        frac_numeric = pd.to_numeric(X["SeasonYear"], errors="coerce").notna().mean()
        if frac_numeric < 0.95 and "SeasonYear" not in cat_cols:
            cat_cols.append("SeasonYear")

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, cat_cols




def save_model_artifacts(
    model: CatBoostRegressor,
    q: float,
    feature_cols: list[str],
    cat_cols: list[str],
    model_path: Path,
    meta_path: Path,
    target_name: str,
    cb_params: dict,
    alpha: float,
) -> None:
    """Persist a fitted CatBoost model and conformal metadata needed to reproduce PIs."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    meta = {
        "target": target_name,
        "alpha": float(alpha),
        "q_conformal": float(q),
        "feature_cols": list(feature_cols),
        "cat_cols": list(cat_cols),
        "cb_params": dict(cb_params),
        "saved_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model_artifacts(model_path: Path, meta_path: Path) -> tuple[CatBoostRegressor, dict]:
    """Load a saved CatBoost model and its conformal metadata."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing saved metadata: {meta_path}")

    cb = CatBoostRegressor()
    cb.load_model(str(model_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return cb, meta


def refit_full_and_impute(
    df_model: pd.DataFrame,
    X_all: pd.DataFrame,
    cat_cols: list[str],
    cb_params: dict,
    alpha: float,
    outdir: Path,
    model_dir: Path,
    save_models: bool,
    impute_draws: int,
    seed: int,
) -> None:
    """Refit models on all observed rows and impute missing y_logC/y_logV rows, saving outputs."""
    rng = np.random.default_rng(seed)

    # Ensure identifier
    out_imp = df_model.copy()
    if "_wq_idx" not in out_imp.columns:
        out_imp["_wq_idx"] = out_imp.index.astype(int)

    # Targets and missing masks
    missC = out_imp["y_logC"].isna()
    missV = out_imp["y_logV"].isna()
    obsC = ~missC
    obsV = ~missV

    # Paths
    model_dir.mkdir(parents=True, exist_ok=True)
    modelC_path = model_dir / "model_logC.cbm"
    metaC_path  = model_dir / "model_logC_meta.json"
    modelV_path = model_dir / "model_logV.cbm"
    metaV_path  = model_dir / "model_logV_meta.json"

    # Concentration
    if missC.any() and obsC.any():
        cbC, qC = fit_catboost_split_conformal(
            X_train=X_all.loc[obsC].copy(),
            y_train=out_imp.loc[obsC, "y_logC"],
            cat_cols=cat_cols,
            alpha=alpha,
            random_state=seed,
            cb_params=cb_params,
            calib_size=0.25,  # matches default; actual calib_size for refit is not used if you later run impute_only
        )
        if save_models:
            save_model_artifacts(
                model=cbC, q=qC, feature_cols=list(X_all.columns), cat_cols=cat_cols,
                model_path=modelC_path, meta_path=metaC_path,
                target_name="logC", cb_params=cb_params, alpha=alpha,
            )

        pred_log, lo_log, hi_log = predict_with_pi_manual(cbC, qC, X_all.loc[missC].copy())
        out_imp["Result_mg_L_pred"] = np.nan
        out_imp.loc[missC, "Result_mg_L_pred"] = safe_expm1(pred_log)
        out_imp["Result_mg_L_pi_low"] = np.nan
        out_imp.loc[missC, "Result_mg_L_pi_low"] = safe_expm1(lo_log)
        out_imp["Result_mg_L_pi_high"] = np.nan
        out_imp.loc[missC, "Result_mg_L_pi_high"] = safe_expm1(hi_log)
        out_imp["q_conformal_logC"] = np.nan
        out_imp.loc[missC, "q_conformal_logC"] = float(qC)
    else:
        # still save observed-only placeholders for consistency
        out_imp["Result_mg_L_pred"] = np.nan
        out_imp["Result_mg_L_pi_low"] = np.nan
        out_imp["Result_mg_L_pi_high"] = np.nan
        out_imp["q_conformal_logC"] = np.nan

    # Volume
    if missV.any() and obsV.any():
        cbV, qV = fit_catboost_split_conformal(
            X_train=X_all.loc[obsV].copy(),
            y_train=out_imp.loc[obsV, "y_logV"],
            cat_cols=cat_cols,
            alpha=alpha,
            random_state=seed,
            cb_params=cb_params,
            calib_size=0.25,
        )
        if save_models:
            save_model_artifacts(
                model=cbV, q=qV, feature_cols=list(X_all.columns), cat_cols=cat_cols,
                model_path=modelV_path, meta_path=metaV_path,
                target_name="logV", cb_params=cb_params, alpha=alpha,
            )

        pred_log, lo_log, hi_log = predict_with_pi_manual(cbV, qV, X_all.loc[missV].copy())
        out_imp["Volume_pred"] = np.nan
        out_imp.loc[missV, "Volume_pred"] = safe_expm1(pred_log)
        out_imp["Volume_pi_low"] = np.nan
        out_imp.loc[missV, "Volume_pi_low"] = safe_expm1(lo_log)
        out_imp["Volume_pi_high"] = np.nan
        out_imp.loc[missV, "Volume_pi_high"] = safe_expm1(hi_log)
        out_imp["q_conformal_logV"] = np.nan
        out_imp.loc[missV, "q_conformal_logV"] = float(qV)
    else:
        out_imp["Volume_pred"] = np.nan
        out_imp["Volume_pi_low"] = np.nan
        out_imp["Volume_pi_high"] = np.nan
        out_imp["q_conformal_logV"] = np.nan

    # Filled columns (only replace missing)
    out_imp["Result_mg_L_filled"] = out_imp["Result_mg_L"]
    out_imp.loc[out_imp["Result_mg_L_filled"].isna(), "Result_mg_L_filled"] = out_imp.loc[out_imp["Result_mg_L_filled"].isna(), "Result_mg_L_pred"]

    out_imp["Volume_filled"] = out_imp["Volume"]
    out_imp.loc[out_imp["Volume_filled"].isna(), "Volume_filled"] = out_imp.loc[out_imp["Volume_filled"].isna(), "Volume_pred"]

    imputed_csv = outdir / "wq_cleaned_ml_imputed.csv"
    out_imp.to_csv(imputed_csv, index=False)
    print(f"[INFO] Wrote imputed dataset: {imputed_csv}")

    # Optional per-row draws (only for imputed rows)
    if impute_draws and impute_draws > 0:
        draw_records = []

        if missC.any() and out_imp["Result_mg_L_pi_low"].notna().any():
            dC = out_imp.loc[missC & out_imp["Result_mg_L_pi_low"].notna(), ["_wq_idx", "Result_mg_L_pi_low", "Result_mg_L_pi_high"]].copy()
            lo = np.log1p(dC["Result_mg_L_pi_low"].to_numpy())
            hi = np.log1p(dC["Result_mg_L_pi_high"].to_numpy())
            draws_log = sample_uniform_interval(lo, hi, int(impute_draws), rng)
            draws = safe_expm1(draws_log)
            for i, idx in enumerate(dC["_wq_idx"].to_numpy()):
                draw_records.append({"_wq_idx": int(idx), "Target": "Result_mg_L", "draws": ",".join(map(str, draws[i, :]))})

        if missV.any() and out_imp["Volume_pi_low"].notna().any():
            dV = out_imp.loc[missV & out_imp["Volume_pi_low"].notna(), ["_wq_idx", "Volume_pi_low", "Volume_pi_high"]].copy()
            lo = np.log1p(dV["Volume_pi_low"].to_numpy())
            hi = np.log1p(dV["Volume_pi_high"].to_numpy())
            draws_log = sample_uniform_interval(lo, hi, int(impute_draws), rng)
            draws = safe_expm1(draws_log)
            for i, idx in enumerate(dV["_wq_idx"].to_numpy()):
                draw_records.append({"_wq_idx": int(idx), "Target": "Volume", "draws": ",".join(map(str, draws[i, :]))})

        if draw_records:
            draws_csv = outdir / "imputed_row_draws.csv"
            pd.DataFrame(draw_records).to_csv(draws_csv, index=False)
            print(f"[INFO] Wrote imputed row draws: {draws_csv}")
def fit_catboost_split_conformal(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: list[str],
    alpha: float,
    random_state: int,
    cb_params: dict,
    calib_size: float = 0.25,
):
    """
    Split conformal regression with symmetric absolute residual score on the log scale.

    Returns:
      model: fitted CatBoostRegressor
      q: residual quantile used for intervals (float)
    """
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=calib_size, random_state=random_state
    )

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
    cb = CatBoostRegressor(**cb_params, random_seed=random_state, cat_features=cat_idx)
    cb.fit(X_tr, y_tr)

    cal_pred = cb.predict(X_cal)
    resid = np.abs(y_cal.to_numpy() - np.asarray(cal_pred))
    # split conformal quantile
    q = float(np.quantile(resid[~np.isnan(resid)], 1 - alpha)) if np.isfinite(resid).any() else float("nan")
    return cb, q


def predict_with_pi_manual(cb: CatBoostRegressor, q: float, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pred = np.asarray(cb.predict(X_test))
    lo = y_pred - q
    hi = y_pred + q
    return y_pred, lo, hi


def sample_uniform_interval(lo: np.ndarray, hi: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)
    u = rng.random((len(lo2), n_draws))
    return lo2[:, None] + u * (hi2[:, None] - lo2[:, None])


def plot_cv_metrics(metrics_df: pd.DataFrame, fig_path: Path) -> None:
    if metrics_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for target in ["logC", "logV"]:
        d = metrics_df.loc[metrics_df["Target"] == target].sort_values("Year_Test")
        if d.empty:
            continue
        ax.plot(d["Year_Test"], d["RMSE"], marker="o", label=target)
    ax.set_title("LOYO CV RMSE by Held-out Year (log scale)")
    ax.set_xlabel("Held-out Year")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)




def save_feature_importance(
    fi_df: pd.DataFrame,
    out_csv: Path,
    out_png: Path,
    top_k: int = 25,
    title: str = "Feature importance (mean across LOYO folds)"
) -> None:
    """Save feature importance table and a horizontal bar plot (top_k)."""
    if fi_df is None or fi_df.empty:
        return
    fi_df = fi_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    fi_df.to_csv(out_csv, index=False)

    top = fi_df.head(int(top_k)).copy()
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(top) + 1.5)))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_title(title)
    ax.set_xlabel("Mean importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Path to cleaned WQ×STIR table (default: <repo>/out/wq_cleaned.csv).")
    ap.add_argument("--repo", type=str, default=None,
                    help="Optional repo root path. If omitted, auto-detected.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Miscoverage rate. alpha=0.05 gives ~95% prediction intervals.")
    ap.add_argument("--draws", type=int, default=2000,
                    help="Monte Carlo draws for annual load propagation per group.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for splitting and Monte Carlo.")
    ap.add_argument("--exclude_flagged", action="store_true",
                    help="If set, drops rows with Flag or Inflow_Flag not NA/blank.")
    ap.add_argument("--calib_size", type=float, default=0.25,
                    help="Fraction of training fold reserved for conformal calibration (split conformal).")

    # CatBoost hyperparameters
    ap.add_argument("--cb_iterations", type=int, default=3000,
                    help="CatBoost boosting iterations.")
    ap.add_argument("--cb_lr", type=float, default=0.05,
                    help="CatBoost learning rate.")
    ap.add_argument("--cb_depth", type=int, default=8,
                    help="CatBoost tree depth.")
    ap.add_argument("--cb_l2", type=float, default=6.0,
                    help="CatBoost L2 leaf regularization.")
    ap.add_argument("--cb_subsample", type=float, default=0.8,
                    help="Row subsample fraction (bagging).")
    ap.add_argument("--cb_colsample_bylevel", type=float, default=0.8,
                    help="Feature subsample fraction per tree level.")
    ap.add_argument("--cb_verbose_every", type=int, default=0,
                    help="If >0, print CatBoost progress every N iterations. 0/False disables.")
    ap.add_argument("--threads", type=int, default=-1,
                    help="CatBoost thread_count. -1 uses all cores.")

    # Output folders (optional override)
    ap.add_argument("--out_subdir", type=str, default="ml_catboost_conformal_loyo",
                    help="Subfolder under <repo>/out/ for outputs.")
    ap.add_argument("--fig_subdir", type=str, default="ml_catboost_conformal_loyo",
                    help="Subfolder under <repo>/figs/ for figures.")

    # Model persistence / imputation
    ap.add_argument(
        "--model_subdir",
        type=str,
        default="models",
        help="Subfolder under <repo>/out/<out_subdir>/ to save/load fitted full-data models and metadata."
    )

    ap.add_argument(
        "--impute_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, after LOYO CV refit on all observed data and impute missing "
            "Result_mg_L and/or Volume rows with saved conformal PIs. "
            "Default: True."
        )
    )

    ap.add_argument(
        "--impute_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, skip LOYO and only perform imputation using previously saved "
            "models/metadata in --model_subdir."
        )
    )

    ap.add_argument(
        "--impute_draws",
        type=int,
        default=2000,
        help=(
            "Number of Monte Carlo draws to generate for imputed rows "
            "(uniform within PI in log space). "
            "Set to 0 to disable. Default: 2000."
        )
    )

    ap.add_argument(
        "--save_models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, save the refit full-data models (CatBoost .cbm) and "
            "conformal metadata for reuse. Default: True."
        )
    )

    # Feature importance outputs
    ap.add_argument("--feature_importance", action="store_true",
                    help="If set, compute and save mean feature importance across LOYO folds for each target.")
    ap.add_argument("--fi_topk", type=int, default=25,
                    help="Top-K features to show in the feature-importance bar plot.")

    # Convenience mode
    ap.add_argument("--fast", action="store_true",
                    help="Enable FAST mode (fewer iterations & draws).")
    ap.add_argument("--fast_iterations", type=int, default=600,
                    help="Iterations used when --fast is set.")
    ap.add_argument("--fast_draws", type=int, default=300,
                    help="MC draws used when --fast is set.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else find_repo_root(Path.cwd())
    data_path = Path(args.data).resolve() if args.data else (repo / "out" / "wq_cleaned.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    outdir = repo / "out" / str(args.out_subdir)
    figdir = repo / "figs" / str(args.fig_subdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root : {repo}")
    print(f"[INFO] Data      : {data_path}")
    print(f"[INFO] Out dir   : {outdir}")
    print(f"[INFO] Fig dir   : {figdir}")

    df = pd.read_csv(data_path, na_values=['NA','NaN','nan',''], keep_default_na=True)
    df = parse_dates(df)
    df = add_time_features(df)
    df = ensure_treatment(df)
    df = coerce_bool(df, "NoRunoff")
    df = coerce_bool(df, "Has_Inflow")


    if "Year" not in df.columns:
        if "Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
        else:
            raise ValueError("Need Year or Date column.")

    if "Result_mg_L" not in df.columns:
        raise ValueError("Missing Result_mg_L")
    if "Volume" not in df.columns:
        raise ValueError("Missing Volume")

    df["y_logC"] = safe_log1p_series(df["Result_mg_L"])
    df["y_logV"] = safe_log1p_series(df["Volume"])

    if "NoRunoff" in df.columns:
        df_model = df.loc[~df["NoRunoff"].fillna(False)].copy()
    else:
        df_model = df.copy()

    if args.exclude_flagged:
        for fc in ["Flag", "Inflow_Flag"]:
            if fc in df_model.columns:
                s = df_model[fc].astype(str).str.strip()
                df_model = df_model.loc[(s.eq("")) | (s.eq("NA")) | (s.eq("nan"))].copy()

    X_all, cat_cols = build_feature_frame(df_model)

    years = sorted(pd.Series(df_model["Year"]).dropna().unique().tolist())
    if len(years) < 2:
        raise ValueError("LOYO requires at least two years.")

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------
    # Explicit control flow
    #   - impute_only: skip LOYO, load saved models, impute, then return
    #   - otherwise: run LOYO, then (optionally) refit+impute at the end
    # ------------------------------------------------------------
    if args.impute_only:
        if args.impute_missing:
            print("[INFO] --impute_only set: skipping LOYO and using saved models/metadata for imputation.")
        else:
            # This case can happen if someone runs: --impute_only --no-impute-missing
            # Make it explicit rather than silently doing nothing.
            raise ValueError("--impute_only requires imputation to be enabled. Remove --no-impute-missing.")

        model_dir = outdir / str(args.model_subdir)
        modelC_path = model_dir / "model_logC.cbm"
        metaC_path  = model_dir / "model_logC_meta.json"
        modelV_path = model_dir / "model_logV.cbm"
        metaV_path  = model_dir / "model_logV_meta.json"

        for p in [modelC_path, metaC_path, modelV_path, metaV_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing saved model artifact: {p}\n"
                    "Run without --impute_only first to train/refit and save models."
                )

        cbC, metaC = load_model_artifacts(modelC_path, metaC_path)
        cbV, metaV = load_model_artifacts(modelV_path, metaV_path)

        # Validate feature columns
        featC = metaC.get("feature_cols", [])
        featV = metaV.get("feature_cols", [])
        if list(X_all.columns) != list(featC):
            raise ValueError(
                "Feature column mismatch for logC model. "
                "This usually means the feature engineering changed since the model was saved. "
                "Re-run without --impute_only to refit and resave the models."
            )
        if list(X_all.columns) != list(featV):
            raise ValueError(
                "Feature column mismatch for logV model. "
                "This usually means the feature engineering changed since the model was saved. "
                "Re-run without --impute_only to refit and resave the models."
            )

        qC = float(metaC["q_conformal"])
        qV = float(metaV["q_conformal"])

        # Predict missing
        missC = df_model["y_logC"].isna()
        missV = df_model["y_logV"].isna()

        out_imp = df_model.copy()
        if "_wq_idx" not in out_imp.columns:
            out_imp["_wq_idx"] = out_imp.index.astype(int)

        if missC.any():
            pred_log = np.asarray(cbC.predict(X_all.loc[missC]))
            lo_log = pred_log - qC
            hi_log = pred_log + qC
            out_imp["Result_mg_L_pred"] = np.nan
            out_imp.loc[missC, "Result_mg_L_pred"] = safe_expm1(pred_log)
            out_imp["Result_mg_L_pi_low"] = np.nan
            out_imp.loc[missC, "Result_mg_L_pi_low"] = safe_expm1(lo_log)
            out_imp["Result_mg_L_pi_high"] = np.nan
            out_imp.loc[missC, "Result_mg_L_pi_high"] = safe_expm1(hi_log)
            out_imp["q_conformal_logC"] = np.nan
            out_imp.loc[missC, "q_conformal_logC"] = qC
        else:
            out_imp["Result_mg_L_pred"] = np.nan
            out_imp["Result_mg_L_pi_low"] = np.nan
            out_imp["Result_mg_L_pi_high"] = np.nan
            out_imp["q_conformal_logC"] = np.nan

        if missV.any():
            pred_log = np.asarray(cbV.predict(X_all.loc[missV]))
            lo_log = pred_log - qV
            hi_log = pred_log + qV
            out_imp["Volume_pred"] = np.nan
            out_imp.loc[missV, "Volume_pred"] = safe_expm1(pred_log)
            out_imp["Volume_pi_low"] = np.nan
            out_imp.loc[missV, "Volume_pi_low"] = safe_expm1(lo_log)
            out_imp["Volume_pi_high"] = np.nan
            out_imp.loc[missV, "Volume_pi_high"] = safe_expm1(hi_log)
            out_imp["q_conformal_logV"] = np.nan
            out_imp.loc[missV, "q_conformal_logV"] = qV
        else:
            out_imp["Volume_pred"] = np.nan
            out_imp["Volume_pi_low"] = np.nan
            out_imp["Volume_pi_high"] = np.nan
            out_imp["q_conformal_logV"] = np.nan

        out_imp["Result_mg_L_filled"] = out_imp["Result_mg_L"]
        out_imp.loc[out_imp["Result_mg_L_filled"].isna(), "Result_mg_L_filled"] = out_imp.loc[
            out_imp["Result_mg_L_filled"].isna(), "Result_mg_L_pred"
        ]

        out_imp["Volume_filled"] = out_imp["Volume"]
        out_imp.loc[out_imp["Volume_filled"].isna(), "Volume_filled"] = out_imp.loc[
            out_imp["Volume_filled"].isna(), "Volume_pred"
        ]

        imputed_csv = outdir / "wq_cleaned_ml_imputed.csv"
        out_imp.to_csv(imputed_csv, index=False)
        print(f"[INFO] Wrote imputed dataset (impute_only): {imputed_csv}")

        # Optional draws
        if args.impute_draws and args.impute_draws > 0:
            rng2 = np.random.default_rng(args.seed + 101)
            draw_records = []
            if missC.any() and out_imp["Result_mg_L_pi_low"].notna().any():
                dC = out_imp.loc[
                    missC & out_imp["Result_mg_L_pi_low"].notna(),
                    ["_wq_idx", "Result_mg_L_pi_low", "Result_mg_L_pi_high"]
                ].copy()
                lo = np.log1p(dC["Result_mg_L_pi_low"].to_numpy())
                hi = np.log1p(dC["Result_mg_L_pi_high"].to_numpy())
                draws_log = sample_uniform_interval(lo, hi, int(args.impute_draws), rng2)
                draws = safe_expm1(draws_log)
                for i, idx in enumerate(dC["_wq_idx"].to_numpy()):
                    draw_records.append({"_wq_idx": int(idx), "Target": "Result_mg_L", "draws": ",".join(map(str, draws[i, :]))})

            if missV.any() and out_imp["Volume_pi_low"].notna().any():
                dV = out_imp.loc[
                    missV & out_imp["Volume_pi_low"].notna(),
                    ["_wq_idx", "Volume_pi_low", "Volume_pi_high"]
                ].copy()
                lo = np.log1p(dV["Volume_pi_low"].to_numpy())
                hi = np.log1p(dV["Volume_pi_high"].to_numpy())
                draws_log = sample_uniform_interval(lo, hi, int(args.impute_draws), rng2)
                draws = safe_expm1(draws_log)
                for i, idx in enumerate(dV["_wq_idx"].to_numpy()):
                    draw_records.append({"_wq_idx": int(idx), "Target": "Volume", "draws": ",".join(map(str, draws[i, :]))})

            if draw_records:
                draws_csv = outdir / "imputed_row_draws.csv"
                pd.DataFrame(draw_records).to_csv(draws_csv, index=False)
                print(f"[INFO] Wrote imputed row draws: {draws_csv}")

        return
    # If we got here, we are doing the full run (LOYO), optionally followed by refit+impute later.


    # FAST mode overrides (keeps CLI stable + documented)
    if args.fast:
        print(f"[INFO] FAST mode enabled: iterations={args.fast_iterations}, draws={args.fast_draws}")
        args.cb_iterations = int(args.fast_iterations)
        args.draws = int(args.fast_draws)

    cb_params = dict(
        loss_function="RMSE",
        iterations=int(args.cb_iterations),
        learning_rate=float(args.cb_lr),
        depth=int(args.cb_depth),
        l2_leaf_reg=float(args.cb_l2),
        subsample=float(args.cb_subsample),
        colsample_bylevel=float(args.cb_colsample_bylevel),
        eval_metric="RMSE",
        thread_count=int(args.threads),
        verbose=(int(args.cb_verbose_every) if int(args.cb_verbose_every) > 0 else False),
    )

    metrics_rows = []
    pred_rows = []
    annual_draw_rows = []

    fi_rows_C = []
    fi_rows_V = []

    pbar = tqdm(years, desc="LOYO folds", unit="year", ncols=110)
    for y_hold in pbar:
        t0 = time.time()

        test_mask = (df_model["Year"] == y_hold)
        train_mask = ~test_mask

        df_train = df_model.loc[train_mask].copy()
        df_test  = df_model.loc[test_mask].copy()

        X_train = X_all.loc[train_mask].copy()
        X_test  = X_all.loc[test_mask].copy()

        yC_train = df_train["y_logC"]
        yV_train = df_train["y_logV"]
        yC_test  = df_test["y_logC"]
        yV_test  = df_test["y_logV"]

        idxC_tr = yC_train.notna()
        idxC_te = yC_test.notna()
        idxV_tr = yV_train.notna()
        idxV_te = yV_test.notna()

        # Fit: concentration (manual split conformal)
        cb_C, qC = fit_catboost_split_conformal(
            X_train.loc[idxC_tr], yC_train.loc[idxC_tr],
            cat_cols=cat_cols, alpha=args.alpha,
            random_state=args.seed + int(y_hold),
            cb_params=cb_params, calib_size=args.calib_size
        )
        predC, loC, hiC = predict_with_pi_manual(cb_C, qC, X_test.loc[idxC_te])

        # Feature importance (per fold)
        try:
            imp = cb_C.get_feature_importance()
            fi_rows_C.append(pd.DataFrame({
                "Year_Test": int(y_hold),
                "feature": list(X_train.columns),
                "importance": imp
            }))
        except Exception:
            pass


        # Fit: volume (manual split conformal)
        cb_V, qV = fit_catboost_split_conformal(
            X_train.loc[idxV_tr], yV_train.loc[idxV_tr],
            cat_cols=cat_cols, alpha=args.alpha,
            random_state=args.seed + 1000 + int(y_hold),
            cb_params=cb_params, calib_size=args.calib_size
        )
        predV, loV, hiV = predict_with_pi_manual(cb_V, qV, X_test.loc[idxV_te])

        try:
            imp = cb_V.get_feature_importance()
            fi_rows_V.append(pd.DataFrame({
                "Year_Test": int(y_hold),
                "feature": list(X_train.columns),
                "importance": imp
            }))
        except Exception:
            pass


        # Metrics (log scale)
        metrics_rows.append({
            "Year_Test": int(y_hold),
            "Target": "logC",
            "n_test": int(idxC_te.sum()),
            "MAE": float(mean_absolute_error(yC_test.loc[idxC_te], predC)) if idxC_te.sum() else np.nan,
            "RMSE": rmse(yC_test.loc[idxC_te].to_numpy(), predC) if idxC_te.sum() else np.nan,
            "R2": float(r2_score(yC_test.loc[idxC_te], predC)) if idxC_te.sum() > 2 else np.nan,
            "q_conformal": qC
        })
        metrics_rows.append({
            "Year_Test": int(y_hold),
            "Target": "logV",
            "n_test": int(idxV_te.sum()),
            "MAE": float(mean_absolute_error(yV_test.loc[idxV_te], predV)) if idxV_te.sum() else np.nan,
            "RMSE": rmse(yV_test.loc[idxV_te].to_numpy(), predV) if idxV_te.sum() else np.nan,
            "R2": float(r2_score(yV_test.loc[idxV_te], predV)) if idxV_te.sum() > 2 else np.nan,
            "q_conformal": qV
        })

        # Sample-level predictions (back-transformed)
        base_cols = [c for c in ["SampleID","Date","Year","Treatment","Analyte","Irrigation","Rep"] if c in df_test.columns]

        dfC = df_test.loc[idxC_te, base_cols].copy()
        dfC["Target"] = "Result_mg_L"
        dfC["y_true"] = safe_expm1(yC_test.loc[idxC_te].to_numpy())
        dfC["y_pred"] = safe_expm1(predC)
        dfC["pi_low"] = safe_expm1(loC)
        dfC["pi_high"] = safe_expm1(hiC)

        dfV = df_test.loc[idxV_te, base_cols].copy()
        dfV["Target"] = "Volume_L"
        dfV["y_true"] = safe_expm1(yV_test.loc[idxV_te].to_numpy())
        dfV["y_pred"] = safe_expm1(predV)
        dfV["pi_low"] = safe_expm1(loV)
        dfV["pi_high"] = safe_expm1(hiV)

        pred_rows.append(dfC)
        pred_rows.append(dfV)

        # Annual load propagation for the TEST YEAR ONLY (CV-valid)
        key_cols = [c for c in ["SampleID","Year","Treatment","Analyte"] if c in df_test.columns]
        C_tab = df_test.loc[idxC_te, key_cols].copy()
        C_tab["loC"] = loC
        C_tab["hiC"] = hiC

        V_tab = df_test.loc[idxV_te, key_cols].copy()
        V_tab["loV"] = loV
        V_tab["hiV"] = hiV

        paired = C_tab.merge(V_tab, on=key_cols, how="inner")

        if not paired.empty:
            C_draw_log = sample_uniform_interval(paired["loC"].to_numpy(), paired["hiC"].to_numpy(), args.draws, rng)
            V_draw_log = sample_uniform_interval(paired["loV"].to_numpy(), paired["hiV"].to_numpy(), args.draws, rng)

            C_draw = safe_expm1(C_draw_log)  # mg/L
            V_draw = safe_expm1(V_draw_log)  # L
            load_draw = C_draw * V_draw      # mg

            g = paired[["Year","Treatment","Analyte"]].copy()
            g["Year"] = pd.to_numeric(g["Year"], errors="coerce").astype("Int64")
            group_id = g.apply(lambda r: f"{int(r['Year'])}|{r['Treatment']}|{r['Analyte']}", axis=1).to_numpy()

            unique_groups, inv = np.unique(group_id, return_inverse=True)
            for gi, gid in enumerate(unique_groups):
                rows = (inv == gi)
                sums = load_draw[rows, :].sum(axis=0)
                year_s, trt, an = gid.split("|", 2)
                annual_draw_rows.append(pd.DataFrame({
                    "Year": int(year_s),
                    "Treatment": trt,
                    "Analyte": an,
                    "Draw": np.arange(args.draws, dtype=int),
                    "AnnualLoad_mg": sums
                }))

        elapsed = time.time() - t0
        pbar.set_postfix_str(f"year={y_hold} test={len(df_test)} paired={len(paired)} fold_s={elapsed:,.1f}")

    # Write outputs
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["Target","Year_Test"])
    metrics_df.to_csv(outdir / "cv_metrics_by_year.csv", index=False)

    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    preds_df.to_csv(outdir / "cv_predictions_samplelevel.csv", index=False)

    if annual_draw_rows:
        annual_draws = pd.concat(annual_draw_rows, ignore_index=True)
        annual_draws.to_csv(outdir / "annual_load_draws.csv", index=False)

        q_lo = args.alpha / 2
        q_hi = 1 - args.alpha / 2
        annual_summary = (
            annual_draws
            .groupby(["Year","Treatment","Analyte"], as_index=False)["AnnualLoad_mg"]
            .agg(
                mean="mean",
                median="median",
                low=lambda s: float(np.quantile(s, q_lo)),
                high=lambda s: float(np.quantile(s, q_hi)),
                n_draws="count"
            )
        )
        annual_summary.to_csv(outdir / "annual_load_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "annual_load_draws.csv", index=False)
        pd.DataFrame().to_csv(outdir / "annual_load_summary.csv", index=False)

    plot_cv_metrics(metrics_df, figdir / "cv_rmse_by_year.png")

    # Feature importance summary (mean across LOYO folds)
    if fi_rows_C:
        fiC = pd.concat(fi_rows_C, ignore_index=True)
        fiC_sum = (fiC.groupby("feature", as_index=False)["importance"]
                        .agg(importance_mean="mean", importance_sd="std")
                        .sort_values("importance_mean", ascending=False))
        save_feature_importance(
            fiC_sum,
            outdir / "feature_importance_logC.csv",
            figdir / "feature_importance_logC.png",
            top_k=args.fi_topk,
            title="Feature importance: logC model (mean across LOYO folds)"
        )
    if fi_rows_V:
        fiV = pd.concat(fi_rows_V, ignore_index=True)
        fiV_sum = (fiV.groupby("feature", as_index=False)["importance"]
                        .agg(importance_mean="mean", importance_sd="std")
                        .sort_values("importance_mean", ascending=False))
        save_feature_importance(
            fiV_sum,
            outdir / "feature_importance_logV.csv",
            figdir / "feature_importance_logV.png",
            top_k=args.fi_topk,
            title="Feature importance: logV model (mean across LOYO folds)"
        )

    
    # Optional: refit on all observed data and impute missing rows, saving models for reuse
    if args.impute_missing:
        model_dir = outdir / str(args.model_subdir)
        refit_full_and_impute(
            df_model=df_model,
            X_all=X_all,
            cat_cols=cat_cols,
            cb_params=cb_params,
            alpha=float(args.alpha),
            outdir=outdir,
            model_dir=model_dir,
            save_models=bool(args.save_models),
            impute_draws=int(args.impute_draws),
            seed=int(args.seed),
        )
    print(f"[DONE] Outputs written to:\n  {outdir}\n  {figdir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] KeyboardInterrupt received.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
