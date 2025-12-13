#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_catboost_conformal_loyo.py

LOYO CV: CatBoost + MAPIE v1 SplitConformalRegressor to model:
  1) OUT concentration: Result_mg_L
  2) OUT runoff volume: Volume (L)

Then propagate uncertainty to annual loads via Monte Carlo:
  AnnualLoad_mg = sum_i (C_i * V_i), where C is mg/L and V is L.

Quality-of-life features
------------------------
- tqdm progress bar across LOYO folds
- Per-fold timing diagnostics (fit + conformalize + predict + aggregation)
- Optional CatBoost training verbosity (prints every N iterations)
- Checkpoint outputs after EACH fold so partial results are preserved

Outputs (created automatically)
-------------------------------
- out/ml_catboost_conformal_loyo/
- figs/ml_catboost_conformal_loyo/

MAPIE note
----------
MAPIE v1 uses conformalizer classes (SplitConformalRegressor). This script
implements train -> conformalize -> predict within each LOYO fold.

Uncertainty note
----------------
Split conformal yields prediction intervals (PI). We propagate PI uncertainty
to annual loads by sampling uniformly within PI bounds in log space. This is
interval-based uncertainty propagation (not a Bayesian posterior).

"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from mapie.regression import SplitConformalRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import matplotlib.pyplot as plt


# -----------------------------
# Repo utilities
# -----------------------------

def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` to find repo root (README.md + out/ + figs/)."""
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
    """Ensure a CT/MT/ST treatment column exists named 'Treatment'."""
    if "Treatment" not in df.columns and "System" in df.columns:
        df["Treatment"] = df["System"]
    return df

def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct X and identify categorical columns.
    Excludes free-text Notes by default for stability/reproducibility.
    """
    desired = [
        "Year", "Irrigation", "Rep",
        "Treatment", "Analyte",
        "Inflow_Result_mg_L",
        "Flag", "Inflow_Flag",
        "FlumeMethod", "MeasureMethod", "IrrMethod", "TSSMethod",
        "Lab",
        "SeasonYear", "Crop",
        "CumAll_STIR_toDate", "Season_STIR_toDate",
        "DaysSincePlant", "DaysUntilHarvest",
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
        if frac_numeric < 0.95:
            if "SeasonYear" not in cat_cols:
                cat_cols.append("SeasonYear")

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X, cat_cols

def split_train_conformal(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    conformal_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=conformal_frac, random_state=random_state)

def fit_split_conformal_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_conf: pd.DataFrame,
    y_conf: pd.Series,
    cat_cols: list[str],
    confidence_level: float,
    random_state: int,
    cb_params: dict
) -> SplitConformalRegressor:
    cb = CatBoostRegressor(**cb_params, random_seed=random_state)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]

    conformal = SplitConformalRegressor(
        estimator=cb,
        confidence_level=confidence_level,
        conformity_score="absolute",
        prefit=False
    )

    conformal.fit(X_train, y_train, fit_params={"cat_features": cat_idx})
    conformal.conformalize(X_conf, y_conf)
    return conformal

def predict_with_pi(
    conformal: SplitConformalRegressor,
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pred, y_int = conformal.predict_interval(X_test)
    lo = y_int[:, 0, 0]
    hi = y_int[:, 1, 0]
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
        ax.plot(d["Year_Test"], d["RMSE"], marker="o", label=target)
    ax.set_title("LOYO CV RMSE by Held-out Year (log scale)")
    ax.set_xlabel("Held-out Year")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

def write_checkpoints(
    outdir: Path,
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    annual_draws_df: pd.DataFrame,
    alpha: float
) -> None:
    """Write cumulative outputs after each fold."""
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(outdir / "cv_metrics_by_year.csv", index=False)

    preds_df.to_csv(outdir / "cv_predictions_samplelevel.csv", index=False)

    if not annual_draws_df.empty:
        annual_draws_df.to_csv(outdir / "annual_load_draws.csv", index=False)

        q_lo = alpha / 2
        q_hi = 1 - alpha / 2
        annual_summary = (
            annual_draws_df
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
        # Ensure files exist even if empty
        pd.DataFrame().to_csv(outdir / "annual_load_draws.csv", index=False)
        pd.DataFrame().to_csv(outdir / "annual_load_summary.csv", index=False)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Path to wq_with_stir_by_season.csv. Default: <repo>/out/wq_with_stir_by_season.csv")
    ap.add_argument("--repo", type=str, default=None,
                    help="Optional repo root path. If omitted, auto-detected.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Miscoverage rate. alpha=0.05 gives ~95% prediction intervals.")
    ap.add_argument("--draws", type=int, default=2000,
                    help="Monte Carlo draws for annual load propagation per group.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_flagged", action="store_true",
                    help="If set, drops rows with Flag or Inflow_Flag not NA/blank.")
    ap.add_argument("--conformal_frac", type=float, default=0.2,
                    help="Fraction of training data reserved for conformalization within each LOYO fold.")
    ap.add_argument("--cb_verbose_every", type=int, default=0,
                    help="If >0, CatBoost prints training progress every N iterations (per model per fold).")
    ap.add_argument("--fast", action="store_true",
                    help="Fast mode for pipeline debugging (fewer iterations and fewer draws).")
    args = ap.parse_args()

    start = Path.cwd()
    repo = Path(args.repo).resolve() if args.repo else find_repo_root(start)

    data_path = Path(args.data).resolve() if args.data else (repo / "out" / "wq_with_stir_by_season.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    outdir = repo / "out" / "ml_catboost_conformal_loyo"
    figdir = repo / "figs" / "ml_catboost_conformal_loyo"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root : {repo}")
    print(f"[INFO] Data      : {data_path}")
    print(f"[INFO] Out dir   : {outdir}")
    print(f"[INFO] Fig dir   : {figdir}")

    # CatBoost parameters (shared)
    iterations = 3000
    draws = args.draws
    if args.fast:
        iterations = 600
        draws = min(draws, 300)
        print(f"[INFO] FAST mode enabled: iterations={iterations}, draws={draws}")

    cb_params = dict(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        subsample=0.8,
        colsample_bylevel=0.8,
        eval_metric="RMSE",
        verbose=(args.cb_verbose_every if args.cb_verbose_every > 0 else False),
    )

    df = pd.read_csv(data_path)
    df = parse_dates(df)
    df = add_time_features(df)
    df = ensure_treatment(df)
    df = coerce_bool(df, "NoRunoff")

    if "Year" not in df.columns:
        if "Date" in df.columns:
            df["Year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
        else:
            raise ValueError("Need Year or Date column.")

    for needed in ["Result_mg_L", "Volume", "Treatment", "Analyte"]:
        if needed not in df.columns:
            raise ValueError(f"Missing required column: {needed}")

    df["y_logC"] = safe_log1p_series(df["Result_mg_L"])
    df["y_logV"] = safe_log1p_series(df["Volume"])

    # Exclude no-runoff rows by default
    if "NoRunoff" in df.columns:
        df_model = df.loc[~df["NoRunoff"].fillna(False)].copy()
    else:
        df_model = df.copy()

    # Optional: drop flagged rows
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
    confidence_level = 1.0 - args.alpha

    metrics_rows: list[dict] = []
    pred_parts: list[pd.DataFrame] = []
    annual_draw_parts: list[pd.DataFrame] = []

    t_global0 = time.perf_counter()

    # tqdm progress bar over folds
    for fold_idx, y_hold in enumerate(tqdm(years, desc="LOYO folds", unit="year"), start=1):
        t0 = time.perf_counter()

        test_mask = (df_model["Year"] == y_hold)
        train_mask = ~test_mask

        df_train_full = df_model.loc[train_mask].copy()
        df_test = df_model.loc[test_mask].copy()

        X_train_full = X_all.loc[train_mask].copy()
        X_test = X_all.loc[test_mask].copy()

        yC_train_full = df_train_full["y_logC"]
        yV_train_full = df_train_full["y_logV"]
        yC_test = df_test["y_logC"]
        yV_test = df_test["y_logV"]

        idxC_tr = yC_train_full.notna()
        idxC_te = yC_test.notna()
        idxV_tr = yV_train_full.notna()
        idxV_te = yV_test.notna()

        # Split conformal inside training partition
        X_C_tr, X_C_cf, y_C_tr, y_C_cf = split_train_conformal(
            X_train_full.loc[idxC_tr], yC_train_full.loc[idxC_tr],
            random_state=args.seed + int(y_hold),
            conformal_frac=args.conformal_frac
        )
        X_V_tr, X_V_cf, y_V_tr, y_V_cf = split_train_conformal(
            X_train_full.loc[idxV_tr], yV_train_full.loc[idxV_tr],
            random_state=args.seed + 1000 + int(y_hold),
            conformal_frac=args.conformal_frac
        )

        t_split = time.perf_counter()

        # Fit + conformalize: concentration
        conformal_C = fit_split_conformal_catboost(
            X_C_tr, y_C_tr, X_C_cf, y_C_cf,
            cat_cols=cat_cols,
            confidence_level=confidence_level,
            random_state=args.seed + int(y_hold),
            cb_params=cb_params
        )
        t_fitC = time.perf_counter()
        predC, loC, hiC = predict_with_pi(conformal_C, X_test.loc[idxC_te])
        t_predC = time.perf_counter()

        # Fit + conformalize: volume
        conformal_V = fit_split_conformal_catboost(
            X_V_tr, y_V_tr, X_V_cf, y_V_cf,
            cat_cols=cat_cols,
            confidence_level=confidence_level,
            random_state=args.seed + 1000 + int(y_hold),
            cb_params=cb_params
        )
        t_fitV = time.perf_counter()
        predV, loV, hiV = predict_with_pi(conformal_V, X_test.loc[idxV_te])
        t_predV = time.perf_counter()

        # Metrics (log scale)
        metrics_rows.append({
            "Fold": fold_idx,
            "Folds_Total": len(years),
            "Year_Test": int(y_hold),
            "Target": "logC",
            "n_test": int(idxC_te.sum()),
            "MAE": float(mean_absolute_error(yC_test.loc[idxC_te], predC)) if idxC_te.sum() else np.nan,
            "RMSE": rmse(yC_test.loc[idxC_te].to_numpy(), predC) if idxC_te.sum() else np.nan,
            "R2": float(r2_score(yC_test.loc[idxC_te], predC)) if idxC_te.sum() > 2 else np.nan
        })
        metrics_rows.append({
            "Fold": fold_idx,
            "Folds_Total": len(years),
            "Year_Test": int(y_hold),
            "Target": "logV",
            "n_test": int(idxV_te.sum()),
            "MAE": float(mean_absolute_error(yV_test.loc[idxV_te], predV)) if idxV_te.sum() else np.nan,
            "RMSE": rmse(yV_test.loc[idxV_te].to_numpy(), predV) if idxV_te.sum() else np.nan,
            "R2": float(r2_score(yV_test.loc[idxV_te], predV)) if idxV_te.sum() > 2 else np.nan
        })

        # Sample-level predictions (original units)
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

        pred_parts.append(dfC)
        pred_parts.append(dfV)

        # Annual load propagation (test year only)
        key_cols = [c for c in ["SampleID","Year","Treatment","Analyte"] if c in df_test.columns]
        C_tab = df_test.loc[idxC_te, key_cols].copy()
        C_tab["loC"] = loC
        C_tab["hiC"] = hiC

        V_tab = df_test.loc[idxV_te, key_cols].copy()
        V_tab["loV"] = loV
        V_tab["hiV"] = hiV

        paired = C_tab.merge(V_tab, on=key_cols, how="inner")

        if not paired.empty:
            C_draw_log = sample_uniform_interval(paired["loC"].to_numpy(), paired["hiC"].to_numpy(), draws, rng)
            V_draw_log = sample_uniform_interval(paired["loV"].to_numpy(), paired["hiV"].to_numpy(), draws, rng)
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
                annual_draw_parts.append(pd.DataFrame({
                    "Year": int(year_s),
                    "Treatment": trt,
                    "Analyte": an,
                    "Draw": np.arange(draws, dtype=int),
                    "AnnualLoad_mg": sums
                }))

        t_end = time.perf_counter()

        # Build cumulative frames and checkpoint
        metrics_df = pd.DataFrame(metrics_rows).sort_values(["Target","Year_Test"])
        preds_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
        annual_draws_df = pd.concat(annual_draw_parts, ignore_index=True) if annual_draw_parts else pd.DataFrame()

        write_checkpoints(outdir, metrics_df, preds_df, annual_draws_df, alpha=args.alpha)

        # Plot updated metrics after each fold
        plot_cv_metrics(metrics_df, figdir / "cv_rmse_by_year.png")

        # Fold diagnostics
        fold_msg = (
            f"[FOLD {fold_idx}/{len(years)}] Year={y_hold} | "
            f"split={t_split - t0:0.1f}s, "
            f"fitC={(t_fitC - t_split):0.1f}s, predC={(t_predC - t_fitC):0.1f}s, "
            f"fitV={(t_fitV - t_predC):0.1f}s, predV={(t_predV - t_fitV):0.1f}s, "
            f"agg={(t_end - t_predV):0.1f}s, total={(t_end - t0):0.1f}s"
        )
        tqdm.write(fold_msg)

    t_global1 = time.perf_counter()
    tqdm.write(f"[DONE] Total elapsed: {(t_global1 - t_global0)/60:0.1f} minutes")
    tqdm.write(f"[DONE] Outputs written to:\n  {outdir}\n  {figdir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] KeyboardInterrupt received. Partial outputs should be saved in out/ml_catboost_conformal_loyo/.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
