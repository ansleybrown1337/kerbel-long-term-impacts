#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from typing import Dict, Any

PRESERVE_TOKENS = {"NA", "NA.IRR", "U", "None"}

def norm_header(c: str) -> str:
    return (
        c.strip()
         .replace("\t", "")
         .replace(" ", "")
         .replace(".", "")
         .replace("/", "")
    )

def normalize_tokens(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series
    def _fix(x):
        if x is None:
            return "NA"
        if isinstance(x, str):
            s = x.strip()
            if s in PRESERVE_TOKENS:
                return s
            if s == "" or s == "#VALUE!":
                return "NA"
            return s
        return x
    return series.map(_fix)

def pick_first_token(s: pd.Series) -> str:
    # choose first non-"NA" token; else "NA"
    for v in s:
        if isinstance(v, str) and v != "NA":
            return v
    return "NA"

# --------------------------
# RL/MDL REFERENCE TABLE(S)
# --------------------------
# Map vendor/era to per-analyte RL/MDL entries
# Provided_Units are those given by the lab/method; we will also standardize to mg/L.
RLMDL_TABLE: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Current ALS (Houston/Holland), ~2023+
    "ALS_2023plus": {
        "NitrateNitrite": {
            "MDL": 0.03, "RL": 0.2, "Provided_Units": "mg/L",
            "Method": "EPA 300.0", "Notes": "ALS current 2023+"
        },
        "Nitrate": {
            "MDL": 0.03, "RL": 0.1, "Provided_Units": "mg/L",
            "Method": "EPA 300.1", "Notes": "ALS current 2023+"
        },
        "Nitrite": {
            "MDL": 0.03, "RL": 0.1, "Provided_Units": "mg/L",
            "Method": "EPA 300.2", "Notes": "ALS current 2023+"
        },
        "Ammonium(NH4)": {
            "MDL": 0.077, "RL": 0.2, "Provided_Units": "mg/L",
            "Method": "EPA 350.1", "Notes": "ALS current 2023+"
        },
        "OrthoP": {
            "MDL": 0.01, "RL": 0.05, "Provided_Units": "mg/L",
            "Method": "SM4500P E (P-Ortho)", "Notes": "ALS current 2023+"
        },
        "TKN": {
            "MDL": 0.1, "RL": 0.5, "Provided_Units": "mg/L",
            "Method": "M4500 NH3 D", "Notes": "ALS current 2023+"
        },
        "TotalP": {
            "MDL": 0.02, "RL": 0.05, "Provided_Units": "mg/L",
            "Method": "SM4500P E (P_TW)", "Notes": "ALS current 2023+"
        },
        "Selenium": {
            "MDL": 0.86, "RL": 2.0, "Provided_Units": "µg/L",
            "Method": "EPA 200.8", "Notes": "ALS current 2023+"
        },
        "TDS": {
            "MDL": 5.0, "RL": 10.0, "Provided_Units": "mg/L",
            "Method": "EPA 160.1", "Notes": "ALS current 2023+"
        },
        "TSS": {  # performed by CSU in-house, but include here for completeness if ALS ran it
            "MDL": 2.5, "RL": 2.5, "Provided_Units": "mg/L",
            "Method": "EPA 160.2", "Notes": "CSU in-house; mirrored"
        },
        # Others (pH, EC, etc.) are outside our 14-analyte long table
    },
    # Old ALS (Fort Collins) pre-2023
    "ALS_pre2023": {
        "Nitrate": {
            "MDL": 0.19, "RL": 0.2, "Provided_Units": "MG/L",
            "Method": "EPA300.0 [300Nitrate]_W", "Notes": "ALS Fort Collins pre-2023"
        },
        "Nitrite": {
            "MDL": 0.11, "RL": 0.15, "Provided_Units": "MG/L",
            "Method": "EPA300.0 [300Nitrite]_W", "Notes": "ALS Fort Collins pre-2023"
        },
        "OrthoP": {
            "MDL": 0.34, "RL": 0.75, "Provided_Units": "MG/L",
            "Method": "EPA300.0 [300OPHOS]_W", "Notes": "ALS Fort Collins pre-2023"
        },
        "Selenium": {
            "MDL": None, "RL": 1.5, "Provided_Units": "UG/L",
            "Method": "EPA200.8 [SE200.8]_W", "Notes": "ALS Fort Collins pre-2023"
        },
        "TotalP": {
            "MDL": 0.033, "RL": 0.05, "Provided_Units": "MG/L",
            "Method": "EPA365.2 [365.2TPHOS]_W", "Notes": "ALS Fort Collins pre-2023"
        },
        # TSS, TDS, etc. not defined here by the old sheet; leave NA unless assumed elsewhere
    },
}

def _to_mg_per_L(value, units):
    if value is None or isinstance(value, str):
        return "NA"
    u = (units or "").strip().lower()
    if u in ("mg/l", "mg\\l", "mgperL", "mg"):
        return float(value)
    if u in ("µg/l", "ug/l", "µg\\l", "ug\\l", "ugl"):
        return float(value) / 1000.0
    # pH or other unitless → NA for mg/L standardization
    return "NA"

def select_rlmdl_source(year: str, lab: str) -> str:
    """
    Decide which RL/MDL table to use.
    - If lab contains 'CSU' ⇒ assume ALS_pre2023 (conservative), but mark via RLMDL_Source later.
    - Else Year ≥ 2023 ⇒ ALS_2023plus
    - Else ⇒ ALS_pre2023
    """
    y = None
    try:
        y = int(str(year))
    except Exception:
        pass

    lab_str = (lab or "").lower()
    is_csu = "csu" in lab_str

    if is_csu:
        return "CSU_assumed_from_ALS_pre2023"
    if y is not None and y >= 2023:
        return "ALS_2023plus"
    return "ALS_pre2023"

def fetch_rlmdl(analyte: str, source_key: str):
    """
    Pull RL/MDL row for a given analyte and source key.
    Returns dict with keys:
      MDL_Provided, RL_Provided, RLMDL_Provided_Units, RLMDL_Method,
      MDL_mg_L, RL_mg_L, RLMDL_Source, RLMDL_Assumed (TRUE/FALSE)
    """
    # Resolve “assumed CSU” into ALS_pre2023 values, but keep source label
    table_key = "ALS_pre2023" if source_key == "CSU_assumed_from_ALS_pre2023" else source_key
    tbl = RLMDL_TABLE.get(table_key, {})
    row = tbl.get(analyte)

    out = {
        "MDL_Provided": "NA",
        "RL_Provided": "NA",
        "RLMDL_Provided_Units": "NA",
        "RLMDL_Method": "NA",
        "MDL_mg_L": "NA",
        "RL_mg_L": "NA",
        "RLMDL_Source": source_key,
        "RLMDL_Assumed": "FALSE" if source_key in ("ALS_2023plus", "ALS_pre2023") else "TRUE",
    }

    if row is None:
        # Not found for this analyte; leave NA; mark assumed if it's CSU or unknown table
        if source_key not in ("ALS_2023plus", "ALS_pre2023"):
            out["RLMDL_Assumed"] = "TRUE"
        return out

    mdl = row.get("MDL")
    rl = row.get("RL")
    units = row.get("Provided_Units")
    method = row.get("Method")

    out["MDL_Provided"] = "NA" if mdl is None else str(mdl)
    out["RL_Provided"] = "NA" if rl is None else str(rl)
    out["RLMDL_Provided_Units"] = units or "NA"
    out["RLMDL_Method"] = method or "NA"
    out["MDL_mg_L"] = _to_mg_per_L(mdl, units)
    out["RL_mg_L"] = _to_mg_per_L(rl, units)
    # Keep as strings for schema consistency
    out["MDL_mg_L"] = "NA" if out["MDL_mg_L"] == "NA" else str(out["MDL_mg_L"])
    out["RL_mg_L"] = "NA" if out["RL_mg_L"] == "NA" else str(out["RL_mg_L"])
    return out

def main():
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data" / "Master_WaterQuality_Kerbel_LastUpdated_10272025.csv"
    dst = repo / "out" / "kerbel_master_concentrations_long.csv"

    # Use row 2 as headers; skip row 3 (units). Keep literal tokens like "NA".
    df = pd.read_csv(src, header=1, skiprows=[2], keep_default_na=False)

    # Normalize headers
    df.columns = [norm_header(c) for c in df.columns]
    cols = list(df.columns)

    # Concentration analytes (normalized)
    conc_cols = [
        "OrthoP","TotalP","ICP","TKN","Nitrate","Nitrite",
        "NitrateNitrite","TotalN","Ammonium(NH4)","TSP",
        "Selenium","NPOC","TDS","TSS"
    ]

    # Drop LOAD block by position (after FlumeMethod .. before TSSMethod)
    flume_idx = cols.index("FlumeMethod")
    tssm_idx = cols.index("TSSMethod")
    load_block_cols = cols[flume_idx + 1 : tssm_idx]

    # IMPORTANT: ensure Volume survives even if it sits inside the positional LOAD block
    # (this protects OUT Volume too, and enables INF Volume pairing).
    if "Volume" in load_block_cols:
        load_block_cols = [c for c in load_block_cols if c != "Volume"]

    # Meta/ID columns = everything not in conc analytes and not in the load block
    id_cols = [c for c in cols if c not in conc_cols and c not in load_block_cols]

    # Keep meta + concentrations only
    df_keep = df[id_cols + conc_cols]

    # Long format
    df_long = df_keep.melt(
        id_vars=id_cols,
        value_vars=conc_cols,
        var_name="Analyte",
        value_name="Result_mg_L"
    )

    # Normalize tokens across ALL text columns
    for c in df_long.columns:
        if df_long[c].dtype == object:
            df_long[c] = normalize_tokens(df_long[c])

    # Identify flow direction:
    def infer_flow(row):
        io = str(row.get("InflowOutflow", "")).strip()
        tr = str(row.get("Treatment", "")).strip()
        if io == "INF" or tr == "INF":
            return "INF"
        return "OUT"

    df_long["FlowDir"] = df_long.apply(infer_flow, axis=1)

    # Split inflow vs outflow (final output keeps OUT only)
    out_rows = df_long[df_long["FlowDir"] == "OUT"].copy()
    inf_rows = df_long[df_long["FlowDir"] == "INF"].copy()

    # Keys for pairing inflow to outflow (ignore Treatment)
    key_cols = ["Year", "Date", "Irrigation", "Rep", "Analyte"]

    # Build inflow lookup for Result, Flag, and Volume
    # (Volume must exist in df_long via id_cols; the safeguard above helps ensure that.)
    inf_agg = {
        "Result_mg_L": pick_first_token,
        "Flag": pick_first_token
    }
    if "Volume" in inf_rows.columns:
        inf_agg["Volume"] = pick_first_token

    inf_lookup = (
        inf_rows.groupby(key_cols, dropna=False)
                .agg(inf_agg)
                .reset_index()
                .rename(columns={
                    "Result_mg_L": "Inflow_Result_mg_L",
                    "Flag": "Inflow_Flag",
                    "Volume": "Inflow_Volume"
                })
    )

    # Merge inflow values/flag/volume onto OUT rows only
    out_joined = out_rows.merge(inf_lookup, on=key_cols, how="left")

    # Fill missing inflow fields with "NA"
    out_joined["Inflow_Result_mg_L"] = out_joined["Inflow_Result_mg_L"].fillna("NA")
    out_joined["Inflow_Flag"] = out_joined["Inflow_Flag"].fillna("NA")
    if "Inflow_Volume" in out_joined.columns:
        out_joined["Inflow_Volume"] = out_joined["Inflow_Volume"].fillna("NA")

    # Add Has_Inflow as TRUE/FALSE text token
    out_joined["Has_Inflow"] = out_joined["Inflow_Result_mg_L"].apply(
        lambda x: "FALSE" if str(x) == "NA" else "TRUE"
    )

    # --------------------------
    # Attach RL/MDL to each row
    # --------------------------
    # Decide source (ALS_2023plus, ALS_pre2023, CSU_assumed_from_ALS_pre2023) per row
    # based on Year and Lab name.
    def _attach(row):
        source_key = select_rlmdl_source(row.get("Year", ""), row.get("Lab", ""))
        rlmdl = fetch_rlmdl(str(row.get("Analyte", "")), source_key)
        return pd.Series(rlmdl)

    rlmdl_df = out_joined.apply(_attach, axis=1)
    out_joined = pd.concat([out_joined, rlmdl_df], axis=1)

    # Ensure all object columns keep tokens
    for c in out_joined.columns:
        if out_joined[c].dtype == object:
            out_joined[c] = normalize_tokens(out_joined[c])

    # Drop helper
    out_joined = out_joined.drop(columns=["FlowDir"])

    # Write
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_joined.to_csv(dst, index=False)

    print(f"[OK] Wrote long-format file → {dst}")
    print(f"[INFO] Dropped load columns (count): {len(load_block_cols)}")
    print(f"[INFO] Melted analytes: {len(conc_cols)}  |  Meta columns kept: {len(id_cols)}")
    print(f"[INFO] Inflow paired onto OUT rows using keys: {key_cols}")
    added_cols = ["Inflow_Result_mg_L", "Inflow_Flag", "Has_Inflow"]
    if "Inflow_Volume" in out_joined.columns:
        added_cols.insert(2, "Inflow_Volume")
    print(f"[INFO] Added columns: {', '.join(added_cols)}")
    print(f"[INFO] RL/MDL columns added: MDL_Provided, RL_Provided, RLMDL_Provided_Units, RLMDL_Method, MDL_mg_L, RL_mg_L, RLMDL_Source, RLMDL_Assumed")
    print(f"[INFO] Output rows (OUT only): {len(out_joined)}")

if __name__ == "__main__":
    main()
