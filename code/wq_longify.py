#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

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

    # Build inflow lookup for Result and Flag
    # Normalize INF Flag tokens too (already normalized above)
    inf_lookup = (
        inf_rows.groupby(key_cols, dropna=False)
                .agg({
                    "Result_mg_L": pick_first_token,
                    "Flag": pick_first_token
                })
                .reset_index()
                .rename(columns={
                    "Result_mg_L": "Inflow_Result_mg_L",
                    "Flag": "Inflow_Flag"
                })
    )

    # Merge inflow values/flag onto OUT rows only
    out_joined = out_rows.merge(inf_lookup, on=key_cols, how="left")

    # Fill missing inflow fields with "NA"
    out_joined["Inflow_Result_mg_L"] = out_joined["Inflow_Result_mg_L"].fillna("NA")
    out_joined["Inflow_Flag"] = out_joined["Inflow_Flag"].fillna("NA")

    # Add Has_Inflow as TRUE/FALSE text token
    out_joined["Has_Inflow"] = out_joined["Inflow_Result_mg_L"].apply(
        lambda x: "FALSE" if str(x) == "NA" else "TRUE"
    )

    # Drop helper
    out_joined = out_joined.drop(columns=["FlowDir"])

    # Write
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_joined.to_csv(dst, index=False)

    print(f"[OK] Wrote long-format file → {dst}")
    print(f"[INFO] Dropped load columns (count): {len(load_block_cols)}")
    print(f"[INFO] Melted analytes: {len(conc_cols)}  |  Meta columns kept: {len(id_cols)}")
    print(f"[INFO] Inflow paired onto OUT rows using keys: {key_cols}")
    print(f"[INFO] Added columns: Inflow_Result_mg_L, Inflow_Flag, Has_Inflow")
    print(f"[INFO] Output rows (OUT only): {len(out_joined)}")

if __name__ == "__main__":
    main()
