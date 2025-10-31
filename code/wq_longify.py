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
    # Only apply to object/text columns
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
        # Leave numerics/booleans as-is
        return x
    return series.map(_fix)

def main():
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data" / "Master_WaterQuality_Kerbel_LastUpdated_10272025.csv"
    dst = repo / "out" / "kerbel_master_concentrations_long.csv"

    # Row 2 = real headers, row 3 = units to skip; DON'T auto-NA text tokens
    df = pd.read_csv(src, header=1, skiprows=[2], keep_default_na=False)

    # Normalize headers
    df.columns = [norm_header(c) for c in df.columns]
    cols = list(df.columns)

    # Concentration analytes (normalized names)
    conc_cols = [
        "OrthoP","TotalP","ICP","TKN","Nitrate","Nitrite",
        "NitrateNitrite","TotalN","Ammonium(NH4)","TSP",
        "Selenium","NPOC","TDS","TSS"
    ]

    # Drop the LOAD block by position (after FlumeMethod .. before TSSMethod)
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

    # Apply schema token normalization to EVERY object column (metadata + analyte + result)
    for c in df_long.columns:
        if df_long[c].dtype == object:
            df_long[c] = normalize_tokens(df_long[c])

    # Write
    dst.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(dst, index=False)

    print(f"[OK] Wrote long-format file → {dst}")
    print(f"[INFO] Dropped load columns (count): {len(load_block_cols)}")
    print(f"[INFO] Melted analytes: {len(conc_cols)}  |  Meta columns kept: {len(id_cols)}")

if __name__ == "__main__":
    main()
