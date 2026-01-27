#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py — Kerbel LTI data pipeline runner (pretty console)

Steps:
  A) Build WQ long-format CSV                  (wq_longify.py)
  B) Build STIR events long + daily aggregates (stir_pipeline.py)
  C) Merge WQ + STIR by crop season            (merge_wq_stir_by_season.py)
  C2) Merge residue (aggregated)               (merge_residue.py)
  D) Bayes-specific cleaning for modeling      (stir_bayes_backend.py)

Key behavior
============
- Runs the FULL pipeline every time (no output-based shortcuts).
- All intermediate CSVs are written under:  <out>/<out-csv-subdir>/
- The Bayes-ready modeling file is written to: <out>/wq_cleaned.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

RESET = ""
BOLD = ""
DIM = ""
FG = {"grey": "", "red": "", "green": "", "yellow": "", "blue": "", "magenta": "", "cyan": "", "white": ""}
CHECK = "✓"
CROSS = "✗"
ARROW = "→"
SKIP = "↷"

try:
    from rich.console import Console
    from rich.table import Table
    from rich.theme import Theme
    console: Optional[Console] = Console(theme=Theme({
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "info": "cyan",
        "muted": "dim",
        "step": "bold blue",
        "title": "bold white",
    }))
    _USE_RICH = True
except Exception:
    console = None
    _USE_RICH = False
    try:
        import colorama
        colorama.just_fix_windows_console()
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        FG = {
            "grey": "\033[90m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
        }
    except Exception:
        pass


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _echo(msg: str, style: str = "info") -> None:
    if _USE_RICH and console:
        console.print(f"[{_ts()}] {msg}", style=style)
    else:
        color = {
            "ok": FG["green"],
            "warn": FG["yellow"],
            "err": FG["red"],
            "info": FG["cyan"],
            "step": FG["blue"],
            "title": FG["white"] + BOLD,
            "muted": DIM,
        }.get(style, "")
        print(f"[{_ts()}] {color}{msg}{RESET}", flush=True)


def _banner():
    msg = "Kerbel LTI — Data Pipeline Runner"
    if _USE_RICH and console:
        console.rule(f"[title]{msg}")
    else:
        bar = "=" * (len(msg) + 4)
        print(f"\n{bar}\n  {msg}\n{bar}\n")


def _check_file(path: Path, label: str | None = None) -> Path:
    p = Path(path)
    if not p.exists():
        _echo(f"{CROSS} {label or 'Required file'} not found: {p}", "err")
        raise FileNotFoundError(f"{label or 'Required file'} not found: {p}")
    return p


def _ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: list[str], step_name: str) -> float:
    cmd_pretty = " ".join(cmd)
    _echo(f"{ARROW} {step_name}: {cmd_pretty}", "step")
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        elapsed = time.monotonic() - t0
        _echo(f"{CROSS} {step_name} FAILED (exit {e.returncode}) [{elapsed:.1f}s]", "err")
        raise
    elapsed = time.monotonic() - t0
    _echo(f"{CHECK} {step_name} finished [{elapsed:.1f}s]", "ok")
    return elapsed


def run_all(
    *,
    wq_raw: str | Path,
    crops: str | Path,
    records: str | Path,
    mapper: str | Path,
    residue: str | Path,
    out_dir: str | Path = "out",
    out_csv_subdir: str = "pipeline_csvs",
    skip_wq: bool = False,
    skip_stir: bool = False,
    skip_merge: bool = False,
    skip_residue: bool = False,
    skip_bayes_clean: bool = False,
    debug: bool = False,
) -> None:
    _banner()
    start = time.monotonic()

    out_dir = _ensure_dir(Path(out_dir))
    csv_dir = _ensure_dir(out_dir / out_csv_subdir)

    wq_long = csv_dir / "kerbel_master_concentrations_long.csv"
    stir_events = csv_dir / "stir_events_long.csv"
    merged_csv = csv_dir / "wq_with_stir_by_season.csv"
    merged_with_residue_csv = csv_dir / "wq_with_stir_by_season_with_residue.csv"
    unmatched_csv = csv_dir / "wq_with_stir_unmatched.csv"
    wq_cleaned = out_dir / "wq_cleaned.csv"

    durations: dict[str, float] = {}

    # --- Step A ---
    if not skip_wq:
        _echo("Step A: WQ longify", "title")
        _check_file(Path("code/wq_longify.py"), "wq_longify.py")
        _check_file(Path(wq_raw), "--wq-raw")
        cmd = [sys.executable, "code/wq_longify.py", "--in", str(wq_raw), "--out", str(wq_long)]
        if debug:
            cmd.append("--debug")
        durations["WQ longify"] = _run(cmd, "WQ longify")
    else:
        _echo(f"{SKIP} Skipping Step A: WQ longify", "muted")
    _check_file(wq_long, "WQ long CSV")

    # --- Step B ---
    if not skip_stir:
        _echo("Step B: STIR pipeline", "title")
        _check_file(Path("code/stir_pipeline.py"), "stir_pipeline.py")
        _check_file(Path(records), "tillage records CSV")
        _check_file(Path(mapper), "tillage mapper CSV")
        cmd = [sys.executable, "code/stir_pipeline.py",
               "--records", str(records),
               "--mapper", str(mapper),
               "--outdir", str(csv_dir)]
        if Path(crops).exists():
            cmd += ["--crop", str(crops)]
        durations["STIR pipeline"] = _run(cmd, "STIR pipeline")
    else:
        _echo(f"{SKIP} Skipping Step B: STIR pipeline", "muted")
    _check_file(stir_events, "STIR events long CSV")

    # --- Step C ---
    if not skip_merge:
        _echo("Step C: Merge WQ + STIR by season", "title")
        _check_file(Path("code/merge_wq_stir_by_season.py"), "merge_wq_stir_by_season.py")
        _check_file(Path(crops), "crop records CSV")
        cmd = [sys.executable, "code/merge_wq_stir_by_season.py",
               "--wq", str(wq_long),
               "--stir", str(stir_events),
               "--crops", str(crops),
               "--out", str(csv_dir)]
        if debug:
            cmd.append("--debug")
        durations["Merge WQ+STIR"] = _run(cmd, "Merge WQ+STIR")
    else:
        _echo(f"{SKIP} Skipping Step C: Merge", "muted")
    _check_file(merged_csv, "Merged WQ×STIR CSV")

    # --- Step C2 (Residue) ---
    if not skip_residue:
        _echo("Step C2: Merge residue (aggregated)", "title")
        _check_file(Path("code/merge_residue.py"), "merge_residue.py")
        _check_file(Path(residue), "residue CSV")
        cmd = [sys.executable, "code/merge_residue.py",
               "--wq", str(merged_csv),
               "--residue", str(residue),
               "--out", str(merged_with_residue_csv)]
        if debug:
            cmd.append("--debug")
        durations["Merge residue"] = _run(cmd, "Merge residue")
    else:
        _echo(f"{SKIP} Skipping Step C2: Residue merge", "muted")

    # choose downstream input
    downstream_in = merged_with_residue_csv if (not skip_residue) else merged_csv
    _check_file(downstream_in, "WQ×STIR (with residue)" if not skip_residue else "WQ×STIR merged")

    # --- Step D ---
    if not skip_bayes_clean:
        _echo("Step D: Bayes-specific cleaning", "title")
        _check_file(Path("code/stir_bayes_backend.py"), "stir_bayes_backend.py")
        cmd = [sys.executable, "code/stir_bayes_backend.py", "--in", str(downstream_in), "--out", str(wq_cleaned)]
        if debug:
            cmd.append("--debug")
        durations["Bayes clean"] = _run(cmd, "Bayes clean")
        _echo(f"{CHECK} Wrote wq_cleaned.csv → {wq_cleaned}", "ok")
    else:
        _echo(f"{SKIP} Skipping Step D: Bayes clean", "muted")
    _check_file(wq_cleaned, "wq_cleaned.csv")

    total_elapsed = time.monotonic() - start
    if _USE_RICH and console:
        table = Table(title="Pipeline Summary", expand=True)
        table.add_column("Step", justify="left", style="bold")
        table.add_column("Output", justify="left")
        table.add_column("Time (s)", justify="right")

        table.add_row("WQ long", str(wq_long), f"{durations.get('WQ longify', 0.0):.1f}" if not skip_wq else "-")
        table.add_row("STIR events long", str(stir_events), f"{durations.get('STIR pipeline', 0.0):.1f}" if not skip_stir else "-")
        table.add_row("Merged WQ×STIR", str(merged_csv), f"{durations.get('Merge WQ+STIR', 0.0):.1f}" if not skip_merge else "-")
        table.add_row("Residue merged", str(downstream_in), f"{durations.get('Merge residue', 0.0):.1f}" if not skip_residue else "-")
        table.add_row("Bayes-cleaned", str(wq_cleaned), f"{durations.get('Bayes clean', 0.0):.1f}" if not skip_bayes_clean else "-")
        if unmatched_csv.exists():
            table.add_row("Unmatched rows (QC)", str(unmatched_csv), "-")
        console.print(table)
        console.print(f"[bold]Total elapsed:[/bold] {total_elapsed:.1f}s")
        console.rule()
    else:
        _echo("Done. Key outputs:", "info")
        _echo(f"  WQ long:            {wq_long}", "info")
        _echo(f"  STIR events long:   {stir_events}", "info")
        _echo(f"  Merged WQ×STIR:     {merged_csv}", "info")
        if not skip_residue:
            _echo(f"  + Residue merged:   {downstream_in}", "info")
        if unmatched_csv.exists():
            _echo(f"  Unmatched rows QC:  {unmatched_csv}", "info")
        _echo(f"  Bayes cleaned:      {wq_cleaned}", "info")
        _echo(f"Total elapsed: {total_elapsed:.1f}s", "info")


def main() -> None:
    p = argparse.ArgumentParser(description="Run Kerbel LTI data pipeline end-to-end (no shortcuts).")
    p.add_argument("--wq-raw", default="data/Master_WaterQuality_Kerbel_LastUpdated_10272025.csv",
                   help="Raw WQ file to longify.")
    p.add_argument("--crops", default="data/crop records.csv", help="Crop records CSV.")
    p.add_argument("--records", default="data/tillage_records.csv", help="Tillage operations log CSV.")
    p.add_argument("--mapper", default="data/tillage_mapper_input.csv", help="Tillage mapper CSV.")
    p.add_argument("--residue", default="data/residue_2011_2025.csv",
                   help="Residue CSV (observed or dummy).")
    p.add_argument("--out", dest="out_dir", default="out", help="Output directory.")
    p.add_argument("--out-csv-subdir", default="pipeline_csvs",
                   help="Subfolder under --out for pipeline intermediate CSVs.")
    p.add_argument("--skip-wq", action="store_true", help="Skip WQ longify step.")
    p.add_argument("--skip-stir", action="store_true", help="Skip STIR pipeline step.")
    p.add_argument("--skip-merge", action="store_true", help="Skip merge step.")
    p.add_argument("--skip-residue", action="store_true", help="Skip residue merge step.")
    p.add_argument("--skip-bayes-clean", action="store_true", help="Skip Bayes-specific cleaning step.")
    p.add_argument("--debug", action="store_true", help="Verbose logging for scripts that support it.")
    args = p.parse_args()

    run_all(
        wq_raw=args.wq_raw,
        crops=args.crops,
        records=args.records,
        mapper=args.mapper,
        residue=args.residue,
        out_dir=args.out_dir,
        out_csv_subdir=args.out_csv_subdir,
        skip_wq=args.skip_wq,
        skip_stir=args.skip_stir,
        skip_merge=args.skip_merge,
        skip_residue=args.skip_residue,
        skip_bayes_clean=args.skip_bayes_clean,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
