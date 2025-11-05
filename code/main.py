
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Kerbel long-term impacts end-to-end runner (pretty console)

Steps:
  A) Make/locate WQ long-format CSV            (wq_longify.py)
  B) Build STIR events long + daily aggregates (stir_pipeline.py)
  C) Merge WQ + STIR by crop season            (merge_wq_stir_by_season.py)

Usage (typical):
  python code/main.py --wq-long out/kerbel_master_concentrations_long.csv --stir-events out/stir_events_long.csv --crops "data/crop records.csv" --records data/tillage_records.csv --mapper data/tillage_mapper_input.csv --out out --debug
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------- Color / styling (works on Windows via colorama if available) ---------
RESET = ""
BOLD = ""
DIM = ""
FG = {
    "grey": "",
    "red": "",
    "green": "",
    "yellow": "",
    "blue": "",
    "magenta": "",
    "cyan": "",
    "white": "",
}
CHECK = "✓"
CROSS = "✗"
ARROW = "→"
SKIP = "↷"

try:
    # Try rich first (best formatting if user has it)
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
    # Fallback to colorama/ANSI
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
        # plain text
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
    msg = "Kerbel LTI — End-to-End Runner"
    if _USE_RICH and console:
        console.rule(f"[title]{msg}")
    else:
        bar = "=" * (len(msg) + 4)
        print(f"\n{bar}\n  {msg}\n{bar}\n")


def _check_file(path: Path, must_exist: bool = True, label: str | None = None) -> Path:
    p = Path(path)
    if must_exist and not p.exists():
        _echo(f"{CROSS} {label or 'Required file'} not found: {p}", "err")
        raise FileNotFoundError(f"{label or 'Required file'} not found: {p}")
    return p


def _ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: list[str], step_name: str) -> float:
    """
    Run a subprocess and return elapsed seconds.
    """
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
    wq_long: str | Path = "out/kerbel_master_concentrations_long.csv",
    stir_events: str | Path = "out/stir_events_long.csv",
    crops: str | Path = "data/crop records.csv",
    records: str | Path = "data/tillage_records.csv",
    mapper: str | Path = "data/tillage_mapper_input.csv",
    out_dir: str | Path = "out",
    wq_raw: str | Path | None = None,
    skip_wq: bool = False,
    skip_stir: bool = False,
    skip_merge: bool = False,
    debug: bool = False,
) -> None:
    """
    Run steps A→B→C. If outputs already exist, you can skip their producers.
    """
    _banner()
    start = time.monotonic()

    out_dir = _ensure_dir(Path(out_dir))
    wq_long = Path(wq_long)
    stir_events = Path(stir_events)
    crops = Path(crops)
    records = Path(records)
    mapper = Path(mapper)

    durations = {}

    # --- Step A: WQ longify (only if not skipping) ---
    if not skip_wq:
        if wq_long.exists() and not wq_raw:
            _echo(f"{CHECK} WQ long exists: {wq_long}", "ok")
        else:
            _echo("Step A: WQ longify", "title")
            _check_file(wq_raw or "", must_exist=True, label="--wq-raw is required to build --wq-long")
            _check_file(Path("code/wq_longify.py"), must_exist=True, label="wq_longify.py")
            cmd = [
                sys.executable, "code/wq_longify.py",
                "--in", str(wq_raw),
                "--out", str(wq_long),
            ]
            if debug:
                cmd.append("--debug")
            durations["WQ longify"] = _run(cmd, "WQ longify")
    else:
        _echo(f"{SKIP} Skipping Step A: WQ longify", "muted")

    _check_file(wq_long, must_exist=True, label="WQ long CSV")

    # --- Step B: STIR pipeline ---
    if not skip_stir:
        _echo("Step B: STIR pipeline", "title")
        _check_file(Path("code/stir_pipeline.py"), must_exist=True, label="stir_pipeline.py")
        _check_file(records, must_exist=True, label="tillage records CSV")
        _check_file(mapper, must_exist=True, label="tillage mapper CSV")

        cmd = [
            sys.executable, "code/stir_pipeline.py",
            "--records", str(records),
            "--mapper", str(mapper),
            "--outdir", str(out_dir),
        ]
        # Note: stir_pipeline.py accepts optional --crop (singular). It does NOT accept --debug.
        if Path(crops).exists():
            cmd += ["--crop", str(crops)]

        durations["STIR pipeline"] = _run(cmd, "STIR pipeline")
        if not stir_events.exists():
            candidate = out_dir / "stir_events_long.csv"
            if candidate.exists():
                stir_events = candidate
        _check_file(stir_events, must_exist=True, label="STIR events CSV")
    else:
        _echo(f"{SKIP} Skipping Step B: STIR pipeline", "muted")
        _check_file(stir_events, must_exist=True, label="STIR events CSV")

    # --- Step C: Merge WQ + STIR by season ---
    if not skip_merge:
        _echo("Step C: Merge WQ + STIR by season", "title")
        _check_file(Path("code/merge_wq_stir_by_season.py"), must_exist=True, label="merge_wq_stir_by_season.py")
        _check_file(crops, must_exist=True, label="crop records CSV")
        cmd = [
            sys.executable, "code/merge_wq_stir_by_season.py",
            "--wq", str(wq_long),
            "--stir", str(stir_events),
            "--crops", str(crops),
            "--out", str(out_dir),
        ]
        if debug:
            cmd.append("--debug")
        durations["Merge WQ+STIR"] = _run(cmd, "Merge WQ+STIR")
    else:
        _echo(f"{SKIP} Skipping Step C: Merge", "muted")

    # --- Final summary ---
    total_elapsed = time.monotonic() - start
    merged_csv = out_dir / "wq_with_stir_by_season.csv"
    unmatched_csv = out_dir / "wq_with_stir_unmatched.csv"

    if _USE_RICH and console:
        table = Table(title="Pipeline Summary", expand=True)
        table.add_column("Step", justify="left", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Output", justify="left")
        table.add_column("Time (s)", justify="right")

        table.add_row(
            "WQ long",
            f"[ok]{CHECK}[/ok]",
            str(wq_long),
            f"{durations.get('WQ longify', 0.0):.1f}" if not skip_wq else "-",
        )
        table.add_row(
            "STIR events long",
            f"[ok]{CHECK}[/ok]",
            str(stir_events),
            f"{durations.get('STIR pipeline', 0.0):.1f}" if not skip_stir else "-",
        )
        table.add_row(
            "Merged WQ+STIR (season)",
            f"[ok]{CHECK}[/ok]" if merged_csv.exists() else f"[err]{CROSS}[/err]",
            f"{merged_csv}" + ("" if merged_csv.exists() else " (missing!)"),
            f"{durations.get('Merge WQ+STIR', 0.0):.1f}" if not skip_merge else "-",
        )
        if unmatched_csv.exists():
            table.add_row(
                "Unmatched rows (QC)",
                f"[ok]{CHECK}[/ok]",
                str(unmatched_csv),
                "-",
            )
        console.print(table)
        console.print(f"[bold]Total elapsed:[/bold] {total_elapsed:.1f}s")
        console.rule()
    else:
        _echo("Done. Key outputs:", "info")
        _echo(f"  WQ long:                  {wq_long}", "info")
        _echo(f"  STIR events long:         {stir_events}", "info")
        _echo(f"  Merged WQ+STIR (season):  {merged_csv} {'(missing!)' if not merged_csv.exists() else ''}", "info")
        if unmatched_csv.exists():
            _echo(f"  Unmatched rows (QC):      {unmatched_csv}", "info")
        _echo(f"Total elapsed: {total_elapsed:.1f}s", "info")


def main() -> None:
    p = argparse.ArgumentParser(description="Run Kerbel LTI pipeline end-to-end.")
    p.add_argument("--wq-long", default="out/kerbel_master_concentrations_long.csv", help="Path to WQ long-format CSV (output of wq_longify.py).")
    p.add_argument("--wq-raw", default=None, help="Raw WQ file to longify (required if --wq-long does not exist and you are not skipping WQ).")
    p.add_argument("--stir-events", default="out/stir_events_long.csv", help="Path to STIR events long CSV (output of stir_pipeline.py).")
    p.add_argument("--crops", default="data/crop records.csv", help="Crop records CSV.")
    p.add_argument("--records", default="data/tillage_records.csv", help="Tillage operations log CSV.")
    p.add_argument("--mapper", default="data/tillage_mapper_input.csv", help="Tillage mapper CSV.")
    p.add_argument("--out", dest="out_dir", default="out", help="Output directory.")
    p.add_argument("--skip-wq", action="store_true", help="Skip WQ longify step.")
    p.add_argument("--skip-stir", action="store_true", help="Skip STIR pipeline step.")
    p.add_argument("--skip-merge", action="store_true", help="Skip merge step.")
    p.add_argument("--debug", action="store_true", help="Verbose logging for step A and C scripts (stir_pipeline.py ignores).")
    args = p.parse_args()

    run_all(
        wq_long=args.wq_long,
        stir_events=args.stir_events,
        crops=args.crops,
        records=args.records,
        mapper=args.mapper,
        out_dir=args.out_dir,
        wq_raw=args.wq_raw,
        skip_wq=args.skip_wq,
        skip_stir=args.skip_stir,
        skip_merge=args.skip_merge,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
