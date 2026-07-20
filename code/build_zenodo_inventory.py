#!/usr/bin/env python3
"""Build non-destructive current-to-release manifests for the Kerbel repository."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


SKIP_DIRS = {".git", ".Rproj.user"}


def norm(path: Path) -> str:
    return path.as_posix()


def git_files(repo: Path, git_exe: str) -> set[str]:
    result = subprocess.run(
        [git_exe, "ls-files"], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def all_relevant_files(repo: Path) -> list[Path]:
    files = []
    for path in repo.rglob("*"):
        relative = path.relative_to(repo)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_file():
            files.append(relative)
    return sorted(files, key=lambda value: value.as_posix().lower())


def workflow(path: str) -> str:
    low = path.lower()
    if "old bayes" in low or "old bayes output" in low:
        return "legacy"
    if "bayes_vs_ml" in low or "bayes-ml" in low or "postprocessing" in low or "comparison" in low:
        return "comparison"
    if low.startswith("code/ml_") or "/ml_catboost" in low or low.startswith("figs/ml_"):
        return "ML"
    if "bayes" in low or low.endswith(".stan") or "m_stir_mogp" in low:
        return "Bayes"
    if low.startswith("code/") or "pipeline_csvs" in low or "wq_cleaned" in low:
        return "pipeline"
    if low.startswith("data/") or low in {"license", "citation.cff", ".zenodo.json", "readme.md", "changelog.md"}:
        return "shared"
    return "shared"


def proposed_path(path: str, flow: str) -> str:
    p = Path(path)
    low = path.lower()
    name = p.name
    if path in {"README.md", "CITATION.cff", ".zenodo.json", "LICENSE", "CHANGELOG.md", ".gitignore", ".gitattributes"}:
        return path
    if low.startswith("environment/"):
        return path
    if low.startswith("data/"):
        return f"data/raw/{name}" if name.lower() != "data_dictionary.csv" else "data/data_dictionary.csv"
    if low.startswith("out/pipeline_csvs/") or low == "out/wq_cleaned.csv":
        return f"data/processed/{name}"
    if low.startswith("code/"):
        folder = {"pipeline": "pipeline", "Bayes": "bayes", "ML": "ml", "comparison": "comparison", "legacy": "legacy"}.get(flow, "shared")
        return f"code/{folder}/{name}"
    if low.startswith("out/ml_catboost_conformal_loyo/"):
        return "results/ml/catboost_loyo/" + path.split("out/ml_catboost_conformal_loyo/", 1)[1]
    if low.startswith("out/bayes_vs_ml_postprocessing_v2p1/"):
        return "results/comparison/v2p1/" + path.split("out/bayes_vs_ml_postprocessing_v2p1/", 1)[1]
    if low.startswith("out/bayes_vs_ml_metrics_v2p1/"):
        return "results/comparison/v2p1/annual_agreement_inputs/" + path.split("out/bayes_vs_ml_metrics_v2p1/", 1)[1]
    if low.startswith("out/"):
        return f"results/bayes/v2p1/{name}" if "v2p1" in low else f"results/legacy/{name}"
    if low.startswith("figs/bayes_vs_ml_postprocessing_v2p1/"):
        return "figures/comparison/v2p1/" + path.split("figs/bayes_vs_ml_postprocessing_v2p1/", 1)[1]
    if low.startswith("figs/annual_bayes_vs_ml_faceted_jpg_v2p1/"):
        return "figures/comparison/v2p1/annual/" + path.split("figs/annual_bayes_vs_ml_faceted_jpg_v2p1/", 1)[1]
    if low.startswith("figs/ml_catboost_conformal_loyo/"):
        return "figures/ml/catboost_loyo/" + path.split("figs/ml_catboost_conformal_loyo/", 1)[1]
    if low.startswith("figs/"):
        return f"figures/bayes/v2p1/{name}" if "v2p1" in low else f"figures/legacy/{name}"
    if low.startswith("docs/workflows/"):
        return f"docs/reproducibility/{name}"
    if low.startswith("docs/"):
        return f"docs/methods/{name}" if "readme" in low or "method" in low else path
    if low.startswith("tests/"):
        return path
    return path


def status(path: str, flow: str) -> str:
    low = path.lower()
    if any(token in low for token in ("old bayes", "v1p", "_v1", "codex_volume_check")):
        return "superseded"
    if any(token in low for token in ("__pycache__", "catboost_info", ".rhistory", ".rdatatmp", ".pyc")):
        return "local-only"
    if low.endswith((".pptx", ".pdf")) and low.startswith("docs/"):
        return "diagnostic"
    if "audit" in low or "diagnostic" in low or "cv_" in low or "feature_importance" in low:
        return "diagnostic"
    if "intermediate" in low or "pipeline_csvs" in low:
        return "intermediate"
    if "v2p1" in low or "ml_catboost_conformal_loyo" in low or low.startswith("data/"):
        return "final"
    return "active"


def generator(path: str, flow: str) -> str:
    low = path.lower()
    if low.startswith("out/bayes_vs_ml_postprocessing_v2p1/") or low.startswith("figs/bayes_vs_ml_postprocessing_v2p1/"):
        return "code/bayes_ml_postprocessing_v2p1.py"
    if "bayes_vs_ml_metrics" in low or "annual_bayes_vs_ml" in low:
        return "code/annual_load_bayes_vs_ml.py"
    if "ml_catboost_conformal_loyo" in low:
        return "code/ml_catboost_conformal_loyo_v2_eventlevel.py or code/ml_postprocess_plots_v2_eventlevel.py"
    if flow == "Bayes" and (low.startswith("out/") or low.startswith("figs/")):
        return "code/stir-bayes-load2p1_nonneg.Rmd"
    if "pipeline_csvs" in low or low == "out/wq_cleaned.csv":
        return "code/run_pipeline.py and pipeline component scripts"
    if low.startswith("data/"):
        return "Source data or manually maintained metadata"
    if low.startswith("docs/") or Path(path).name in {"README.md", "CITATION.cff", ".zenodo.json", "CHANGELOG.md"}:
        return "Manually maintained documentation/metadata"
    return "See workflow documentation"


def recommendation(path: str, flow: str, state: str) -> tuple[str, str]:
    low = path.lower()
    name = Path(path).name.lower()
    exclude_reasons = [
        (("old bayes" in low or state == "superseded"), "Superseded or legacy artifact; preserve only in development history."),
        (("__pycache__" in low or name.endswith((".pyc", ".pyo"))), "Python cache/bytecode is local and reproducible."),
        (("catboost_info" in low), "CatBoost local training telemetry is not an article-facing result."),
        ((name.endswith(".exe") or name.endswith((".dll", ".so", ".dylib"))), "Platform-specific compiled binary; rebuild from source."),
        (("out_cmdstanr" in low or name.endswith(".rds")), "Large local CmdStan/fit object is not essential to the saved-draw release route."),
        (("codex_volume_check" in low), "Temporary validation directory; validated products are elsewhere."),
        ((name in {".rhistory", ".rdatatmp"}), "Local session state."),
        ((name.endswith(".inspect.ndjson")), "Spreadsheet verification support file; not a scientific artifact."),
        ((low.startswith("docs/") and name.endswith((".pptx", ".pdf"))), "Meeting/presentation artifact is redundant with release tables and documentation."),
        ((name == "predictions_from_saved_models.csv"), "Byte-identical provenance alias of canonical wq_cleaned_ml_imputed.csv."),
        ((low.startswith("figs/") and "v2p1" not in low and "ml_catboost_conformal_loyo" not in low), "Superseded or redundant figure version."),
    ]
    for condition, reason in exclude_reasons:
        if condition:
            return "exclude", reason
    return "include", "Active source, input, validated result, diagnostic required for interpretation, or release documentation."


def manuscript_use(path: str, flow: str, state: str) -> str:
    low = path.lower()
    if "master_cumulative" in low:
        return "Primary cumulative-load manuscript review table"
    if "sensitivity" in low:
        return "Bayesian nonnegative-load sensitivity"
    if "spearman" in low:
        return "Temporal agreement table"
    if "coverage" in low or "cv_" in low:
        return "ML calibration/performance support"
    if "feature_importance" in low:
        return "Descriptive ML interpretation"
    if low.startswith("figs/"):
        return "Candidate figure or diagnostic visual"
    if flow == "pipeline":
        return "Analytical dataset provenance"
    if flow in {"Bayes", "ML", "comparison"} and state in {"final", "diagnostic"}:
        return f"{flow} methods/results support"
    return "Repository/reproducibility support or no direct manuscript use"


def dependencies(path: str, flow: str) -> str:
    low = path.lower()
    if low.endswith(".rmd") or low.endswith(".r") or low.endswith(".stan"):
        return "R; R Markdown; CmdStanR/CmdStan as applicable; see Bayesian workflow"
    if low.endswith(".py"):
        return "Python; see workflow README and environment metadata"
    if low.endswith(".csv") and flow == "comparison":
        return "Saved Bayes/ML artifacts; code/bayes_ml_postprocessing_v2p1.py"
    return "See generating script and workflow README"


def hardcoded_risk(path: str, flow: str) -> str:
    low = path.lower()
    if low.endswith((".py", ".r", ".rmd")):
        return "high: update relative input/output references before moving"
    if low.endswith((".md", ".json", ".cff")):
        return "medium: update links and documented commands after moving"
    return "low: data/result path is consumed by scripts or manifests"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--git-exe", default="git")
    args = parser.parse_args()
    repo = args.repo.resolve()
    tracked = git_files(repo, args.git_exe)
    records = []
    for relative in all_relevant_files(repo):
        current = norm(relative)
        flow = workflow(current)
        state = status(current, flow)
        include, reason = recommendation(current, flow, state)
        records.append({
            "current_path": current,
            "proposed_release_path": proposed_path(current, flow),
            "workflow": flow,
            "status": state,
            "tracked_in_initial_repository": current in tracked,
            "file_size_bytes": (repo / relative).stat().st_size,
            "generating_script": generator(current, flow),
            "manuscript_use": manuscript_use(current, flow, state),
            "zenodo_recommendation": include,
            "recommendation_reason": reason,
            "dependencies": dependencies(current, flow),
            "hard_coded_path_risk": hardcoded_risk(current, flow),
        })
    manifest = pd.DataFrame(records).sort_values(["workflow", "current_path"], key=lambda s: s.astype(str).str.lower())
    manifest.to_csv(repo / "docs" / "output_manifest.csv", index=False)
    release = manifest.loc[manifest["zenodo_recommendation"] == "include"].copy()
    release.insert(0, "release_order", range(1, len(release) + 1))
    release.to_csv(repo / "docs" / "zenodo_release_manifest.csv", index=False)
    print(f"[OK] Inventory rows: {len(manifest)}")
    print(f"[OK] Recommended Zenodo include rows: {len(release)}")


if __name__ == "__main__":
    main()
