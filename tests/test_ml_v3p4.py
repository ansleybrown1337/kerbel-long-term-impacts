from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from comparison.bayes_ml_comparison_v3p4_physical_event import (  # noqa: E402
    bayes_referenced_axis_upper,
)


ML_PATH = REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p4_physical_event.py"
PLOT_PATH = REPO / "code" / "ml" / "ml_postprocess_plots_v3p4_physical_event.py"
REGENERATE_PATH = REPO / "code" / "ml" / "ml_regenerate_from_saved_models_v3p4.py"
COMPARISON_PATH = (
    REPO / "code" / "comparison" / "bayes_ml_comparison_v3p4_physical_event.py"
)
ML_WORKFLOW_PATH = REPO / "docs" / "workflows" / "ml_v3p4_physical_event.md"
COMPARISON_WORKFLOW_PATH = (
    REPO / "docs" / "workflows" / "comparison_v3p4_physical_event.md"
)
CONFIG_PATH = REPO / "config" / "physical_event_v3p4.json"
COMPACTION_PATH = REPO / "data" / "furrow_tire_compaction_records.csv"


def test_v3p4_ml_and_comparison_release_files_are_isolated() -> None:
    for path in (
        ML_PATH,
        PLOT_PATH,
        REGENERATE_PATH,
        COMPARISON_PATH,
        ML_WORKFLOW_PATH,
        COMPARISON_WORKFLOW_PATH,
        CONFIG_PATH,
        COMPACTION_PATH,
    ):
        assert path.is_file(), path

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["workflow_version"] == "v3p4_physical_event"
    assert config["versions"] == {
        "bayesian": "v3p3_physical_event",
        "ml": "v3p4_physical_event",
        "comparison": "v3p4_physical_event",
    }
    assert config["output_roots"]["ml_results"] == (
        "results/ml/v3p4_physical_event"
    )
    assert config["output_roots"]["ml_figures"] == (
        "figures/ml/v3p4_physical_event"
    )
    assert config["output_roots"]["comparison_results"] == (
        "results/comparison/v3p4_physical_event"
    )
    assert config["output_roots"]["comparison_figures"] == (
        "figures/comparison/v3p4_physical_event"
    )


def test_v3p4_compaction_roster_is_exactly_the_approved_scope() -> None:
    with COMPACTION_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    found = {(int(row["Year"]), row["Treatment"]) for row in rows}
    assert found == {
        (2021, "ST"),
        (2022, "MT"),
        (2022, "ST"),
        (2023, "MT"),
        (2023, "ST"),
        (2024, "MT"),
        (2024, "ST"),
        (2025, "MT"),
        (2025, "ST"),
    }
    assert all(row["RepScope"].lower() == "both" for row in rows)
    assert all(row["FurrowTireCompaction"] == "1" for row in rows)
    assert all(
        row["Timing"] == "after_residue_before_first_runoff" for row in rows
    )


def test_v3p4_ml_uses_compaction_in_both_prediction_models() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    concentration_block = source.split(
        "concentration_desired = [", 1
    )[1].split("volume_desired = [", 1)[0]
    volume_block = source.split("volume_desired = [", 1)[1].split(
        "# Never create event-level concentration averages", 1
    )[0]
    assert '"FurrowTireCompaction"' in concentration_block
    assert '"FurrowTireCompaction"' in volume_block
    assert "attach_furrow_tire_compaction(events, repo)" in source
    assert "attach_compaction_to_concentration_rows(df, events)" in source
    assert "required_concentration_features - set(features_c)" in source
    assert '"FurrowTireCompaction" not in features_v' in source
    assert '"included_in_logC": "FurrowTireCompaction" in features_c' in source
    assert '"included_in_logV": "FurrowTireCompaction" in features_v' in source


def test_v3p4_ml_feature_contract_removes_only_the_duplicate_limit() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    concentration_block = source.split(
        "concentration_desired = [", 1
    )[1].split("volume_desired = [", 1)[0]
    volume_block = source.split("volume_desired = [", 1)[1].split(
        "# Never create event-level concentration averages", 1
    )[0]

    assert '"MDL_mg_L"' in concentration_block
    assert '"RL_mg_L"' in concentration_block
    assert '"Result_lod_mg_L"' not in concentration_block
    assert '"DaysSincePlant"' in volume_block
    assert '"DaysUntilHarvest"' not in volume_block
    assert config["ml_feature_contract"] == {
        "concentration_required": [
            "MDL_mg_L", "RL_mg_L", "FurrowTireCompaction"
        ],
        "concentration_excluded_duplicate": "Result_lod_mg_L",
        "volume_required_plant_timing": "DaysSincePlant",
        "volume_excluded_harvest_timing": "DaysUntilHarvest",
        "feature_importance_top_k": 20,
    }


def test_v3p4_importance_figures_are_top_20_and_word_readable() -> None:
    plots = PLOT_PATH.read_text(encoding="utf-8")
    comparison = COMPARISON_PATH.read_text(encoding="utf-8")
    assert '"feature_importance_top_k"' in plots
    assert '"feature_importance_top_k"' in comparison
    assert '0.55 * len(data) + 2.0' in plots
    assert '0.55 * len(rows) + 2.5' in comparison
    assert 'axis.tick_params(axis="y", labelsize=12)' in plots
    assert 'axis.tick_params(axis="y", labelsize=12)' in comparison
    assert "top {top_k} of {total_features} inputs" in comparison


def test_v3p4_primary_comparison_restores_ml_ribbons_on_bayes_scale() -> None:
    comparison = COMPARISON_PATH.read_text(encoding="utf-8")
    upper = bayes_referenced_axis_upper(
        pd.Series([8.0, 10.0]),
        pd.Series([7.0, 9.0]),
        pd.Series([6.0]),
        pd.Series([5.0]),
    )
    assert upper == pytest.approx(10.8)
    assert comparison.count('axis.set_ylim(0, axis_upper)') == 2
    assert 'alpha=0.17 if method == "Bayes" else 0.13' in comparison
    assert '"primary_annual_figure_ml_prediction_ribbon_shown": True' in comparison
    assert '"annual_runoff_volume_ml_prediction_ribbon_shown": True' in comparison
    assert "ML prediction-interval bounds excluded" in comparison


def test_v3p4_ml_has_hard_support_checks_and_preflight() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    assert "APPROVED_COMPACTION_ASSIGNMENTS" in source
    assert 'config["expected_compacted_events"]' in source
    assert 'config["expected_compacted_events_with_genuine_volume"]' in source
    assert '"--preflight-only"' in source
    assert "PRE-FLIGHT PASSED" in source
    assert "tire_compaction_event_audit_ml_v3p4_physical_event.csv" in source
    assert "tire_compaction_audit_summary_ml_v3p4_physical_event.csv" in source


def test_v3p4_entrypoints_and_output_names_are_current() -> None:
    ml = ML_PATH.read_text(encoding="utf-8")
    plots = PLOT_PATH.read_text(encoding="utf-8")
    regenerate = REGENERATE_PATH.read_text(encoding="utf-8")
    comparison = COMPARISON_PATH.read_text(encoding="utf-8")
    ml_workflow = ML_WORKFLOW_PATH.read_text(encoding="utf-8")
    comparison_workflow = COMPARISON_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "ml_postprocess_plots_v3p4_physical_event" in ml
    assert "run_manifest_ml_v3p4_physical_event.json" in ml
    assert "event_analyte_draw_ledger_ml_v3p4.csv" in ml
    assert "volume_observation_training_table_v3p4.csv" in plots
    assert "ml_catboost_conformal_loyo_v3p4_physical_event.py" in regenerate
    assert "Bayesian v3p3 and ML v3p4" in comparison
    assert "physical_event_v3p4.json" in comparison
    assert "FurrowTireCompaction" in ml_workflow
    assert "the concentration (`logC`) and runoff-volume (`logV`)" in ml_workflow
    assert "completed ML `v3p4_physical_event` outputs" in comparison_workflow


def test_v3p4_uses_shared_release_data_contract() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    assert "CORRECTED_VERSION as DATA_CONTRACT_VERSION" in source
    assert 'PHYSICAL_EVENT_CONFIG["data_contract"]["preflight_directory"]' in source
    assert (
        'preflight_metadata.get("workflow_version") != DATA_CONTRACT_VERSION'
        in source
    )
