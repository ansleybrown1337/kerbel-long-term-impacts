from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ML_PATH = REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p3_physical_event.py"
PLOT_PATH = REPO / "code" / "ml" / "ml_postprocess_plots_v3p3_physical_event.py"
REGENERATE_PATH = REPO / "code" / "ml" / "ml_regenerate_from_saved_models_v3p3.py"
COMPARISON_PATH = (
    REPO / "code" / "comparison" / "bayes_ml_comparison_v3p3_physical_event.py"
)
ML_WORKFLOW_PATH = REPO / "docs" / "workflows" / "ml_v3p3_physical_event.md"
COMPARISON_WORKFLOW_PATH = (
    REPO / "docs" / "workflows" / "comparison_v3p3_physical_event.md"
)
CONFIG_PATH = REPO / "config" / "physical_event_v3p3.json"
COMPACTION_PATH = REPO / "data" / "furrow_tire_compaction_records.csv"


def test_v3p3_ml_and_comparison_release_files_are_isolated() -> None:
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
    assert config["versions"] == {
        "bayesian": "v3p3_physical_event",
        "ml": "v3p3_physical_event",
        "comparison": "v3p3_physical_event",
    }
    assert config["output_roots"]["ml_results"] == (
        "results/ml/v3p3_physical_event"
    )
    assert config["output_roots"]["ml_figures"] == (
        "figures/ml/v3p3_physical_event"
    )
    assert config["output_roots"]["comparison_results"] == (
        "results/comparison/v3p3_physical_event"
    )
    assert config["output_roots"]["comparison_figures"] == (
        "figures/comparison/v3p3_physical_event"
    )


def test_v3p3_compaction_roster_is_exactly_the_approved_scope() -> None:
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


def test_v3p3_ml_uses_compaction_in_volume_only() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    concentration_block = source.split(
        "concentration_desired = [", 1
    )[1].split("volume_desired = [", 1)[0]
    volume_block = source.split("volume_desired = [", 1)[1].split(
        "# Never create event-level concentration averages", 1
    )[0]
    assert "FurrowTireCompaction" not in concentration_block
    assert '"FurrowTireCompaction"' in volume_block
    assert "attach_furrow_tire_compaction(events, repo)" in source
    assert '"FurrowTireCompaction" in features_c' in source
    assert '"FurrowTireCompaction" not in features_v' in source
    assert '"included_in_logC": "FurrowTireCompaction" in features_c' in source
    assert '"included_in_logV": "FurrowTireCompaction" in features_v' in source


def test_v3p3_ml_has_hard_support_checks_and_preflight() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    assert "APPROVED_COMPACTION_ASSIGNMENTS" in source
    assert 'config["expected_compacted_events"]' in source
    assert 'config["expected_compacted_events_with_genuine_volume"]' in source
    assert '"--preflight-only"' in source
    assert "PRE-FLIGHT PASSED" in source
    assert "tire_compaction_event_audit_ml_v3p3_physical_event.csv" in source
    assert "tire_compaction_audit_summary_ml_v3p3_physical_event.csv" in source


def test_v3p3_entrypoints_and_output_names_are_current() -> None:
    ml = ML_PATH.read_text(encoding="utf-8")
    plots = PLOT_PATH.read_text(encoding="utf-8")
    regenerate = REGENERATE_PATH.read_text(encoding="utf-8")
    comparison = COMPARISON_PATH.read_text(encoding="utf-8")
    ml_workflow = ML_WORKFLOW_PATH.read_text(encoding="utf-8")
    comparison_workflow = COMPARISON_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "ml_postprocess_plots_v3p3_physical_event" in ml
    assert "run_manifest_ml_v3p3_physical_event.json" in ml
    assert "event_analyte_draw_ledger_ml_v3p3.csv" in ml
    assert "volume_observation_training_table_v3p3.csv" in plots
    assert "ml_catboost_conformal_loyo_v3p3_physical_event.py" in regenerate
    assert "Bayesian v3p3 and ML v3p3" in comparison
    assert "physical_event_v3p3.json" in comparison
    assert "FurrowTireCompaction" in ml_workflow
    assert "runoff-volume (`logV`) feature set only" in ml_workflow
    assert "completed ML `v3p3_physical_event` outputs" in comparison_workflow


def test_v3p3_still_uses_corrected_v3p1_data_contract() -> None:
    source = ML_PATH.read_text(encoding="utf-8")
    assert "CORRECTED_VERSION as DATA_CONTRACT_VERSION" in source
    assert '/ "physical_event_v3p1"' in source
    assert (
        'preflight_metadata.get("workflow_version") != DATA_CONTRACT_VERSION'
        in source
    )
