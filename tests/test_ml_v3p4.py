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
    annual_signed_ct_relative_products,
    bayes_referenced_axis_upper,
    event_concentration_one_to_one_table,
    management_period_sensitivity_products,
    primary_ct_relative_plot_data,
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


def test_management_period_sensitivity_is_reproducible_and_signed(tmp_path: Path) -> None:
    annual_load_rows = []
    annual_volume_rows = []
    for year in range(2011, 2026):
        period_multiplier = 1.0 if year <= 2020 else 2.0
        for treatment, treatment_multiplier in {"CT": 1.0, "MT": 0.8, "ST": 0.7}.items():
            for analyte in ("TSS", "TP", "TN"):
                annual_load_rows.append(
                    {
                        "Method": "ML",
                        "Scenario": "full_record_model_only",
                        "Year": year,
                        "Analyte": analyte,
                        "Treatment": treatment,
                        "primary_center": period_multiplier * treatment_multiplier,
                    }
                )
            for method, scenario in (
                ("Bayes", "model_only"),
                ("ML", "full_record_model_only"),
            ):
                annual_volume_rows.append(
                    {
                        "Method": method,
                        "Scenario": scenario,
                        "Year": year,
                        "Treatment": treatment,
                        "center_kL": period_multiplier * treatment_multiplier,
                    }
                )

    bayes_template = pd.DataFrame(
        {
            "analyte": ["TSS", "TP", "TN"],
            "CT_mod_load_sum_mean_kg": [10.0, 10.0, 10.0],
            "MT_mod_load_sum_mean_kg": [8.0, 8.0, 8.0],
            "ST_mod_load_sum_mean_kg": [7.0, 7.0, 7.0],
            "MT_pct_red_mean": [12.5, 12.5, 12.5],
            "ST_pct_red_mean": [22.5, 22.5, 22.5],
        }
    )
    for prefix in (
        "pre_tire_compaction_era_2011_2020",
        "tire_compaction_era_2021_2025",
    ):
        bayes_template.to_csv(
            tmp_path
            / f"{prefix}_total_loads_kg_with_pct_reductions_v3p3_physical_event.csv",
            index=False,
        )

    result = management_period_sensitivity_products(
        pd.DataFrame(annual_load_rows),
        pd.DataFrame(annual_volume_rows),
        tmp_path,
    )

    assert len(result) == 48
    assert not result.duplicated(["Outcome", "Period", "Method", "Treatment"]).any()
    bayes_mt = result.loc[
        result["Outcome"].eq("TSS")
        & result["Period"].eq("2011-2020")
        & result["Method"].eq("Bayes")
        & result["Treatment"].eq("MT"),
        "PercentChangeFromCT",
    ].item()
    ml_mt = result.loc[
        result["Outcome"].eq("TSS")
        & result["Period"].eq("2011-2020")
        & result["Method"].eq("ML")
        & result["Treatment"].eq("MT"),
        "PercentChangeFromCT",
    ].item()
    assert bayes_mt == pytest.approx(-12.5)
    assert ml_mt == pytest.approx(-20.0)


def test_management_period_figure_has_no_title_and_centers_legend_above() -> None:
    source = COMPARISON_PATH.read_text(encoding="utf-8")
    plot_block = source.split(
        "def plot_management_period_sensitivity(", 1
    )[1].split("def make_figures(", 1)[0]
    assert "figure.suptitle" not in plot_block
    assert "Labels show MT or ST percent change relative to CT" not in plot_block
    assert 'loc="upper center"' in plot_block
    assert 'bbox_to_anchor=(0.5, 0.995)' in plot_block
    assert '"postprocessing" / "pre_post_2021_management_sensitivity"' in plot_block


def test_primary_ct_relative_plot_is_focused_and_reproducible() -> None:
    rows = []
    for method, scenario in (
        ("Bayes", "model_only"),
        ("ML", "full_record_model_only"),
    ):
        for analyte in ("TSS", "TP", "TN", "NH4"):
            for treatment in ("MT", "ST"):
                rows.append(
                    {
                        "Method": method,
                        "Scenario": scenario,
                        "Analyte": analyte,
                        "ComparisonTreatment": treatment,
                        "primary_center": 1.0,
                        "lower_95": 0.0,
                        "upper_95": 2.0,
                    }
                )
    result = primary_ct_relative_plot_data(pd.DataFrame(rows))
    assert len(result) == 12
    assert set(result["Analyte"].astype(str)) == {"TSS", "TP", "TN"}
    assert not result.duplicated(
        ["Method", "Analyte", "ComparisonTreatment"]
    ).any()

    source = COMPARISON_PATH.read_text(encoding="utf-8")
    plot_block = source.split("def plot_ct_relative(", 1)[1].split(
        "def plot_performance_comparison(", 1
    )[0]
    assert "figure.suptitle" not in plot_block
    assert 'loc="upper center"' in plot_block
    assert '"cumulative_primary_analyte_differences_vs_ct"' in plot_block
    assert 'colors = {"MT":' in plot_block


def test_annual_signed_ct_relative_uses_within_draw_ratios_and_ml_point_center() -> None:
    draw_rows = []
    treatment_values = {
        "CT": [10.0, 20.0],
        "MT": [8.0, 22.0],
        "ST": [12.0, 18.0],
    }
    for method, scenario in (
        ("Bayes", "model_only"),
        ("ML", "full_record_model_only"),
    ):
        for treatment, values in treatment_values.items():
            for draw, value in enumerate(values, start=1):
                draw_rows.append(
                    {
                        "Method": method,
                        "Scenario": scenario,
                        "Year": 2021,
                        "Outcome": "TSS",
                        "Treatment": treatment,
                        "Draw": draw,
                        "ModeledAnnualTotal": value,
                    }
                )
    point_rows = [
        {
            "Method": "ML",
            "Scenario": "full_record_model_only",
            "Year": 2021,
            "Outcome": "TSS",
            "Treatment": treatment,
            "PointAnnualTotal": value,
        }
        for treatment, value in {"CT": 10.0, "MT": 9.0, "ST": 11.0}.items()
    ]
    raw, summary = annual_signed_ct_relative_products(
        pd.DataFrame(draw_rows), pd.DataFrame(point_rows)
    )
    bayes_mt = summary.loc[
        summary["Method"].eq("Bayes")
        & summary["ComparisonTreatment"].eq("MT")
    ].iloc[0]
    ml_mt = summary.loc[
        summary["Method"].eq("ML")
        & summary["ComparisonTreatment"].eq("MT")
    ].iloc[0]
    assert bayes_mt["median"] == pytest.approx(-5.0)
    assert ml_mt["primary_center"] == pytest.approx(-10.0)
    assert ml_mt["primary_center_type"] == (
        "deterministic_mean_per_plot_annual_contrast"
    )
    assert raw.loc[
        raw["Method"].eq("Bayes")
        & raw["ComparisonTreatment"].eq("ST"),
        "PercentDifferenceRelativeToCT",
    ].tolist() == pytest.approx([20.0, -10.0])
    assert summary["n_invalid_percent_draws"].eq(0).all()


def test_annual_signed_ct_relative_figure_marks_2021_without_title() -> None:
    source = COMPARISON_PATH.read_text(encoding="utf-8")
    plot_block = source.split(
        "def plot_annual_signed_ct_relative(", 1
    )[1].split("def plot_performance_comparison(", 1)[0]
    assert "figure.suptitle" not in plot_block
    assert 'label="Tire compaction begins (2021)"' in plot_block
    assert "compaction_boundary" in plot_block
    assert 'colors = {"MT": "#2A7F62", "ST": "#2E6EAA"}' in plot_block
    assert 'ecolor=colors[treatment]' in plot_block
    assert 'linewidth=2.0' in plot_block
    assert 'set_hatch' not in plot_block
    assert 'Nonpositive CT draws omitted from ratio' not in plot_block
    assert 'loc="upper center"' in plot_block
    assert '"annual_signed_differences_vs_ct"' in plot_block


def test_event_one_to_one_table_resolves_observations_by_physical_event() -> None:
    analytes = ["NH4", "NO3", "NO2", "OP", "Se", "TDS", "TKN", "TN", "TP", "TSS"]
    bayes_rows = []
    ml_rows = []
    for index, analyte in enumerate(analytes, start=1):
        event_id = f"PE_{index}"
        for observed in [float(index), float(index + 2)]:
            bayes_rows.append(
                {
                    "PhysicalEventID": event_id,
                    "Analyte": analyte,
                    "Year": 2020,
                    "Treatment": "MT",
                    "Observed": observed,
                    "Predicted": float(index + 0.5),
                }
            )
        ml_rows.append(
            {
                "Target": "Result_mg_L",
                "PhysicalEventID": event_id,
                "Analyte": analyte,
                "Year": 2020,
                "Treatment": "MT",
                "y_true": float(index + 1),
                "y_pred": float(index + 1.5),
                "n_source_observations": 2,
            }
        )
    result = event_concentration_one_to_one_table(
        pd.DataFrame(bayes_rows), pd.DataFrame(ml_rows)
    )
    assert len(result) == 20
    assert result["Analyte"].nunique() == 10
    assert not result.duplicated(
        ["Method", "PhysicalEventID", "Analyte"]
    ).any()
    assert result["Observed_mg_L"].eq(
        result["Analyte"].map({analyte: index + 1 for index, analyte in enumerate(analytes, 1)})
    ).all()


def test_event_one_to_one_figures_are_square_and_use_identical_symlog_axes() -> None:
    source = COMPARISON_PATH.read_text(encoding="utf-8")
    plot_block = source.split(
        "def plot_event_concentration_one_to_one(", 1
    )[1].split("def plot_performance_comparison(", 1)[0]
    assert 'axis.set_box_aspect(1)' in plot_block
    assert 'axis.set_xscale(' in plot_block and 'axis.set_yscale(' in plot_block
    assert '"symlog"' in plot_block
    assert 'axis.set_xlim(0, axis_maximum)' in plot_block
    assert 'axis.set_ylim(0, axis_maximum)' in plot_block
    assert 'label="One-to-one"' in plot_block
    assert '"event_concentration_one_to_one"' in plot_block


def test_ml_compaction_sensitivity_is_explicitly_noncausal() -> None:
    source = COMPARISON_PATH.read_text(encoding="utf-8")
    block = source.split(
        "def ml_furrow_compaction_predictive_sensitivity(", 1
    )[1].split("def management_period_sensitivity_products(", 1)[0]
    assert 'without_compaction["FurrowTireCompaction"] = 0' in block
    assert 'len(event_rows) != 120' in block
    assert 'clusters are Year x Irrigation' in block
    assert '"paired predictive sensitivity, not a causal effect' in block


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
