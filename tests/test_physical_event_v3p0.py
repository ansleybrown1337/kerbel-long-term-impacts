from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from shared.physical_event import (  # noqa: E402
    add_concentration_observation_id,
    add_physical_event_id,
    build_event_analyte_load_ledger,
    build_event_analyte_point_load_ledger,
    build_volume_observation_table,
    event_balanced_weights,
    loyo_masks,
    resolve_prediction_draws,
    split_event_groups,
    validate_corrected_artifact_metadata,
)
from comparison.bayes_ml_comparison_v3p0_physical_event import (  # noqa: E402
    STUDY_YEARS,
    attach_primary_centers,
    bayes_negative_sensitivity,
    coverage_by_year_target,
    normalize_ledger,
    normalize_point_ledger,
    overall_nrmse_table,
    performance_table,
    read_observed_annual_summary,
)
from ml.ml_postprocess_plots_v3p0_physical_event import observed_annual_loads  # noqa: E402
from bayes.stir_bayes_backend import clean_wq_stir  # noqa: E402


def rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_wq_idx": [1, 2],
            "Date": ["2020-07-01", "2020-07-01"],
            "Year": [2020, 2020],
            "Irrigation": [1, 1],
            "Rep": [1, 1],
            "Treatment": ["CT", "CT"],
            "SampleID": ["A", "A-D"],
            "MeasureMethod": ["Ruler", "Ruler"],
            "FlumeMethod": ["7 V", "7 V"],
            "NoRunoff": [False, False],
            "Duplicate": [False, True],
            "Analyte": ["TotalN", "TotalP"],
            "Result_mg_L": [2.0, 3.0],
            "Volume": [100.0, 100.0],
        }
    )


def test_row_duplication_does_not_add_volume_observation() -> None:
    base = rows().iloc[[0]].copy()
    duplicated = pd.concat([base, base.assign(_wq_idx=99, Analyte="TotalP")], ignore_index=True)
    base_volume, _ = build_volume_observation_table(base)
    duplicated_volume, _ = build_volume_observation_table(duplicated)
    assert len(base_volume) == len(duplicated_volume) == 1
    assert duplicated_volume.loc[0, "source_row_count"] == 2


def test_genuine_parallel_volume_observations_are_retained() -> None:
    frame = rows().copy()
    frame.loc[1, "MeasureMethod"] = "Transducer"
    frame.loc[1, "Volume"] = 101.0
    volume, reports = build_volume_observation_table(frame)
    assert len(volume) == 2
    assert reports["ambiguous_volume_observations"].empty


def test_copied_volume_value_is_deduplicated() -> None:
    volume, reports = build_volume_observation_table(rows())
    assert len(volume) == 1
    assert len(reports["copied_volume_values"]) == 2
    assert reports["copied_volume_values"]["source_row_count"].eq(2).all()


def test_confirmed_zero_is_observed_and_missing_is_not() -> None:
    frame = rows().copy()
    frame["Volume"] = [0.0, np.nan]
    frame["NoRunoff"] = [True, False]
    frame["Date"] = ["2020-07-01", "2020-07-02"]
    volume, reports = build_volume_observation_table(frame)
    assert volume["Volume"].tolist() == [0.0]
    assert len(reports["events_without_volume_observation"]) == 1


def test_calibration_split_contains_whole_events() -> None:
    ids = pd.Series(["A", "A", "B", "B", "C", "C", "D", "D"])
    proper, calibration = split_event_groups(ids, 0.25, 7)
    assert set(ids.iloc[proper]).isdisjoint(set(ids.iloc[calibration]))


def test_loyo_heldout_year_is_excluded() -> None:
    years = pd.Series([2020, 2020, 2021, 2021])
    ids = pd.Series(["A", "A", "B", "B"])
    train, test = loyo_masks(years, ids, 2021)
    assert years.loc[train].eq(2021).sum() == 0
    assert years.loc[test].eq(2021).all()


def test_event_balanced_weights_sum_to_one() -> None:
    frame = add_physical_event_id(rows())
    frame = pd.concat([frame, frame.iloc[[0]].assign(_wq_idx=3)], ignore_index=True)
    weights = event_balanced_weights(frame, ["PhysicalEventID", "Analyte"])
    sums = weights.groupby([frame["PhysicalEventID"], frame["Analyte"]]).sum()
    assert np.allclose(sums, 1.0)


def test_postprediction_resolution_supports_median_mean_and_priority() -> None:
    draws = pd.DataFrame(
        {
            "PhysicalEventID": ["A", "A", "A"],
            "Analyte": ["TN", "TN", "TN"],
            "Draw": [0, 0, 0],
            "Method": ["low", "high", "high"],
            "value": [1.0, 5.0, 7.0],
        }
    )
    groups = ["PhysicalEventID", "Analyte", "Draw"]
    assert resolve_prediction_draws(draws, group_columns=groups, value_column="value").loc[0, "value"] == 5
    assert resolve_prediction_draws(draws, group_columns=groups, value_column="value", method="mean").loc[0, "value"] == pytest.approx(13 / 3)
    priority = resolve_prediction_draws(
        draws, group_columns=groups, value_column="value", method="method_priority",
        method_column="Method", method_priority=["high", "low"],
    )
    assert priority.loc[0, "value"] == 6


def test_annual_load_is_invariant_to_copied_concentration_prediction() -> None:
    base = pd.DataFrame(
        {"PhysicalEventID": ["A"], "Analyte": ["TN"], "Draw": [0], "Concentration_mg_L": [2.0]}
    )
    copied = pd.concat([base, base], ignore_index=True)
    copied = resolve_prediction_draws(
        copied,
        group_columns=["PhysicalEventID", "Analyte", "Draw"],
        value_column="Concentration_mg_L",
    )
    volume = pd.DataFrame({"PhysicalEventID": ["A"], "Draw": [0], "Volume_L": [100.0]})
    assert build_event_analyte_load_ledger(base, volume)["Load_kg"].sum() == build_event_analyte_load_ledger(copied, volume)["Load_kg"].sum()


def test_point_load_requires_one_resolved_event_analyte_and_volume() -> None:
    concentration = pd.DataFrame({
        "PhysicalEventID": ["A"],
        "Year": [2020],
        "Treatment": ["CT"],
        "Analyte": ["TN"],
        "Concentration_mg_L": [2.0],
    })
    volume = pd.DataFrame({"PhysicalEventID": ["A"], "Volume_L": [100.0]})
    ledger = build_event_analyte_point_load_ledger(concentration, volume)
    assert ledger.loc[0, "Load_kg"] == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="must first resolve"):
        build_event_analyte_point_load_ledger(
            pd.concat([concentration, concentration], ignore_index=True), volume
        )
    with pytest.raises(ValueError, match="must first resolve"):
        build_event_analyte_point_load_ledger(
            concentration, pd.concat([volume, volume], ignore_index=True)
        )


def test_comparison_uses_ml_point_total_without_moving_draw_interval() -> None:
    summary = pd.DataFrame({
        "Method": ["Bayes", "ML"],
        "Scenario": ["model_only", "full_record_model_only"],
        "Year": [2020, 2020],
        "Analyte": ["TN", "TN"],
        "Treatment": ["CT", "CT"],
        "median": [2.0, 20.0],
        "lower_95": [1.0, 10.0],
        "upper_95": [3.0, 30.0],
    })
    points = pd.DataFrame({
        "Method": ["ML"],
        "Scenario": ["full_record_model_only"],
        "Year": [2020],
        "Analyte": ["TN"],
        "Treatment": ["CT"],
        "PointTotal_kg": [4.0],
    })
    result = attach_primary_centers(
        summary,
        points,
        keys=["Method", "Scenario", "Year", "Analyte", "Treatment"],
        point_column="PointTotal_kg",
    ).set_index("Method")
    assert result.loc["Bayes", "primary_center"] == 2.0
    assert result.loc["ML", "primary_center"] == 4.0
    assert result.loc["ML", "median"] == 20.0
    assert result.loc["ML", "lower_95"] == 10.0
    assert result.loc["ML", "upper_95"] == 30.0
    assert not bool(result.loc["ML", "primary_center_within_draw_interval"])


def test_comparison_point_ledger_rejects_method_multiplication() -> None:
    point = pd.DataFrame({
        "PhysicalEventID": ["A", "A"],
        "Analyte": ["TN", "TN"],
        "Year": [2011, 2011],
        "Treatment": ["CT", "CT"],
        "Load_kg": [1.0, 1.0],
        "SampleMethod": ["grab", "composite"],
    })
    with pytest.raises(ValueError, match="duplicate PhysicalEventID x Analyte"):
        normalize_point_ledger(point, "ML", "full_record", require_complete_years=False)


def test_overall_nrmse_pools_treatments_and_includes_volume() -> None:
    concentration = pd.DataFrame({
        "Target": ["Result_mg_L"] * 4,
        "Analyte": ["TN"] * 4,
        "Treatment": ["CT", "CT", "MT", "MT"],
        "y_true": [1.0, 3.0, 2.0, 4.0],
        "y_pred": [1.0, 2.0, 2.0, 6.0],
        "pi_low": [0.0] * 4,
        "pi_high": [7.0] * 4,
    })
    volume = pd.DataFrame({
        "Target": ["Volume_L", "Volume_L"],
        "Treatment": ["CT", "MT"],
        "y_true": [100.0, 200.0],
        "y_pred": [90.0, 220.0],
        "pi_low": [50.0, 100.0],
        "pi_high": [150.0, 300.0],
    })
    performance = pd.concat([
        performance_table(concentration, "ML", "Result_mg_L"),
        performance_table(volume, "ML", "Volume_L"),
    ], ignore_index=True)
    overall = overall_nrmse_table(performance)
    assert overall["DisplayTarget"].tolist() == ["TN", "Volume"]
    tn = overall.loc[overall["DisplayTarget"].eq("TN")].iloc[0]
    expected_rmse = np.sqrt((0.0**2 + (-1.0)**2 + 0.0**2 + 2.0**2) / 4)
    assert tn["NRMSE_mean_observed"] == pytest.approx(expected_rmse / 2.5)


def test_comparison_refuses_legacy_and_incomplete_year_metadata(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"workflow_version": "v2p1", "event_unit": "PhysicalEventID", "years": [2020]}))
    with pytest.raises(ValueError, match="Legacy"):
        validate_corrected_artifact_metadata([legacy], expected_years=[2020])
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"workflow_version": "v3p0_physical_event", "event_unit": "PhysicalEventID", "years": [2020]}))
    with pytest.raises(ValueError, match="Incomplete year"):
        validate_corrected_artifact_metadata([incomplete], expected_years=[2020, 2021])


def test_concentration_observation_ids_remain_row_unique() -> None:
    identified = add_concentration_observation_id(rows())
    assert identified["ConcentrationObservationID"].nunique() == len(identified)


def test_physical_event_id_is_stable_across_date_representations() -> None:
    text = rows().iloc[[0]].copy()
    timestamp = text.copy()
    timestamp["Date"] = pd.to_datetime(timestamp["Date"])
    timestamp["Year"] = timestamp["Year"].astype(float)
    assert (
        add_physical_event_id(text).loc[text.index[0], "PhysicalEventID"]
        == add_physical_event_id(timestamp).loc[timestamp.index[0], "PhysicalEventID"]
    )


def test_comparison_requires_complete_primary_but_allows_incomplete_loyo_ledger() -> None:
    ledger = pd.DataFrame({
        "PhysicalEventID": ["A"], "Analyte": ["TN"], "Year": [2011],
        "Treatment": ["CT"], "Draw": [0], "Load_kg": [1.0],
    })
    with pytest.raises(ValueError, match="required exactly"):
        normalize_ledger(ledger, "ML", "full_record")
    normalized = normalize_ledger(
        ledger, "ML", "loyo", require_complete_years=False
    )
    assert normalized["Year"].tolist() == [2011]


def test_bayes_negative_sensitivity_truncates_annual_not_source_draws() -> None:
    annual = pd.DataFrame({
        "Method": ["Bayes", "Bayes"],
        "Scenario": ["model_only", "model_only"],
        "Year": [2011, 2012], "Analyte": ["TN", "TN"],
        "Treatment": ["CT", "CT"], "Draw": [1, 1],
        "Load_kg": [-1.0, 2.0],
    })
    original = annual.copy(deep=True)
    draws, summary = bayes_negative_sensitivity(annual)
    values = draws.set_index("Scenario")["CumulativeLoad_kg"].to_dict()
    assert values["raw_annual_draws_display_floor_only"] == pytest.approx(1.0)
    assert values["annual_draw_truncation_at_zero"] == pytest.approx(2.0)
    assert summary["n_negative_annual_draws"].eq(1).all()
    pd.testing.assert_frame_equal(annual, original)


def test_loyo_coverage_reports_absent_target_years_as_zero_n_and_na() -> None:
    diagnostics = pd.DataFrame({
        "Target": ["Volume_L"], "Year": [2011], "y_true": [10.0],
        "pi_low": [8.0], "pi_high": [12.0],
    })
    coverage = coverage_by_year_target(diagnostics, "ML")
    assert coverage["Year"].tolist() == STUDY_YEARS
    absent = coverage.loc[coverage["Year"].eq(2012)].iloc[0]
    assert absent["n"] == 0
    assert pd.isna(absent["IntervalCoverage"])


def test_observed_plot_ledger_does_not_multiply_event_volume_by_sample_rows(
    tmp_path: Path,
) -> None:
    frame = rows().copy()
    repeated_tn = frame.iloc[[0]].assign(_wq_idx=3, SampleID="B", Result_mg_L=4.0)
    frame = pd.concat([frame, repeated_tn], ignore_index=True)
    source = tmp_path / "observed.csv"
    frame.to_csv(source, index=False)
    annual = observed_annual_loads(source)
    assert annual["observed_load_mg"].sum() == pytest.approx(600.0)
    assert annual["observed_physical_events"].eq(1).all()


def test_v3p0_model_entrypoints_share_final_cleaned_input() -> None:
    backend = (REPO / "code" / "bayes" / "stir-bayes-backend.R").read_text(
        encoding="utf-8"
    )
    rmd = (
        REPO / "code" / "bayes" / "stir-bayes-load_v3p0_physical_event.Rmd"
    ).read_text(encoding="utf-8")
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p0_physical_event.py"
    ).read_text(encoding="utf-8")

    assert 'load_wq_stir <- function(path = "out/wq_cleaned.csv"' in backend
    assert "wq_with_stir_by_season.csv" not in backend
    assert "if (interactive())" not in backend
    assert 'file.path(repo_root, "out", "wq_cleaned.csv")' in rmd
    assert 'repo / "out" / "wq_cleaned.csv"' in ml


def test_final_cleaning_preserves_storm_event_labels() -> None:
    cleaned = clean_wq_stir(pd.DataFrame({"Irrigation": ["1", "S1", "S2", "2"]}))
    assert cleaned["Irrigation"].astype("string").tolist() == ["1", "S1", "S2", "2"]
    assert cleaned.loc[cleaned["Irrigation"].isin(["S1", "S2"]), "IRR_z"].isna().all()
    assert cleaned.loc[cleaned["Irrigation"].isin(["1", "2"]), "IRR_z"].notna().all()


def test_bayesian_batch_runner_is_single_pass_and_uses_final_input() -> None:
    batch = (
        REPO / "code" / "bayes" / "stir-bayes-load_v3p0_physical_event.R"
    ).read_text(encoding="utf-8")

    assert batch.startswith("#!/usr/bin/env Rscript")
    assert 'file.path(repo_root, "out", "wq_cleaned.csv")' in batch
    assert "fit <- mod$sample(" in batch
    assert "force_recompile = .batch_force_recompile" in batch
    assert batch.count("cmdstan_dashboard(fit)") == 1
    assert 'Sys.getenv("BAYES_REUSE_FIT", "false")' in batch
    assert "fit <- readRDS(fit_rds_path)" in batch
    assert "as_plain_numeric_matrix" in batch
    assert "as.numeric(V_mean + V_sd * muV_z_ev)" in batch
    assert "as.numeric(c_mean + c_sd * muC_z)" in batch
    assert "pick_key_vars_v3p0" not in batch
    assert "bad_pars <-" not in batch
    assert "overall_diagnostics_bayes_v3p0_physical_event.csv" in batch


def test_v3p0_outputs_are_isolated_in_version_folders() -> None:
    config = json.loads(
        (REPO / "config" / "physical_event_v3p0.json").read_text(encoding="utf-8")
    )
    assert config["output_roots"] == {
        "bayesian_results": "results/bayes/v3p0_physical_event",
        "ml_results": "results/ml/v3p0_physical_event",
        "comparison_results": "results/comparison/v3p0_physical_event",
        "bayesian_figures": "figures/bayes/v3p0_physical_event",
        "ml_figures": "figures/ml/v3p0_physical_event",
        "comparison_figures": "figures/comparison/v3p0_physical_event",
    }


def test_comparison_observed_annual_summary_converts_grams_to_kg(tmp_path: Path) -> None:
    source = pd.DataFrame({
        "source": ["Observed", "Bayes_Modeled"],
        "Year": [2020, 2020],
        "analyte": ["TN", "TN"],
        "treatment": ["CT", "CT"],
        "load_mean": [2500.0, 9999.0],
        "load_low": [2000.0, 9999.0],
        "load_high": [3000.0, 9999.0],
    })
    source.to_csv(
        tmp_path / "annual_load_summary_bayes_plus_observed_v3p0_physical_event.csv",
        index=False,
    )
    observed = read_observed_annual_summary(tmp_path)
    assert observed["center_kg"].tolist() == [2.5]
    assert observed["lower_95_kg"].tolist() == [2.0]
    assert observed["upper_95_kg"].tolist() == [3.0]


def test_complete_v3p0_figure_suites_are_wired_to_entrypoints() -> None:
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p0_physical_event.py"
    ).read_text(encoding="utf-8")
    comparison = (
        REPO / "code" / "comparison" / "bayes_ml_comparison_v3p0_physical_event.py"
    ).read_text(encoding="utf-8")
    assert "generate_postprocess_figure_suite(repo, output_dir, figure_dir, data_path, args)" in ml
    assert '"--figures-only"' in comparison
    assert "axis.fill_between" in comparison
    assert 'markerfacecolor="none"' in comparison
    assert "read_observed_annual_summary(bayes_dir)" in comparison


def test_ml_uncertainty_uses_empirical_calibration_residuals_not_uniform_bounds() -> None:
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p0_physical_event.py"
    ).read_text(encoding="utf-8")
    assert "empirical_calibration_residual_draws" in ml
    assert "calibration_signed_residuals_log" in ml
    assert "weighted_resampling_of_signed_log_scale_calibration_residuals" in ml
    assert "uniform_log_draws" not in ml
    assert "legacy_uniform_between_conformal_bounds_used\": False" in ml
