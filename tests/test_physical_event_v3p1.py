from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BAYES_V3P1_ARCHIVE = REPO / "old_code" / "versions" / "v3p1_physical_event"
sys.path.insert(0, str(REPO / "code"))

from shared.physical_event import (  # noqa: E402
    add_concentration_observation_id,
    add_physical_event_id,
    aggregate_replicate_mean,
    build_event_date_audit,
    build_event_analyte_load_ledger,
    build_event_analyte_point_load_ledger,
    build_volume_observation_table,
    event_balanced_weights,
    loyo_masks,
    observed_annual_plot_summary,
    resolve_prediction_draws,
    split_event_groups,
    validate_corrected_artifact_metadata,
    yearly_irrigation_roster,
)
from comparison.bayes_ml_comparison_v3p2_physical_event import (  # noqa: E402
    STUDY_YEARS,
    attach_primary_centers,
    bayes_negative_sensitivity,
    coverage_by_year_target,
    ct_relative,
    ml_annual_volume_products,
    normalize_ledger,
    normalize_point_ledger,
    overall_nrmse_table,
    observed_annual_load_completeness,
    observed_annual_volume_completeness,
    performance_table,
)
from ml.ml_postprocess_plots_v3p1_physical_event import observed_annual_loads  # noqa: E402
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


def test_date_distinct_volume_observations_are_not_discarded() -> None:
    frame = rows().copy()
    frame.loc[1, "Date"] = "2020-07-02"
    volume, _ = build_volume_observation_table(frame)
    assert len(volume) == 2
    assert volume["PhysicalEventID"].nunique() == 1


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
    frame["Rep"] = [1, 2]
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
        "Rep": [1, 1],
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
        performance_table(concentration, "ML", "Result_mg_L", "synthetic"),
        performance_table(volume, "ML", "Volume_L", "synthetic"),
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
    incomplete.write_text(json.dumps({"workflow_version": "v3p1_physical_event", "event_unit": "PhysicalEventID", "years": [2020]}))
    with pytest.raises(ValueError, match="Incomplete year"):
        validate_corrected_artifact_metadata([incomplete], expected_years=[2020, 2021])


def test_comparison_accepts_explicit_bayes_v3p2_ml_v3p1_pairing(
    tmp_path: Path,
) -> None:
    bayes = tmp_path / "bayes.json"
    ml = tmp_path / "ml.json"
    bayes.write_text(
        json.dumps(
            {
                "workflow_version": "v3p2_physical_event",
                "event_unit": "PhysicalEventID",
                "years": [2020],
            }
        )
    )
    ml.write_text(
        json.dumps(
            {
                "workflow_version": "v3p1_physical_event",
                "event_unit": "PhysicalEventID",
                "years": [2020],
            }
        )
    )
    records = validate_corrected_artifact_metadata(
        [bayes, ml],
        expected_years=[2020],
        expected_versions=["v3p2_physical_event", "v3p1_physical_event"],
    )
    assert [record["workflow_version"] for record in records] == [
        "v3p2_physical_event",
        "v3p1_physical_event",
    ]


def test_concentration_observation_ids_remain_row_unique() -> None:
    identified = add_concentration_observation_id(rows())
    assert identified["ConcentrationObservationID"].nunique() == len(identified)


def test_changing_date_alone_does_not_change_physical_event_id() -> None:
    text = rows().iloc[[0]].copy()
    timestamp = text.copy()
    timestamp["Date"] = "2020-07-02"
    assert (
        add_physical_event_id(text).loc[text.index[0], "PhysicalEventID"]
        == add_physical_event_id(timestamp).loc[timestamp.index[0], "PhysicalEventID"]
    )


def test_different_replicates_are_different_physical_events() -> None:
    frame = pd.concat(
        [rows().iloc[[0]], rows().iloc[[0]].assign(_wq_idx=99, Rep=2)],
        ignore_index=True,
    )
    identified = add_physical_event_id(frame)
    assert identified["PhysicalEventID"].nunique() == 2


def test_multi_date_rows_merge_and_select_unique_volume_event_date() -> None:
    frame = pd.concat(
        [
            rows().iloc[[0]],
            rows().iloc[[0]].assign(
                _wq_idx=99, Date="2020-07-02", Volume=np.nan, Result_mg_L=4.0
            ),
        ],
        ignore_index=True,
    )
    identified = add_physical_event_id(frame)
    assert identified["PhysicalEventID"].nunique() == 1
    volume, _ = build_volume_observation_table(frame)
    event_dates, multi_date, conflicts = build_event_date_audit(frame, volume)
    assert event_dates.loc[0, "EventDate"] == "2020-07-01"
    assert event_dates.loc[0, "EventDateResolution"] == (
        "unique_genuine_volume_observation_date"
    )
    assert len(multi_date) == 1
    assert conflicts.empty


def test_corrected_current_dataset_has_528_dynamic_roster_events() -> None:
    cleaned = pd.read_csv(REPO / "out" / "wq_cleaned.csv", low_memory=False)
    identified = add_physical_event_id(cleaned)
    events = identified[
        ["PhysicalEventID", "Year", "Irrigation", "Rep", "Treatment"]
    ].drop_duplicates()
    assert events["PhysicalEventID"].nunique() == 528
    numeric = pd.to_numeric(events["Irrigation"], errors="coerce")
    assert numeric.notna().sum() == 510
    assert numeric.isna().sum() == 18


def test_yearly_expected_irrigation_roster_uses_actual_labels() -> None:
    events = pd.DataFrame(
        {
            "Year": [2020, 2020, 2021, 2021, 2021],
            "Irrigation": ["1", "S1", "1", "2", "S2"],
        }
    )
    roster = yearly_irrigation_roster(events)
    assert set(roster.loc[roster["Year"].eq(2020), "Irrigation"]) == {"1", "S1"}
    assert set(roster.loc[roster["Year"].eq(2021), "Irrigation"]) == {
        "1", "2", "S2"
    }


def test_annual_load_sums_within_rep_before_averaging() -> None:
    ledger = pd.DataFrame(
        {
            "Method": ["Bayes"] * 4,
            "Scenario": ["model_only"] * 4,
            "Year": [2020] * 4,
            "Analyte": ["TN"] * 4,
            "Treatment": ["CT"] * 4,
            "Rep": [1, 1, 2, 2],
            "Draw": [1] * 4,
            "PhysicalEventID": ["E1", "E2", "E3", "E4"],
            "Load_kg": [4.0, 6.0, 7.0, 7.0],
        }
    )
    _, draws = aggregate_replicate_mean(
        ledger,
        value_column="Load_kg",
        group_columns=["Method", "Scenario", "Year", "Analyte", "Treatment"],
        draw_column="Draw",
        plot_total_column="PlotAnnualLoad_kg",
        treatment_mean_column="Load_kg",
    )
    assert draws.loc[0, "Load_kg"] == pytest.approx(12.0)
    assert draws.loc[0, "n_replicate_plots"] == 2


def test_ct_relative_reduction_uses_treatment_mean_draws() -> None:
    annual = pd.DataFrame(
        {
            "Method": ["Bayes"] * 3,
            "Scenario": ["model_only"] * 3,
            "Year": [2020] * 3,
            "Analyte": ["TN"] * 3,
            "Treatment": ["CT", "MT", "ST"],
            "Draw": [1] * 3,
            "Load_kg": [12.0, 9.0, 6.0],
        }
    )
    cumulative = annual.rename(columns={"Load_kg": "CumulativeLoad_kg"})
    raw, _ = ct_relative(cumulative)
    reductions = raw.set_index("ComparisonTreatment")[
        "PercentDifferenceRelativeToCT"
    ]
    assert reductions["MT"] == pytest.approx(25.0)
    assert reductions["ST"] == pytest.approx(50.0)


def test_comparison_requires_complete_primary_but_allows_incomplete_loyo_ledger() -> None:
    ledger = pd.DataFrame({
        "PhysicalEventID": ["A"], "Analyte": ["TN"], "Year": [2011],
        "Treatment": ["CT"], "Rep": [1], "Draw": [0], "Load_kg": [1.0],
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
    assert annual["n_complete_plots"].eq(1).all()


def test_v3p1_model_entrypoints_share_final_cleaned_input() -> None:
    backend = (REPO / "code" / "bayes" / "stir-bayes-backend.R").read_text(
        encoding="utf-8"
    )
    rmd = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "stir-bayes-load_v3p1_physical_event.Rmd"
    ).read_text(encoding="utf-8")
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p1_physical_event.py"
    ).read_text(encoding="utf-8")

    assert 'load_wq_stir <- function(path = "out/wq_cleaned.csv"' in backend
    assert "wq_with_stir_by_season.csv" not in backend
    assert "if (interactive())" not in backend
    assert "stir-bayes-load_v3p1_physical_event.R" in rmd
    assert "source(batch_script, chdir = FALSE)" in rmd
    assert 'repo / "out" / "wq_cleaned.csv"' in ml


def test_final_cleaning_preserves_storm_event_labels() -> None:
    cleaned = clean_wq_stir(pd.DataFrame({"Irrigation": ["1", "S1", "S2", "2"]}))
    assert cleaned["Irrigation"].astype("string").tolist() == ["1", "S1", "S2", "2"]
    assert cleaned.loc[cleaned["Irrigation"].isin(["S1", "S2"]), "IRR_z"].isna().all()
    assert cleaned.loc[cleaned["Irrigation"].isin(["1", "2"]), "IRR_z"].notna().all()


def test_bayesian_batch_runner_is_single_pass_and_uses_final_input() -> None:
    batch = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "stir-bayes-load_v3p1_physical_event.R"
    ).read_text(encoding="utf-8")

    assert batch.startswith("#!/usr/bin/env Rscript")
    assert 'file.path(repo_root, "out", "wq_cleaned.csv")' in batch
    assert "fit <- mod$sample(" in batch
    assert "force_recompile = .batch_force_recompile" in batch
    assert batch.count("cmdstan_dashboard(fit)") == 1
    assert 'Sys.getenv("BAYES_REUSE_FIT", "false")' in batch
    assert 'Sys.getenv("BAYES_REUSE_QUICK_DIAGNOSTICS", "false")' in batch
    assert "fit <- readRDS(fit_rds_path)" in batch
    assert "as_plain_numeric_matrix" in batch
    assert "post$C_true_event" in batch
    assert "post$V_true_event" in batch
    assert "concentration_mg_L <- pmax(" in batch
    assert "volume_L <- pmax(" in batch
    assert "Load_g = .data$Concentration_mg_L * .data$Volume_L / 1000" in batch
    assert "quick_sampling_diagnostics" in batch
    assert "rhat_threshold = 1.04" in batch
    assert '"total_parameters"' in batch
    assert '"parameters_rhat_gt_1p04"' in batch
    assert '"parameters_rhat_at_or_below_1p04"' in batch
    assert "make_key_parameter_map" in batch
    assert '"beta_res_V", "Residue effect on runoff volume"' in batch
    assert "attachments = c(quick_dashboard_png, quick_diagnostics_csv)" in batch
    assert "if (sampled_new_fit)" in batch
    assert "Reusing compact diagnostics already generated" in batch
    assert "pick_key_vars_v3p1" not in batch
    assert "bad_pars <-" not in batch
    assert "overall_diagnostics_bayes_v3p1_physical_event.csv" in batch
    assert "quantile_color =" not in batch
    assert "quantile_linetype =" not in batch
    assert "quantile_linewidth =" not in batch
    assert batch.count('vline_colour = "black"') == 6
    assert "event_prediction_base <- physical_event_tbl %>%" in batch
    assert "write_csv_retry <- function" in batch
    assert "readr::write_csv(" not in batch
    assert 'filter(as.character(.data$source) == "Modeled")' in batch


def test_saved_fit_residue_diagnostic_is_read_only_and_targeted() -> None:
    diagnostic = (
        BAYES_V3P1_ARCHIVE / "code" / "bayes" / "diagnose_saved_fit_v3p1.R"
    ).read_text(encoding="utf-8")

    assert "readRDS(fit_path)" in diagnostic
    assert "mod$sample(" not in diagnostic
    assert "cmdstan_model(" not in diagnostic
    assert '"beta_res_V"' in diagnostic
    assert '"residue_parameter_diagnostics_v3p1_physical_event.csv"' in diagnostic
    assert '"residue_chain_diagnostics_v3p1_physical_event.csv"' in diagnostic
    assert '"residue_volume_by_divergence_v3p1_physical_event.csv"' in diagnostic
    assert (
        '"residue_volume_parameter_correlations_v3p1_physical_event.csv"'
        in diagnostic
    )


def test_v3p1_outputs_are_isolated_in_version_folders() -> None:
    config = json.loads(
        (REPO / "config" / "physical_event_v3p1.json").read_text(encoding="utf-8")
    )
    assert config["output_roots"] == {
        "bayesian_results": "results/bayes/v3p1_physical_event",
        "ml_results": "results/ml/v3p1_physical_event",
        "comparison_results": "results/comparison/v3p1_physical_event",
        "bayesian_figures": "figures/bayes/v3p1_physical_event",
        "ml_figures": "figures/ml/v3p1_physical_event",
        "comparison_figures": "figures/comparison/v3p1_physical_event",
    }
    assert config["bayesian_prediction_resolution"] == {
        "concentration": "latent_event_analyte_truth",
        "volume": "latent_physical_event_truth",
        "observed_values": "likelihood_and_reference_only",
    }


def test_observed_annual_load_uses_complete_plot_mean_and_range() -> None:
    expected = pd.DataFrame(
        {
            "Year": [2020] * 4,
            "Irrigation": [1, 2, 1, 2],
            "Rep": [1, 1, 2, 2],
            "Analyte": ["TN"] * 4,
            "Treatment": ["CT"] * 4,
            "PhysicalEventID": ["E1", "E2", "E3", "E4"],
        }
    )
    observed_events = pd.DataFrame(
        {
            "Year": [2020] * 4,
            "Irrigation": [1, 2, 1, 2],
            "Rep": [1, 1, 2, 2],
            "Analyte": ["TN"] * 4,
            "Treatment": ["CT"] * 4,
            "PhysicalEventID": ["E1", "E2", "E3", "E4"],
            "Load_kg": [4.0, 6.0, 7.0, 7.0],
        }
    )
    audit = observed_annual_load_completeness(
        expected,
        observed_events,
    )
    assert audit["n_complete_plots"].eq(2).all()
    assert audit["ObservedAnnualLoad_kg"].eq(12.0).all()
    assert audit["ObservedAnnualLoadRangeLow_kg"].eq(10.0).all()
    assert audit["ObservedAnnualLoadRangeHigh_kg"].eq(14.0).all()
    assert audit["ObservedIntervalType"].eq(
        "descriptive replicate minimum-to-maximum range"
    ).all()


def test_one_complete_plot_has_value_without_interval_and_incomplete_subtotal_is_excluded() -> None:
    expected = pd.DataFrame(
        {
            "Year": [2020] * 4,
            "Irrigation": [1, 2, 1, 2],
            "Rep": [1, 1, 2, 2],
            "Analyte": ["TN"] * 4,
            "Treatment": ["CT"] * 4,
            "PhysicalEventID": ["E1", "E2", "E3", "E4"],
        }
    )
    observed = pd.DataFrame(
        {
            "Year": [2020] * 3,
            "Irrigation": [1, 2, 1],
            "Rep": [1, 1, 2],
            "Analyte": ["TN"] * 3,
            "Treatment": ["CT"] * 3,
            "PhysicalEventID": ["E1", "E2", "E3"],
            "Load_kg": [4.0, 6.0, 1000.0],
        }
    )
    audit = observed_annual_load_completeness(expected, observed)
    assert audit["n_complete_plots"].eq(1).all()
    assert audit["ObservedAnnualLoad_kg"].eq(10.0).all()
    assert audit["ObservedAnnualLoadRangeLow_kg"].isna().all()
    assert audit["ObservedAnnualLoadRangeHigh_kg"].isna().all()
    incomplete = audit.loc[audit["Rep"].eq(2)].iloc[0]
    assert incomplete["ObservedIncompleteSubtotal_kg"] == 1000.0
    assert pd.isna(incomplete["ObservedReplicateAnnualLoad_kg"])


def test_no_complete_plots_produce_na() -> None:
    expected = pd.DataFrame(
        {
            "Year": [2020, 2020],
            "Irrigation": [1, 2],
            "Rep": [1, 1],
            "Treatment": ["CT", "CT"],
            "PhysicalEventID": ["E1", "E2"],
        }
    )
    observed = pd.DataFrame(
        {
            "Year": [2020],
            "Irrigation": [1],
            "Rep": [1],
            "Treatment": ["CT"],
            "PhysicalEventID": ["E1"],
            "Volume_L": [100.0],
        }
    )
    audit = observed_annual_plot_summary(
        expected,
        observed,
        value_column="Volume_L",
    )
    assert audit["n_complete_plots"].eq(0).all()
    assert audit["TreatmentMean"].isna().all()


def test_plot_completeness_applies_year_wide_irrigation_roster() -> None:
    expected = pd.DataFrame(
        {
            "Year": [2020, 2020, 2020],
            "Irrigation": [1, 2, 1],
            "Rep": [1, 1, 2],
            "Treatment": ["CT", "CT", "CT"],
            "PhysicalEventID": ["E1", "E2", "E3"],
        }
    )
    observed = expected.assign(Volume_L=[4.0, 6.0, 10.0])

    audit = observed_annual_plot_summary(
        expected,
        observed,
        value_column="Volume_L",
    )

    rep2 = audit.loc[audit["Rep"].eq(2)].iloc[0]
    assert rep2["ExpectedEventCount"] == 2
    assert rep2["ObservedEventCount"] == 1
    assert rep2["MissingIrrigationLabels"] == "2"
    assert not rep2["PlotComplete"]
    assert pd.isna(rep2["ReplicateAnnualValue"])


def test_ml_annual_volume_draws_sum_within_rep_before_averaging() -> None:
    point = pd.DataFrame(
        {
            "PhysicalEventID": ["E1", "E2", "E3", "E4"],
            "Year": [2020] * 4,
            "Treatment": ["CT"] * 4,
            "Rep": [1, 1, 2, 2],
            "Volume_L": [4.0, 6.0, 5.0, 9.0],
            "Analyte": ["TN"] * 4,
        }
    )
    draw0 = point.assign(Draw=0)
    draw1 = point.assign(Draw=1, Volume_L=point["Volume_L"] * 2)
    draws = pd.concat([draw0, draw1], ignore_index=True)

    annual = ml_annual_volume_products(point, draws)

    assert annual.loc[0, "PointTreatmentMeanVolume_L"] == 12.0
    assert annual.loc[0, "draw_mean_L"] == 18.0
    assert annual.loc[0, "n_replicate_plots"] == 2
    assert annual.loc[0, "annual_reporting_unit"] == "mean_per_treatment_plot"


def test_complete_v3p1_figure_suites_are_wired_to_entrypoints() -> None:
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p1_physical_event.py"
    ).read_text(encoding="utf-8")
    comparison = (
        REPO / "code" / "comparison" / "bayes_ml_comparison_v3p2_physical_event.py"
    ).read_text(encoding="utf-8")
    assert "generate_postprocess_figure_suite(repo, output_dir, figure_dir, data_path, args)" in ml
    assert '"--figures-only"' in comparison
    assert "axis.fill_between" in comparison
    assert 'markerfacecolor="none"' in comparison
    assert "observed_annual_plot_summary" in comparison


def test_ml_uncertainty_uses_empirical_calibration_residuals_not_uniform_bounds() -> None:
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p1_physical_event.py"
    ).read_text(encoding="utf-8")
    assert "empirical_calibration_residual_draws" in ml
    assert "calibration_signed_residuals_log" in ml
    assert "weighted_resampling_of_signed_log_scale_calibration_residuals" in ml
    assert "uniform_log_draws" not in ml
    assert "legacy_uniform_between_conformal_bounds_used\": False" in ml


def test_ml_separated_run_has_nonreconstructing_final_model_stage() -> None:
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p1_physical_event.py"
    ).read_text(encoding="utf-8")
    workflow = (
        REPO / "docs" / "workflows" / "ml_v3p1_physical_event.md"
    ).read_text(encoding="utf-8")

    assert '"--fit_final_models_only"' in ml
    assert "if args.fit_final_models_only:" in ml
    assert "reconstruction not started" in ml
    assert "--fit_final_models_only --no-figures" in workflow
    assert workflow.index("--no-impute_missing") < workflow.index(
        "--fit_final_models_only"
    ) < workflow.index("ml_regenerate_from_saved_models_v3p1.py")


def test_primary_models_predict_observed_and_missing_points_without_substitution() -> None:
    ml = (
        REPO / "code" / "ml" / "ml_catboost_conformal_loyo_v3p1_physical_event.py"
    ).read_text(encoding="utf-8")
    plots = (
        REPO / "code" / "ml" / "ml_postprocess_plots_v3p1_physical_event.py"
    ).read_text(encoding="utf-8")
    comparison = (
        REPO / "code" / "comparison" / "bayes_ml_comparison_v3p2_physical_event.py"
    ).read_text(encoding="utf-8")

    assert '"primary_uses_observed_value_substitution": False' in ml
    assert "default=False" in plots
    assert "event_analyte_draw_ledger_observed_plus_imputed_sensitivity.csv" not in comparison
    assert "event_analyte_point_ledger_observed_plus_imputed_sensitivity.csv" not in comparison


def test_bayes_models_prespecified_ten_analytes_and_all_subset_rows() -> None:
    expected = {
        "OP", "TP", "NO3", "TN", "TSS",
        "TKN", "NH4", "Se", "NO2", "TDS",
    }
    ml_only = {"ICP", "TSP", "NPOC", "NOx"}
    source = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "stir-bayes-load_v3p1_physical_event.R"
    ).read_text(encoding="utf-8")
    active_analytes = source.split("analytes_keep <- c(", 1)[1].split(")", 1)[0]
    assert all(f'"{analyte}"' in active_analytes for analyte in expected)
    assert all(f'"{analyte}"' not in active_analytes for analyte in ml_only)
    assert "observed_values_substituted_into_modeled_products = FALSE" in source
    assert "complete 528 physical events x 10 prespecified Bayes analytes" in source
    assert "tidyr::crossing(" in source
    assert "nrow(d_pred) == E_n * A_n" in source
    assert "drop_unmapped_analytes = FALSE" in source
    assert "drop_missing_c_stats = FALSE" in source

    stan = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "m_stir_mogp_v3p1_physical_event.stan"
    ).read_text(encoding="utf-8")
    assert "matrix[E_n, A_n] C_true_event" in stan
    assert "cholesky_factor_corr[A_n] L_corr_C_event" in stan
    assert "vector[EA_n] CIN_EA" in stan
    assert "if (is_C_miss[i] == 0)" in stan


def test_bayes_v3p1_prior_and_observation_layers_match_approved_structure() -> None:
    stan = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "m_stir_mogp_v3p1_physical_event.stan"
    ).read_text(encoding="utf-8")
    batch = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "stir-bayes-load_v3p1_physical_event.R"
    ).read_text(encoding="utf-8")

    parameters = stan.split("parameters {", 1)[1].split(
        "transformed parameters {", 1
    )[0]
    concentration_process = stan.split(
        "mu_C_event[e, a] = alpha[a]", 1
    )[1].split("C_true_event =", 1)[0]

    assert "vector[A_n] beta_vol;" in stan
    assert "real mu_beta_vol;" in parameters
    assert "real<lower=0> sigma_beta_vol;" in parameters
    assert "cholesky_factor_corr[A_n] L_corr_C_event;" in parameters
    assert "matrix[E_n, A_n] Z_C_event;" in parameters
    assert "gamma_B[a, B[r]]" in concentration_process
    assert "gamma_S" not in concentration_process
    assert "gamma_F" not in concentration_process
    assert "beta_dup" not in concentration_process
    assert "+ beta_dup[A[i]] * DUP[i]" in stan
    assert "+ gamma_S[A[i], S[i]]" in stan
    assert "+ gamma_F[A[i], Fu[i]]" in stan
    assert "Lab" not in stan
    assert "MeasureMethod" not in stan
    assert "CIN_EA_impute ~ std_normal();" in stan
    assert "VIN_event_impute ~ std_normal();" in stan
    assert "C_cens_limit[censored_idx] <- (" in batch
    assert batch.count("C_cens_limit[censored_idx] <- (") == 1
    assert "do not calculate censoring a second time" in batch


def test_bayes_postprocessing_uses_exported_volume_column_name() -> None:
    source = (
        BAYES_V3P1_ARCHIVE
        / "code"
        / "bayes"
        / "stir-bayes-load_v3p1_physical_event.R"
    ).read_text(encoding="utf-8")
    assert "plot_volume_L = sum(.data$Volume_L)" in source
    assert "plot_volume_L = sum(.data$volume_L)" not in source
