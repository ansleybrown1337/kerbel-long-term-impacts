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
    PHYSICAL_EVENT_KEY,
    add_physical_event_id,
    aggregate_replicate_mean,
    build_event_analyte_load_ledger,
    build_volume_observation_table,
    event_balanced_weights,
    resolve_prediction_draws,
    yearly_irrigation_roster,
)


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


def test_physical_event_key_excludes_date_sample_and_analyte() -> None:
    frame = rows()
    frame.loc[1, "Date"] = "2020-07-02"
    identified = add_physical_event_id(frame)
    assert PHYSICAL_EVENT_KEY == ["Year", "Irrigation", "Rep", "Treatment"]
    assert identified["PhysicalEventID"].nunique() == 1


def test_replicates_are_distinct_physical_events() -> None:
    frame = rows()
    frame.loc[1, "Rep"] = 2
    identified = add_physical_event_id(frame)
    assert identified["PhysicalEventID"].nunique() == 2


def test_copied_volume_rows_do_not_add_observations() -> None:
    volume, reports = build_volume_observation_table(rows())
    assert len(volume) == 1
    assert len(reports["copied_volume_values"]) == 2
    assert reports["copied_volume_values"]["source_row_count"].eq(2).all()


def test_genuine_parallel_volume_methods_are_retained() -> None:
    frame = rows()
    frame.loc[1, "MeasureMethod"] = "Transducer"
    frame.loc[1, "Volume"] = 101.0
    volume, reports = build_volume_observation_table(frame)
    assert len(volume) == 2
    assert reports["ambiguous_volume_observations"].empty


def test_confirmed_zero_is_observed_and_missing_is_not() -> None:
    frame = rows()
    frame["Volume"] = [0.0, np.nan]
    frame["NoRunoff"] = [True, False]
    frame["Date"] = ["2020-07-01", "2020-07-02"]
    frame["Rep"] = [1, 2]
    volume, reports = build_volume_observation_table(frame)
    assert volume["Volume"].tolist() == [0.0]
    assert len(reports["events_without_volume_observation"]) == 1


def test_event_balanced_weights_sum_to_one_within_event_analyte() -> None:
    frame = add_physical_event_id(rows())
    frame = pd.concat([frame, frame.iloc[[0]].assign(_wq_idx=3)], ignore_index=True)
    weights = event_balanced_weights(frame, ["PhysicalEventID", "Analyte"])
    sums = weights.groupby([frame["PhysicalEventID"], frame["Analyte"]]).sum()
    assert np.allclose(sums, 1.0)


def test_prediction_resolution_occurs_after_prediction() -> None:
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
    median = resolve_prediction_draws(
        draws, group_columns=groups, value_column="value"
    )
    assert median.loc[0, "value"] == 5
    mean = resolve_prediction_draws(
        draws, group_columns=groups, value_column="value", method="mean"
    )
    assert mean.loc[0, "value"] == pytest.approx(13 / 3)


def test_event_load_ledger_is_unique_and_uses_physical_units() -> None:
    concentration = pd.DataFrame(
        {
            "PhysicalEventID": ["A"],
            "Analyte": ["TN"],
            "Draw": [1],
            "Concentration_mg_L": [2.0],
        }
    )
    volume = pd.DataFrame(
        {"PhysicalEventID": ["A"], "Draw": [1], "Volume_L": [100.0]}
    )
    ledger = build_event_analyte_load_ledger(concentration, volume)
    assert ledger.loc[0, "Load_kg"] == pytest.approx(0.0002)
    with pytest.raises(ValueError, match="must first resolve"):
        build_event_analyte_load_ledger(
            pd.concat([concentration, concentration], ignore_index=True), volume
        )


def test_annual_aggregation_sums_within_rep_then_averages_plots() -> None:
    frame = pd.DataFrame(
        {
            "Year": [2020, 2020, 2020, 2020],
            "Treatment": ["CT"] * 4,
            "Analyte": ["TN"] * 4,
            "Rep": [1, 1, 2, 2],
            "Draw": [1] * 4,
            "Load_kg": [1.0, 2.0, 10.0, 20.0],
        }
    )
    plots, treatment = aggregate_replicate_mean(
        frame,
        value_column="Load_kg",
        group_columns=["Year", "Treatment", "Analyte"],
        draw_column="Draw",
    )
    assert sorted(plots["PlotTotal"].tolist()) == [3.0, 30.0]
    assert treatment.loc[0, "TreatmentMean"] == 16.5
    assert treatment.loc[0, "n_replicate_plots"] == 2


def test_yearly_roster_uses_actual_labels_within_year() -> None:
    frame = pd.DataFrame(
        {
            "Year": [2020, 2020, 2021, 2021],
            "Irrigation": [1, "S1", 1, 2],
        }
    )
    roster = yearly_irrigation_roster(frame)
    labels_2020 = set(roster.loc[roster["Year"].eq(2020), "Irrigation"].astype(str))
    labels_2021 = set(roster.loc[roster["Year"].eq(2021), "Irrigation"].astype(str))
    assert labels_2020 == {"1", "S1"}
    assert labels_2021 == {"1", "2"}


def test_included_preflight_matches_corrected_release_contract() -> None:
    preflight = REPO / "results" / "preflight"
    metadata = json.loads(
        (preflight / "preflight_metadata.json").read_text(encoding="utf-8")
    )
    events = pd.read_csv(preflight / "physical_events.csv", low_memory=False)
    assert metadata["workflow_version"] == "v3p4_physical_event"
    assert metadata["input"] == "out/wq_cleaned.csv"
    assert metadata["ready_for_model_execution"] is True
    assert metadata["blocking_rows"] == 0
    assert metadata["physical_events"] == 528
    assert metadata["numeric_irrigation_events"] == 510
    assert metadata["storm_events"] == 18
    assert len(events) == 528
    assert events["PhysicalEventID"].nunique() == 528
    assert (
        preflight / "BLOCKING_REVIEW.csv"
    ).read_text(encoding="utf-8").strip() == "BlockingFinding"


def test_cleaned_release_table_preserves_storm_labels() -> None:
    cleaned = pd.read_csv(REPO / "out" / "wq_cleaned.csv", usecols=["Irrigation"])
    labels = set(cleaned["Irrigation"].dropna().astype(str))
    assert {"S1", "S2"}.issubset(labels)


def test_v3p3_manifest_and_diagnostics_preserve_release_qualification() -> None:
    results = REPO / "results" / "bayes" / "v3p3_physical_event"
    manifest = json.loads(
        (results / "run_manifest_bayes_v3p3_physical_event.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = pd.read_csv(results / "quick_diagnostics_v3p3_physical_event.csv")
    metrics = diagnostics.loc[diagnostics["category"].eq("global")].set_index(
        "metric"
    )["value"]
    assert manifest["workflow_version"] == "v3p3_physical_event"
    assert manifest["data_contract_version"] == "v3p4_physical_event"
    assert manifest["physical_event_key"] == PHYSICAL_EVENT_KEY
    assert manifest["annual_reporting_unit"] == "mean_per_treatment_plot"
    assert metrics["parameters_rhat_gt_1p04"] == 20
    assert metrics["divergences"] == 28
