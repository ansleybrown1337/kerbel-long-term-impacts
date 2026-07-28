from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "code" / "pipeline" / "merge_wq_stir_by_season.py"
SPEC = importlib.util.spec_from_file_location("merge_wq_stir_by_season", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seasonal_stir = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seasonal_stir)


def _stir_cumulative() -> pd.DataFrame:
    stir = pd.DataFrame(
        {
            "System": ["CT"] * 5,
            "Date": pd.to_datetime(
                [
                    "2020-10-10",  # preceding harvest date
                    "2020-11-01",  # fall tillage after harvest
                    "2021-03-01",  # spring tillage before planting
                    "2021-04-15",  # operation on planting date
                    "2021-05-01",  # in-season operation
                ]
            ),
            "STIR_val": [5.0, 10.0, 20.0, 30.0, 40.0],
            "STIR_nonharvest_same_day": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    return seasonal_stir.compute_cumulative_stir(stir)


def _wq_season(previous_harvest: str | None = "2020-10-10") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Treatment": ["CT"],
            "Date": pd.to_datetime(["2021-06-01"]),
            "PlantDate": pd.to_datetime(["2021-04-15"]),
            "HarvestDate": pd.to_datetime(["2021-10-15"]),
            "PreviousHarvestDate": pd.to_datetime([previous_harvest]),
            "SeasonYear": [2021],
        }
    )


def test_postharvest_stir_carries_fall_and_preplant_operations_forward() -> None:
    merged = seasonal_stir.merge_stir_with_wq(
        _wq_season(),
        _stir_cumulative(),
        season_anchor="postharvest",
    )

    # The preceding harvest-day operation stays with the preceding crop.
    # Every later operation through the runoff sample belongs to the new crop.
    assert merged.loc[0, "CumAll_STIR_toDate"] == pytest.approx(105.0)
    assert merged.loc[0, "Season_STIR_toDate"] == pytest.approx(100.0)
    assert merged.loc[0, "Season_STIR_StartDate"] == pd.Timestamp("2020-10-10")
    assert not bool(merged.loc[0, "Season_STIR_LeftCensored"])
    assert merged.loc[0, "Season_STIR_BoundaryDayCarryover"] == pytest.approx(0.0)


def test_nonharvest_operations_on_harvest_date_carry_to_next_crop() -> None:
    stir = _stir_cumulative()
    stir.loc[
        stir["Date"].eq(pd.Timestamp("2020-10-10")),
        "STIR_nonharvest_same_day",
    ] = 2.0

    merged = seasonal_stir.merge_stir_with_wq(
        _wq_season(),
        stir,
        season_anchor="postharvest",
    )

    # Of the boundary-date STIR total of 5, two units are a non-harvest
    # operation and therefore carry into the following crop.
    assert merged.loc[0, "Season_STIR_toDate"] == pytest.approx(102.0)
    assert merged.loc[0, "Season_STIR_BoundaryDayCarryover"] == pytest.approx(2.0)


def test_read_stir_separates_harvest_from_same_day_followup_operations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stir.csv"
    pd.DataFrame(
        {
            "Date": ["2020-10-10", "2020-10-10"],
            "System": ["CT", "CT"],
            "Operation (verbatim)": ["Harvest Grain Corn", "Shred Stalks"],
            "STIR_Event": [5.0, 7.0],
        }
    ).to_csv(source, index=False)

    daily = seasonal_stir.read_stir(str(source))

    assert daily.loc[0, "STIR_val"] == pytest.approx(12.0)
    assert daily.loc[0, "STIR_nonharvest_same_day"] == pytest.approx(7.0)


def test_calendar_and_plant_anchors_do_not_match_postharvest_definition() -> None:
    postharvest = seasonal_stir.merge_stir_with_wq(
        _wq_season(),
        _stir_cumulative(),
        season_anchor="postharvest",
    )
    calendar = seasonal_stir.merge_stir_with_wq(
        _wq_season(),
        _stir_cumulative(),
        season_anchor="calendar",
    )
    plant = seasonal_stir.merge_stir_with_wq(
        _wq_season(),
        _stir_cumulative(),
        season_anchor="plant",
    )

    assert postharvest.loc[0, "Season_STIR_toDate"] == pytest.approx(100.0)
    assert calendar.loc[0, "Season_STIR_toDate"] == pytest.approx(90.0)
    assert plant.loc[0, "Season_STIR_toDate"] == pytest.approx(70.0)


def test_first_observed_season_is_explicitly_left_censored() -> None:
    merged = seasonal_stir.merge_stir_with_wq(
        _wq_season(previous_harvest=None),
        _stir_cumulative(),
        season_anchor="postharvest",
    )

    assert merged.loc[0, "Season_STIR_toDate"] == pytest.approx(105.0)
    assert pd.isna(merged.loc[0, "Season_STIR_StartDate"])
    assert bool(merged.loc[0, "Season_STIR_LeftCensored"])


def test_crop_windows_carry_the_preceding_harvest_boundary() -> None:
    crops = pd.DataFrame(
        {
            "PlantDate": pd.to_datetime(["2020-04-15", "2021-04-15"]),
            "HarvestDate": pd.to_datetime(["2020-10-10", "2021-10-15"]),
            "SeasonYear": [2020, 2021],
            "Crop": ["grain corn", "barley"],
        }
    )
    wq = pd.DataFrame(
        {
            "Treatment": ["CT", "MT"],
            "Date": pd.to_datetime(["2021-06-01", "2021-06-01"]),
        }
    )

    attached = seasonal_stir.attach_season_windows(wq, crops)

    assert set(attached["Treatment"]) == {"CT", "MT"}
    assert set(attached["Crop"]) == {"barley"}
    assert set(attached["previous_crop"]) == {"grain corn"}
    assert set(attached["PreviousHarvestDate"]) == {pd.Timestamp("2020-10-10")}
