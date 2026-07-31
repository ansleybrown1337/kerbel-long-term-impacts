import importlib.util
from pathlib import Path
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "code" / "pipeline" / "merge_residue.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("merge_residue_v3p1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _season(
    treatment: str,
    plant: str,
    harvest: str,
    crop: str,
    previous_harvest: str,
    previous_crop: str,
) -> dict:
    return {
        "Treatment": treatment,
        "PlantDate": plant,
        "HarvestDate": harvest,
        "SeasonYear": pd.Timestamp(plant).year,
        "Crop": crop,
        "PreviousHarvestDate": previous_harvest,
        "previous_crop": previous_crop,
    }


def test_confirmed_residue_dates_map_only_to_spring_crop():
    rows = [
        _season("CT", "2018-05-18", "2018-09-04", "dry beans", "2017-10-13", "silage corn"),
        _season("CT", "2018-09-15", "2019-08-02", "winter wheat", "2018-09-04", "dry beans"),
        _season("CT", "2023-05-02", "2023-09-08", "silage corn", "2022-10-24", "grain corn"),
        _season("CT", "2023-09-13", "2024-08-02", "winter wheat", "2023-09-08", "silage corn"),
    ]
    wq = pd.DataFrame(rows)
    dates = MODULE.read_measurement_dates(
        REPO / "data" / "residue_measurement_dates_v3p1.csv"
    )
    seasons = MODULE.build_season_table(wq, dates)

    bean = seasons.loc[seasons["Crop"].eq("dry beans")].iloc[0]
    wheat_2018 = seasons.loc[
        seasons["PlantDate"].eq(pd.Timestamp("2018-09-15"))
    ].iloc[0]
    silage = seasons.loc[seasons["Crop"].eq("silage corn")].iloc[0]
    wheat_2023 = seasons.loc[
        seasons["PlantDate"].eq(pd.Timestamp("2023-09-13"))
    ].iloc[0]

    assert bean["ResidueModelDate"] == pd.Timestamp("2018-05-22")
    assert silage["ResidueModelDate"] == pd.Timestamp("2023-05-01")
    assert wheat_2018["ResidueModelDate"] == pd.Timestamp("2018-09-15")
    assert wheat_2023["ResidueModelDate"] == pd.Timestamp("2023-09-13")
    assert bool(bean["ResidueAssignmentEligible"])
    assert bool(silage["ResidueAssignmentEligible"])
    assert not bool(wheat_2018["ResidueAssignmentEligible"])
    assert not bool(wheat_2023["ResidueAssignmentEligible"])


def test_full_data_has_90_plot_seasons_54_observed_residue_units():
    wq_path = REPO / "out" / "pipeline_csvs" / "wq_with_stir_by_season.csv"
    stir_path = REPO / "out" / "pipeline_csvs" / "stir_events_long.csv"
    if not wq_path.exists() or not stir_path.exists():
        return

    wq = pd.read_csv(wq_path)
    residue = pd.read_csv(REPO / "data" / "residue_2011_2025.csv")
    dates = MODULE.read_measurement_dates(
        REPO / "data" / "residue_measurement_dates_v3p1.csv"
    )
    _, audit = MODULE.merge_residue(
        wq,
        MODULE.aggregate_residue(residue),
        dates,
        stir_path,
    )

    assert len(audit) == 90
    assert audit["Residue_PercentCover"].notna().sum() == 54
    winter = audit["Crop"].astype(str).str.lower().eq("winter wheat")
    assert audit.loc[winter, "Residue_PercentCover"].isna().all()
    assert audit["Residue_STIR_toMeasurement"].notna().all()
