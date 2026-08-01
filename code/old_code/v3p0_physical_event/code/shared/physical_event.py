"""Physical-event identities and post-prediction resolution for v3p0.

The scientific unit contract is deliberately centralized here so the Bayesian,
ML, and comparison workflows cannot silently invent different event keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "physical_event_v3p0.json"
if not _CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Shared physical-event configuration is absent: {_CONFIG_PATH}")
PHYSICAL_EVENT_CONFIG = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
PHYSICAL_EVENT_KEY = list(PHYSICAL_EVENT_CONFIG["physical_event_key"])
VOLUME_PROVENANCE_COLUMNS = list(PHYSICAL_EVENT_CONFIG["volume_provenance_columns"])
ANALYTE_COLUMN_CANDIDATES = ["analyte_abbr", "Analyte"]
CORRECTED_VERSION = str(PHYSICAL_EVENT_CONFIG["workflow_version"])


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required column(s): {', '.join(missing)}")


def _normal(value: object) -> str:
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".15g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return " ".join(str(value).strip().split())


def _stable_id(prefix: str, values: Iterable[object]) -> str:
    payload = "\x1f".join(_normal(value) for value in values).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _truthy(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: False
        if pd.isna(value)
        else str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
    )


def add_physical_event_id(
    df: pd.DataFrame,
    *,
    allow_missing_key: bool = False,
) -> pd.DataFrame:
    """Return a copy with a deterministic ``PhysicalEventID``.

    Missing key values are rejected in model-facing use. The audit calls this
    with ``allow_missing_key=True`` so incomplete records can be reported.
    """

    _require_columns(df, PHYSICAL_EVENT_KEY, "Physical-event input")
    output = df.copy()
    missing = output[PHYSICAL_EVENT_KEY].isna() | output[PHYSICAL_EVENT_KEY].apply(
        lambda column: column.astype(str).str.strip().eq("")
    )
    if missing.any(axis=1).any() and not allow_missing_key:
        counts = missing.sum().loc[lambda values: values.gt(0)].to_dict()
        raise ValueError(
            "PhysicalEventID cannot be constructed from incomplete keys; "
            f"missing counts: {counts}. Run the physical-event preflight audit."
        )
    identity = output[PHYSICAL_EVENT_KEY].copy()
    # The audit reads Date from CSV as text, while ML engineering converts it
    # to Timestamp. Canonicalize it here so every workflow creates the same ID.
    parsed_date = pd.to_datetime(identity["Date"], errors="coerce")
    canonical_date = identity["Date"].astype("string")
    canonical_date.loc[parsed_date.notna()] = parsed_date.loc[
        parsed_date.notna()
    ].dt.strftime("%Y-%m-%d")
    identity["Date"] = canonical_date
    output["PhysicalEventID"] = [
        _stable_id("PE", values)
        for values in identity.itertuples(index=False, name=None)
    ]
    return output


def add_concentration_observation_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign one ID to every analyte result row without collapsing rows."""

    output = df.copy()
    if "_wq_idx" in output.columns:
        source = output["_wq_idx"]
        if source.isna().any() or source.astype(str).duplicated().any():
            raise ValueError("_wq_idx must be complete and unique when present.")
        output["ConcentrationObservationID"] = [
            _stable_id("CO", [value]) for value in source
        ]
    else:
        output["ConcentrationObservationID"] = [
            _stable_id("CO", [position, *values])
            for position, values in enumerate(
                output.reindex(columns=PHYSICAL_EVENT_KEY).itertuples(index=False, name=None)
            )
        ]
    if output["ConcentrationObservationID"].duplicated().any():
        raise AssertionError("ConcentrationObservationID is not unique.")
    return output


def build_volume_observation_table(
    df: pd.DataFrame,
    *,
    provenance_columns: Sequence[str] = VOLUME_PROVENANCE_COLUMNS,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Deduplicate copied volume values without pre-fit aggregation.

    Exact copies across analyte/sample/duplicate rows collapse when physical
    event, available measurement provenance, and numeric value all match.
    Different methods or different recorded values remain separate observation
    rows. Multiple values within identical available provenance are reported as
    blocking ambiguities because the cleaned table lacks a stronger source ID.
    """

    _require_columns(df, [*PHYSICAL_EVENT_KEY, "Volume"], "Volume input")
    work = add_physical_event_id(df, allow_missing_key=True)
    work["Volume"] = pd.to_numeric(work["Volume"], errors="coerce")
    available_provenance = [column for column in provenance_columns if column in work.columns]
    if not available_provenance:
        work["_VolumeProvenance"] = "unspecified"
        available_provenance = ["_VolumeProvenance"]

    no_runoff = _truthy(work["NoRunoff"]) if "NoRunoff" in work.columns else pd.Series(False, index=work.index)
    synthetic_zero = work["Volume"].isna() & no_runoff
    work.loc[synthetic_zero, "Volume"] = 0.0
    work["_confirmed_zero"] = no_runoff & work["Volume"].eq(0)
    candidates = work.loc[work["Volume"].notna()].copy()

    identity_columns = ["PhysicalEventID", *available_provenance, "Volume"]
    candidates["VolumeObservationID"] = [
        _stable_id("VO", values)
        for values in candidates[identity_columns].itertuples(index=False, name=None)
    ]
    copy_counts = (
        candidates.groupby("VolumeObservationID", dropna=False)
        .size()
        .rename("source_row_count")
        .reset_index()
    )
    observations = (
        candidates.sort_index()
        .drop_duplicates("VolumeObservationID", keep="first")
        .merge(copy_counts, on="VolumeObservationID", how="left", validate="one_to_one")
    )
    keep = [
        "VolumeObservationID", "PhysicalEventID", *PHYSICAL_EVENT_KEY,
        *available_provenance, "Volume", "_confirmed_zero", "source_row_count",
    ]
    source_columns = [
        column for column in ["_wq_idx", "SampleID", "Duplicate", "NoRunoff", "Notes"]
        if column in observations.columns
    ]
    observations = observations[[*keep, *source_columns]].reset_index(drop=True)

    provenance_group = ["PhysicalEventID", *available_provenance]
    provenance_stats = (
        observations.groupby(provenance_group, dropna=False)
        .agg(
            n_distinct_values=("Volume", "nunique"),
            volume_values=("Volume", lambda values: ";".join(format(v, ".15g") for v in sorted(set(values)))),
            n_volume_observations=("VolumeObservationID", "size"),
        )
        .reset_index()
    )
    ambiguous_keys = provenance_stats.loc[provenance_stats["n_distinct_values"].gt(1)]
    ambiguous = observations.merge(ambiguous_keys, on=provenance_group, how="inner")

    status = (
        work.groupby("PhysicalEventID", dropna=False)
        .agg(
            any_no_runoff=("_confirmed_zero", "any"),
            any_zero=("Volume", lambda values: values.eq(0).any()),
            any_positive=("Volume", lambda values: values.gt(0).any()),
        )
        .reset_index()
    )
    zero_conflicts = status.loc[
        status["any_positive"] & (status["any_zero"] | status["any_no_runoff"])
    ]
    missing_key = work.loc[
        work[PHYSICAL_EVENT_KEY].isna().any(axis=1),
        [column for column in [*PHYSICAL_EVENT_KEY, "PhysicalEventID", "_wq_idx", "SampleID", "Volume", *available_provenance] if column in work.columns],
    ].drop_duplicates()
    copied_ids = copy_counts.loc[copy_counts["source_row_count"].gt(1)]
    copied = candidates.merge(
        copied_ids, on="VolumeObservationID", how="inner", validate="many_to_one"
    )
    copied_columns = [
        column for column in [
            "VolumeObservationID", "source_row_count", "PhysicalEventID",
            *PHYSICAL_EVENT_KEY, *available_provenance, "Volume", "_confirmed_zero",
            "_wq_idx", "SampleID", "Analyte", "analyte_abbr", "SampleMethod",
            "Duplicate", "Lab", "NoRunoff", "Notes",
        ] if column in copied.columns
    ]
    copied = copied[copied_columns].sort_values(
        ["PhysicalEventID", "VolumeObservationID", *(
            ["_wq_idx"] if "_wq_idx" in copied_columns else []
        )]
    ).reset_index(drop=True)
    no_observation = (
        work.loc[~work["PhysicalEventID"].isin(observations["PhysicalEventID"]), [*PHYSICAL_EVENT_KEY, "PhysicalEventID"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    reports = {
        "ambiguous_volume_observations": ambiguous,
        "zero_missing_conflicts": zero_conflicts,
        "missing_physical_key": missing_key,
        "copied_volume_values": copied,
        "events_without_volume_observation": no_observation,
    }
    if strict:
        blocking = {name: len(table) for name, table in reports.items() if name in {
            "ambiguous_volume_observations", "zero_missing_conflicts", "missing_physical_key"
        } and not table.empty}
        if blocking:
            raise ValueError(f"Blocking physical-event volume audit findings: {blocking}")
    if observations["VolumeObservationID"].duplicated().any():
        raise AssertionError("VolumeObservationID is not unique after deduplication.")
    return observations, reports


def event_balanced_weights(df: pd.DataFrame, group_columns: Sequence[str]) -> pd.Series:
    """Weights that sum to one within each scientific event group."""

    _require_columns(df, group_columns, "Weight input")
    sizes = df.groupby(list(group_columns), dropna=False)[group_columns[0]].transform("size")
    weights = 1.0 / sizes.astype(float)
    sums = weights.groupby([df[column] for column in group_columns], dropna=False).sum()
    if not np.allclose(sums.to_numpy(dtype=float), 1.0):
        raise AssertionError("Event-balanced weights do not sum to one.")
    return weights


def split_event_groups(
    physical_event_ids: pd.Series,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split proper-training/calibration positions by whole physical events."""

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one.")
    ids = physical_event_ids.astype(str).to_numpy()
    unique = np.unique(ids)
    if len(unique) < 2:
        raise ValueError("At least two physical events are required for calibration.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_calibration = min(max(1, int(np.ceil(len(unique) * calibration_fraction))), len(unique) - 1)
    calibration_ids = set(shuffled[:n_calibration])
    calibration = np.array([value in calibration_ids for value in ids])
    proper = ~calibration
    if set(ids[proper]) & set(ids[calibration]):
        raise AssertionError("A PhysicalEventID crossed the training/calibration split.")
    return np.flatnonzero(proper), np.flatnonzero(calibration)


def loyo_masks(
    years: pd.Series,
    physical_event_ids: pd.Series,
    heldout_year: int,
) -> tuple[pd.Series, pd.Series]:
    """Return proper outer-fold masks and verify held-out-event exclusion."""

    numeric_years = pd.to_numeric(years, errors="raise").astype(int)
    test = numeric_years.eq(int(heldout_year))
    train = ~test
    train_events = set(physical_event_ids.loc[train].astype(str))
    test_events = set(physical_event_ids.loc[test].astype(str))
    if train_events & test_events:
        raise ValueError("A PhysicalEventID appears in both held-out and training rows.")
    if numeric_years.loc[train].eq(int(heldout_year)).any():
        raise AssertionError("Held-out year leaked into LOYO training.")
    return train, test


def resolve_prediction_draws(
    draws: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    value_column: str,
    method: str = "median",
    method_column: str | None = None,
    method_priority: Sequence[str] = (),
) -> pd.DataFrame:
    """Resolve row predictions only after prediction, never before fitting."""

    _require_columns(draws, [*group_columns, value_column], "Prediction draws")
    work = draws.copy()
    if method == "method_priority":
        if not method_column or method_column not in work.columns:
            raise ValueError("method_priority requires an available method_column.")
        if not method_priority:
            raise ValueError("method_priority was requested but the hierarchy is empty.")
        rank = {str(value): position for position, value in enumerate(method_priority)}
        work["_priority"] = work[method_column].astype(str).map(rank).fillna(len(rank))
        best = work.groupby(list(group_columns), dropna=False)["_priority"].transform("min")
        work = work.loc[work["_priority"].eq(best)]
        method = "median"
    reducers = {"median": "median", "mean": "mean"}
    if method not in reducers:
        raise ValueError("method must be median, mean, or method_priority.")
    resolved = (
        work.groupby(list(group_columns), dropna=False, as_index=False)[value_column]
        .agg(reducers[method])
    )
    if resolved.duplicated(list(group_columns)).any():
        raise AssertionError("Post-prediction resolution did not produce unique groups.")
    return resolved


def build_event_analyte_load_ledger(
    concentration_draws: pd.DataFrame,
    volume_draws: pd.DataFrame,
    *,
    concentration_column: str = "Concentration_mg_L",
    volume_column: str = "Volume_L",
) -> pd.DataFrame:
    """Create the sole v3p0 physical event-analyte-draw load ledger."""

    analyte = next((column for column in ANALYTE_COLUMN_CANDIDATES if column in concentration_draws), None)
    if analyte is None:
        raise ValueError("Concentration draws need analyte_abbr or Analyte.")
    c_key = ["PhysicalEventID", analyte, "Draw"]
    v_key = ["PhysicalEventID", "Draw"]
    _require_columns(concentration_draws, [*c_key, concentration_column], "Concentration draws")
    _require_columns(volume_draws, [*v_key, volume_column], "Volume draws")
    if concentration_draws.duplicated(c_key).any():
        raise ValueError("Concentration draws must first resolve to one event-analyte-draw row.")
    if volume_draws.duplicated(v_key).any():
        raise ValueError("Volume draws must first resolve to one event-draw row.")
    ledger = concentration_draws.merge(volume_draws, on=v_key, how="inner", validate="many_to_one")
    ledger["Load_kg"] = (
        pd.to_numeric(ledger[concentration_column], errors="raise")
        * pd.to_numeric(ledger[volume_column], errors="raise")
        / 1_000_000.0
    )
    if ledger.duplicated(c_key).any():
        raise AssertionError("Event-analyte-draw load ledger is not unique.")
    return ledger


def build_event_analyte_point_load_ledger(
    concentration_points: pd.DataFrame,
    volume_points: pd.DataFrame,
    *,
    concentration_column: str = "Concentration_mg_L",
    volume_column: str = "Volume_L",
) -> pd.DataFrame:
    """Create one deterministic point load per physical event and analyte.

    Row-level concentration predictions and genuine volume-observation
    predictions must be resolved before this function is called. Method,
    sampler, duplicate, and laboratory rows are therefore never summation
    dimensions.
    """

    analyte = next(
        (column for column in ANALYTE_COLUMN_CANDIDATES if column in concentration_points),
        None,
    )
    if analyte is None:
        raise ValueError("Concentration points need analyte_abbr or Analyte.")
    c_key = ["PhysicalEventID", analyte]
    v_key = ["PhysicalEventID"]
    _require_columns(
        concentration_points,
        [*c_key, concentration_column],
        "Concentration point predictions",
    )
    _require_columns(
        volume_points,
        [*v_key, volume_column],
        "Volume point predictions",
    )
    if concentration_points.duplicated(c_key).any():
        raise ValueError(
            "Concentration point predictions must first resolve to one "
            "PhysicalEventID x Analyte row."
        )
    if volume_points.duplicated(v_key).any():
        raise ValueError(
            "Volume point predictions must first resolve to one PhysicalEventID row."
        )
    ledger = concentration_points.merge(
        volume_points,
        on=v_key,
        how="inner",
        validate="many_to_one",
    )
    ledger["Load_kg"] = (
        pd.to_numeric(ledger[concentration_column], errors="raise")
        * pd.to_numeric(ledger[volume_column], errors="raise")
        / 1_000_000.0
    )
    if ledger.duplicated(c_key).any():
        raise AssertionError("Event-analyte point-load ledger is not unique.")
    return ledger


def validate_corrected_artifact_metadata(
    metadata_paths: Sequence[str | Path],
    *,
    expected_years: Sequence[int] | None = None,
) -> list[Mapping[str, object]]:
    """Refuse legacy, absent, or incomplete comparison inputs."""

    records: list[Mapping[str, object]] = []
    for path_like in metadata_paths:
        path = Path(path_like)
        if not path.is_file():
            raise FileNotFoundError(f"Required corrected artifact metadata is absent: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("workflow_version") != CORRECTED_VERSION:
            raise ValueError(f"Legacy or incompatible artifact refused: {path}")
        if record.get("event_unit") != "PhysicalEventID":
            raise ValueError(f"Artifact does not declare the physical-event contract: {path}")
        if expected_years is not None:
            found = sorted(int(value) for value in record.get("years", []))
            wanted = sorted(int(value) for value in expected_years)
            if found != wanted:
                raise ValueError(f"Incomplete year coverage in {path}: found {found}; expected {wanted}")
        records.append(record)
    return records
