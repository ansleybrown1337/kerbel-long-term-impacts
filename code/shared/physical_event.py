"""Physical-event identities, audits, and aggregation for the public workflow.

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


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "physical_event_v3p4.json"
if not _CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Shared physical-event configuration is absent: {_CONFIG_PATH}")
PHYSICAL_EVENT_CONFIG = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
PHYSICAL_EVENT_KEY = list(PHYSICAL_EVENT_CONFIG["physical_event_key"])
VOLUME_PROVENANCE_COLUMNS = list(PHYSICAL_EVENT_CONFIG["volume_provenance_columns"])
EVENT_DATE_COLUMN = str(PHYSICAL_EVENT_CONFIG["event_date"]["source_column"])
EVENT_LEVEL_PREDICTOR_COLUMNS = list(
    PHYSICAL_EVENT_CONFIG["event_level_predictor_columns"]
)
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


def _canonical_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    canonical = series.astype("string")
    canonical.loc[parsed.notna()] = parsed.loc[parsed.notna()].dt.strftime("%Y-%m-%d")
    return canonical


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

    _require_columns(
        df,
        [*PHYSICAL_EVENT_KEY, EVENT_DATE_COLUMN, "Volume"],
        "Volume input",
    )
    work = add_physical_event_id(df, allow_missing_key=True)
    work[EVENT_DATE_COLUMN] = _canonical_date_series(work[EVENT_DATE_COLUMN])
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

    # Date is observation metadata rather than physical-event identity. It is
    # nevertheless part of a genuine volume-observation identity so a value
    # recorded on another date is never discarded merely because its method and
    # numeric value match.
    identity_columns = [
        "PhysicalEventID", EVENT_DATE_COLUMN, *available_provenance, "Volume"
    ]
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
        EVENT_DATE_COLUMN,
        *available_provenance, "Volume", "_confirmed_zero", "source_row_count",
    ]
    source_columns = [
        column for column in ["_wq_idx", "SampleID", "Duplicate", "NoRunoff", "Notes"]
        if column in observations.columns
    ]
    observations = observations[[*keep, *source_columns]].reset_index(drop=True)

    provenance_group = [
        "PhysicalEventID", EVENT_DATE_COLUMN, *available_provenance
    ]
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


def build_event_date_audit(
    df: pd.DataFrame,
    volume_observations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select one deterministic EventDate and audit multi-date merges.

    A date attached to exactly one genuine volume observation is preferred.
    Otherwise the earliest valid contributing date is used. Predictor
    conflicts are reported separately and are blocking for model execution.
    """

    _require_columns(
        df,
        [*PHYSICAL_EVENT_KEY, EVENT_DATE_COLUMN],
        "Event-date input",
    )
    rows = add_physical_event_id(df, allow_missing_key=True)
    if "ConcentrationObservationID" not in rows:
        rows = add_concentration_observation_id(rows)
    rows[EVENT_DATE_COLUMN] = _canonical_date_series(rows[EVENT_DATE_COLUMN])
    if volume_observations is None:
        volume_observations, _ = build_volume_observation_table(rows)
    volume = volume_observations.copy()
    if EVENT_DATE_COLUMN in volume:
        volume[EVENT_DATE_COLUMN] = _canonical_date_series(volume[EVENT_DATE_COLUMN])

    volume_by_event = {
        event_id: group
        for event_id, group in volume.groupby("PhysicalEventID", dropna=False)
    }
    records: list[dict[str, object]] = []
    conflict_records: list[dict[str, object]] = []
    group_columns = ["PhysicalEventID", *PHYSICAL_EVENT_KEY]
    for identity, group in rows.groupby(group_columns, dropna=False, sort=True):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        record = dict(zip(group_columns, identity_values))
        parsed_dates = pd.to_datetime(group[EVENT_DATE_COLUMN], errors="coerce")
        contributing_dates = sorted(
            parsed_dates.loc[parsed_dates.notna()].dt.strftime("%Y-%m-%d").unique()
        )
        event_volume = volume_by_event.get(record["PhysicalEventID"])
        volume_ids: list[str] = []
        volume_dates: list[str] = []
        if event_volume is not None:
            volume_ids = sorted(
                event_volume["VolumeObservationID"].dropna().astype(str).unique()
            )
            parsed_volume_dates = pd.to_datetime(
                event_volume[EVENT_DATE_COLUMN], errors="coerce"
            )
            volume_dates = sorted(
                parsed_volume_dates.loc[parsed_volume_dates.notna()]
                .dt.strftime("%Y-%m-%d")
                .unique()
            )
        if len(volume_ids) == 1 and len(volume_dates) == 1:
            event_date = volume_dates[0]
            decision = "unique_genuine_volume_observation_date"
        elif contributing_dates:
            event_date = contributing_dates[0]
            decision = (
                "earliest_valid_date_multiple_or_no_genuine_volume_observations"
            )
        else:
            event_date = pd.NA
            decision = "blocking_no_valid_date"

        conflict_columns: list[str] = []
        conflict_details: list[str] = []
        for column in EVENT_LEVEL_PREDICTOR_COLUMNS:
            if column not in group:
                continue
            values = sorted(
                {
                    _normal(value)
                    for value in group[column]
                    if pd.notna(value) and str(value).strip()
                }
            )
            if len(values) > 1:
                conflict_columns.append(column)
                conflict_details.append(f"{column}={' | '.join(values)}")
                conflict_records.append(
                    {
                        **record,
                        "Predictor": column,
                        "DistinctValues": " | ".join(values),
                        "ContributingDates": ";".join(contributing_dates),
                    }
                )

        record.update(
            {
                "EventDate": event_date,
                "EventDateResolution": decision,
                "ContributingDates": ";".join(contributing_dates),
                "n_contributing_dates": len(contributing_dates),
                "ConcentrationObservationIDs": ";".join(
                    sorted(
                        group["ConcentrationObservationID"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                ),
                "VolumeObservationIDs": ";".join(volume_ids),
                "n_genuine_volume_observations": len(volume_ids),
                "PredictorConflictColumns": ";".join(conflict_columns),
                "PredictorConflictDetails": ";".join(conflict_details),
                "HasBlockingPredictorConflict": bool(conflict_columns),
            }
        )
        records.append(record)

    event_dates = pd.DataFrame(records)
    if event_dates.duplicated("PhysicalEventID").any():
        raise AssertionError("Event-date audit is not unique by PhysicalEventID.")
    multi_date = event_dates.loc[
        event_dates["n_contributing_dates"].gt(1)
    ].reset_index(drop=True)
    conflicts = pd.DataFrame(
        conflict_records,
        columns=[
            *group_columns,
            "Predictor",
            "DistinctValues",
            "ContributingDates",
        ],
    )
    return event_dates, multi_date, conflicts


def yearly_irrigation_roster(events: pd.DataFrame) -> pd.DataFrame:
    """Return the actual distinct Irrigation labels independently by Year."""

    _require_columns(events, ["Year", "Irrigation"], "Irrigation roster")
    roster = (
        events[["Year", "Irrigation"]]
        .drop_duplicates()
        .sort_values(["Year", "Irrigation"], key=lambda values: values.astype(str))
        .reset_index(drop=True)
    )
    return roster


def aggregate_replicate_mean(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
    draw_column: str | None = None,
    plot_total_column: str = "PlotTotal",
    treatment_mean_column: str = "TreatmentMean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum within Rep, then average replicate-specific annual plot totals."""

    required = [*group_columns, "Rep", value_column]
    if draw_column:
        required.append(draw_column)
    _require_columns(frame, required, "Mean-per-plot aggregation")
    plot_groups = [*group_columns, "Rep", *([draw_column] if draw_column else [])]
    plot_totals = (
        frame.groupby(plot_groups, dropna=False, as_index=False)
        .agg(**{plot_total_column: (value_column, "sum")})
    )
    mean_groups = [*group_columns, *([draw_column] if draw_column else [])]
    treatment_means = (
        plot_totals.groupby(mean_groups, dropna=False, as_index=False)
        .agg(
            **{
                treatment_mean_column: (plot_total_column, "mean"),
                "n_replicate_plots": ("Rep", "nunique"),
            }
        )
    )
    return plot_totals, treatment_means


def observed_annual_plot_summary(
    expected_events: pd.DataFrame,
    observed_event_values: pd.DataFrame,
    *,
    value_column: str,
    analysis_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize complete replicate plots and descriptive replicate ranges."""

    plot_key = ["Year", "Treatment", "Rep", *analysis_columns]
    event_columns = [*plot_key, "Irrigation", "PhysicalEventID"]
    _require_columns(expected_events, event_columns, "Expected event roster")
    _require_columns(
        observed_event_values,
        [*plot_key, "Irrigation", "PhysicalEventID", value_column],
        "Observed event values",
    )
    actual_events = expected_events[event_columns].drop_duplicates().copy()
    yearly_roster = actual_events[["Year", "Irrigation"]].drop_duplicates()
    plot_groups = actual_events[plot_key].drop_duplicates()
    expected = plot_groups.merge(
        yearly_roster,
        on="Year",
        how="left",
        validate="many_to_many",
    ).merge(
        actual_events,
        on=[*plot_key, "Irrigation"],
        how="left",
        validate="one_to_one",
    )
    observed = observed_event_values[
        [*plot_key, "Irrigation", "PhysicalEventID", value_column]
    ].copy()
    observed[value_column] = pd.to_numeric(observed[value_column], errors="coerce")
    observed = observed.loc[observed[value_column].notna()]
    if observed.duplicated([*plot_key, "Irrigation"]).any():
        raise ValueError("Observed event values are not unique by plot and event.")

    expected_plot = (
        expected.groupby(plot_key, dropna=False, as_index=False)
        .agg(
            ExpectedEventCount=("Irrigation", "nunique"),
            ExpectedIrrigationLabels=(
                "Irrigation",
                lambda values: ";".join(sorted(values.astype(str).unique())),
            ),
        )
    )
    observed_plot = (
        observed.groupby(plot_key, dropna=False, as_index=False)
        .agg(
            ObservedEventCount=("Irrigation", "nunique"),
            IncompleteObservedSubtotal=(value_column, "sum"),
        )
    )
    missing = expected.merge(
        observed[[*plot_key, "Irrigation"]].drop_duplicates(),
        on=[*plot_key, "Irrigation"],
        how="left",
        indicator=True,
    )
    missing = (
        missing.loc[missing["_merge"].eq("left_only")]
        .groupby(plot_key, dropna=False, as_index=False)
        .agg(
            MissingIrrigationLabels=(
                "Irrigation",
                lambda values: ";".join(sorted(values.astype(str).unique())),
            )
        )
    )
    plot = expected_plot.merge(observed_plot, on=plot_key, how="left").merge(
        missing, on=plot_key, how="left"
    )
    plot["ObservedEventCount"] = plot["ObservedEventCount"].fillna(0).astype(int)
    plot["MissingIrrigationLabels"] = plot["MissingIrrigationLabels"].fillna("")
    plot["PlotComplete"] = plot["ObservedEventCount"].eq(plot["ExpectedEventCount"])
    plot["ReplicateAnnualValue"] = plot["IncompleteObservedSubtotal"].where(
        plot["PlotComplete"]
    )

    treatment_key = ["Year", "Treatment", *analysis_columns]
    treatment = (
        plot.groupby(treatment_key, dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "n_complete_plots": int(group["PlotComplete"].sum()),
                    "TreatmentMean": group.loc[
                        group["PlotComplete"], "ReplicateAnnualValue"
                    ].mean(),
                    "SampleSD": group.loc[
                        group["PlotComplete"], "ReplicateAnnualValue"
                    ].std(ddof=1),
                    "SE": (
                        group.loc[
                            group["PlotComplete"], "ReplicateAnnualValue"
                        ].std(ddof=1)
                        / np.sqrt(int(group["PlotComplete"].sum()))
                        if int(group["PlotComplete"].sum()) >= 2
                        else np.nan
                    ),
                    "Minimum": (
                        group.loc[
                            group["PlotComplete"], "ReplicateAnnualValue"
                        ].min()
                        if int(group["PlotComplete"].sum()) >= 1
                        else np.nan
                    ),
                    "Maximum": (
                        group.loc[
                            group["PlotComplete"], "ReplicateAnnualValue"
                        ].max()
                        if int(group["PlotComplete"].sum()) >= 1
                        else np.nan
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    treatment["RangeLow"] = treatment["Minimum"].where(
        treatment["n_complete_plots"].ge(2)
    )
    treatment["RangeHigh"] = treatment["Maximum"].where(
        treatment["n_complete_plots"].ge(2)
    )
    treatment["IntervalType"] = np.where(
        treatment["n_complete_plots"].ge(2),
        "descriptive replicate minimum-to-maximum range",
        "none",
    )
    treatment["CompletenessDefinition"] = (
        "every expected physical event has observed concentration and genuine "
        "runoff volume, or confirmed zero runoff"
    )
    return (
        plot.merge(treatment, on=treatment_key, how="left", validate="many_to_one")
        .sort_values(plot_key)
        .reset_index(drop=True)
    )


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
    """Create the physical event-analyte-draw load ledger."""

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
    expected_versions: Sequence[str] | None = None,
) -> list[Mapping[str, object]]:
    """Refuse absent, incompatible, or incomplete comparison inputs.

    ``expected_versions`` permits the explicitly versioned Bayes v3p3 and ML
    v3p4 inputs. Omitting it uses the shared release contract.
    """

    records: list[Mapping[str, object]] = []
    if expected_versions is not None and len(expected_versions) != len(metadata_paths):
        raise ValueError(
            "expected_versions must contain one workflow version per metadata path."
        )
    for index, path_like in enumerate(metadata_paths):
        path = Path(path_like)
        if not path.is_file():
            raise FileNotFoundError(f"Required corrected artifact metadata is absent: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        expected_version = (
            expected_versions[index]
            if expected_versions is not None
            else CORRECTED_VERSION
        )
        if record.get("workflow_version") != expected_version:
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
