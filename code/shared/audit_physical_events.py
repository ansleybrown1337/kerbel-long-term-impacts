#!/usr/bin/env python3
"""Audit the shared physical-event data contract without fitting models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.physical_event import (  # noqa: E402
    CORRECTED_VERSION,
    EVENT_DATE_COLUMN,
    PHYSICAL_EVENT_CONFIG,
    PHYSICAL_EVENT_KEY,
    VOLUME_PROVENANCE_COLUMNS,
    add_concentration_observation_id,
    add_physical_event_id,
    build_event_date_audit,
    build_volume_observation_table,
    yearly_irrigation_roster,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Current cleaned WQ CSV.")
    parser.add_argument("--output-dir", required=True, help="New preflight output directory.")
    return parser.parse_args()


def _write(table: pd.DataFrame, output_dir: Path, name: str) -> None:
    table.to_csv(output_dir / name, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    input_display = (
        input_path.relative_to(repo_root).as_posix()
        if input_path.is_relative_to(repo_root)
        else str(input_path)
    )
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path, low_memory=False)
    rows = add_concentration_observation_id(add_physical_event_id(raw, allow_missing_key=True))
    volume_observations, reports = build_volume_observation_table(raw)
    event_dates, multi_date_events, predictor_conflicts = build_event_date_audit(
        rows, volume_observations
    )
    event_date_map = event_dates[["PhysicalEventID", "EventDate"]]
    rows = rows.merge(
        event_date_map,
        on="PhysicalEventID",
        how="left",
        validate="many_to_one",
    )
    volume_observations = volume_observations.merge(
        event_date_map,
        on="PhysicalEventID",
        how="left",
        validate="many_to_one",
    )
    available_provenance = [column for column in VOLUME_PROVENANCE_COLUMNS if column in raw]

    physical_events = (
        rows.groupby(["PhysicalEventID", *PHYSICAL_EVENT_KEY], dropna=False)
        .agg(
            concentration_rows=("ConcentrationObservationID", "size"),
            analytes=("analyte_abbr" if "analyte_abbr" in rows else "Analyte", "nunique"),
            nonmissing_concentrations=("Result_mg_L", lambda values: values.notna().sum()),
            nonmissing_volume_candidate_rows=("Volume", lambda values: values.notna().sum()),
        )
        .reset_index()
    )
    volume_counts = (
        volume_observations.groupby("PhysicalEventID", dropna=False)
        .agg(
            genuine_volume_observations=("VolumeObservationID", "size"),
            volume_methods=("MeasureMethod", "nunique") if "MeasureMethod" in volume_observations else ("VolumeObservationID", "size"),
        )
        .reset_index()
    )
    physical_events = physical_events.merge(volume_counts, on="PhysicalEventID", how="left")
    physical_events = physical_events.merge(
        event_dates[
            [
                "PhysicalEventID",
                "EventDate",
                "EventDateResolution",
                "ContributingDates",
                "n_contributing_dates",
                "HasBlockingPredictorConflict",
            ]
        ],
        on="PhysicalEventID",
        how="left",
        validate="one_to_one",
    )
    physical_events[["genuine_volume_observations", "volume_methods"]] = physical_events[
        ["genuine_volume_observations", "volume_methods"]
    ].fillna(0).astype(int)

    concentration_export_columns = [
        column for column in [
            "ConcentrationObservationID", "PhysicalEventID", *PHYSICAL_EVENT_KEY,
            EVENT_DATE_COLUMN, "EventDate",
            "_wq_idx", "SampleID", "Duplicate", "Analyte", "analyte_abbr",
            "Result_mg_L", "Volume", *available_provenance, "NoRunoff",
        ] if column in rows
    ]
    concentration_observations = rows[concentration_export_columns].copy()
    physical_events_by_year_treatment = (
        physical_events.groupby(["Year", "Treatment"], dropna=False, as_index=False)
        .agg(
            physical_events=("PhysicalEventID", "nunique"),
            concentration_rows=("concentration_rows", "sum"),
            genuine_volume_observations=("genuine_volume_observations", "sum"),
        )
        .sort_values(["Year", "Treatment"])
    )
    analyte_column = "analyte_abbr" if "analyte_abbr" in rows else "Analyte"
    concentration_group_columns = [
        column for column in [analyte_column, "Year", "SampleMethod", "Duplicate"]
        if column in rows
    ]
    concentration_observation_counts = (
        rows.groupby(concentration_group_columns, dropna=False, as_index=False)
        .agg(
            concentration_observations=("ConcentrationObservationID", "size"),
            nonmissing_concentrations=("Result_mg_L", lambda values: values.notna().sum()),
            physical_events=("PhysicalEventID", "nunique"),
        )
        .sort_values(concentration_group_columns)
    )

    copied_candidate_rows = int(
        volume_observations["source_row_count"].sub(1).clip(lower=0).sum()
    )
    candidate_rows_before = int(pd.to_numeric(raw["Volume"], errors="coerce").notna().sum())
    if "NoRunoff" in raw:
        confirmed_missing_zero = raw["Volume"].isna() & raw["NoRunoff"].astype(str).str.lower().eq("true")
        candidate_rows_before += int(confirmed_missing_zero.sum())

    volume_method_counts = (
        volume_observations.groupby(available_provenance or ["PhysicalEventID"], dropna=False)
        .agg(
            genuine_volume_observations=("VolumeObservationID", "size"),
            physical_events=("PhysicalEventID", "nunique"),
            copied_source_rows=("source_row_count", lambda values: values.sub(1).clip(lower=0).sum()),
        )
        .reset_index()
    )
    method_signature = (
        volume_observations[available_provenance]
        .astype("string")
        .fillna("<MISSING>")
        .agg("|".join, axis=1)
        if available_provenance
        else pd.Series("unspecified", index=volume_observations.index)
    )
    event_volume_methods = (
        volume_observations.assign(VolumeMethodSignature=method_signature)
        .groupby("PhysicalEventID", as_index=False)
        .agg(
            genuine_volume_observations=("VolumeObservationID", "size"),
            distinct_volume_methods=("VolumeMethodSignature", "nunique"),
            volume_method_signatures=("VolumeMethodSignature", lambda values: ";".join(sorted(set(values)))),
        )
    )
    events_with_multiple_volume_methods = event_volume_methods.loc[
        event_volume_methods["distinct_volume_methods"].gt(1)
    ].merge(
        physical_events[["PhysicalEventID", *PHYSICAL_EVENT_KEY]],
        on="PhysicalEventID", how="left", validate="one_to_one",
    )
    multiplicity = (
        physical_events.groupby("Year", dropna=False)
        .agg(
            physical_events=("PhysicalEventID", "nunique"),
            concentration_rows=("concentration_rows", "sum"),
            nonmissing_concentrations=("nonmissing_concentrations", "sum"),
            genuine_volume_observations=("genuine_volume_observations", "sum"),
            events_with_multiple_volume_observations=("genuine_volume_observations", lambda values: values.gt(1).sum()),
        )
        .reset_index()
    )
    irrigation_roster = yearly_irrigation_roster(physical_events)
    irrigation_roster_by_year = (
        irrigation_roster.groupby("Year", dropna=False, as_index=False)
        .agg(
            expected_irrigation_labels=(
                "Irrigation",
                lambda values: ";".join(sorted(values.astype(str).unique())),
            ),
            expected_irrigation_count=("Irrigation", "nunique"),
        )
    )

    status = (
        rows.groupby(["PhysicalEventID", *PHYSICAL_EVENT_KEY], dropna=False)
        .agg(
            volume_candidate_rows=("Volume", lambda values: values.notna().sum()),
            zero_rows=("Volume", lambda values: pd.to_numeric(values, errors="coerce").eq(0).sum()),
            positive_rows=("Volume", lambda values: pd.to_numeric(values, errors="coerce").gt(0).sum()),
            missing_volume_rows=("Volume", lambda values: values.isna().sum()),
            confirmed_no_runoff_rows=("NoRunoff", lambda values: values.astype(str).str.lower().eq("true").sum()) if "NoRunoff" in rows else ("Volume", lambda values: 0),
        )
        .reset_index()
        .merge(volume_counts[["PhysicalEventID", "genuine_volume_observations"]], on="PhysicalEventID", how="left")
    )
    status["genuine_volume_observations"] = status["genuine_volume_observations"].fillna(0).astype(int)
    status["likelihood_status"] = "observed"
    status.loc[status["genuine_volume_observations"].eq(0), "likelihood_status"] = "no_volume_likelihood_row"
    status.loc[status["genuine_volume_observations"].gt(0) & status["zero_rows"].gt(0), "likelihood_status"] = "confirmed_zero_observation"

    blocking_parts: list[pd.DataFrame] = []
    for finding, table in [
        ("missing_physical_event_key", reports["missing_physical_key"]),
        ("ambiguous_same_provenance_volume", reports["ambiguous_volume_observations"]),
        ("zero_positive_conflict", reports["zero_missing_conflicts"]),
        (
            "missing_event_date",
            event_dates.loc[event_dates["EventDate"].isna()],
        ),
        ("conflicting_event_level_predictor", predictor_conflicts),
    ]:
        if not table.empty:
            part = table.copy()
            part.insert(0, "BlockingFinding", finding)
            blocking_parts.append(part)
    expected_roster = PHYSICAL_EVENT_CONFIG["corrected_roster"]
    irrigation_numeric = pd.to_numeric(
        physical_events["Irrigation"], errors="coerce"
    )
    roster_counts = {
        "total": int(physical_events["PhysicalEventID"].nunique()),
        "numeric": int(irrigation_numeric.notna().sum()),
        "storm": int(irrigation_numeric.isna().sum()),
    }
    observed_storm_labels = sorted(
        physical_events.loc[
            irrigation_numeric.isna(), "Irrigation"
        ].astype(str).unique()
    )
    configured_storm_labels = sorted(
        str(value) for value in PHYSICAL_EVENT_CONFIG["storm_handling"]["labels"]
    )
    unexpected_storm_labels = sorted(
        set(observed_storm_labels) - set(configured_storm_labels)
    )
    if unexpected_storm_labels:
        blocking_parts.append(
            pd.DataFrame(
                [
                    {
                        "BlockingFinding": "unexpected_nonnumeric_irrigation_label",
                        "PhysicalEventID": pd.NA,
                        "unexpected_labels": ";".join(unexpected_storm_labels),
                        "configured_storm_labels": ";".join(
                            configured_storm_labels
                        ),
                    }
                ]
            )
        )
    expected_counts = {
        "total": int(expected_roster["expected_total_events"]),
        "numeric": int(expected_roster["expected_numeric_irrigation_events"]),
        "storm": int(expected_roster["expected_storm_events"]),
    }
    if roster_counts != expected_counts:
        blocking_parts.append(
            pd.DataFrame(
                [
                    {
                        "BlockingFinding": "corrected_roster_count_mismatch",
                        "PhysicalEventID": pd.NA,
                        "observed_total_events": roster_counts["total"],
                        "expected_total_events": expected_counts["total"],
                        "observed_numeric_irrigation_events": roster_counts["numeric"],
                        "expected_numeric_irrigation_events": expected_counts["numeric"],
                        "observed_storm_events": roster_counts["storm"],
                        "expected_storm_events": expected_counts["storm"],
                    }
                ]
            )
        )
    blocking_review = pd.concat(blocking_parts, ignore_index=True, sort=False) if blocking_parts else pd.DataFrame(columns=["BlockingFinding"])

    summary_records = [
        ("workflow_version", CORRECTED_VERSION),
        ("cleaned_rows", len(raw)),
        ("physical_events", physical_events["PhysicalEventID"].nunique()),
        ("numeric_irrigation_events", roster_counts["numeric"]),
        ("storm_events", roster_counts["storm"]),
        ("multi_date_physical_events", len(multi_date_events)),
        ("event_level_predictor_conflicts", len(predictor_conflicts)),
        ("concentration_observations", len(concentration_observations)),
        ("nonmissing_concentration_observations", int(pd.to_numeric(raw["Result_mg_L"], errors="coerce").notna().sum())),
        ("volume_candidate_rows_before_dedup", candidate_rows_before),
        ("genuine_volume_observations_after_dedup", len(volume_observations)),
        ("copied_volume_rows_removed", copied_candidate_rows),
        ("events_with_volume_observation", volume_observations["PhysicalEventID"].nunique()),
        ("events_without_volume_observation", len(reports["events_without_volume_observation"])),
        ("confirmed_zero_volume_observations", int(volume_observations["Volume"].eq(0).sum())),
        ("blocking_rows", len(blocking_review)),
        ("blocking_physical_events", blocking_review["PhysicalEventID"].nunique() if "PhysicalEventID" in blocking_review else 0),
        ("ready_for_model_execution", str(blocking_review.empty).lower()),
    ]
    summary = pd.DataFrame(summary_records, columns=["metric", "value"])

    _write(summary, output_dir, "preflight_summary.csv")
    _write(physical_events, output_dir, "physical_events.csv")
    _write(physical_events_by_year_treatment, output_dir, "physical_events_by_year_treatment.csv")
    _write(concentration_observations, output_dir, "concentration_observations.csv")
    _write(concentration_observation_counts, output_dir, "concentration_observation_counts.csv")
    _write(volume_observations, output_dir, "volume_observations.csv")
    _write(volume_method_counts, output_dir, "volume_methods.csv")
    _write(event_volume_methods, output_dir, "volume_observations_per_physical_event.csv")
    _write(events_with_multiple_volume_methods, output_dir, "events_with_multiple_volume_methods.csv")
    _write(reports["copied_volume_values"], output_dir, "copied_volume_values.csv")
    _write(reports["ambiguous_volume_observations"], output_dir, "ambiguous_volume_observations.csv")
    _write(reports["events_without_volume_observation"], output_dir, "events_without_volume_observation.csv")
    _write(status, output_dir, "zero_missing_volume_status.csv")
    _write(multiplicity, output_dir, "event_multiplicity_by_year.csv")
    _write(irrigation_roster, output_dir, "yearly_irrigation_roster.csv")
    _write(
        irrigation_roster_by_year,
        output_dir,
        "yearly_irrigation_roster_summary.csv",
    )
    _write(event_dates, output_dir, "event_date_audit.csv")
    _write(multi_date_events, output_dir, "multi_date_event_audit.csv")
    _write(
        predictor_conflicts,
        output_dir,
        "event_level_predictor_conflicts.csv",
    )
    _write(blocking_review, output_dir, "BLOCKING_REVIEW.csv")
    metadata = {
        "workflow_version": CORRECTED_VERSION,
        "event_unit": "PhysicalEventID",
        "audit_only": True,
        "input": input_display,
        "physical_event_key": PHYSICAL_EVENT_KEY,
        "date_is_physical_event_identity": False,
        "event_date_selection_rule": PHYSICAL_EVENT_CONFIG["event_date"][
            "selection_rule"
        ],
        "physical_events": roster_counts["total"],
        "numeric_irrigation_events": roster_counts["numeric"],
        "storm_events": roster_counts["storm"],
        "observed_storm_labels": observed_storm_labels,
        "multi_date_physical_events": len(multi_date_events),
        "event_level_predictor_conflicts": len(predictor_conflicts),
        "irrigation_roster_rule": expected_roster["irrigation_roster_rule"],
        "seasonal_stir": PHYSICAL_EVENT_CONFIG["seasonal_stir"],
        "storm_handling": PHYSICAL_EVENT_CONFIG["storm_handling"],
        "annual_reporting": PHYSICAL_EVENT_CONFIG["annual_reporting"],
        "volume_provenance_columns_available": available_provenance,
        "ready_for_model_execution": blocking_review.empty,
        "blocking_rows": len(blocking_review),
        "years": sorted(int(value) for value in rows["Year"].dropna().unique()),
    }
    (output_dir / "preflight_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    audit_readme = f"""# Physical-event preflight audit

Input: `{input_display}`

- Physical events: {physical_events['PhysicalEventID'].nunique():,}
- Numeric-irrigation events: {roster_counts['numeric']:,}
- Recorded S1/S2 storm events: {roster_counts['storm']:,}
- Multi-date physical events: {len(multi_date_events):,}
- Event-level predictor conflicts: {len(predictor_conflicts):,}
- Concentration rows: {len(concentration_observations):,}
- Genuine volume observations: {len(volume_observations):,}
- Copied volume rows removed: {copied_candidate_rows:,}
- Events without a volume observation: {len(reports['events_without_volume_observation']):,}
- Blocking review rows: {len(blocking_review):,}
- Ready for model execution: `{str(blocking_review.empty).lower()}`

`PhysicalEventID` uses `{' + '.join(PHYSICAL_EVENT_KEY)}`; Date is observation
metadata and does not change the identity. `EventDate` uses the date of a unique
genuine volume observation when available and otherwise the earliest valid
contributing date. `VolumeObservationID` uses the physical-event identity,
observation Date, available measurement provenance, and exact recorded value.
`SampleID` is not used to manufacture volume observations.

Review `BLOCKING_REVIEW.csv` before model execution. Detailed source rows for
copied values are in `copied_volume_values.csv`; they remain concentration rows
but contribute only one genuine volume observation per deterministic identity.
The full multi-date merge provenance is in `multi_date_event_audit.csv`.

`Season_STIR_toDate` uses
`{PHYSICAL_EVENT_CONFIG['seasonal_stir']['interval']}`: post-harvest and
pre-plant operations are assigned to the following crop. The first observed
season is explicitly left-censored because its preceding harvest date is not
available.
"""
    (output_dir / "AUDIT_README.md").write_text(audit_readme, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nPreflight outputs: {output_dir}")
    if not blocking_review.empty:
        print("BLOCKED: review BLOCKING_REVIEW.csv before model execution.")


if __name__ == "__main__":
    main()
