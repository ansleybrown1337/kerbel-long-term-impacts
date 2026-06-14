# Bayesian v2p1 Unit-of-Analysis Notes

## Purpose

Version v2p1 preserves the working v1p8 nonnegative workflow while correcting
unit-of-analysis problems for outflow volume, inflow volume, and residue.
Concentration and inflow concentration remain at the analyte-row level.

The latent outflow-volume process and observation likelihood now operate once
per conflict-free volume-measurement event unit. Inflow volume is represented
and imputed once per physical plot runoff event, independently of the number of
sample or volume-measurement records. Residue is modeled and imputed once per
planting-season plot unit. Shared event and residue-unit values are mapped back
to analyte rows for the concentration model and load calculations.

The direct STIR estimand, scalar `beta_vol`, analyte-specific `beta_stir`,
multi-output year GP, censoring, residue effects, and in-model imputation for
missing CIN, VIN, and residue are retained at their appropriate units.

## Completed-Run Status

The first full v2p1 run completed successfully on June 14, 2026 and produced
the expected model outputs.

## Event-Volume Key

The event-volume key is created after the active analyte filter and after storm
rows without ordinal irrigation numbers are removed:

`Date + Year + Irrigation + Rep + Treatment + SampleID + MeasureMethod`

The workflow stops before sampling if a key group contains more than one
distinct nonmissing `volume_z` value or conflicting event-level volume-process
predictors. The key deliberately excludes analyte, laboratory duplicate,
composite status, sampler method, and flume method because they do not identify
a distinct volume observation. `MeasureMethod` remains in the key because
removing it produces conflicting nonmissing volume observations for otherwise
identical event/sample groups.

## Physical VIN-Event Key

VIN is shared at the physical plot runoff-event level:

`Date + Year + Irrigation + Rep + Treatment`

The workflow stops if this key produces conflicting nonmissing standardized
VIN values. A `VIN_E` mapping links each volume-measurement event to one
physical VIN event, ensuring missing VIN receives exactly one imputation
parameter and one `normal(0,1)` prior per physical event.

## Residue-Unit Key

Residue is derived once at planting for each plot-season. The residue-unit key
is:

`PlantDate + Year + Treatment + Rep + Crop + previous_crop`

The residue submodel uses the earliest modeled runoff event's
`stir_season_z` within each residue unit. This avoids using later within-season
STIR changes as parents of a planting-time residue measurement. The workflow
stops if a residue unit contains conflicting residue observations, previous
crop values, or earliest-event STIR values.

## Changed Data Interface

The v2p1 Stan data interface replaces row-level outflow-volume and VIN inputs
with:

- `E_n`: number of event-volume units
- `E`: analyte-row to event-volume mapping
- `E_rep_row`: representative analyte row for event-level predictors
- `VOL_event`: one standardized outflow-volume observation per event unit
- `N_VOL_event_miss` and `VOL_event_missidx`: event-level missingness inputs
- `VIN_n`: number of physical plot runoff events carrying VIN
- `VIN_E`: volume-measurement event to physical VIN-event mapping
- `VIN_event`: one standardized inflow-volume predictor per physical event
- `N_VIN_event_miss` and `VIN_event_missidx`: event-level VIN imputation inputs

It replaces row-level residue likelihood/imputation inputs with:

- `R_n`: number of planting-season residue units
- `R`: analyte-row to residue-unit mapping
- `RES_unit`: one observed or missing residue value per residue unit
- `STIR_res_unit`: earliest-event seasonal STIR predictor per residue unit
- `PrevCr_res_unit`: previous-crop index per residue unit
- `N_RES_unit_miss` and `RES_unit_missidx`: residue-unit imputation inputs

Stan retains row-level `mu_V`, `V_true`, `VOL_rep`, `VIN_merge`, `mu_res`,
`RES_star`, and `RES_rep01` aliases for downstream compatibility.

## Downstream Prediction Policy

Every downstream reconstruction of the outflow-volume mean includes
`beta_vin * vin_z`. Annual prediction frames carry observed standardized VIN
at the event level. When event VIN is missing, downstream summaries use
`vin_z = 0`, the center of the model's `normal(0,1)` VIN imputation prior.
Hypothetical STIR-to-load scenarios expose `vin_z` explicitly and also default
to zero.

This does not add a VIN observation likelihood. In Stan, missing VIN has one
event-level `normal(0,1)` imputation prior and observed VIN remains a supplied
event-level predictor.

## Audit Outputs

Before compilation and sampling, the v2p1 Rmd writes:

- `out/event_volume_audit_summary_v2p1.csv`
- `out/event_volume_audit_largest_groups_v2p1.csv`
- `out/event_volume_audit_conflicts_v2p1.csv`
- `out/vin_event_audit_conflicts_v2p1.csv`
- `out/event_predictor_repetition_audit_v2p1.csv`
- `out/residue_unit_audit_summary_v2p1.csv`
- `out/residue_unit_audit_conflicts_v2p1.csv`

All model outputs, figures, diagnostics, and the saved fit use v2p1 names.
The completion-email behavior is preserved and reports `model_version = "v2p1"`.

