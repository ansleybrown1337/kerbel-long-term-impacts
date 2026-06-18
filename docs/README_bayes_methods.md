# Bayesian STIR-Water Quality Model

**Current implementation:** v2p1

**Selected model for inference:** v2p1

**Stan model:** `code/m_stir_mogp_v2p1.stan`

**Driver analysis:** `code/stir-bayes-load2p1_nonneg.Rmd`

**Primary dataset:** `out/wq_cleaned.csv`

## Overview

This document describes the hierarchical Bayesian model used to analyze
2011-2025 edge-of-field water-quality responses to Soil Tillage Intensity
Rating (STIR) at the Kerbel long-term tillage experiment at CSU ARDEC.

The model jointly represents runoff concentration and runoff volume, propagates
their uncertainty into event and annual loads, and accounts for:

- analyte-specific STIR, inflow concentration, irrigation, duplicate, residue,
  crop, block, sampler, and flume effects;
- a multi-output Gaussian process over year;
- latent true concentration and volume separated from observation error;
- concentration values censored below reporting limits; and
- missing inflow concentration, inflow volume, outflow volume, and residue.

Inference uses Hamiltonian Monte Carlo with the No-U-Turn Sampler in Stan.

## Current Status

v2p1 is the latest implemented and completed workflow and is the selected model
for formal inference and publication. It corrects pseudo-replication in v1p8 by
assigning each variable to its scientific unit of analysis before fitting:

| Variable or process | v2p1 model unit |
| --- | --- |
| Outflow concentration (`C`) | Analyte row |
| Inflow concentration (`CIN`) | Analyte row |
| Outflow volume (`VOL`) | Volume-measurement event |
| Inflow volume (`VIN`) | Physical plot runoff event |
| Residue (`RES`) | Planting-season plot |

The first completed v2p1 run on June 14, 2026 converged better than the earlier
row-replicated formulation. It retained the following diagnostic limitations:

- 54 divergent transitions across 4,000 post-warmup draws;
- no maximum-treedepth transitions;
- one chain with E-BFMI below 0.3 (`0.267`);
- 9 research-facing parameters with R-hat above 1.01;
- worst research-facing R-hat: `1.141`.

Despite these remaining diagnostics, v2p1 is the selected formal model because
its scientific unit-of-analysis structure is preferred over earlier
row-replicated workflows.

## Model Units And Mappings

### Analyte rows

Concentration and inflow concentration remain analyte-row-level because their
values and missingness can differ by analyte. In the completed v2p1 run:

- `N = 10,804` analyte rows;
- `5,488` concentration rows were missing;
- `5,551` inflow-concentration rows were missing.

### Volume-measurement events

Outflow volume is modeled once per conflict-free volume-measurement event.
The event key is:

`Date + Year + Irrigation + Rep + Treatment + SampleID + MeasureMethod`

The row-to-volume-event mapping is `E`. The completed run contained:

- `E_n = 1,000` volume-measurement events;
- 642 observed event volumes;
- 358 missing event volumes;
- zero conflicting event-volume groups.

This replaces approximately 10.85 repeated analyte-row copies per observed
volume with one volume likelihood contribution.

### Physical VIN events

Inflow volume is shared once per physical plot runoff event. The VIN key is:

`Date + Year + Irrigation + Rep + Treatment`

`VIN_E` maps each volume-measurement event to one physical VIN event. The
completed run contained:

- `VIN_n = 517` physical VIN events;
- 324 observed VIN events;
- 193 missing VIN events;
- zero conflicting VIN-event groups.

Missing VIN has one parameter and one `normal(0,1)` prior per missing physical
event. VIN has no observation likelihood because no separate data model informs
it.

### Planting-season residue units

Residue is derived once at planting for each plot-season. Its key is:

`PlantDate + Year + Treatment + Rep + Crop + previous_crop`

The row-to-residue mapping is `R`. The residue model uses the earliest modeled
runoff event's standardized seasonal STIR within each residue unit. The
completed run contained:

- `R_n = 102` planting-season plot units;
- 66 observed residue units;
- 36 missing residue units;
- zero conflicting residue units.

This replaces approximately 119.68 repeated analyte-row residue likelihood
contributions per observed residue unit with one contribution.

## Model Specification

All concentration and volume outcomes supplied to Stan are standardized.
Residue is supplied as a proportion in `(0,1)`.

Treatment is not included as a separate causal predictor. Management enters
the outcome models through measured STIR and the other explicitly represented
covariates. Accordingly, direct STIR coefficients are interpreted conditional
on inflow conditions, volume, residue, crop, design variables, and the modeled
year structure.

### Concentration model

For analyte row `i`, with analyte `a = A[i]`:

```math
\begin{aligned}
\mu_{C,i} ={}&
\alpha_a
+ \beta_{\mathrm{stir},a}\,\mathrm{STIR}_i
+ \beta_{\mathrm{cin},a}\,\mathrm{CIN}^{*}_i
+ \beta_{\mathrm{vol}}\,V_{\mathrm{true},i} \\
&+ \beta_{\mathrm{irr},a}\,\mathrm{IRR}_{z,i}
+ \beta_{\mathrm{dup},a}\,\mathrm{DUP}_i
+ \beta_{\mathrm{res},a}\,\mathrm{RES}^{*}_i \\
&+ \gamma_{a,\mathrm{Cr}[i]}
+ \gamma_{a,\mathrm{B}[i]}
+ \gamma_{a,\mathrm{S}[i]}
+ \gamma_{a,\mathrm{Fu}[i]}
+ f_{Y[i],a}.
\end{aligned}
```

Latent true concentration is non-centered:

```math
C_{\mathrm{true},i}
= \mu_{C,i} + \sigma_{\mathrm{analyte}}z_{C,i},
\qquad z_{C,i} \sim \mathrm{Normal}(0,1).
```

Observed uncensored concentrations use an analyte-specific observation-error
likelihood. Censored observations use a left-censored normal likelihood at the
row-specific reporting limit transformed to the concentration z scale.

### Volume model

For volume-measurement event `e`, representative row `r = E_rep_row[e]`, and
physical VIN event `v = VIN_E[e]`:

```math
\mu_{V,e}
= a_V
+ b_V\,\mathrm{STIR}_r
+ \beta_{\mathrm{vin}}\,\mathrm{VIN}^{*}_v
+ \beta_{\mathrm{res},V}\,\mathrm{RES}^{*}_r
+ \gamma^{(V)}_{\mathrm{Cr}[r]}.
```

Latent true volume and its observation likelihood are applied once per
volume-measurement event:

```math
V_{\mathrm{true},e}
= \mu_{V,e} + \sigma_V z_{V,e},
\qquad
\mathrm{VOL}_e
\sim \mathrm{Normal}(V_{\mathrm{true},e},\sigma_{\mathrm{VOL,obs}}).
```

The event-level volume state is mapped back to analyte rows before entering the
concentration model.

### Residue model

For planting-season residue unit `u`:

```math
\mu_{\mathrm{res},u}
= \mathrm{logit}(\mathrm{res\_base})
+ b_{\mathrm{res,stir}}\,\mathrm{STIR}_{\mathrm{plant},u}
+ \gamma^{(\mathrm{res})}_{\mathrm{PrevCr}[u]}.
```

Observed residue contributes one logit-normal likelihood per residue unit.
Missing residue is parameterized on the unconstrained logit scale and receives
one unit-level model contribution before being mapped back to analyte rows.

### Missing predictors

- Missing `CIN` remains analyte-row-level with
  `CIN_impute ~ normal(0,1)`.
- Missing `VIN` is physical-event-level with
  `VIN_event_impute ~ normal(0,1)`.
- There is no VIN observation likelihood.
- Missing residue is informed by the planting-season residue submodel.

## Temporal Structure

Year-by-analyte effects use a separable multi-output Gaussian process:

```math
\mathrm{vec}(F_{\mathrm{year}})
\sim
\mathrm{Normal}(0,\Sigma_A \otimes K_{\mathrm{year}}).
```

The year covariance uses the squared-exponential kernel implemented by
`cov_GPL2`, while a Cholesky-factor correlation structure allows analytes to
share temporal information.

## Load Propagation And Predictions

Event loads are computed after back-transforming concentration and volume:

```math
L_{e,a} = C_{e,a}V_e.
```

Annual loads sum each analyte-event once. All downstream reconstructions of the
volume mean include the VIN term. For downstream annual summaries, missing VIN
uses `vin_z = 0`, the center of its standardized `normal(0,1)` prior.
Hypothetical STIR-to-load scenarios expose `vin_z` explicitly and default to
zero.

## Generated Quantities And Compatibility Aliases

Stan generates event-level replicated volume, residue-unit replicated residue,
and row-level replicated concentration. Shared event and residue-unit states
are also mapped to row-level aliases (`mu_V`, `V_true`, `VOL_rep`, `VIN_merge`,
`mu_res`, `RES_star`, and `RES_rep01`) for downstream compatibility. These
aliases do not create additional likelihood contributions.

Primary v2p1 outputs include:

- `out/annual_load_summary_bayes_v2p1.csv`
- `out/annual_load_summary_bayes_plus_observed_v2p1.csv`
- `out/annual_load_draws_bayes_v2p1.csv`
- `out/study_period_total_loads_kg_v2p1.csv`
- `out/annual_volume_kL_wide_modeled_v2p1.csv`

## Guardrails And Audit Outputs

The v2p1 driver stops before compilation if event or residue mappings contain
conflicting values or invalid indices. It writes:

- `out/event_volume_audit_summary_v2p1.csv`
- `out/event_volume_audit_largest_groups_v2p1.csv`
- `out/event_volume_audit_conflicts_v2p1.csv`
- `out/vin_event_audit_conflicts_v2p1.csv`
- `out/event_predictor_repetition_audit_v2p1.csv`
- `out/residue_unit_audit_summary_v2p1.csv`
- `out/residue_unit_audit_conflicts_v2p1.csv`

The serialized completed fit is:

`code/out_cmdstanr/fit_mogp_v2p1.rds`

## Version History

v2p1 retains the v1p8 scientific structure while correcting unit-of-analysis
pseudo-replication for outflow volume, VIN, and residue. See
`docs/bayes-model_versions.md` for the full model lineage and convergence
status, and `docs/README_bayes_methods_v2p1_notes.md` for the detailed v2p1
interface and audit notes.
