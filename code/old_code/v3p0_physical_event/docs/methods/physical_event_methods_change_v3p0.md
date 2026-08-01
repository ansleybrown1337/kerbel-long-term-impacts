# Methods change note: physical-event indexing v3p0

The correction changes hydrologic indexing and physical aggregation, not the accepted Bayesian predictor structure.

Previously, local event-volume keys included `SampleID` and `MeasureMethod`. Because first-flush, outflow, and duplicate chemistry rows can describe the same plot runoff event, that key could assign the full runoff volume more than once. v3p0 defines the physical event as `Date + Year + Irrigation + Rep + Treatment`.

For Bayesian inference, all analyte rows remain row-level. Each physical event has one latent true runoff volume. Zero or more genuine observed-volume rows map to that latent state through the existing common runoff-volume observation error. Missing volume supplies no likelihood row; confirmed zero remains observed. No method coefficient, method bias, method-specific standard deviation, or new parameter was introduced. Existing STIR, inflow-volume, residue, crop, concentration, censoring, random-effect, GP, prior, and causal structures are retained.

For ML, concentration rows and genuine volume observations remain their respective training units with event-balanced weights. LOYO excludes the held-out year and calibration splits whole physical events. Final load is constructed after row/observation prediction by resolving to one concentration per event-analyte-draw and one volume per event-draw.

This change prevents chemistry-record multiplicity from multiplying physical runoff volume. It does not assert that discrepant observations are erroneous; audit conflicts and cross-model disagreements are review diagnostics only.
