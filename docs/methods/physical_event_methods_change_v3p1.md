# Methods change note: physical-event indexing v3p1

The correction changes hydrologic indexing and physical aggregation, not the accepted Bayesian predictor structure.

Previously, local event-volume keys included `SampleID` and `MeasureMethod`, and v3p0 still included `Date` in its physical-event identity. Because first-flush, outflow, duplicate chemistry rows, and adjacent-date records can describe the same plot runoff event, those keys could split or multiply one plot event. v3p1 defines the physical event as `Year + Irrigation + Rep + Treatment`; Rep 1 and Rep 2 remain separate.

For a corrected event containing several dates, all legitimate concentration and genuine volume observations are retained. The event date is the date associated with a unique genuine volume observation when available; otherwise it is the earliest valid date. The preflight exports all contributing dates and observation IDs and blocks unresolved event-predictor conflicts. The current corrected roster contains 528 events: 510 numeric-irrigation plot events and 18 S1/S2 storm plot events.

For Bayesian inference, all analyte rows remain row-level. Each physical event has one latent true runoff volume. Zero or more genuine observed-volume rows map to that latent state through the existing common runoff-volume observation error. Missing volume supplies no likelihood row; confirmed zero remains observed. No method coefficient, method bias, method-specific standard deviation, or new parameter was introduced. Existing inflow-volume, residue, crop, concentration, censoring, random-effect, GP, prior, and causal structures are retained. The seasonal STIR input is corrected to begin after the preceding crop's named harvest operation, including same-day follow-up, fall, and pre-plant operations in the following crop's exposure.

For ML, concentration rows and genuine volume observations remain their respective training units with event-balanced weights. LOYO excludes the held-out year and calibration splits whole physical events. Final load is constructed after row/observation prediction by resolving to one concentration per event-analyte-draw and one volume per event-draw.

Annual and cumulative model products now represent mean load per treatment plot. Within each draw, physical-event values are summed within Rep and the annual replicate totals are averaged; CT-relative reductions are calculated from those treatment-mean draws. Event-level RMSE/NRMSE remains in its original event-level unit.

Primary observed annual points no longer use pooled event bootstrap resampling. Completeness is evaluated per replicate plot using each year's actual irrigation roster. Two complete plots produce their mean and descriptive replicate range, one complete plot produces its value without an interval and with a distinct marker, and no complete plots produce no primary point. Incomplete subtotals remain audit-only.

S1/S2 remain distinct Irrigation labels. Bayes retains the existing numeric codes 11/12 for its linear irrigation term, while ML retains categorical storm labels. That cross-framework difference is documented as a v3p1 limitation.

This change prevents chemistry-record multiplicity and date fragmentation from multiplying physical runoff volume. It does not assert that discrepant observations are erroneous; audit conflicts and cross-model disagreements are review diagnostics only.
