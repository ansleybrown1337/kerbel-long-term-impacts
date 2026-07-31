# Central estimates and uncertainty intervals

- Bayes line: posterior median of annual physical-event load draws; band: 95% posterior credible interval.
- ML line: deterministic mean of replicate-specific annual plot totals; band: 95% Monte Carlo empirical calibration-residual prediction interval. Signed log-scale residuals are resampled from the physical-event-grouped split-conformal calibration set, with performance evaluated by outer LOYO. The ML line is not the median of the propagated draws.
- Observed markers: completeness is evaluated independently within Year x Treatment x Rep x Analyte. Two complete plots are averaged and their minimum-to-maximum range is shown descriptively; one complete plot is shown with an x marker and no interval; no complete plots produce no point. Incomplete subtotals remain audit-only and never enter the primary mean. The replicate range is not a confidence interval.

- Annual runoff-volume figure: Bayesian posterior mean with 95% credible interval, deterministic ML mean of replicate annual plot volumes without an ML ribbon, and the observed complete-plot mean/range; all volumes are shown in kL. Observed volume completeness is assessed within each replicate plot.

The primary annual comparison figures retain the Bayesian 95% credible-interval ribbon and observed replicate ranges, while omitting the ML prediction ribbon so that Bayes and ML centers remain readable on a common linear scale. Full model intervals are retained in the supplemental_uncertainty figure folder, using separate Bayes and ML rows with method-specific linear y-axis scales.

RMSE/NRMSE evaluation is deliberately split into two ML tracks. The primary Bayes-versus-ML performance figures retain the unchanged Bayesian posterior-predictive fit metrics and use ML full-record physical-event reconstruction points. Separate tables and figures retain ML outer-LOYO held-out validation metrics. ML concentration metrics are resolved by PhysicalEventID x Analyte; volume metrics are resolved by PhysicalEventID. Both use median resolution, and neither is calculated against partial observed annual subtotals.

A prediction interval quantifies uncertainty for predicted outcomes. It is not a confidence interval for a fitted parameter. The event-level split-conformal coverage guarantee does not automatically become a 95% frequentist coverage guarantee for the summed annual Monte Carlo band. The saved ml_point_center_interval_alignment_audit.csv reports whether each point total lies within its propagated draw interval.
