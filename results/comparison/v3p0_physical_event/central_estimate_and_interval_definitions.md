# Central estimates and uncertainty intervals

- Bayes line: posterior median of annual physical-event load draws; band: 95% posterior credible interval.
- ML line: deterministic sum of physical-event point loads; band: 95% Monte Carlo empirical calibration-residual prediction interval. Signed log-scale residuals are resampled from the physical-event-grouped split-conformal calibration set, with performance evaluated by outer LOYO. The ML line is not the median of the propagated draws.
- Observed markers: corrected physical-event observed annual summaries, kept separate from both model-only products; error bars are the existing event-bootstrap 95% intervals.

- Annual runoff-volume figure: Bayesian posterior mean with 95% credible interval, deterministic ML sum of physical-event point volumes without an ML ribbon, and observed event-bootstrap mean with 95% confidence interval; all volumes are shown in kL.

The primary annual comparison figures retain the Bayesian 95% credible-interval ribbon and observed bootstrap intervals, while omitting the ML prediction ribbon so that Bayes and ML centers remain readable on a common linear scale. Full model intervals are retained in the `supplemental_uncertainty` figure folder, using separate Bayes and ML rows with method-specific linear y-axis scales.

A prediction interval quantifies uncertainty for predicted outcomes. It is not a confidence interval for a fitted parameter. The event-level split-conformal coverage guarantee does not automatically become a 95% frequentist coverage guarantee for the summed annual Monte Carlo band. The saved ml_point_center_interval_alignment_audit.csv reports whether each point total lies within its propagated draw interval.
