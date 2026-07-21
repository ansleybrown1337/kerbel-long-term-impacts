# RMSE and NRMSE evaluation tracks

- Bayesian RMSE and NRMSE are unchanged posterior-predictive fit diagnostics. No Bayesian LOYO calculation is introduced.
- The primary Bayes-versus-ML table uses ML full-record physical-event reconstruction points: the deterministic concentration and volume predictions that underlie the annual point-load sums, matched only where observations exist.
- ML outer-LOYO RMSE and NRMSE are retained in separate validation tables and figures. They quantify held-out-year prediction performance and are not used as the ML center in the annual reconstruction figures.
- To prevent measurement-method or sampler-method copies from being counted as separate load-producing events, ML concentration diagnostics are resolved by PhysicalEventID x Analyte and volume diagnostics by PhysicalEventID using the same median resolution as the point-load workflow. Legitimate source rows remain unchanged in the model inputs.
- RMSE is reported in mg/L for concentration and L for event volume. NRMSE is RMSE divided by the mean observed value within the displayed group. Metrics are not calculated against partial observed annual subtotals.
