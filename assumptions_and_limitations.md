# Assumptions and Limitations

Assumptions
-----------
- Data completeness: the Brent price time series accurately reflects traded prices and has no systematic gaps after cleaning.
- Temporal alignment: event dates recorded in `events.csv` represent the approximate start or most relevant date for each event; many events have extended durations.
- Model form: change points are modeled as discrete regime boundaries where parameters (e.g., mean, variance) change abruptly.
- Independence within regimes: a simple model assumes conditional independence within each regime (or limited autocorrelation); if strong autocorrelation exists, the model will be extended.

Limitations
-----------
- Correlation vs causation: temporal co-occurrence of a change point and an event does not establish causality—other factors may be responsible.
- Dating uncertainty: events and market reactions can be lagged; single-day event dates are approximate anchors, not exact causal timestamps.
- Model misspecification risk: if returns are heavy-tailed or heteroskedastic, a Gaussian likelihood may be inappropriate; robust likelihoods or volatility models may be required.
- Multiple change points: posterior support for many change points may indicate overfitting—use regularization or priors that penalize excessive segmentation.
- Confounders and omitted variables: macroeconomic indicators, inventory data, and demand-side shocks may influence prices and are not included here unless explicitly modeled.
- Data frequency and aggregation: daily vs weekly aggregation affects detection sensitivity; choose frequency that balances noise and temporal resolution.
- Computational cost: Bayesian sampling (NUTS) for complex models can be slow; plan for subsampling or variational approximations if needed.

Mitigations
-----------
- Sensitivity analyses: compare results across priors, frequencies, and likelihood choices.
- Posterior predictive checks: verify that the model can reproduce key features of the data.
- Synthetic tests: simulate series with known change points to validate detection performance.
- Conservative interpretation: report credible intervals for change-point timing and avoid firm causal claims without further causal analysis.
