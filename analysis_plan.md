# Analysis Plan — Bayesian Change Point Analysis of Brent Oil Prices

Purpose
-------
This document outlines a concise, reproducible workflow to detect structural breaks in Brent crude oil prices using Bayesian change point models (PyMC). The goal is to identify regime shifts in price dynamics and relate them (carefully) to major geopolitical and economic events.

Data sources
------------
- Historical Brent oil prices (daily or weekly) — cleaned time series with `date` and `price`.
- Event registry: structured CSV of major geopolitical/OPEC/economic events with approximate start dates (provided as `events.csv`).

High-level workflow
-------------------
1. Data ingestion and cleaning
   - Load price series, align to business days or weekly frequency.
   - Fill / mark missing values; log-transform prices if appropriate.
2. Exploratory analysis (EDA)
   - Visualize series, compute rolling means/std, detect obvious outliers.
   - Trend decomposition (loess / STL) to highlight seasonality/trend.
   - Stationarity tests (ADF, KPSS) on level and returns; inspect autocorrelation and partial autocorrelation.
   - Volatility analysis: rolling volatility and GARCH-style diagnostics (optional).
3. Model selection and specification
   - Choose model family: piecewise-Gaussian on log-prices or change points on mean/variance of returns.
   - Typical Bayesian change point model: K change points (K unknown or using a sparse prior), with regime-specific parameters (mean, sd) and priors that reflect domain knowledge.
   - Use informative but weak priors for stability; test sensitivity.
4. Bayesian inference
   - Implement model in PyMC (or numpyro) and sample with NUTS; run multiple chains and check convergence diagnostics (R-hat, ESS).
   - Posterior predictive checks to validate fit.
5. Event alignment and interpretation
   - Compare posterior change point distributions to `events.csv` dates.
   - Report overlap/temporal proximity but avoid automatic causal claims.
6. Robustness and sensitivity
   - Vary K priors, different data frequencies (daily vs weekly), and alternative likelihoods (heavy-tailed errors).
   - Test model on synthetic data with known change points.
7. Communication and delivery
   - Dashboard with interactive timeline, posterior change point probabilities, and event overlays.
   - Written report summarizing methods, key changes, and limitations.

Time-series diagnostics that inform modeling choices
-----------------------------------------------
- Trend: Strong trend suggests modeling log-prices or detrending before change-point detection.
- Stationarity: Non-stationarity motivates modeling changes in mean/variance rather than assuming constant parameters.
- Volatility patterns: Heteroskedasticity suggests regime-dependent variance or modeling returns rather than levels.

Change point model description (conceptual)
-----------------------------------------
- Purpose: identify times where the generative parameters of the series (e.g., mean, variance, autoregressive coefficients) shift.
- A simple model: assume the series is divided into contiguous segments; within each segment observations come from a Normal(mean_i, sd_i) where the i indexes the regime.
- Priors: place priors on the number of change points (or use a large K with sparsity), on location (uniform or time-informed), and on regime parameters.
- Inference yields posterior distributions over change-point locations, regime parameters, and predictive distributions.

Expected outputs
----------------
- Posterior distributions for change point dates (credible intervals).
- Posterior estimates of regime parameters (means, variances) and their uncertainty.
- Posterior predictive checks and fit diagnostics.
- Visualizations: price series with shaded credible intervals for change regions; event overlay from `events.csv`.

Limitations and interpretation guidance
--------------------------------------
- Statistical association vs causation: a temporal alignment between a change point and an event is suggestive but not proof of causality. Confounders and delayed responses are common.
- Dating imprecision: posterior credible intervals reflect uncertainty; avoid pegging abrupt events to single-day causal statements.
- Model misspecification: if the chosen likelihood or priors are poor matches to the data (e.g., heavy tails, autocorrelation), change-point inferences may be biased.
- Multiple testing / overfitting: allowing many change-points without regularization risks finding spurious breaks.

Communication channels and formats
---------------------------------
- Technical stakeholders: Jupyter notebooks and reproducible PyMC code (Git repo) with detailed methods and diagnostics.
- Executive summary: 1–2 page PDF report highlighting key regime shifts and practical implications.
- Interactive dashboard: Web app (Streamlit / Dash / Panel) showing the timeline, posterior densities, and event overlays for non-technical stakeholders.
- Short briefing slide deck for meetings.

Next steps
----------
1. Run EDA on the supplied Brent price series (generate plots, stationarity results).
2. Iterate on a PyMC change point model for weekly returns; test sensitivity.
3. Build the dashboard skeleton and integrate event overlay from `events.csv`.

References and further reading
-----------------------------
- Barry, D. & Hartigan, J. A. (1993). A Bayesian Analysis for Change Point Problems.
- Fearnhead, P. (2006). Exact and efficient Bayesian inference for multiple change-point problems.
- Turner et al. (2013). Bayesian methods for change point detection in time series.
- PyMC documentation: change point modeling examples.

Prepared by: Analysis team — deliverables bundle (plan + events CSV + assumptions)
