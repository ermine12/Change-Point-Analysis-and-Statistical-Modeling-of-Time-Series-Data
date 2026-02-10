# Brent Oil: Change Point Analysis – Analysis Plan

Version: 1.0

## Objectives

- Identify and quantify structural breaks (change points) in Brent oil prices using Bayesian inference (PyMC).
- Associate detected changes with major events (geopolitics, OPEC decisions, macro shocks) and communicate probabilistic impacts.
- Produce a concise, reproducible workflow, a well-documented notebook, and an interactive dashboard to explore results.

## Data

- Primary: BrentOilPrices.csv (Date, Price) – daily/regular frequency assumed. We will verify frequency and handle gaps/holidays.
- Events: events.csv – curated list of 10–15+ market-relevant events with dates, category, and brief description.

## End-to-End Workflow

1) Data ingestion and validation
   - Load BrentOilPrices.csv; coerce Date to datetime; set index; sort chronologically.
   - Inspect duplicates, missing dates/prices; decide on forward-fill or leave gaps; document.
   - Optionally filter by a timeframe relevant to events if needed for clarity.

2) Exploratory Data Analysis (EDA)
   - Plot raw Price over time with key descriptive stats (min/max/percentiles).
   - Compute log prices and log returns r_t = log(P_t) - log(P_{t-1}).
   - Visualize returns and rolling volatility (e.g., 21-day std) to assess clustering.
   - Stationarity diagnostics on Price vs. log returns (ADF, KPSS where applicable; discuss interpretation rather than strict pass/fail).
   - Comment on trends, volatility regimes, and implications for modeling the level vs. returns.

3) Modeling strategy
   - Baseline model: single change point in the mean of (a) level or (b) returns depending on EDA/stationarity.
     - If Price level shows strong non-stationarity, prefer modeling log returns; otherwise, level with a local-mean model.
   - Define a discrete uniform prior for change point τ over valid index positions.
   - Define parameters before/after τ (μ1, μ2). Start with shared σ; consider heteroskedastic extension later.
   - Likelihood: Normal with mean determined by pm.math.switch(time_index < τ, μ1, μ2).
   - Consider weakly-informative priors centered on empirical summaries.
   - Inference via pm.sample(); check convergence (R-hat, ESS), and posterior predictive checks as needed.

4) Interpretation and event mapping
   - Summarize posterior of τ (date range, MAP/median, credible intervals). Visualize posterior mass over dates.
   - Compare posterior of μ1 vs μ2 to quantify pre/post differences; compute effect size and % change.
   - Overlay events from events.csv; identify plausible associations (date alignment ± window). Emphasize correlation-in-time, not causation.

5) Sensitivity and robustness (targeted)
   - Sensitivity of τ to:
     - Data transformation (level vs. log returns)
     - Prior width on μ and σ
     - Excluding periods of extreme volatility
   - Optional: multiple change points (e.g., product partition models, Bayesian online change detection) as future work.

6) Communication and delivery
   - Notebook: Complete, reproducible EDA + PyMC modeling + plots.
   - Dashboard: Flask backend serving data/model outputs; React frontend for interactive exploration.
   - Report components integrated into README and assumptions_and_limitations.md.
   - Clear statements of uncertainty and limitations (no causal claims; event matching is heuristic).

## Understanding the Model and Data

References (core reading list)
- Gelman et al., Bayesian Data Analysis (Ch. on hierarchical models and posterior checks)
- Carlin & Louis, Bayesian Methods for Data Analysis (discrete parameters, mixture models)
- PyMC documentation: discrete variables, change point examples, pm.math.switch
- Time series texts: Enders or Hamilton (stationarity, volatility, structural breaks)
- Complementary: BOCPD (Adams & MacKay), Markov-switching (Hamilton), VAR introductions

Time series properties and modeling implications
- Trend: If strong deterministic/stochastic trend exists in levels, prefer modeling returns or de-trended series.
- Stationarity: Returns often closer to stationary than levels; this motivates modeling returns for a mean-shift model.
- Volatility clustering: A constant-σ Normal likelihood is a simplification; volatility regimes may warrant extensions (e.g., separate σ1, σ2, or GARCH-like approaches). We will start simple and note limitations.

## Change Point Model: Purpose and Mechanics

- Purpose: Detect structural breaks where the data-generating process changes (e.g., mean level/return), signaling regime shifts linked to market events or policy decisions.
- Construction (baseline):
  - τ ~ DiscreteUniform(1, T-1)
  - μ1, μ2 ~ Normal(prior_center, prior_scale)
  - σ ~ HalfNormal(scale)
  - For each t: μ_t = switch(t < τ, μ1, μ2)
  - y_t ~ Normal(μ_t, σ)
- Outputs:
  - Posterior over τ → implied date(s) of change
  - Posterior over μ1, μ2 (and optionally σ1, σ2) → pre/post regime summaries
  - Credible intervals, effect sizes, and posterior predictive checks

## Expected Outputs and Their Interpretation

- Date estimate(s) of change point with credible interval on τ.
- Pre/post parameter distributions (means; optionally variances) with probabilistic contrasts (e.g., P(μ2 > μ1)).
- Plots: price series with annotated τ; posterior τ histogram/density; trace plots; parameter posteriors.
- Event alignment table: nearest events, lag windows, and qualitative linkage. Note: temporal association ≠ causation.

## Communication Plan

- Primary channels:
  - Technical notebook (Jupyter) with narrative markdown and code.
  - Interactive dashboard: Flask (APIs) + React (visuals) with event overlays, filters, and date selectors.
  - Repository README: setup, run instructions, and summary findings.
  - Brief executive summary (bulleted insights, figures) for stakeholders.
- Artifacts: analysis_plan.md, assumptions_and_limitations.md, events.csv, notebooks, src backend/frontend, screenshots.

## Reproducibility and Project Structure

- Notebooks: 01_EDA_and_data_prep.ipynb, 02_pymc_change_point.ipynb (parameterized cells; ensure deterministic seeding where practical).
- Data: raw BrentOilPrices.csv (read-only); events.csv maintained under version control.
- Code: src/change_point/model.py for PyMC model utilities; scripts/notebooks generate and cache outputs consumed by the dashboard.
- Environment: requirements.txt with pinned versions; instructions in README to create environment and run.

## Next Steps Checklist

- [ ] Finalize events.csv with ≥15 curated entries.
- [ ] Complete EDA on level vs. returns and pick baseline target.
- [ ] Implement PyMC single-τ model; run and validate convergence.
- [ ] Create plots (τ posterior, parameter posteriors, annotated series).
- [ ] Map τ to events and draft quantified impact statements.
- [ ] Expose outputs via Flask APIs; build React UI with interactive overlays.
- [ ] Update README with run instructions and screenshots.
