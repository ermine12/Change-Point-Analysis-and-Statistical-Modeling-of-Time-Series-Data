# Task 1: Laying the Foundation for Analysis

## 1. Data Analysis Workflow
The analysis of Brent oil prices proceeds through the following structured steps:

1.  **Data Ingestion & Cleaning**: Load historical Brent oil prices, handle inconsistent date formats, and ensure chronological order.
2.  **Exploratory Data Analysis (EDA)**:
    *   Visual inspection of price levels for major shocks and trends.
    *   Transformation to log returns to achieve stationarity.
    *   Volatility analysis to identify clustering.
3.  **Bayesian Change Point Modeling**:
    *   Define a PyMC model with a discrete switch point $\tau$.
    *   Model the transition of parameters (mean, volatility) before and after the switch.
    *   MCMC sampling for posterior distribution estimation.
4.  **Inference & Interpretation**:
    *   Evaluate model convergence ($R$-hat, Trace plots).
    *   Identify the most probable change point dates.
    *   Quantify the statistical shift in prices/returns.
5.  **Event Correlation**:
    *   Map detected change points to researched geopolitical and economic events.
    *   Synthesize insights for stakeholders.

## 2. Event Data Summary
A structured dataset of 15 key market events has been compiled in `events.csv`. Notable events include:
- **1990 Gulf War**: Iraq's invasion of Kuwait causing a major supply disruption.
- **2008 Global Financial Crisis**: Severe demand collapse and price crash.
- **2014 US Shale Boom**: Market regime shift due to increased non-OPEC supply.
- **2020 COVID-19 Pandemic**: Record demand shock and negative price signals.
- **2022 Russia-Ukraine Conflict**: Major geopolitical risk premium and supply uncertainty.

## 3. Assumptions and Limitations
### Assumptions:
- **Data Integrity**: Historical prices reflect the true market equilibrium at each timestamp.
- **Discrete Transitions**: The model assumes shifts in market regimes occur at specific, identifiable points in time.
- **Stationarity in Returns**: Log returns follow a process that is locally stationary within regimes.

### Limitations:
- **Correlation vs. Causation**: While the model identifies statistical structural breaks that coincide with external events, it does not strictly prove that a specific event *caused* the shift, as multiple factors (confounders) are always at play.
- **Dating Precision**: Market reactions to news may be anticipatory or lagged, meaning the statistical $\tau$ might not align exactly with the calendar date of an announcement.
- **Omitted Variables**: This model focuses primarily on price series; it does not explicitly incorporate external variables like GDP or inventory levels.

## 4. Time Series Properties
Based on our analysis of data from 1987 to 2022:
- **Non-Stationarity**: The price level shows a strong stochastic trend (ADF p-value ≈ 0.29).
- **Log Return Stability**: Log returns are highly stationary (ADF p-value < 1e-28), making them suitable for mean-shift detection models.
- **Volatility Clustering**: Periods of extreme price movement are followed by similar high-volatility events, particularly around the 2008 and 2020 crises.

## 5. Communication Channels
Results will be communicated via:
- **Interactive Dashboard**: A Flask-React application for visual exploration.
- **Technical Jupyter Notebook**: A comprehensive, reproducible record of the modeling process.
- **Strategic Report**: Integrating quantified impacts of events for executive decision-making.

## 6. Change Point Models: Purpose and Outputs
Change point models identify "structural breaks" where the underlying parameters of a time series change. 
**Expected Outputs**:
- **Switch Point ($\tau$)**: The estimated date of the regime shift.
- **Regime Parameters**: Shift in mean ($\mu$) or volatility ($\sigma$) across the break.
- **Credible Intervals**: Probabilistic uncertainty around the timing of the shift.
