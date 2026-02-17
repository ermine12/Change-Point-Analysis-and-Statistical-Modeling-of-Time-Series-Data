# Change Point Analysis and Statistical Modeling of Time Series Data

Quantifies structural breaks (regime shifts) in time series with a focus on Brent oil prices, using Bayesian inference to detect change points, assess uncertainty, and connect shifts to real-world events for decision-making in risk, trading, and strategy.

## Business Problem
Oil and macro time series often experience structural breaks due to shocks (geopolitics, pandemics, policy). Conventional models that assume stationarity miss these regime shifts, leading to mispriced risk, delayed reactions, and poor forecast calibration.

This project provides a reproducible pipeline to detect and explain regime shifts so that:
- Risk teams can adapt VaR/limits to evolving regimes
- Portfolio and trading teams can align strategies to current market state
- Analysts can explain shifts with context from curated event data

## Solution Overview
- Bayesian change point models (PyMC) on log returns to ensure local stationarity
- Single and multiple change point detection (recursive segmentation)
- Automated convergence diagnostics (R-hat, ESS) and posterior summaries
- Event correlation by aligning detected change dates with curated events
- Programmatic API, Streamlit dashboard, and optional React/Vite frontend

## Key Results
- Detected the COVID-19 crash regime change around 2020-04-23 (± ~5 days)
- Multiple historically aligned shifts (e.g., 2008 crisis, 2014 shale boom, 1990 Gulf War)
- Automated diagnostics ensure reliability; validation uses faster settings and flags convergence when samples are low

Example impact framing (adapt per data):
- Metric 1: 5–10% change in mean daily returns across regimes (directionally consistent with crisis vs recovery)
- Metric 2: Operational savings: hours reduced for manual event alignment using automated detection and overlays
- Metric 3: Faster iteration: refactoring to a class-based model cut iteration time from ~2 hours to ~15 minutes

## Quick Start
```bash
# Clone
git clone https://github.com/USERNAME/REPO.git
cd REPO

# (Recommended) Create a virtual environment
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# cmd
.venv\Scripts\activate
# bash (Git Bash)
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Validate environment and config
python verify_config.py

# Run the model (writes JSON outputs under data/)
python src/run_model.py

# Launch Streamlit dashboard
streamlit run streamlit_app.py

# Optional: start the API server
python src/app.py
```

## Project Structure
```
.
├── analysis_plan.md
├── assumptions_and_limitations.md
├── data/
│   └── model_results_single.json
├── events.csv
├── notebooks/
│   ├── 01_change_point_analysis.ipynb
│   └── 01a_EDA_and_prep_clean.ipynb
├── reports/
│   ├── Day1_Day2_Implementation_Report.md
│   └── Technical_Report.md  # this file is generated in reports/
├── src/
│   ├── app.py                # API server
│   ├── analyze_properties.py # EDA/utilities
│   ├── change_point/
│   │   ├── bayesian_model.py # Core Bayesian model class
│   │   └── model.py          # Utilities/helpers
│   ├── config.py
│   ├── run_model.py          # Entrypoint to run model(s)
│   └── validate_model.py
├── streamlit_app.py          # Interactive dashboard
├── dashboard/                # Optional React/Vite frontend
│   ├── index.html
│   └── src/
├── tests/                    # Unit/integration tests
├── requirements.txt
├── pytest.ini
└── README.md
```

## Demo
- Streamlit app: run `streamlit run streamlit_app.py` and open the local URL.
- Optional React/Vite dashboard (if using):
  - `cd dashboard && npm install && npm run dev` (requires Node.js)

Add a short screen recording/GIF of the Streamlit dashboard showing change points and event overlays to this section.

## Technical Details
- Data: Brent oil historical prices (daily). Preprocessing includes datetime parsing, sorting, optional gap handling, and log return computation `r_t = log(P_t / P_{t-1})`. Event data is curated in events.csv with dates and categories.
- Model: Single and multi change point models using PyMC. Priors are weakly informative; change point τ is discrete uniform across valid indices. Likelihood uses regime means before/after τ with optional heteroskedasticity (σ1, σ2). Inference via NUTS for continuous parameters and Metropolis for τ.
- Evaluation: Convergence checks (R-hat < 1.05 target, ESS > 400), posterior summaries and credible intervals for τ, parameter contrasts (Δμ), and posterior predictive checks. Multi-CP detection via recursive segmentation with `max_change_points` and `min_segment_size` safeguards.

## Future Improvements
- Add heavy-tailed likelihoods (Student-t) and explicit volatility regimes or GARCH
- Incorporate exogenous covariates (macro, inventories) for causal analysis and explainability
- Variational inference / SMC for faster multi-change point inference on long series
- Formal event-match scoring and windowed alignment in the API
- Export artifacts for BI tools and batch scoring pipelines

## Author
- Name: Your Name
- LinkedIn: https://www.linkedin.com/in/yourprofile
- Email: your.email@example.com
- GitHub: https://github.com/yourusername

---

# API (Brief)
- GET /api/prices – recent prices
- GET /api/events – curated events
- GET /api/change-points – detected change points and diagnostics

For full details, see tests/test_api.py and src/app.py.
