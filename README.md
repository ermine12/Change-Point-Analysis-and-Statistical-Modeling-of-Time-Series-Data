# Change Point Analysis — Brent Oil Prices

Quick start
-----------
1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Run the EDA notebook:

```bash
jupyter lab notebooks/01_EDA_and_data_prep.ipynb
```

What this repo contains
- `notebooks/01_EDA_and_data_prep.ipynb`: Download/clean Brent prices, run basic EDA and stationarity tests.
- `events.csv`: curated list of geopolitical/economic events for overlay/interpretation.
- `analysis_plan.md`, `assumptions_and_limitations.md`: planning and documentation.
- `requirements.txt`: Python package requirements.

Next steps (suggested)
- Implement `notebooks/02_pymc_change_point.ipynb` to build a Bayesian change-point model on weekly returns.
- Add an interactive dashboard (Streamlit/Dash) to visualize posterior change-point probabilities with event overlays.
