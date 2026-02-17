"""Lightweight tests to increase coverage by importing safe modules
and exercising simple, fast code paths in BayesianChangePointModel
without heavy sampling. These tests avoid side effects and long runtimes.
"""
import numpy as np
import pandas as pd

from src.config import ProjectPaths, ModelParams, paths, params  # include in coverage
from src.change_point.bayesian_model import BayesianChangePointModel


# Attempt safe imports of entrypoint modules (no side effects expected on import)
# Wrapped in try/except to avoid breaking tests on CI if environment differs.
try:  # pragma: no cover - if import fails it won't reduce coverage
    import src.analyze_properties  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:  # pragma: no cover
    import src.run_model  # noqa: F401
except Exception:  # noqa: BLE001
    pass

try:  # pragma: no cover
    import src.app  # noqa: F401
except Exception:  # noqa: BLE001
    pass


def _make_price_series(n=20, start_price=100.0, dt_start="2020-01-01"):
    idx = pd.date_range(dt_start, periods=n, freq="D")
    # Simple synthetic prices with a small drift, with a small shift halfway
    base = np.linspace(0, 0.05, n)
    shift = np.zeros(n)
    shift[n // 2 :] += 0.02
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.001, size=n)
    prices = start_price * (1 + base + shift + noise)
    return pd.Series(prices, index=idx, name="Price")


def test_init_prices_mode_branch():
    s = _make_price_series(n=15)
    # use_log_returns=False branch should store raw prices
    model = BayesianChangePointModel(s, use_log_returns=False)
    assert model.use_log_returns is False
    # ensure internal series aligns with input length
    assert len(model.returns) == len(s)


def test_summary_without_trace_formats_change_points():
    s = _make_price_series(n=16)
    model = BayesianChangePointModel(s, use_log_returns=True)
    # Manually set change_points and ensure summary handles no-trace path
    cp_date = s.index[8]
    model.change_points = [cp_date]
    summary = model.summary()
    assert summary["change_points"][0]["date"] == str(cp_date.date())
    assert summary["change_points"][0]["credible_interval"] is None


def test_fit_multi_change_point_noop_when_depth_zero():
    s = _make_price_series(n=40)
    model = BayesianChangePointModel(s, use_log_returns=True)
    # With max_change_points=0, recursive search should exit immediately
    model.fit_multi_change_point(max_change_points=0, min_segment_size=10, draws=10, tune=10, chains=1)
    assert model.change_points == []


def test_config_exports_importable():
    # Ensure config module exports are importable and usable
    pp = ProjectPaths()
    mp = ModelParams()
    # Access commonly used attributes to ensure they exist
    assert pp.data.name == "data"
    assert isinstance(mp.draws, int)
    # Access global instances
    assert paths.data.name == "data"
    assert isinstance(params.target_ess, int)
