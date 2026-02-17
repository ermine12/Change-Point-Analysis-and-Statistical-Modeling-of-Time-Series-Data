import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting smoke test...")

try:
    print("Importing config...")
    from src.config import paths, params
    print("Config imported.")

    print("Importing BayesianChangePointModel...")
    from src.change_point.bayesian_model import BayesianChangePointModel
    print("Model imported.")

    print("Creating dummy data...")
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    prices = np.random.normal(100, 10, 10)
    df_prices = pd.Series(prices, index=dates)

    print("Initializing model...")
    model = BayesianChangePointModel(df_prices, use_log_returns=True)
    print("Model initialized successfully.")
    
    print("Checking internal state...")
    assert model.returns is not None
    assert len(model.returns) == 9 # 10 prices -> 9 returns
    print("Internal state OK.")

    print("SMOKE TEST PASSED")

except Exception as e:
    print(f"SMOKE TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
