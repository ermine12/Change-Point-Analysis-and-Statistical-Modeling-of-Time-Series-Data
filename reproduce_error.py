
import pandas as pd
from src.change_point.bayesian_model import BayesianChangePointModel
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

try:
    short_series = pd.Series(
        [10, 11, 12, 13, 14],
        index=pd.date_range('2020-01-01', periods=5, freq='D')
    )
    BayesianChangePointModel(short_series)
except ValueError as e:
    print(f"Caught expected ValueError: {e}")
except Exception as e:
    print(f"Caught unexpected exception: {type(e).__name__}: {e}")
else:
    print("Did not raise any exception!")
