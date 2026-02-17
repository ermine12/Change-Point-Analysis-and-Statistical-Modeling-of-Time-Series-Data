"""Configuration module for Change Point Analysis project.

This module defines project paths and model parameters using dataclasses
to ensure type safety and easy configuration.
"""
from dataclasses import dataclass
from pathlib import Path
import os

# Base directory is the project root (2 levels up from src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class ProjectPaths:
    """Project directory and file paths."""
    base: Path = BASE_DIR
    data: Path = base / "data"
    reports: Path = base / "reports"
    src: Path = base / "src"
    tests: Path = base / "tests"
    
    # Files
    brent_prices: Path = data / "BrentOilPrices.csv"
    events: Path = base / "events.csv"
    model_results_single: Path = data / "model_results_single.json"
    model_results_multi: Path = data / "model_results_multi.json"
    model_results_canonical: Path = data / "model_results.json"
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.data.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelParams:
    """Default parameters for Bayesian Change Point Model."""
    draws: int = 2000
    tune: int = 1000
    chains: int = 2
    random_seed: int = 42
    target_rhat: float = 1.05
    target_ess: int = 400

# Global configuration validator
paths = ProjectPaths()
params = ModelParams()
