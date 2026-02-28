"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def config(project_root: Path) -> dict:
    """Load and return the pipeline configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_monthly_df() -> pd.DataFrame:
    """Create a small synthetic monthly DataFrame for testing.

    Returns:
        DataFrame with realistic column names and 36 rows (3 years).
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.DataFrame(
        {
            "gdp": np.cumsum(np.random.randn(36) * 0.3) + 0.5,
            "cpi": np.cumsum(np.random.randn(36) * 0.2) + 2.0,
            "unemployment_rate": np.clip(
                4.0 + np.cumsum(np.random.randn(36) * 0.1), 3.0, 8.0
            ),
            "bank_rate": np.clip(
                0.5 + np.cumsum(np.random.randn(36) * 0.05), 0.1, 5.0
            ),
            "retail_sales_index": 100 + np.cumsum(np.random.randn(36) * 0.5),
            "industrial_production_index": 100 + np.cumsum(
                np.random.randn(36) * 0.3
            ),
            "m4_growth": 3.0 + np.random.randn(36) * 1.0,
            "mortgage_approvals": 60 + np.random.randn(36) * 5.0,
        },
        index=dates,
    )
