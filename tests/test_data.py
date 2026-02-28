"""Tests for data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import Preprocessor


class TestPreprocessor:
    """Test suite for the Preprocessor class."""

    def test_load_sample_data_returns_dataframe(
        self, config: dict, project_root: Path
    ) -> None:
        """Sample data loading should return a non-empty DataFrame."""
        preprocessor = Preprocessor(config, project_root)
        df = preprocessor.load_sample_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_sample_data_has_expected_columns(
        self, config: dict, project_root: Path
    ) -> None:
        """Sample data should contain all expected economic indicators."""
        preprocessor = Preprocessor(config, project_root)
        df = preprocessor.load_sample_data()
        expected_cols = {
            "gdp", "cpi", "unemployment_rate", "bank_rate",
            "retail_sales_index", "industrial_production_index",
            "m4_growth", "mortgage_approvals",
        }
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

    def test_no_missing_values_after_processing(
        self, config: dict, project_root: Path
    ) -> None:
        """Preprocessor should impute all missing values."""
        preprocessor = Preprocessor(config, project_root)
        df = preprocessor.load_sample_data()
        assert df.isna().sum().sum() == 0, "Processed data contains NaN values"

    def test_save_and_load_roundtrip(
        self, config: dict, project_root: Path, tmp_path: Path
    ) -> None:
        """Saving and loading should produce identical data."""
        config = config.copy()
        config["paths"] = config["paths"].copy()
        config["paths"]["processed_data"] = str(tmp_path / "processed")

        preprocessor = Preprocessor(config, project_root)
        df_original = preprocessor.load_sample_data()
        preprocessor.save(df_original, "test_roundtrip.parquet")
        df_loaded = preprocessor.load("test_roundtrip.parquet")

        pd.testing.assert_frame_equal(df_original, df_loaded, check_freq=False)

    def test_interpolate_quarterly_to_monthly(self) -> None:
        """Quarterly interpolation should produce monthly observations."""
        quarterly_dates = pd.date_range("2020-01-01", periods=8, freq="QS")
        df = pd.DataFrame({
            "date": quarterly_dates,
            "value": [0.5, 0.3, 0.4, 0.6, 0.2, -0.1, 0.3, 0.5],
        })
        result = Preprocessor._interpolate_quarterly(df, "value")
        # Should have roughly 3x as many rows (months between quarters)
        assert len(result) >= len(df)
        assert "date" in result.columns
        assert "value" in result.columns
