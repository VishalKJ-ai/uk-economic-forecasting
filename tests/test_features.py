"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for the FeatureEngineer class."""

    def test_transform_adds_lag_features(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should add lag columns for configured periods."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        # Check that at least some lag features exist
        lag_cols = [c for c in result.columns if "_lag" in c]
        assert len(lag_cols) > 0, "No lag features were created"

        # Verify specific lag column exists
        assert "gdp_lag1" in result.columns
        assert "cpi_lag12" in result.columns

    def test_transform_adds_rolling_features(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should add rolling mean and std columns."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        rolling_cols = [c for c in result.columns if "_roll" in c]
        assert len(rolling_cols) > 0, "No rolling features were created"

    def test_transform_adds_change_features(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should add MoM and YoY change columns."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        mom_cols = [c for c in result.columns if "_mom" in c]
        yoy_cols = [c for c in result.columns if "_yoy" in c]
        assert len(mom_cols) > 0, "No MoM features were created"
        assert len(yoy_cols) > 0, "No YoY features were created"

    def test_no_nans_after_transform(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should drop rows with NaN (from lagging/rolling)."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)
        assert result.isna().sum().sum() == 0

    def test_prepare_supervised_shapes(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """prepare_supervised should produce correct train/test shapes."""
        engineer = FeatureEngineer(config)
        features_df = engineer.transform(sample_monthly_df)

        X_train, y_train, X_test, y_test = engineer.prepare_supervised(
            features_df,
            target="gdp",
            horizon=1,
            train_end="2021-12-31",
            test_start="2022-01-01",
        )

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) > 0
        assert X_train.shape[1] == X_test.shape[1]

    def test_prepare_supervised_no_leakage(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Training data should not contain dates after train_end."""
        engineer = FeatureEngineer(config)
        features_df = engineer.transform(sample_monthly_df)

        X_train, _, X_test, _ = engineer.prepare_supervised(
            features_df,
            target="gdp",
            horizon=1,
            train_end="2021-12-31",
            test_start="2022-01-01",
        )

        assert X_train.index.max() <= pd.Timestamp("2021-12-31")
        assert X_test.index.min() >= pd.Timestamp("2022-01-01")

    def test_transform_adds_spread_features(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should add real_rate_approx and excess_money_growth spreads."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        assert "real_rate_approx" in result.columns, (
            "real_rate_approx spread feature missing"
        )
        assert "excess_money_growth" in result.columns, (
            "excess_money_growth spread feature missing"
        )

    def test_transform_adds_interaction_features(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should add multiplicative interaction terms."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        interaction_cols = [c for c in result.columns if "_x_" in c]
        assert len(interaction_cols) > 0, "No interaction features were created"
        assert "bank_rate_x_cpi" in result.columns

    def test_transform_adds_calendar_features(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should add month and quarter calendar features."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        assert "month" in result.columns, "month calendar feature missing"
        assert "quarter" in result.columns, "quarter calendar feature missing"
        assert result["month"].min() >= 1
        assert result["month"].max() <= 12
        assert result["quarter"].min() >= 1
        assert result["quarter"].max() <= 4

    def test_prepare_supervised_raises_on_missing_target(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """prepare_supervised should raise KeyError for a non-existent target."""
        engineer = FeatureEngineer(config)
        features_df = engineer.transform(sample_monthly_df)

        with pytest.raises(KeyError, match="not_a_real_column"):
            engineer.prepare_supervised(
                features_df,
                target="not_a_real_column",
                horizon=1,
                train_end="2021-12-31",
                test_start="2022-01-01",
            )

    def test_transform_output_fewer_rows_than_input(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should drop rows with NaN from lagging, so output is shorter."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        assert len(result) < len(sample_monthly_df), (
            "Transform should drop rows where lag/rolling introduced NaN"
        )

    def test_no_infinities_after_transform(
        self, config: dict, sample_monthly_df: pd.DataFrame
    ) -> None:
        """Transform should replace infinities from pct_change on zero values."""
        engineer = FeatureEngineer(config)
        result = engineer.transform(sample_monthly_df)

        assert not np.isinf(result.values).any(), (
            "Infinity values remain after feature engineering"
        )
