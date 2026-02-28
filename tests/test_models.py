"""Tests for the forecasting models and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import Evaluator


class TestEvaluator:
    """Test suite for the Evaluator metrics."""

    def test_compute_metrics_perfect_prediction(self) -> None:
        """Perfect predictions should yield zero error."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = Evaluator.compute_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mape"] == pytest.approx(0.0, abs=1e-10)

    def test_compute_metrics_known_values(self) -> None:
        """Metrics should match hand-calculated values."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.5, 2.5, 3.5])

        metrics = Evaluator.compute_metrics(y_true, y_pred)
        # MAE = mean(|0.5|, |0.5|, |0.5|) = 0.5
        assert metrics["mae"] == pytest.approx(0.5, abs=1e-10)
        # RMSE = sqrt(mean(0.25, 0.25, 0.25)) = 0.5
        assert metrics["rmse"] == pytest.approx(0.5, abs=1e-10)

    def test_compute_metrics_handles_zero_actuals(self) -> None:
        """MAPE should handle zero actual values gracefully."""
        y_true = pd.Series([0.0, 1.0, 2.0])
        y_pred = pd.Series([0.1, 1.1, 2.1])
        metrics = Evaluator.compute_metrics(y_true, y_pred)
        # MAPE computed only on non-zero actuals
        assert np.isfinite(metrics["mape"])

    def test_dm_test_identical_models(self) -> None:
        """DM test with identical predictions should yield p > 0.05."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
        dm_stat, p_value = Evaluator._dm_test(y_true, pred, pred)
        assert p_value >= 0.05, "Identical models should not be significantly different"

    def test_dm_test_different_models(self) -> None:
        """DM test should return a finite statistic and p-value."""
        np.random.seed(42)
        y_true = np.random.randn(50)
        pred_1 = y_true + np.random.randn(50) * 0.1  # Good model
        pred_2 = y_true + np.random.randn(50) * 1.0  # Bad model
        dm_stat, p_value = Evaluator._dm_test(y_true, pred_1, pred_2)
        assert np.isfinite(dm_stat)
        assert 0.0 <= p_value <= 1.0

    def test_compute_metrics_returns_all_keys(self) -> None:
        """Metrics dictionary should contain exactly rmse, mae, mape."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 2.2, 3.3])
        metrics = Evaluator.compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"rmse", "mae", "mape"}

    def test_compute_metrics_all_zeros_mape_is_nan(self) -> None:
        """MAPE should be NaN when all actual values are zero."""
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = pd.Series([0.1, 0.2, 0.3])
        metrics = Evaluator.compute_metrics(y_true, y_pred)
        assert np.isnan(metrics["mape"])

    def test_record_result_stores_entry(self, config: dict, tmp_path: Path) -> None:
        """record_result should accumulate entries in metrics_store."""
        config = config.copy()
        config["paths"] = config["paths"].copy()
        config["paths"]["figures"] = str(tmp_path / "figures")
        config["paths"]["forecasts"] = str(tmp_path / "forecasts")
        evaluator = Evaluator(config, tmp_path)

        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.1, 2.1, 3.1])
        metrics = Evaluator.compute_metrics(y_true, y_pred)
        evaluator.record_result("TestModel", "gdp", 1, metrics, y_true, y_pred)

        assert len(evaluator.metrics_store) == 1
        assert evaluator.metrics_store[0]["model"] == "TestModel"
        assert evaluator.metrics_store[0]["target"] == "gdp"
        assert evaluator.metrics_store[0]["horizon"] == 1
