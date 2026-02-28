"""Tests for the pipeline orchestrator and configuration loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


class TestConfigLoading:
    """Test suite for pipeline configuration."""

    def test_config_loads_successfully(self, project_root: Path) -> None:
        """The config file should parse without errors."""
        config_path = project_root / "config" / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

    def test_config_has_required_sections(self, config: dict) -> None:
        """Config must contain all top-level sections used by the pipeline."""
        required_keys = {
            "data_sources", "date_range", "features", "split",
            "forecast_horizons", "targets", "models", "evaluation",
            "paths", "logging",
        }
        assert required_keys.issubset(set(config.keys())), (
            f"Missing config sections: {required_keys - set(config.keys())}"
        )

    def test_config_targets_are_strings(self, config: dict) -> None:
        """Pipeline targets should be a list of strings."""
        targets = config["targets"]
        assert isinstance(targets, list)
        assert all(isinstance(t, str) for t in targets)

    def test_config_forecast_horizons_are_positive_ints(self, config: dict) -> None:
        """Forecast horizons should be positive integers."""
        horizons = config["forecast_horizons"]
        assert isinstance(horizons, list)
        assert all(isinstance(h, int) and h > 0 for h in horizons)

    def test_config_split_dates_are_ordered(self, config: dict) -> None:
        """train_end must precede test_start."""
        train_end = pd.Timestamp(config["split"]["train_end"])
        test_start = pd.Timestamp(config["split"]["test_start"])
        assert train_end < test_start, "train_end must be before test_start"

    def test_config_model_sections_exist(self, config: dict) -> None:
        """Model hyperparameter sections must exist for all three models."""
        assert "arima" in config["models"]
        assert "xgboost" in config["models"]
        assert "neural_net" in config["models"]

    def test_config_interaction_pairs_reference_valid_columns(
        self, config: dict, project_root: Path
    ) -> None:
        """Interaction pairs should reference columns that exist in sample data."""
        from src.data.preprocessor import Preprocessor

        preprocessor = Preprocessor(config, project_root)
        df = preprocessor.load_sample_data()
        columns = set(df.columns)

        for pair in config["features"]["interaction_pairs"]:
            col_a, col_b = pair
            assert col_a in columns, (
                f"Interaction column '{col_a}' not in processed data columns"
            )
            assert col_b in columns, (
                f"Interaction column '{col_b}' not in processed data columns"
            )
