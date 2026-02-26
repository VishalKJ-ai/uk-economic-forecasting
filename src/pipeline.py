"""Pipeline orchestrator — the main entry point for the forecasting system.

Usage::

    python -m src.pipeline --mode sample   # Run full pipeline with sample data
    python -m src.pipeline --mode update   # Fetch latest data from APIs
    python -m src.pipeline --mode train    # Retrain all models
    python -m src.pipeline --mode forecast # Generate predictions
    python -m src.pipeline --mode full     # update + train + forecast

Modes:
    sample   Run end-to-end using bundled sample data (no API keys needed).
    update   Fetch the latest data from ONS and BoE APIs.
    train    Retrain all models on the latest processed data.
    forecast Generate predictions from saved models.
    full     Execute update, train, and forecast in sequence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# ── project root detection ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Explicit path to config.yaml.  Defaults to
            ``<project_root>/config/config.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _setup_logging(config: Dict[str, Any]) -> None:
    """Configure the root logger from the pipeline configuration.

    Args:
        config: Full pipeline configuration dictionary.
    """
    log_cfg = config.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get(
            "format",
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        ),
        datefmt=log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S"),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                PROJECT_ROOT / log_cfg.get("file", "pipeline.log"),
                mode="a",
            ),
        ],
    )


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════

def step_update(config: Dict[str, Any]) -> pd.DataFrame:
    """Fetch latest data from ONS and BoE APIs and process it.

    Args:
        config: Pipeline configuration.

    Returns:
        Processed DataFrame saved to disk.
    """
    from src.data.ons_client import ONSClient
    from src.data.boe_client import BoEClient
    from src.data.preprocessor import Preprocessor

    logger.info("=" * 70)
    logger.info("STEP: DATA UPDATE (fetching from APIs)")
    logger.info("=" * 70)

    start = config["date_range"]["start"]
    end = config["date_range"]["end"]

    ons = ONSClient(config)
    boe = BoEClient(config)

    try:
        ons_data = ons.fetch_all(start, end)
        boe_data = boe.fetch_all(start, end)
    finally:
        ons.close()
        boe.close()

    preprocessor = Preprocessor(config, PROJECT_ROOT)
    df = preprocessor.process_api_data(ons_data, boe_data)
    preprocessor.save(df)
    return df


def step_load_sample(config: Dict[str, Any]) -> pd.DataFrame:
    """Load and process sample data (no API access required).

    Args:
        config: Pipeline configuration.

    Returns:
        Processed DataFrame saved to disk.
    """
    from src.data.preprocessor import Preprocessor

    logger.info("=" * 70)
    logger.info("STEP: LOAD SAMPLE DATA")
    logger.info("=" * 70)

    preprocessor = Preprocessor(config, PROJECT_ROOT)
    df = preprocessor.load_sample_data()
    preprocessor.save(df)
    return df


def step_feature_engineering(
    config: Dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply feature engineering to the processed data.

    Args:
        config: Pipeline configuration.
        df: Processed monthly DataFrame.

    Returns:
        Feature-engineered DataFrame.
    """
    from src.data.feature_engineer import FeatureEngineer

    logger.info("=" * 70)
    logger.info("STEP: FEATURE ENGINEERING")
    logger.info("=" * 70)

    engineer = FeatureEngineer(config)
    features_df = engineer.transform(df)
    return features_df


def step_train(
    config: Dict[str, Any],
    features_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Train all models on all targets and horizons.

    Args:
        config: Pipeline configuration.
        features_df: Feature-engineered DataFrame.

    Returns:
        Dictionary of trained model objects keyed by
        ``"<model>_<target>_h<horizon>"``.
    """
    from src.data.feature_engineer import FeatureEngineer
    from src.models.arima import ARIMAModel
    from src.models.neural_net import NeuralNetModel
    from src.evaluation.evaluator import Evaluator

    try:
        from src.models.xgboost_model import XGBoostModel
        _xgboost_available = True
    except (ImportError, OSError):
        _xgboost_available = False
        logger.warning("XGBoost unavailable — skipping XGBoost models")

    logger.info("=" * 70)
    logger.info("STEP: MODEL TRAINING")
    logger.info("=" * 70)

    engineer = FeatureEngineer(config)
    evaluator = Evaluator(config, PROJECT_ROOT)
    targets = config["targets"]
    horizons = config["forecast_horizons"]
    train_end = config["split"]["train_end"]
    test_start = config["split"]["test_start"]
    models_path = PROJECT_ROOT / config["paths"]["models"]

    trained_models: Dict[str, Any] = {}

    for target in targets:
        for horizon in horizons:
            logger.info("-" * 60)
            logger.info("Training — target=%s, horizon=%d months", target, horizon)
            logger.info("-" * 60)

            try:
                X_train, y_train, X_test, y_test = engineer.prepare_supervised(
                    features_df, target, horizon, train_end, test_start,
                )
            except KeyError as e:
                logger.error("Target column missing: %s — skipping", e)
                continue

            if len(X_train) < 20 or len(X_test) < 3:
                logger.warning(
                    "Insufficient data for %s h=%d (train=%d, test=%d) — skipping",
                    target, horizon, len(X_train), len(X_test),
                )
                continue

            # ── ARIMA (univariate — uses only the target column) ────
            try:
                arima = ARIMAModel(config)
                y_train_uni = features_df.loc[features_df.index <= train_end, target].dropna()
                arima.fit(y_train_uni, target, horizon)
                arima_preds = arima.predict_horizon(
                    features_df.loc[features_df.index >= test_start, target].dropna()
                )
                # Align predictions with y_test
                common_idx = y_test.index.intersection(arima_preds.index)
                if len(common_idx) > 0:
                    arima_metrics = arima.evaluate(y_test.loc[common_idx], arima_preds.loc[common_idx])
                    evaluator.record_result(
                        "ARIMA", target, horizon, arima_metrics,
                        y_true=y_test.loc[common_idx],
                        y_pred=arima_preds.loc[common_idx],
                    )
                arima.save(models_path)
                trained_models[f"arima_{target}_h{horizon}"] = arima
            except Exception:
                logger.exception("ARIMA training failed for %s h=%d", target, horizon)

            # ── XGBoost ─────────────────────────────────────────────
            try:
                if not _xgboost_available:
                    raise ImportError("XGBoost library not available")
                xgb_model = XGBoostModel(config)
                xgb_model.fit(X_train, y_train, target, horizon)
                xgb_preds = xgb_model.predict(X_test)
                xgb_metrics = xgb_model.evaluate(y_test, xgb_preds)
                y_pred_series = pd.Series(xgb_preds, index=y_test.index)
                evaluator.record_result(
                    "XGBoost", target, horizon, xgb_metrics,
                    y_true=y_test,
                    y_pred=y_pred_series,
                )
                # Feature importance
                importance_df = xgb_model.get_feature_importance(top_n=20)
                evaluator.plot_feature_importance(importance_df, target, horizon)
                xgb_model.save(models_path)
                trained_models[f"xgboost_{target}_h{horizon}"] = xgb_model
            except Exception:
                logger.exception("XGBoost training failed for %s h=%d", target, horizon)

            # ── Neural Network ──────────────────────────────────────
            try:
                nn_model = NeuralNetModel(config)
                nn_model.fit(X_train, y_train, target, horizon)
                nn_preds = nn_model.predict(X_test)
                nn_metrics = nn_model.evaluate(y_test, nn_preds)
                y_pred_series = pd.Series(nn_preds, index=y_test.index)
                evaluator.record_result(
                    "NeuralNet", target, horizon, nn_metrics,
                    y_true=y_test,
                    y_pred=y_pred_series,
                )
                nn_model.save(models_path)
                trained_models[f"neural_net_{target}_h{horizon}"] = nn_model
            except Exception:
                logger.exception("NeuralNet training failed for %s h=%d", target, horizon)

            # ── Plots and comparisons for this target/horizon ───────
            evaluator.plot_actual_vs_predicted(target, horizon)
            evaluator.plot_error_distribution(target, horizon)
            evaluator.diebold_mariano_test(target, horizon)
            evaluator.save_forecasts(target, horizon)

    # ── Final comparison table ──────────────────────────────────────────
    evaluator.print_comparison_table()

    logger.info("Training complete — %d models trained", len(trained_models))
    return trained_models


def step_forecast(
    config: Dict[str, Any],
    features_df: pd.DataFrame,
) -> None:
    """Generate forecasts using saved models.

    Args:
        config: Pipeline configuration.
        features_df: Feature-engineered DataFrame.
    """
    from src.models.xgboost_model import XGBoostModel
    from src.models.neural_net import NeuralNetModel

    logger.info("=" * 70)
    logger.info("STEP: FORECAST GENERATION")
    logger.info("=" * 70)

    models_path = PROJECT_ROOT / config["paths"]["models"]
    forecasts_path = PROJECT_ROOT / config["paths"]["forecasts"]
    forecasts_path.mkdir(parents=True, exist_ok=True)

    targets = config["targets"]
    horizons = config["forecast_horizons"]

    # Use the most recent data point as the forecast input
    latest_features = features_df.iloc[[-1]]

    for target in targets:
        forecast_rows = []
        for horizon in horizons:
            # Try XGBoost first (typically best performer)
            xgb_path = models_path / f"xgboost_{target}_h{horizon}.joblib"
            if xgb_path.exists():
                xgb_model = XGBoostModel(config)
                xgb_model.load(xgb_path)
                pred = xgb_model.predict(latest_features)[0]
                forecast_rows.append({
                    "target": target,
                    "horizon_months": horizon,
                    "model": "XGBoost",
                    "forecast": pred,
                    "forecast_date": str(latest_features.index[0].date()),
                })
                logger.info(
                    "Forecast — %s h=%d | XGBoost=%.4f", target, horizon, pred
                )

            # Neural Net
            nn_path = models_path / f"neural_net_{target}_h{horizon}.pt"
            if nn_path.exists():
                nn_model = NeuralNetModel(config)
                nn_model.load(nn_path)
                pred = nn_model.predict(latest_features)[0]
                forecast_rows.append({
                    "target": target,
                    "horizon_months": horizon,
                    "model": "NeuralNet",
                    "forecast": pred,
                    "forecast_date": str(latest_features.index[0].date()),
                })
                logger.info(
                    "Forecast — %s h=%d | NeuralNet=%.4f", target, horizon, pred
                )

        if forecast_rows:
            df = pd.DataFrame(forecast_rows)
            filepath = forecasts_path / f"latest_forecast_{target}.csv"
            df.to_csv(filepath, index=False)
            logger.info("Latest forecasts saved → %s", filepath)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="uk-economic-forecasting",
        description=(
            "Automated ETL + ML pipeline for forecasting UK macroeconomic "
            "indicators.  Supports data ingestion from ONS and BoE APIs, "
            "model training (ARIMA, XGBoost, Neural Net), and evaluation."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["update", "train", "forecast", "full", "sample"],
        default="sample",
        help=(
            "Pipeline mode: "
            "'update' fetches fresh API data, "
            "'train' retrains models, "
            "'forecast' generates predictions, "
            "'full' runs update+train+forecast, "
            "'sample' runs end-to-end with bundled sample data (default)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the pipeline."""
    args = parse_args()
    config = _load_config(args.config)
    _setup_logging(config)

    logger.info("=" * 70)
    logger.info("UK Economic Forecasting Pipeline")
    logger.info("Mode: %s", args.mode)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("=" * 70)

    try:
        if args.mode == "sample":
            # Full pipeline with sample data
            df = step_load_sample(config)
            features_df = step_feature_engineering(config, df)
            step_train(config, features_df)
            step_forecast(config, features_df)

        elif args.mode == "update":
            step_update(config)

        elif args.mode == "train":
            from src.data.preprocessor import Preprocessor
            preprocessor = Preprocessor(config, PROJECT_ROOT)
            df = preprocessor.load()
            features_df = step_feature_engineering(config, df)
            step_train(config, features_df)

        elif args.mode == "forecast":
            from src.data.preprocessor import Preprocessor
            preprocessor = Preprocessor(config, PROJECT_ROOT)
            df = preprocessor.load()
            features_df = step_feature_engineering(config, df)
            step_forecast(config, features_df)

        elif args.mode == "full":
            df = step_update(config)
            features_df = step_feature_engineering(config, df)
            step_train(config, features_df)
            step_forecast(config, features_df)

        logger.info("=" * 70)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 70)

    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
