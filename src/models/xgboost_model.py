"""XGBoost gradient-boosted tree model for multivariate economic forecasting.

Unlike ARIMA, XGBoost can exploit the full feature matrix — lags, rolling
statistics, interaction terms, and cross-variable signals — to predict each
target indicator.

Key design choices:
- Expanding-window time series cross-validation (never train on future data).
- Hyperparameter tuning via time series CV, not random k-fold.
- Feature importance extraction for model interpretability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
try:
    import xgboost as xgb
except (ImportError, OSError) as _xgb_err:
    xgb = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "XGBoost could not be loaded (%s). XGBoostModel will be unavailable.",
        _xgb_err,
    )

logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost regressor for macroeconomic forecasting.

    Attributes:
        config: XGBoost-specific hyperparameter configuration.
        model: The fitted ``xgb.XGBRegressor`` (or ``None``).
        target: Target variable name.
        horizon: Forecast horizon in months.
        feature_names: Ordered list of feature column names after fitting.
        best_params: Best hyperparameters found during tuning.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise XGBoost model from configuration.

        Args:
            config: Full pipeline configuration.  The ``models.xgboost``
                sub-key is used.

        Raises:
            ImportError: If xgboost library is not available.
        """
        if xgb is None:
            raise ImportError(
                "XGBoost is not available. Install it with: pip install xgboost. "
                "On macOS you may also need: brew install libomp"
            )
        xgb_cfg = config["models"]["xgboost"]
        self.param_grid: Dict[str, List[Any]] = xgb_cfg["param_grid"]
        self.cv_folds: int = xgb_cfg["cv_folds"]
        self.early_stopping_rounds: int = xgb_cfg["early_stopping_rounds"]

        self.model: Optional[xgb.XGBRegressor] = None
        self.target: Optional[str] = None
        self.horizon: Optional[int] = None
        self.feature_names: Optional[List[str]] = None
        self.best_params: Optional[Dict[str, Any]] = None

        logger.info(
            "XGBoost initialised — cv_folds=%d, param_grid keys=%s",
            self.cv_folds,
            list(self.param_grid.keys()),
        )

    # ── public API ──────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target: str,
        horizon: int,
    ) -> "XGBoostModel":
        """Fit XGBoost with time series cross-validated hyperparameter search.

        Rather than exhaustive grid search (which is expensive), we evaluate
        a deterministic subset of the parameter grid using expanding-window
        time series splits.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            target: Name of the target variable.
            horizon: Forecast horizon in months.

        Returns:
            Self, for method chaining.
        """
        self.target = target
        self.horizon = horizon
        self.feature_names = list(X_train.columns)

        logger.info(
            "Fitting XGBoost on '%s' — %d samples, %d features, horizon=%d",
            target, len(X_train), len(self.feature_names), horizon,
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        best_score = np.inf
        best_params: Dict[str, Any] = {}

        # Generate a manageable set of parameter combinations
        param_combos = self._generate_param_combos()

        for params in param_combos:
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]

                model = xgb.XGBRegressor(
                    **params,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)

            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        logger.info(
            "XGBoost CV best RMSE=%.4f with params=%s", best_score, best_params
        )

        # Refit on the full training set with best params
        self.best_params = best_params
        self.model = xgb.XGBRegressor(
            **best_params,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X_train, y_train, verbose=False)

        logger.info("XGBoost final model fitted on %d training samples", len(X_train))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given feature matrix.

        Args:
            X: Feature matrix with the same columns as the training data.

        Returns:
            Array of predicted values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted — call .fit() first")

        predictions = self.model.predict(X)
        logger.debug("XGBoost predicted %d samples for '%s'", len(predictions), self.target)
        return predictions

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with RMSE, MAE, and MAPE.
        """
        from src.evaluation.evaluator import Evaluator

        y_pred_series = pd.Series(y_pred, index=y_true.index)
        metrics = Evaluator.compute_metrics(y_true, y_pred_series)
        logger.info(
            "XGBoost evaluation ('%s' h=%d) — RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%",
            self.target, self.horizon,
            metrics["rmse"], metrics["mae"], metrics["mape"],
        )
        return metrics

    def get_feature_importance(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Extract feature importance scores.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with columns ``["feature", "importance"]`` sorted
            descending by importance.
        """
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model not fitted")

        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        })
        importance_df = (
            importance_df
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return importance_df

    def save(self, path: Path) -> Path:
        """Persist the fitted model to disk.

        Args:
            path: Directory in which to save the model.

        Returns:
            Path to the saved file.
        """
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"xgboost_{self.target}_h{self.horizon}.joblib"
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "best_params": self.best_params,
                "target": self.target,
                "horizon": self.horizon,
            },
            filepath,
        )
        logger.info("XGBoost model saved → %s", filepath)
        return filepath

    def load(self, filepath: Path) -> "XGBoostModel":
        """Load a previously saved model.

        Args:
            filepath: Path to the joblib file.

        Returns:
            Self, for method chaining.
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.best_params = data["best_params"]
        self.target = data["target"]
        self.horizon = data["horizon"]
        logger.info("XGBoost model loaded ← %s", filepath)
        return self

    # ── internal helpers ────────────────────────────────────────────────
    def _generate_param_combos(self) -> List[Dict[str, Any]]:
        """Generate a subset of hyperparameter combinations.

        To keep training time manageable, we sample the middle and
        extreme values from each grid dimension rather than doing a full
        Cartesian product.

        Returns:
            List of parameter dictionaries.
        """
        combos: List[Dict[str, Any]] = []

        # Take first, middle, and last from each list
        def sample_values(values: list) -> list:
            if len(values) <= 3:
                return values
            return [values[0], values[len(values) // 2], values[-1]]

        n_est_vals = sample_values(self.param_grid["n_estimators"])
        depth_vals = sample_values(self.param_grid["max_depth"])
        lr_vals = sample_values(self.param_grid["learning_rate"])

        for n_est in n_est_vals:
            for depth in depth_vals:
                for lr in lr_vals:
                    combos.append({
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "learning_rate": lr,
                        "subsample": self.param_grid["subsample"][-1],
                        "colsample_bytree": self.param_grid["colsample_bytree"][-1],
                    })

        logger.debug("Generated %d parameter combinations for tuning", len(combos))
        return combos
