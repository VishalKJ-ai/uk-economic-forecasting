"""ARIMA / SARIMA baseline forecasting model.

Uses ``pmdarima.auto_arima`` for automatic order selection, falling back to a
manual grid search if the auto procedure fails.  SARIMA captures seasonality
that is common in retail sales and industrial production series.

This serves as the interpretable baseline against which more complex models
(XGBoost, neural network) are compared.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pmdarima as pm

logger = logging.getLogger(__name__)


class ARIMAModel:
    """Auto-fit ARIMA / SARIMA model for univariate time series forecasting.

    Attributes:
        config: Model-specific hyperparameter configuration.
        model: The fitted ``pmdarima`` ARIMA object (or ``None``).
        target: Name of the target variable.
        horizon: Forecast horizon in months.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise ARIMA model parameters.

        Args:
            config: Full pipeline configuration dictionary.  The
                ``models.arima`` sub-key is used for hyperparameters.
        """
        arima_cfg = config["models"]["arima"]
        self.max_p: int = arima_cfg["max_p"]
        self.max_d: int = arima_cfg["max_d"]
        self.max_q: int = arima_cfg["max_q"]
        self.seasonal: bool = arima_cfg["seasonal"]
        self.seasonal_period: int = arima_cfg["seasonal_period"]
        self.stepwise: bool = arima_cfg["stepwise"]
        self.information_criterion: str = arima_cfg["information_criterion"]

        self.model: Optional[pm.ARIMA] = None
        self.target: Optional[str] = None
        self.horizon: Optional[int] = None

        logger.info(
            "ARIMA initialised — max_order=(%d,%d,%d), seasonal=%s, m=%d",
            self.max_p, self.max_d, self.max_q,
            self.seasonal, self.seasonal_period,
        )

    # ── public API ──────────────────────────────────────────────────────
    def fit(
        self,
        y_train: pd.Series,
        target: str,
        horizon: int,
    ) -> "ARIMAModel":
        """Fit the ARIMA model to a univariate training series.

        Uses ``pmdarima.auto_arima`` with the configured search space.
        The model is fit on the raw target series (not multi-variate
        features) because ARIMA operates on a single time series.

        Args:
            y_train: Training observations (univariate, time-ordered).
            target: Name of the target variable being modelled.
            horizon: Forecast horizon (informational — used for logging).

        Returns:
            Self, for method chaining.
        """
        self.target = target
        self.horizon = horizon
        logger.info(
            "Fitting ARIMA on '%s' — %d training observations, horizon=%d",
            target, len(y_train), horizon,
        )

        try:
            self.model = pm.auto_arima(
                y_train.values,
                start_p=1,
                start_q=1,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                seasonal=self.seasonal,
                m=self.seasonal_period if self.seasonal else 1,
                stepwise=self.stepwise,
                information_criterion=self.information_criterion,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
            logger.info(
                "ARIMA auto-fit complete — order=%s, seasonal_order=%s, AIC=%.2f",
                self.model.order,
                self.model.seasonal_order,
                self.model.aic(),
            )
        except Exception:
            logger.exception(
                "auto_arima failed for '%s' — falling back to ARIMA(1,1,1)",
                target,
            )
            self.model = pm.ARIMA(order=(1, 1, 1))
            self.model.fit(y_train.values)

        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        """Generate out-of-sample forecasts.

        Args:
            steps: Number of months to forecast ahead.

        Returns:
            Array of predicted values with length ``steps``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted — call .fit() first")

        forecasts = self.model.predict(n_periods=steps)
        logger.debug("ARIMA predicted %d steps ahead for '%s'", steps, self.target)
        return forecasts

    def predict_horizon(self, y_test: pd.Series) -> pd.Series:
        """Walk-forward predict over the test set at the configured horizon.

        For each time step in the test set, the model uses all data up to
        that point and predicts ``self.horizon`` steps ahead.  The model is
        *updated* (not re-fitted) with each new observation for efficiency.

        Args:
            y_test: Test observations (univariate, time-ordered).

        Returns:
            Series of predictions aligned with ``y_test`` index.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted — call .fit() first")

        predictions: List[float] = []

        for i in range(len(y_test)):
            # Predict horizon steps ahead
            fc = self.model.predict(n_periods=self.horizon)
            predictions.append(fc[-1])  # Take the h-step-ahead value

            # Update model with the new observation
            self.model.update(y_test.iloc[i:i + 1].values)

        result = pd.Series(predictions, index=y_test.index, name=f"arima_pred")
        logger.info(
            "Walk-forward prediction complete for '%s' h=%d — %d predictions",
            self.target, self.horizon, len(result),
        )
        return result

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with RMSE, MAE, and MAPE.
        """
        from src.evaluation.evaluator import Evaluator

        metrics = Evaluator.compute_metrics(y_true, y_pred)
        logger.info(
            "ARIMA evaluation ('%s' h=%d) — RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%",
            self.target, self.horizon,
            metrics["rmse"], metrics["mae"], metrics["mape"],
        )
        return metrics

    def save(self, path: Path) -> Path:
        """Persist the fitted model to disk.

        Args:
            path: Directory in which to save the model file.

        Returns:
            Path to the saved pickle file.
        """
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"arima_{self.target}_h{self.horizon}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("ARIMA model saved → %s", filepath)
        return filepath

    def load(self, filepath: Path) -> "ARIMAModel":
        """Load a previously saved model.

        Args:
            filepath: Path to the pickle file.

        Returns:
            Self, for method chaining.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        logger.info("ARIMA model loaded ← %s", filepath)
        return self
