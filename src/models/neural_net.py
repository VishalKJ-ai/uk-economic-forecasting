"""PyTorch feedforward neural network for economic forecasting.

Architecture: Input -> 128 -> 64 -> 32 -> Output
Each hidden layer uses BatchNorm, ReLU, and Dropout(0.3).
Optimised with Adam and a ReduceLROnPlateau scheduler.
Early stopping prevents overfitting on the small economic dataset.

This model captures non-linear relationships between macroeconomic
variables that tree-based and linear models might miss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class _ForecastNet(nn.Module):
    """Feedforward network for regression.

    Architecture:
        Input(n_features) -> Linear(128) -> BN -> ReLU -> Dropout(0.3)
                           -> Linear(64)  -> BN -> ReLU -> Dropout(0.3)
                           -> Linear(32)  -> BN -> ReLU -> Dropout(0.3)
                           -> Linear(1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_layers: List[int],
        dropout: float,
    ) -> None:
        """Build the network.

        Args:
            n_features: Number of input features.
            hidden_layers: List of hidden layer sizes (e.g. [128, 64, 32]).
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = n_features

        for h_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch_size, n_features)``.

        Returns:
            Predictions of shape ``(batch_size, 1)``.
        """
        return self.network(x)


class NeuralNetModel:
    """PyTorch neural network wrapper for the forecasting pipeline.

    Handles training, prediction, evaluation, and model persistence with
    the same interface as the ARIMA and XGBoost models.

    Attributes:
        config: Neural-net-specific hyperparameter configuration.
        model: The PyTorch ``_ForecastNet`` instance.
        scaler: StandardScaler fitted on training features.
        target: Target variable name.
        horizon: Forecast horizon in months.
        device: Torch device (CPU or CUDA).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise neural network model from configuration.

        Args:
            config: Full pipeline configuration.  The ``models.neural_net``
                sub-key is used.
        """
        nn_cfg = config["models"]["neural_net"]
        self.hidden_layers: List[int] = nn_cfg["hidden_layers"]
        self.dropout: float = nn_cfg["dropout"]
        self.learning_rate: float = nn_cfg["learning_rate"]
        self.weight_decay: float = nn_cfg["weight_decay"]
        self.batch_size: int = nn_cfg["batch_size"]
        self.max_epochs: int = nn_cfg["max_epochs"]
        self.patience: int = nn_cfg["patience"]
        self.lr_scheduler_cfg: Dict[str, Any] = nn_cfg["lr_scheduler"]

        self.model: Optional[_ForecastNet] = None
        self.scaler: Optional[StandardScaler] = None
        self.target: Optional[str] = None
        self.horizon: Optional[int] = None
        self.feature_names: Optional[List[str]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "NeuralNet initialised — layers=%s, dropout=%.2f, device=%s",
            self.hidden_layers, self.dropout, self.device,
        )

    # ── public API ──────────────────────────────────────────────────────
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target: str,
        horizon: int,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "NeuralNetModel":
        """Train the neural network with early stopping.

        If no validation set is provided, the last 20% of training data is
        held out for early stopping.

        Args:
            X_train: Training feature matrix.
            y_train: Training target values.
            target: Name of the target variable.
            horizon: Forecast horizon in months.
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Self, for method chaining.
        """
        self.target = target
        self.horizon = horizon
        self.feature_names = list(X_train.columns)

        logger.info(
            "Fitting NeuralNet on '%s' — %d samples, %d features, horizon=%d",
            target, len(X_train), len(self.feature_names), horizon,
        )

        # Create validation split if not provided
        if X_val is None or y_val is None:
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]
            logger.info(
                "Auto validation split — train=%d, val=%d", len(X_train), len(X_val)
            )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.values)
        X_val_scaled = self.scaler.transform(X_val.values)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.FloatTensor(y_train.values.reshape(-1, 1)).to(self.device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_t = torch.FloatTensor(y_val.values.reshape(-1, 1)).to(self.device)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(X_train_t)),
            shuffle=False,  # Time series — preserve order within batches
        )

        # Build model
        n_features = X_train_scaled.shape[1]
        self.model = _ForecastNet(
            n_features=n_features,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_cfg["factor"],
            patience=self.lr_scheduler_cfg["patience"],
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # ── Training ────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validation ──────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_t)
                val_loss = criterion(val_predictions, y_val_t).item()

            scheduler.step(val_loss)
            mean_train_loss = np.mean(train_losses)

            if (epoch + 1) % 20 == 0:
                logger.debug(
                    "Epoch %3d/%d — train_loss=%.6f, val_loss=%.6f, lr=%.2e",
                    epoch + 1, self.max_epochs,
                    mean_train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            # ── Early stopping ──────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d — best val_loss=%.6f",
                    epoch + 1, best_val_loss,
                )
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        logger.info(
            "NeuralNet training complete — final val_loss=%.6f after %d epochs",
            best_val_loss, epoch + 1,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given feature matrix.

        Args:
            X: Feature matrix with the same columns as training data.

        Returns:
            Array of predicted values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted — call .fit() first")

        self.model.eval()
        X_scaled = self.scaler.transform(X.values)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()

        logger.debug(
            "NeuralNet predicted %d samples for '%s'", len(predictions), self.target
        )
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
            "NeuralNet evaluation ('%s' h=%d) — RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%",
            self.target, self.horizon,
            metrics["rmse"], metrics["mae"], metrics["mape"],
        )
        return metrics

    def save(self, path: Path) -> Path:
        """Persist the model, scaler, and metadata.

        Args:
            path: Directory in which to save the model.

        Returns:
            Path to the saved checkpoint file.
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"neural_net_{self.target}_h{self.horizon}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "n_features": len(self.feature_names),
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "feature_names": self.feature_names,
                "target": self.target,
                "horizon": self.horizon,
            },
            filepath,
        )
        logger.info("NeuralNet model saved → %s", filepath)
        return filepath

    def load(self, filepath: Path) -> "NeuralNetModel":
        """Load a previously saved model.

        Args:
            filepath: Path to the checkpoint file.

        Returns:
            Self, for method chaining.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.feature_names = checkpoint["feature_names"]
        self.target = checkpoint["target"]
        self.horizon = checkpoint["horizon"]

        self.model = _ForecastNet(
            n_features=checkpoint["n_features"],
            hidden_layers=checkpoint["hidden_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]
        self.scaler.n_features_in_ = checkpoint["n_features"]
        self.scaler.var_ = checkpoint["scaler_scale"] ** 2

        logger.info("NeuralNet model loaded ← %s", filepath)
        return self
