"""Model evaluation, comparison, and visualisation.

Provides:
- Standard regression metrics (RMSE, MAE, MAPE).
- Formatted comparison tables (model x horizon x metric).
- Actual vs predicted plots.
- Forecast error distribution plots.
- Diebold-Mariano test for comparing predictive accuracy.

All figures are saved as high-quality PNGs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Matplotlib config — must happen before import of pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Figure quality
_FIG_DPI = 150
_FIG_FORMAT = "png"


class Evaluator:
    """Evaluate and compare forecasting models.

    Attributes:
        figures_path: Directory for saving plots.
        forecasts_path: Directory for saving forecast CSVs.
        metrics_store: Accumulated metrics for the comparison table.
    """

    def __init__(self, config: Dict[str, Any], project_root: Path) -> None:
        """Initialise the evaluator.

        Args:
            config: Full pipeline configuration.
            project_root: Project root directory.
        """
        self.figures_path: Path = project_root / config["paths"]["figures"]
        self.forecasts_path: Path = project_root / config["paths"]["forecasts"]
        self.figures_path.mkdir(parents=True, exist_ok=True)
        self.forecasts_path.mkdir(parents=True, exist_ok=True)
        self.significance_level: float = config["evaluation"]["diebold_mariano_significance"]
        self.metrics_store: List[Dict[str, Any]] = []

        logger.info(
            "Evaluator initialised — figures=%s, forecasts=%s",
            self.figures_path, self.forecasts_path,
        )

    # ── metrics ─────────────────────────────────────────────────────────
    @staticmethod
    def compute_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, float]:
        """Compute RMSE, MAE, and MAPE.

        MAPE is computed only on non-zero actual values to avoid division
        by zero (common in near-zero GDP growth periods).

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with keys ``"rmse"``, ``"mae"``, ``"mape"``.
        """
        errors = y_true.values - y_pred.values
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        # MAPE — guard against zero actuals
        nonzero_mask = np.abs(y_true.values) > 1e-8
        if nonzero_mask.sum() > 0:
            mape = float(
                np.mean(np.abs(errors[nonzero_mask] / y_true.values[nonzero_mask])) * 100
            )
        else:
            mape = float("nan")

        return {"rmse": rmse, "mae": mae, "mape": mape}

    def record_result(
        self,
        model_name: str,
        target: str,
        horizon: int,
        metrics: Dict[str, float],
        y_true: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
    ) -> None:
        """Store a model's evaluation result for later comparison.

        Args:
            model_name: Human-readable model name (e.g. ``"ARIMA"``).
            target: Target variable name.
            horizon: Forecast horizon in months.
            metrics: Dict of metric values.
            y_true: Actual values (stored for DM test and plotting).
            y_pred: Predicted values.
        """
        entry = {
            "model": model_name,
            "target": target,
            "horizon": horizon,
            **metrics,
        }
        if y_true is not None and y_pred is not None:
            entry["_y_true"] = y_true
            entry["_y_pred"] = y_pred

        self.metrics_store.append(entry)
        logger.info(
            "Recorded result — %s | %s | h=%d | RMSE=%.4f | MAE=%.4f | MAPE=%.2f%%",
            model_name, target, horizon,
            metrics["rmse"], metrics["mae"], metrics["mape"],
        )

    # ── comparison table ────────────────────────────────────────────────
    def print_comparison_table(self) -> pd.DataFrame:
        """Print a formatted comparison table of all recorded results.

        Returns:
            DataFrame with columns [model, target, horizon, rmse, mae, mape].
        """
        rows = [
            {k: v for k, v in entry.items() if not k.startswith("_")}
            for entry in self.metrics_store
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("No results to compare")
            return df

        df = df.sort_values(["target", "horizon", "rmse"]).reset_index(drop=True)

        table_str = tabulate(
            df,
            headers="keys",
            tablefmt="fancy_grid",
            floatfmt=(".0f", "", "", ".0f", ".4f", ".4f", ".2f"),
            showindex=False,
        )
        logger.info("\n\nModel Comparison\n%s\n", table_str)

        # Save to CSV
        csv_path = self.forecasts_path / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Comparison table saved → %s", csv_path)

        return df

    # ── plotting ────────────────────────────────────────────────────────
    def plot_actual_vs_predicted(
        self,
        target: str,
        horizon: int,
    ) -> Optional[Path]:
        """Plot actual vs predicted for all models on a given target/horizon.

        Args:
            target: Target variable name.
            horizon: Forecast horizon.

        Returns:
            Path to the saved figure, or ``None`` if no data.
        """
        entries = [
            e for e in self.metrics_store
            if e["target"] == target
            and e["horizon"] == horizon
            and "_y_true" in e
        ]
        if not entries:
            logger.warning("No data for actual-vs-predicted plot (%s h=%d)", target, horizon)
            return None

        fig, ax = plt.subplots(figsize=(12, 5))
        y_true = entries[0]["_y_true"]
        ax.plot(y_true.index, y_true.values, "k-", linewidth=2, label="Actual", zorder=5)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, entry in enumerate(entries):
            y_pred = entry["_y_pred"]
            color = colors[i % len(colors)]
            ax.plot(
                y_pred.index, y_pred.values,
                "--", color=color, linewidth=1.5,
                label=f"{entry['model']} (RMSE={entry['rmse']:.4f})",
            )

        ax.set_title(f"{target.upper()} — Actual vs Predicted (h={horizon}m)", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel(target.upper())
        ax.legend(loc="best", frameon=True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.tight_layout()

        filepath = self.figures_path / f"actual_vs_pred_{target}_h{horizon}.{_FIG_FORMAT}"
        fig.savefig(filepath, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved actual-vs-predicted plot → %s", filepath)
        return filepath

    def plot_error_distribution(
        self,
        target: str,
        horizon: int,
    ) -> Optional[Path]:
        """Plot forecast error distributions for all models.

        Args:
            target: Target variable name.
            horizon: Forecast horizon.

        Returns:
            Path to the saved figure, or ``None`` if no data.
        """
        entries = [
            e for e in self.metrics_store
            if e["target"] == target
            and e["horizon"] == horizon
            and "_y_true" in e
        ]
        if not entries:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        for entry in entries:
            errors = entry["_y_true"].values - entry["_y_pred"].values
            ax.hist(
                errors, bins=15, alpha=0.5,
                label=f"{entry['model']} (MAE={entry['mae']:.4f})",
                edgecolor="white",
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(
            f"{target.upper()} — Forecast Error Distribution (h={horizon}m)",
            fontsize=14,
        )
        ax.set_xlabel("Error (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        filepath = self.figures_path / f"error_dist_{target}_h{horizon}.{_FIG_FORMAT}"
        fig.savefig(filepath, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved error distribution plot → %s", filepath)
        return filepath

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        target: str,
        horizon: int,
    ) -> Path:
        """Plot XGBoost feature importance.

        Args:
            importance_df: DataFrame with ``feature`` and ``importance`` columns.
            target: Target variable name.
            horizon: Forecast horizon.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            importance_df["feature"][::-1],
            importance_df["importance"][::-1],
            color="#1f77b4",
            edgecolor="white",
        )
        ax.set_title(
            f"XGBoost Feature Importance — {target.upper()} (h={horizon}m)",
            fontsize=14,
        )
        ax.set_xlabel("Importance (Gain)")
        fig.tight_layout()

        filepath = self.figures_path / f"feature_importance_{target}_h{horizon}.{_FIG_FORMAT}"
        fig.savefig(filepath, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance plot → %s", filepath)
        return filepath

    # ── statistical tests ───────────────────────────────────────────────
    def diebold_mariano_test(
        self,
        target: str,
        horizon: int,
    ) -> Optional[pd.DataFrame]:
        """Run pairwise Diebold-Mariano tests between all models.

        The DM test checks whether the difference in forecast accuracy
        between two models is statistically significant.

        Args:
            target: Target variable name.
            horizon: Forecast horizon.

        Returns:
            DataFrame of pairwise DM test results, or ``None``.
        """
        entries = [
            e for e in self.metrics_store
            if e["target"] == target
            and e["horizon"] == horizon
            and "_y_true" in e
        ]
        if len(entries) < 2:
            logger.info("Need at least 2 models for DM test — skipping")
            return None

        results = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                e1, e2 = entries[i], entries[j]
                dm_stat, p_value = self._dm_test(
                    e1["_y_true"].values,
                    e1["_y_pred"].values,
                    e2["_y_pred"].values,
                )
                significant = p_value < self.significance_level
                results.append({
                    "model_1": e1["model"],
                    "model_2": e2["model"],
                    "target": target,
                    "horizon": horizon,
                    "dm_statistic": dm_stat,
                    "p_value": p_value,
                    "significant": significant,
                })

        df = pd.DataFrame(results)
        logger.info(
            "Diebold-Mariano results for %s h=%d:\n%s",
            target, horizon,
            tabulate(df, headers="keys", tablefmt="simple", showindex=False),
        )
        return df

    @staticmethod
    def _dm_test(
        y_true: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute the Diebold-Mariano test statistic.

        Uses squared error as the loss function.  The test statistic is
        asymptotically standard normal under the null of equal predictive
        accuracy.

        Args:
            y_true: Actual values.
            pred_1: Predictions from model 1.
            pred_2: Predictions from model 2.

        Returns:
            Tuple of (DM statistic, p-value).
        """
        e1 = y_true - pred_1
        e2 = y_true - pred_2
        d = e1 ** 2 - e2 ** 2  # Loss differential

        n = len(d)
        mean_d = np.mean(d)

        # Newey-West variance estimate with lag 1
        gamma_0 = np.var(d, ddof=1)
        if n > 1:
            gamma_1 = np.cov(d[:-1], d[1:])[0, 1]
        else:
            gamma_1 = 0.0

        var_d = (gamma_0 + 2 * gamma_1) / n
        if var_d <= 0:
            return 0.0, 1.0

        dm_stat = mean_d / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        return float(dm_stat), float(p_value)

    # ── forecast export ─────────────────────────────────────────────────
    def save_forecasts(
        self,
        target: str,
        horizon: int,
    ) -> Optional[Path]:
        """Save forecast results as a CSV.

        Args:
            target: Target variable name.
            horizon: Forecast horizon.

        Returns:
            Path to saved CSV, or ``None``.
        """
        entries = [
            e for e in self.metrics_store
            if e["target"] == target
            and e["horizon"] == horizon
            and "_y_true" in e
        ]
        if not entries:
            return None

        df = pd.DataFrame({"date": entries[0]["_y_true"].index})
        df["actual"] = entries[0]["_y_true"].values
        for entry in entries:
            df[f"{entry['model']}_pred"] = entry["_y_pred"].values

        filepath = self.forecasts_path / f"forecast_{target}_h{horizon}.csv"
        df.to_csv(filepath, index=False)
        logger.info("Saved forecasts → %s", filepath)
        return filepath
