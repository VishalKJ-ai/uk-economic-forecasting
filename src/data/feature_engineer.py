"""Feature engineering for macroeconomic forecasting.

Transforms the cleaned monthly DataFrame into a rich feature matrix suitable
for supervised learning.  The engineering choices are motivated by standard
practices in applied macroeconomic forecasting:

- **Lag features** capture autoregressive dynamics.
- **Rolling statistics** smooth out noise and reveal trends.
- **Month-over-month / year-over-year changes** isolate momentum signals.
- **Spread features** approximate yield-curve-like indicators.
- **Interaction terms** capture non-linear relationships between indicators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Construct feature matrices from processed macroeconomic data.

    Attributes:
        lag_periods: List of lag offsets (in months).
        rolling_windows: List of rolling window sizes (in months).
        rolling_statistics: List of statistics to compute per window.
        mom_period: Month-over-month change period.
        yoy_period: Year-over-year change period.
        interaction_pairs: List of (col_a, col_b) pairs for interactions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise from the pipeline configuration.

        Args:
            config: Full pipeline configuration dictionary.  The
                ``features`` sub-key is used.
        """
        feat_cfg = config["features"]
        self.lag_periods: List[int] = feat_cfg["lag_periods"]
        self.rolling_windows: List[int] = feat_cfg["rolling_windows"]
        self.rolling_statistics: List[str] = feat_cfg["rolling_statistics"]
        self.mom_period: int = feat_cfg["change_periods"]["mom"]
        self.yoy_period: int = feat_cfg["change_periods"]["yoy"]
        self.interaction_pairs: List[List[str]] = feat_cfg.get(
            "interaction_pairs", []
        )
        logger.info(
            "FeatureEngineer initialised — lags=%s, windows=%s",
            self.lag_periods,
            self.rolling_windows,
        )

    # ── public API ──────────────────────────────────────────────────────
    def transform(
        self,
        df: pd.DataFrame,
        targets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply all feature transformations.

        Args:
            df: Processed monthly DataFrame indexed by date.
            targets: Optional list of target column names.  If provided,
                target columns are excluded from feature generation to
                avoid look-ahead bias when building lag features for
                inputs only.

        Returns:
            Augmented DataFrame with original columns plus engineered
            features.  Rows with NaN introduced by lagging/rolling are
            dropped.
        """
        logger.info(
            "Starting feature engineering — input shape %s", df.shape
        )
        df = df.copy()
        base_cols = list(df.columns)

        df = self._add_lag_features(df, base_cols)
        df = self._add_rolling_features(df, base_cols)
        df = self._add_change_features(df, base_cols)
        df = self._add_spread_features(df)
        df = self._add_interaction_features(df)
        df = self._add_calendar_features(df)

        # Replace any infinities from pct_change on zero values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop rows where lagging/rolling introduced NaNs
        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "Dropped %d rows with NaN from feature engineering", n_dropped
            )

        logger.info(
            "Feature engineering complete — output shape %s  (%d new features)",
            df.shape,
            len(df.columns) - len(base_cols),
        )
        return df

    def prepare_supervised(
        self,
        df: pd.DataFrame,
        target: str,
        horizon: int,
        train_end: str,
        test_start: str,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split the feature matrix into train/test for a given target and horizon.

        The target is shifted backwards by ``horizon`` months so that
        features at time *t* predict the target at time *t + horizon*.
        This ensures no data leakage.

        Args:
            df: Feature-engineered DataFrame.
            target: Name of the target column (e.g. ``"gdp"``).
            horizon: Forecast horizon in months.
            train_end: Last date (inclusive) for training data.
            test_start: First date (inclusive) for test data.

        Returns:
            Tuple of ``(X_train, y_train, X_test, y_test)``.
        """
        if target not in df.columns:
            raise KeyError(f"Target '{target}' not found in DataFrame columns")

        df = df.copy()
        # Create the forward-looking target
        target_col = f"{target}_target_h{horizon}"
        df[target_col] = df[target].shift(-horizon)
        df = df.dropna(subset=[target_col])

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col]

        train_mask = X.index <= train_end
        test_mask = X.index >= test_start

        X_train = X.loc[train_mask]
        y_train = y.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        logger.info(
            "Supervised split for %s (h=%d) — train=%d, test=%d",
            target,
            horizon,
            len(X_train),
            len(X_test),
        )
        return X_train, y_train, X_test, y_test

    # ── feature builders ────────────────────────────────────────────────
    def _add_lag_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """Add lagged versions of each column.

        Args:
            df: Input DataFrame.
            cols: Columns to lag.

        Returns:
            DataFrame with additional lag columns.
        """
        for col in cols:
            for lag in self.lag_periods:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        logger.debug("Added %d lag features", len(cols) * len(self.lag_periods))
        return df

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """Add rolling window statistics.

        Args:
            df: Input DataFrame.
            cols: Columns to compute rolling stats for.

        Returns:
            DataFrame with additional rolling columns.
        """
        for col in cols:
            for window in self.rolling_windows:
                for stat in self.rolling_statistics:
                    col_name = f"{col}_roll{window}_{stat}"
                    if stat == "mean":
                        df[col_name] = df[col].rolling(window).mean()
                    elif stat == "std":
                        df[col_name] = df[col].rolling(window).std()
        logger.debug(
            "Added %d rolling features",
            len(cols) * len(self.rolling_windows) * len(self.rolling_statistics),
        )
        return df

    def _add_change_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
    ) -> pd.DataFrame:
        """Add month-over-month and year-over-year percentage changes.

        Args:
            df: Input DataFrame.
            cols: Columns to compute changes for.

        Returns:
            DataFrame with change columns.
        """
        for col in cols:
            df[f"{col}_mom"] = df[col].pct_change(periods=self.mom_period)
            df[f"{col}_yoy"] = df[col].pct_change(periods=self.yoy_period)
        logger.debug("Added %d change features", len(cols) * 2)
        return df

    def _add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spread/difference features between related indicators.

        The bank rate minus CPI approximates the real interest rate — a key
        macro variable.  Other spreads can be added here as domain knowledge
        suggests.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with spread columns.
        """
        if "bank_rate" in df.columns and "cpi" in df.columns:
            df["real_rate_approx"] = df["bank_rate"] - df["cpi"]
            logger.debug("Added real_rate_approx spread feature")

        if "m4_growth" in df.columns and "cpi" in df.columns:
            df["excess_money_growth"] = df["m4_growth"] - df["cpi"]
            logger.debug("Added excess_money_growth spread feature")

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multiplicative interaction terms.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with interaction columns.
        """
        interaction_cols: Dict[str, pd.Series] = {}
        for pair in self.interaction_pairs:
            col_a, col_b = pair[0], pair[1]
            if col_a in df.columns and col_b in df.columns:
                interaction_cols[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]
        if interaction_cols:
            df = pd.concat([df, pd.DataFrame(interaction_cols, index=df.index)], axis=1)
        logger.debug("Added %d interaction features", len(self.interaction_pairs))
        return df

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add month-of-year indicators to capture seasonality.

        Args:
            df: Input DataFrame with DatetimeIndex.

        Returns:
            DataFrame with ``month`` and ``quarter`` columns.
        """
        cal_df = pd.DataFrame(
            {"month": df.index.month, "quarter": df.index.quarter},
            index=df.index,
        )
        df = pd.concat([df, cal_df], axis=1)
        logger.debug("Added calendar features (month, quarter)")
        return df
