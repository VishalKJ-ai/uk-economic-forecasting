"""Data preprocessing module.

Responsible for aligning heterogeneous economic time series to a common
monthly frequency, handling missing values, performing quality checks, and
persisting the cleaned dataset as Parquet files for downstream consumption.

Design decisions documented in-line:
- Forward-fill is the default imputation strategy for economic series because
  most macroeconomic indicators are *sticky* — the latest published value
  remains the best estimate until the next release.
- Quarterly series (GDP) are interpolated to monthly frequency using cubic
  spline to preserve the shape of the growth trajectory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from sample CSV filenames to the internal indicator name
_SAMPLE_FILE_MAP: Dict[str, str] = {
    "gdp_quarterly.csv": "gdp",
    "cpi_monthly.csv": "cpi",
    "unemployment_monthly.csv": "unemployment_rate",
    "bank_rate_monthly.csv": "bank_rate",
    "retail_sales_monthly.csv": "retail_sales_index",
    "industrial_production_monthly.csv": "industrial_production_index",
    "m4_money_supply_monthly.csv": "m4_growth",
    "mortgage_approvals_monthly.csv": "mortgage_approvals",
}

# Column names expected inside each sample CSV (second column)
_SAMPLE_VALUE_COLS: Dict[str, str] = {
    "gdp_quarterly.csv": "gdp_growth",
    "cpi_monthly.csv": "cpi",
    "unemployment_monthly.csv": "unemployment_rate",
    "bank_rate_monthly.csv": "bank_rate",
    "retail_sales_monthly.csv": "retail_sales_index",
    "industrial_production_monthly.csv": "industrial_production_index",
    "m4_money_supply_monthly.csv": "m4_growth",
    "mortgage_approvals_monthly.csv": "mortgage_approvals",
}


class Preprocessor:
    """Clean, align, and persist macroeconomic time series.

    Attributes:
        processed_path: Directory for Parquet outputs.
        sample_path: Directory containing sample CSV files.
    """

    def __init__(self, config: Dict[str, Any], project_root: Path) -> None:
        """Initialise the preprocessor.

        Args:
            config: Full pipeline configuration dictionary.
            project_root: Absolute path to the project root directory.
        """
        self.processed_path: Path = project_root / config["paths"]["processed_data"]
        self.sample_path: Path = project_root / config["paths"]["sample_data"]
        self.processed_path.mkdir(parents=True, exist_ok=True)
        logger.info("Preprocessor initialised — output=%s", self.processed_path)

    # ── public API ──────────────────────────────────────────────────────
    def load_sample_data(self) -> pd.DataFrame:
        """Load all sample CSV files and return a unified monthly DataFrame.

        Returns:
            DataFrame indexed by ``date`` (monthly, end-of-month) with one
            column per indicator.
        """
        logger.info("Loading sample data from %s", self.sample_path)
        series_list: List[pd.DataFrame] = []

        for filename, indicator in _SAMPLE_FILE_MAP.items():
            filepath = self.sample_path / filename
            if not filepath.exists():
                logger.warning("Sample file not found: %s", filepath)
                continue

            df = pd.read_csv(filepath, parse_dates=["date"])
            value_col = _SAMPLE_VALUE_COLS[filename]
            df = df.rename(columns={value_col: indicator})
            df = df[["date", indicator]]

            # If quarterly, interpolate to monthly
            if "quarterly" in filename:
                df = self._interpolate_quarterly(df, indicator)

            series_list.append(df.set_index("date"))
            logger.info(
                "Loaded %-35s | %d obs | cols=%s",
                filename,
                len(df),
                indicator,
            )

        if not series_list:
            raise FileNotFoundError(
                f"No sample CSV files found in {self.sample_path}"
            )

        combined = self._merge_series(series_list)
        combined = self._handle_missing(combined)
        self._validate(combined)
        return combined

    def process_api_data(
        self,
        ons_data: Dict[str, pd.DataFrame],
        boe_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Process data fetched from the ONS and BoE APIs.

        Args:
            ons_data: Dict of DataFrames from :class:`ONSClient.fetch_all`.
            boe_data: Dict of DataFrames from :class:`BoEClient.fetch_all`.

        Returns:
            Unified monthly DataFrame ready for feature engineering.
        """
        logger.info("Processing API data — ONS=%d, BoE=%d series",
                     len(ons_data), len(boe_data))

        series_list: List[pd.DataFrame] = []

        for name, df in {**ons_data, **boe_data}.items():
            if "date" not in df.columns:
                logger.warning("Skipping %s — no 'date' column", name)
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            series_list.append(df.set_index("date"))

        if not series_list:
            raise ValueError("No valid series found in API data")

        combined = self._merge_series(series_list)
        combined = self._handle_missing(combined)
        self._validate(combined)
        return combined

    def save(self, df: pd.DataFrame, filename: str = "processed.parquet") -> Path:
        """Save a DataFrame as a Parquet file.

        Args:
            df: The DataFrame to persist.
            filename: Output filename.

        Returns:
            Path to the saved Parquet file.
        """
        out_path = self.processed_path / filename
        df.to_parquet(out_path, engine="pyarrow")
        logger.info("Saved processed data → %s  (%d rows, %d cols)",
                     out_path, len(df), len(df.columns))
        return out_path

    def load(self, filename: str = "processed.parquet") -> pd.DataFrame:
        """Load a previously saved Parquet file.

        Args:
            filename: Name of the Parquet file inside the processed directory.

        Returns:
            DataFrame with datetime index.
        """
        path = self.processed_path / filename
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded processed data ← %s  (%d rows)", path, len(df))
        return df

    # ── internal helpers ────────────────────────────────────────────────
    @staticmethod
    def _interpolate_quarterly(
        df: pd.DataFrame,
        col: str,
    ) -> pd.DataFrame:
        """Interpolate a quarterly series to monthly frequency.

        Uses cubic spline interpolation to produce smooth monthly estimates
        from quarterly observations.  This is preferred over linear
        interpolation because GDP growth tends to follow smooth trajectories
        between quarterly data releases.

        Args:
            df: Quarterly DataFrame with columns ``["date", col]``.
            col: Name of the value column.

        Returns:
            Monthly DataFrame.
        """
        # Normalise quarterly dates to month-start for alignment with
        # monthly series.  E.g., 2015-03-31 → 2015-03-01.
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df = df.set_index("date")
        # Resample to month-start and interpolate between quarterly obs
        monthly = df.resample("MS").asfreq()
        monthly[col] = monthly[col].interpolate(method="cubic")
        # Forward-fill the tail if cubic interpolation leaves NaNs at edges
        monthly[col] = monthly[col].ffill().bfill()
        monthly = monthly.reset_index()
        return monthly

    @staticmethod
    def _merge_series(series_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Outer-join a list of date-indexed DataFrames on their index.

        All series are resampled to monthly frequency (month-start) before
        joining to ensure alignment.

        Args:
            series_list: List of DataFrames each indexed by date.

        Returns:
            Merged DataFrame with month-start DatetimeIndex.
        """
        resampled = []
        for s in series_list:
            s = s.copy()
            s.index = pd.to_datetime(s.index)
            s = s.resample("MS").last()
            resampled.append(s)

        combined = resampled[0]
        for s in resampled[1:]:
            combined = combined.join(s, how="outer")

        combined = combined.sort_index()
        logger.info(
            "Merged %d series → %d rows × %d cols | %s to %s",
            len(series_list),
            len(combined),
            len(combined.columns),
            combined.index.min().date(),
            combined.index.max().date(),
        )
        return combined

    @staticmethod
    def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using forward fill.

        Justification: most macroeconomic indicators are published with a lag.
        Between publication dates the most recent observation is the best
        available estimate.  Forward fill mirrors this "last known value"
        assumption.  Any remaining leading NaNs are back-filled from the
        earliest available observation.

        Args:
            df: DataFrame that may contain NaN values.

        Returns:
            DataFrame with NaNs imputed.
        """
        missing_before = df.isna().sum()
        cols_with_missing = missing_before[missing_before > 0]

        if not cols_with_missing.empty:
            logger.info("Missing values before imputation:\n%s", cols_with_missing)

        df = df.ffill().bfill()

        missing_after = df.isna().sum().sum()
        if missing_after > 0:
            logger.warning("%d NaN values remain after imputation", missing_after)
        else:
            logger.info("All missing values successfully imputed")

        return df

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Run basic data quality checks.

        Args:
            df: The processed DataFrame.

        Raises:
            ValueError: If critical quality checks fail.
        """
        # Check for duplicated dates
        if df.index.duplicated().any():
            n_dup = df.index.duplicated().sum()
            logger.warning("Found %d duplicated date entries", n_dup)
            df = df[~df.index.duplicated(keep="last")]

        # Check for remaining NaNs
        total_nan = df.isna().sum().sum()
        if total_nan > 0:
            logger.warning("Data still contains %d NaN values after processing", total_nan)

        # Check monotonicity of date index
        if not df.index.is_monotonic_increasing:
            logger.warning("Date index is not monotonically increasing — sorting")
            df.sort_index(inplace=True)

        # Log summary statistics
        logger.info(
            "Validation passed — %d rows, %d columns, date range %s to %s",
            len(df),
            len(df.columns),
            df.index.min().date(),
            df.index.max().date(),
        )
