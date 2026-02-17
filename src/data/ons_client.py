"""ONS (Office for National Statistics) API client.

Fetches macroeconomic time series data from the ONS API including GDP growth,
CPI inflation, unemployment rate, retail sales index, and industrial
production index.  Handles pagination, rate limiting, retries, and
normalisation into tidy pandas DataFrames.

References:
    ONS API docs: https://developer.ons.gov.uk/
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class ONSClient:
    """Client for the Office for National Statistics API.

    Attributes:
        base_url: Root URL of the ONS API.
        datasets: Mapping of indicator names to ONS dataset metadata.
        rate_limit: Minimum seconds between consecutive requests.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the ONS client from the pipeline configuration.

        Args:
            config: The full pipeline configuration dictionary.  The
                ``data_sources.ons`` sub-key is used.
        """
        ons_cfg = config["data_sources"]["ons"]
        self.base_url: str = ons_cfg["base_url"]
        self.datasets: Dict[str, Dict[str, str]] = ons_cfg["datasets"]
        self.rate_limit: float = ons_cfg.get("rate_limit_seconds", 1.0)
        self.max_retries: int = ons_cfg.get("max_retries", 3)
        self.timeout: int = ons_cfg.get("timeout_seconds", 30)
        self._last_request_time: float = 0.0
        self._session = requests.Session()

        logger.info(
            "ONS client initialised — base_url=%s, datasets=%s",
            self.base_url,
            list(self.datasets.keys()),
        )

    # ── public API ──────────────────────────────────────────────────────
    def fetch_all(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all configured ONS datasets.

        Args:
            start_date: ISO-format start date (inclusive).
            end_date: ISO-format end date (inclusive).

        Returns:
            Dictionary mapping indicator names (e.g. ``"gdp"``) to
            DataFrames with columns ``["date", <indicator>]``.
        """
        results: Dict[str, pd.DataFrame] = {}
        for name, meta in self.datasets.items():
            try:
                df = self.fetch_series(name, meta, start_date, end_date)
                if df is not None and not df.empty:
                    results[name] = df
                    logger.info(
                        "ONS | %-25s | %d observations fetched",
                        name,
                        len(df),
                    )
                else:
                    logger.warning("ONS | %-25s | no data returned", name)
            except Exception:
                logger.exception("ONS | %-25s | fetch failed", name)
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def fetch_series(
        self,
        name: str,
        meta: Dict[str, str],
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single ONS time series.

        Args:
            name: Human-readable indicator name (e.g. ``"gdp"``).
            meta: ONS dataset metadata dict with keys ``dataset_id``,
                ``edition``, ``series_id``, ``frequency``.
            start_date: ISO-format start date (inclusive).
            end_date: ISO-format end date (inclusive).

        Returns:
            DataFrame with columns ``["date", <name>]``, or ``None`` on
            failure.
        """
        self._respect_rate_limit()
        url = (
            f"{self.base_url}/timeseries/{meta['series_id']}/dataset/"
            f"{meta['dataset_id']}/data"
        )
        logger.debug("ONS request: GET %s", url)

        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return self._parse_response(name, data, meta["frequency"])

    # ── parsing helpers ─────────────────────────────────────────────────
    def _parse_response(
        self,
        name: str,
        data: Dict[str, Any],
        frequency: str,
    ) -> Optional[pd.DataFrame]:
        """Parse the ONS JSON response into a DataFrame.

        Args:
            name: Indicator name used as the value column header.
            data: Raw JSON dict from the ONS API.
            frequency: ``"monthly"`` or ``"quarterly"``.

        Returns:
            Tidy DataFrame or ``None`` if no observations were parsed.
        """
        key = "months" if frequency == "monthly" else "quarters"
        observations: List[Dict[str, Any]] = data.get(key, [])
        if not observations:
            logger.warning("No '%s' key found in ONS response for %s", key, name)
            return None

        records = []
        for obs in observations:
            try:
                date_str = obs.get("date", "")
                value_str = obs.get("value", "")
                if date_str and value_str:
                    records.append(
                        {"date": pd.to_datetime(date_str), name: float(value_str)}
                    )
            except (ValueError, TypeError):
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ── rate-limit helper ───────────────────────────────────────────────
    def _respect_rate_limit(self) -> None:
        """Sleep if necessary to honour the configured rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()
        logger.debug("ONS client session closed")
