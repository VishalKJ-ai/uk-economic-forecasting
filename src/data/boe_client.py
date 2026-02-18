"""Bank of England Statistical Interactive Database API client.

Fetches monetary and financial time series data from the BoE including the
Bank Rate, M4 money supply growth, mortgage approvals, and consumer credit.

References:
    BoE IADB: https://www.bankofengland.co.uk/boeapps/database/
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class BoEClient:
    """Client for the Bank of England Statistical Interactive Database.

    The BoE IADB exposes CSV downloads parameterised by series code and date
    range.  This client constructs the appropriate query URLs, downloads the
    CSV payloads, and parses them into pandas DataFrames.

    Attributes:
        base_url: Root URL of the BoE IADB CSV endpoint.
        series: Mapping of indicator names to BoE series metadata.
        rate_limit: Minimum seconds between consecutive requests.
        timeout: HTTP request timeout in seconds.
    """

    # The BoE CSV endpoint differs from the browsing UI URL.
    _CSV_ENDPOINT = (
        "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the BoE client from the pipeline configuration.

        Args:
            config: Full pipeline configuration dict.  The
                ``data_sources.boe`` sub-key is used.
        """
        boe_cfg = config["data_sources"]["boe"]
        self.base_url: str = boe_cfg.get("base_url", self._CSV_ENDPOINT)
        self.series: Dict[str, Dict[str, str]] = boe_cfg["series"]
        self.rate_limit: float = boe_cfg.get("rate_limit_seconds", 1.0)
        self.max_retries: int = boe_cfg.get("max_retries", 3)
        self.timeout: int = boe_cfg.get("timeout_seconds", 30)
        self._last_request_time: float = 0.0
        self._session = requests.Session()

        logger.info(
            "BoE client initialised — base_url=%s, series=%s",
            self.base_url,
            list(self.series.keys()),
        )

    # ── public API ──────────────────────────────────────────────────────
    def fetch_all(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch all configured BoE series.

        Args:
            start_date: ISO-format start date.
            end_date: ISO-format end date.

        Returns:
            Dictionary mapping indicator names to DataFrames with columns
            ``["date", <indicator>]``.
        """
        results: Dict[str, pd.DataFrame] = {}
        for name, meta in self.series.items():
            try:
                df = self.fetch_series(name, meta, start_date, end_date)
                if df is not None and not df.empty:
                    results[name] = df
                    logger.info(
                        "BoE | %-25s | %d observations fetched",
                        name,
                        len(df),
                    )
                else:
                    logger.warning("BoE | %-25s | no data returned", name)
            except Exception:
                logger.exception("BoE | %-25s | fetch failed", name)
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
        """Fetch a single BoE time series.

        The BoE IADB returns CSV data when queried with the correct
        parameters.

        Args:
            name: Human-readable indicator name.
            meta: BoE series metadata with key ``series_code``.
            start_date: ISO-format start date.
            end_date: ISO-format end date.

        Returns:
            DataFrame with columns ``["date", <name>]``, or ``None``.
        """
        self._respect_rate_limit()

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        params = {
            "csv.x": "yes",
            "Datefrom": start.strftime("%d/%b/%Y"),
            "Dateto": end.strftime("%d/%b/%Y"),
            "SeriesCodes": meta["series_code"],
            "CSVF": "TN",
            "UsingCodes": "Y",
            "VPD": "Y",
            "VFD": "N",
        }

        logger.debug("BoE request: GET %s params=%s", self.base_url, params)
        response = self._session.get(
            self.base_url, params=params, timeout=self.timeout
        )
        response.raise_for_status()

        return self._parse_csv(name, response.text, meta["series_code"])

    # ── parsing ─────────────────────────────────────────────────────────
    def _parse_csv(
        self,
        name: str,
        csv_text: str,
        series_code: str,
    ) -> Optional[pd.DataFrame]:
        """Parse BoE CSV response into a DataFrame.

        Args:
            name: Indicator name to use as the value column.
            csv_text: Raw CSV text from the BoE response.
            series_code: Expected column header in the CSV.

        Returns:
            Tidy DataFrame or ``None`` if parsing fails.
        """
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            logger.exception("Failed to parse BoE CSV for %s", name)
            return None

        if df.empty:
            return None

        # The BoE CSV typically has columns: DATE, <series_code>
        date_col = [c for c in df.columns if "date" in c.lower()]
        value_col = [c for c in df.columns if series_code in c]

        if not date_col or not value_col:
            # Fall back to positional columns
            if len(df.columns) >= 2:
                date_col = [df.columns[0]]
                value_col = [df.columns[1]]
            else:
                logger.warning(
                    "BoE CSV for %s has unexpected columns: %s",
                    name,
                    list(df.columns),
                )
                return None

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df[date_col[0]], dayfirst=True, errors="coerce")
        result[name] = pd.to_numeric(df[value_col[0]], errors="coerce")
        result = result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return result

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
        logger.debug("BoE client session closed")
