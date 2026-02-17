"""
FRED (Federal Reserve Economic Data) API fetcher.

Endpoint: https://api.stlouisfed.org/fred/series/observations
Docs: https://fred.stlouisfed.org/docs/api/fred/

Requires an API key (free registration at https://fred.stlouisfed.org/docs/api/api_key.html).
Covers US-specific stats and some international series that FRED mirrors.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.fetchers.base import BaseFetcher, Observation

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"


class FredFetcher(BaseFetcher):
    """Fetches time series from the FRED API."""

    provider_name = "fred"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self._api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._timeout = timeout
        if not self._api_key:
            logger.warning(
                "No FRED API key found. Set FRED_API_KEY env var or pass api_key. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    async def health_check(self) -> bool:
        """Verify FRED API connectivity."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{FRED_BASE_URL}/series",
                    params={
                        "series_id": "GDP",
                        "api_key": self._api_key,
                        "file_type": "json",
                    },
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.error("FRED health check failed: %s", exc)
            return False

    async def fetch_series(
        self,
        series_id: str,
        country_iso3: str = "USA",
        start_date: str = "2020-01-01",
        end_date: str = "2026-12-31",
    ) -> list[Observation]:
        """
        Fetch a specific FRED series.

        Args:
            series_id: FRED series identifier (e.g., "CPILFESL", "DFEDTARU")
            country_iso3: ISO-3 code of the country this series represents
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        from config.settings import COUNTRY_CODES

        if not self._api_key:
            logger.error("Cannot fetch FRED data without API key")
            return []

        observations: list[Observation] = []

        # Resolve country name
        country_name = country_iso3
        for name, codes in COUNTRY_CODES.items():
            if codes["iso3"] == country_iso3:
                country_name = name
                break

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                url = f"{FRED_BASE_URL}/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                    "observation_start": start_date,
                    "observation_end": end_date,
                    "sort_order": "desc",
                }
                resp = await client.get(url, params=params)
                resp.raise_for_status()

                data = resp.json()
                now_iso = datetime.now(timezone.utc).isoformat()

                for record in data.get("observations", []):
                    value_str = record.get("value", ".")
                    if value_str == "." or value_str is None:
                        continue

                    try:
                        value = float(value_str)
                    except ValueError:
                        continue

                    obs = Observation(
                        country=country_name,
                        country_iso3=country_iso3,
                        stat_name=series_id,
                        node_id=f"{country_iso3}_FRED_{series_id}_{record['date']}",
                        category="",  # filled by pipeline
                        value=value,
                        period=record["date"],
                        unit="",  # FRED doesn't consistently provide units
                        source=f"FRED ({series_id})",
                        source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                        retrieved_at=now_iso,
                    )
                    observations.append(obs)

                logger.info(
                    "FRED: %s for %s → %d observations",
                    series_id, country_iso3, len(observations),
                )

        except httpx.HTTPStatusError as exc:
            logger.error(
                "FRED HTTP error for %s: %s", series_id, exc.response.status_code
            )
        except Exception as exc:
            logger.error("FRED fetch failed for %s: %s", series_id, exc)

        return observations

    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Adapter to match the BaseFetcher interface.
        For FRED, 'indicator' is the series_id.
        """
        return await self.fetch_series(
            series_id=indicator,
            country_iso3=country_iso3,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31",
        )
