"""
World Bank Open Data API fetcher.

Endpoint: https://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}
Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

Covers approximately 20 of the 50 tracked statistics including:
- GDP growth, debt ratios, current account, FDI, labor participation,
  population growth, NPLs, trade data, energy consumption, and more.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.fetchers.base import BaseFetcher, Observation

logger = logging.getLogger(__name__)

WB_BASE_URL = "https://api.worldbank.org/v2"


class WorldBankFetcher(BaseFetcher):
    """Fetches indicators from the World Bank Open Data API."""

    provider_name = "world_bank"

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def health_check(self) -> bool:
        """Ping the World Bank API."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{WB_BASE_URL}/country/US/indicator/NY.GDP.MKTP.KD.ZG",
                    params={"format": "json", "per_page": 1},
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.error("World Bank health check failed: %s", exc)
            return False

    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Fetch a World Bank indicator for a given country.

        The WB API uses ISO-2 codes in the URL but we accept ISO-3
        and convert internally. If the country code mapping fails,
        we pass ISO-3 directly (WB accepts both in most cases).
        """
        from config.settings import COUNTRY_CODES, COUNTRIES

        # Resolve ISO-2 code for URL
        iso2 = None
        for country_name, codes in COUNTRY_CODES.items():
            if codes["iso3"] == country_iso3:
                iso2 = codes["iso2"]
                break
        country_code = iso2 or country_iso3

        observations: list[Observation] = []

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                url = f"{WB_BASE_URL}/country/{country_code}/indicator/{indicator}"
                params = {
                    "format": "json",
                    "date": f"{start_year}:{end_year}",
                    "per_page": 50,
                }
                resp = await client.get(url, params=params)
                resp.raise_for_status()

                data = resp.json()

                # WB API returns [metadata, records] — records can be None
                if not data or len(data) < 2 or data[1] is None:
                    logger.warning(
                        "No data returned for %s / %s", country_iso3, indicator
                    )
                    return observations

                records = data[1]
                now_iso = datetime.now(timezone.utc).isoformat()

                for record in records:
                    if record.get("value") is None:
                        continue

                    # Resolve the country name for the node_id
                    country_name = country_iso3
                    for name, codes in COUNTRY_CODES.items():
                        if codes["iso3"] == country_iso3:
                            country_name = name
                            break

                    obs = Observation(
                        country=country_name,
                        country_iso3=country_iso3,
                        stat_name=indicator,
                        node_id=f"{country_iso3}_{indicator}_{record['date']}",
                        category="",  # filled by pipeline
                        value=float(record["value"]),
                        period=str(record["date"]),
                        unit=record.get("unit", ""),
                        source=f"World Bank ({indicator})",
                        source_url=url,
                        retrieved_at=now_iso,
                    )
                    observations.append(obs)

                logger.info(
                    "World Bank: %s/%s → %d observations",
                    country_iso3, indicator, len(observations),
                )

        except httpx.HTTPStatusError as exc:
            logger.error(
                "World Bank HTTP error for %s/%s: %s",
                country_iso3, indicator, exc.response.status_code,
            )
        except Exception as exc:
            logger.error(
                "World Bank fetch failed for %s/%s: %s",
                country_iso3, indicator, exc,
            )

        return observations
