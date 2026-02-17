"""
EIA (U.S. Energy Information Administration) International API fetcher.

Endpoint: https://api.eia.gov/v2/international/data/
Docs: https://www.eia.gov/opendata/browser/international

Free API key required: https://www.eia.gov/opendata/

Covers country-level energy statistics for crude oil production,
natural gas production, and similar energy time series.

EIA country codes largely match ISO-3 (USA, CHN, RUS, SAU, etc.) but
differ for some countries — see _COUNTRY_CODE_MAP overrides below.

Note: No stats currently have api_provider="eia" — this fetcher is
infrastructure for future stats. Wire stat definitions in settings.py:

  StatDefinition(
      stat_id=N, name="crude_production", ...
      api_provider="eia",
      eia_facets={"productId": "53", "activityId": "1"},
  )

EIA product/activity codes (common):
  productId=53  Crude oil and lease condensate
  productId=26  Natural gas
  activityId=1  Production
  activityId=2  Imports
  activityId=3  Exports
  activityId=5  Consumption
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.fetchers.base import BaseFetcher, Observation

logger = logging.getLogger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2/international/data"

# EIA country codes that differ from ISO-3
# Most ISO-3 codes work directly with EIA, these are the exceptions
_COUNTRY_CODE_MAP: dict[str, str] = {
    "GBR": "UKIN",   # United Kingdom
    "KOR": "SKOR",   # South Korea
    "SAU": "SARAB",  # Saudi Arabia
}

# Default facets: crude oil production if not otherwise specified
_DEFAULT_FACETS = {"productId": "53", "activityId": "1"}


class EiaFetcher(BaseFetcher):
    """Fetches country-level energy statistics from the EIA International API."""

    provider_name = "eia"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        self._api_key = api_key or os.environ.get("EIA_API_KEY", "")
        self._timeout = timeout
        if not self._api_key:
            logger.warning(
                "No EIA API key found. Set EIA_API_KEY env var. "
                "Get a free key at https://www.eia.gov/opendata/"
            )

    async def health_check(self) -> bool:
        """Verify EIA API connectivity."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    EIA_BASE_URL,
                    params={
                        "api_key": self._api_key,
                        "frequency": "annual",
                        "data[0]": "value",
                        "facets[countryRegionId][]": "USA",
                        "facets[productId][]": "53",
                        "facets[activityId][]": "1",
                        "length": 1,
                    },
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.error("EIA health check failed: %s", exc)
            return False

    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Fetch an EIA indicator for a single country.

        Args:
            country_iso3: ISO-3 code (e.g., "USA", "SAU")
            indicator:    Encodes facets as "productId:activityId"
                          (e.g., "53:1" for crude oil production)
                          or a stat_name for future registry lookup.
            start_year:   First year to include
            end_year:     Last year to include
        """
        from config.settings import COUNTRY_CODES

        if not self._api_key:
            logger.error("Cannot fetch EIA data without API key")
            return []

        # Resolve EIA country code
        eia_country = _COUNTRY_CODE_MAP.get(country_iso3, country_iso3)

        country_name = country_iso3
        for name, codes in COUNTRY_CODES.items():
            if codes["iso3"] == country_iso3:
                country_name = name
                break

        # Parse indicator string "productId:activityId" or use defaults
        facets = _DEFAULT_FACETS.copy()
        if ":" in indicator:
            parts = indicator.split(":", 1)
            facets["productId"] = parts[0]
            facets["activityId"] = parts[1]

        params = {
            "api_key": self._api_key,
            "frequency": "annual",
            "data[0]": "value",
            "facets[countryRegionId][]": eia_country,
            "facets[productId][]": facets["productId"],
            "facets[activityId][]": facets["activityId"],
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 20,  # max years
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(EIA_BASE_URL, params=params)
                if resp.status_code == 404:
                    logger.debug("EIA: no data for %s/%s (404)", country_iso3, indicator)
                    return []
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as exc:
            logger.error(
                "EIA HTTP error for %s/%s: %s",
                country_iso3, indicator, exc.response.status_code,
            )
            return []
        except Exception as exc:
            logger.error("EIA fetch failed for %s/%s: %s", country_iso3, indicator, exc)
            return []

        # Parse response
        response_data = data.get("response", {})
        records = response_data.get("data", [])
        if not records:
            logger.debug("EIA: no records for %s/%s", country_iso3, indicator)
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        result: list[Observation] = []

        for record in records:
            period_str = str(record.get("period", ""))
            value_raw = record.get("value")
            if value_raw is None:
                continue
            try:
                value = float(value_raw)
                year = int(period_str[:4])
            except (TypeError, ValueError):
                continue

            if year < start_year or year > end_year:
                continue

            unit = record.get("unit", "")
            source_url = (
                f"https://www.eia.gov/opendata/browser/international"
                f"?product={facets['productId']}&activity={facets['activityId']}"
            )

            result.append(Observation(
                country=country_name,
                country_iso3=country_iso3,
                stat_name=indicator,
                node_id=f"{country_iso3}_{indicator}_{period_str}",
                category="",
                value=value,
                period=period_str,
                unit=unit,
                source=f"EIA International ({indicator})",
                source_url=source_url,
                retrieved_at=now_iso,
            ))

        logger.info(
            "EIA: %s/%s → %d observations",
            country_iso3, indicator, len(result),
        )
        return result
