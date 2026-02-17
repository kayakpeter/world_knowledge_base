"""
IMF DataMapper API fetcher.

Endpoint: https://www.imf.org/external/datamapper/api/v1/{indicator}
Docs: https://www.imf.org/external/datamapper/api/v1/

No API key required. Returns all countries in a single call — we fetch
once per indicator and extract the requested country by IMF numeric code.
The full-indicator call is cached in-process to avoid redundant fetches
when the pipeline processes the same indicator across multiple countries.

Covers 4 registered stats:
  stat_id=2  debt_to_gdp          → GGXWDG_NGDP
  stat_id=3  primary_deficit      → GGXONLB_NGDP
  stat_id=9  public_investment_ratio → GGX_NGDP
  stat_id=10 output_gap           → NGAP_NPGDP
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional  # noqa: F401 (used in _fetch_indicator_all_countries return type)

import httpx

from ingestion.fetchers.base import BaseFetcher, Observation

logger = logging.getLogger(__name__)

IMF_BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

# In-process cache: indicator_code → raw API response dict
# Prevents N×20 calls when the same indicator is fetched for 20 countries
_indicator_cache: dict[str, dict] = {}
_cache_lock: Optional[asyncio.Lock] = None


def _get_cache_lock() -> asyncio.Lock:
    """Lazily create the lock bound to the running event loop."""
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


class ImfFetcher(BaseFetcher):
    """Fetches fiscal/macro indicators from the IMF DataMapper API."""

    provider_name = "imf"

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def health_check(self) -> bool:
        """Ping the IMF DataMapper API."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{IMF_BASE_URL}/GGXWDG_NGDP",
                    params={"periods": "2023"},
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.error("IMF health check failed: %s", exc)
            return False

    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Fetch an IMF DataMapper indicator for a single country.

        The API returns all countries at once; we cache the full response
        per indicator and extract the requested country. The IMF uses its
        own numeric country codes (stored as COUNTRY_CODES[name]["imf"]).

        Args:
            country_iso3: ISO-3 code (e.g., "USA", "DEU")
            indicator:    IMF indicator code (e.g., "GGXWDG_NGDP")
            start_year:   First year to include
            end_year:     Last year to include
        """
        from config.settings import COUNTRY_CODES

        # IMF DataMapper uses ISO-3 codes (e.g., "USA", "DEU") as country keys
        country_name = country_iso3
        for name, codes in COUNTRY_CODES.items():
            if codes["iso3"] == country_iso3:
                country_name = name
                break

        raw = await self._fetch_indicator_all_countries(indicator)
        if raw is None:
            return []

        # Navigate: values → {indicator} → {iso3} → {year: value}
        indicator_data = raw.get("values", {}).get(indicator, {})
        country_data = indicator_data.get(country_iso3, {})

        if not country_data:
            logger.debug(
                "IMF: no data for %s in indicator %s",
                country_iso3, indicator,
            )
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        observations: list[Observation] = []

        for year_str, value in country_data.items():
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year < start_year or year > end_year:
                continue
            if value is None:
                continue
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue

            obs = Observation(
                country=country_name,
                country_iso3=country_iso3,
                stat_name=indicator,
                node_id=f"{country_iso3}_{indicator}_{year_str}",
                category="",  # filled by pipeline
                value=float_value,
                period=year_str,
                unit="percent",
                source=f"IMF DataMapper ({indicator})",
                source_url=f"{IMF_BASE_URL}/{indicator}",
                retrieved_at=now_iso,
            )
            observations.append(obs)

        logger.info(
            "IMF: %s/%s → %d observations",
            country_iso3, indicator, len(observations),
        )
        return observations

    async def _fetch_indicator_all_countries(self, indicator: str) -> Optional[dict]:
        """
        Fetch all-country data for an indicator, using in-process cache.
        Returns the raw API response dict, or None on error.
        """
        lock = _get_cache_lock()
        async with lock:
            if indicator in _indicator_cache:
                return _indicator_cache[indicator]

        # Fetch outside the lock so parallel calls don't serialize
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                url = f"{IMF_BASE_URL}/{indicator}"
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            async with lock:
                _indicator_cache[indicator] = data
            logger.info("IMF: fetched all-country data for %s", indicator)
            return data

        except httpx.HTTPStatusError as exc:
            logger.error(
                "IMF HTTP error for indicator %s: %s",
                indicator, exc.response.status_code,
            )
        except Exception as exc:
            logger.error("IMF fetch failed for indicator %s: %s", indicator, exc)

        return None
