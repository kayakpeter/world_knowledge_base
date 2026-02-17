"""
OECD SDMX-JSON API fetcher.

Primary endpoint: https://stats.oecd.org/sdmx-json/data/{dataset}/{key}/all
  → Redirects to sdmx.oecd.org (OECD migrated their API in 2024/2025)
New endpoint pattern:
  SOCX_AGG → OECD.ELS.SPD,DSD_SOCX_AGG@DF_SOCX_AGG
  REV       → migrated to World Bank (GC.TAX.TOTL.GD.ZS) in this codebase

Note: The OECD API migration from stats.oecd.org to sdmx.oecd.org changed
the dataflow IDs and key structures. The old API dropped dimension filters
on redirect, causing 422 errors. This fetcher handles the known-working
datasets and logs gracefully when the API is unreachable.

No API key required. OECD data is only available for member countries.
Non-members in our 20-country set (CHN, IND, RUS, BRA, IDN, SAU) will
return empty — this is expected.

Currently active:
  stat_id=7  entitlement_spend  → SOCX_AGG via direct new-API endpoint
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from ingestion.fetchers.base import BaseFetcher, Observation

logger = logging.getLogger(__name__)

OECD_OLD_BASE_URL = "https://stats.oecd.org/sdmx-json/data"
OECD_NEW_BASE_URL = "https://sdmx.oecd.org/public/rest/data"

# Map stat_name → (new_agency_dataflow, old_dataset, old_key_template)
# OECD migrated from stats.oecd.org to sdmx.oecd.org in 2024-2025.
# We try the new endpoint first, fall back to old (which may redirect and fail).
_STAT_DATASET_MAP: dict[str, tuple[str, str, str]] = {
    "entitlement_spend": (
        # New API: agency.dataflow — SOCX dataset
        "OECD.ELS.SPD,DSD_SOCX_AGG@DF_SOCX_AGG",
        # Old API: dataset name and key template
        "SOCX_AGG",
        "TOTAL.PUBLIC.PCTGDP.TOTAL.{country}",
    ),
    # tax_to_gdp moved to World Bank in settings.py — kept here as reference
    "tax_to_gdp": (
        "OECD,DF_REV",
        "REV",
        "1000.{country}.1000",
    ),
}

# OECD member countries from our 20-country set (ISO-3 codes)
# Non-members will get no data from this API
_OECD_MEMBERS = {
    "USA", "DEU", "JPN", "GBR", "FRA", "ITA", "CAN",
    "ESP", "MEX", "AUS", "KOR", "TUR", "NLD", "POL",
}


class OecdFetcher(BaseFetcher):
    """Fetches social/fiscal indicators from the OECD SDMX-JSON API."""

    provider_name = "oecd"

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def health_check(self) -> bool:
        """Ping the OECD API with a small request."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(
                    f"{OECD_OLD_BASE_URL}/REV/1000.USA.1000/all",
                    params={"startTime": "2022", "endTime": "2022",
                            "dimensionAtObservation": "allDimensions",
                            "format": "json"},
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.error("OECD health check failed: %s", exc)
            return False

    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Fetch an OECD indicator for a single country.

        Args:
            country_iso3: ISO-3 code (e.g., "USA")
            indicator:    stat_name used to look up dataset/key mapping
                          (e.g., "entitlement_spend", "tax_to_gdp")
            start_year:   First year to include
            end_year:     Last year to include
        """
        from config.settings import COUNTRY_CODES

        # Non-OECD members have no data
        if country_iso3 not in _OECD_MEMBERS:
            logger.debug(
                "OECD: %s is not an OECD member — skipping %s",
                country_iso3, indicator,
            )
            return []

        if indicator not in _STAT_DATASET_MAP:
            logger.warning(
                "OECD: no dataset mapping for stat '%s' — add it to _STAT_DATASET_MAP",
                indicator,
            )
            return []

        new_flow, old_dataset, old_key_template = _STAT_DATASET_MAP[indicator]
        old_key = old_key_template.format(country=country_iso3)

        country_name = country_iso3
        for name, codes in COUNTRY_CODES.items():
            if codes["iso3"] == country_iso3:
                country_name = name
                break

        # Try new API endpoint first (sdmx.oecd.org), then fall back to old
        attempts = [
            (f"{OECD_NEW_BASE_URL}/{new_flow}/{country_iso3}",
             {"startPeriod": str(start_year), "endPeriod": str(end_year),
              "dimensionAtObservation": "allDimensions", "format": "jsondata"}),
            (f"{OECD_OLD_BASE_URL}/{old_dataset}/{old_key}/all",
             {"startTime": str(start_year), "endTime": str(end_year),
              "dimensionAtObservation": "allDimensions", "format": "json"}),
        ]

        data = None
        source_url = attempts[0][0]

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=False) as client:
            for url, params in attempts:
                try:
                    resp = await client.get(url, params=params)
                    if resp.status_code in (301, 302):
                        # Redirect drops key — the old API is fully deprecated
                        logger.debug("OECD: %s redirected — API migrated, skipping", url)
                        continue
                    if resp.status_code == 404:
                        logger.debug("OECD: 404 for %s/%s", country_iso3, indicator)
                        return []
                    if resp.status_code in (422, 500, 501):
                        logger.debug("OECD: %s for %s — API error, skipping",
                                     resp.status_code, indicator)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    source_url = url
                    break

                except httpx.HTTPStatusError as exc:
                    logger.debug("OECD HTTP %s for %s/%s",
                                 exc.response.status_code, country_iso3, indicator)
                except Exception as exc:
                    logger.debug("OECD fetch error for %s/%s: %s", country_iso3, indicator, exc)

        if data is None:
            logger.info(
                "OECD: no data available for %s/%s (API migration in progress)",
                country_iso3, indicator,
            )
            return []

        return self._parse_sdmx_json(
            data=data,
            country_name=country_name,
            country_iso3=country_iso3,
            stat_name=indicator,
            source_url=source_url,
            start_year=start_year,
            end_year=end_year,
        )

    def _parse_sdmx_json(
        self,
        data: dict,
        country_name: str,
        country_iso3: str,
        stat_name: str,
        source_url: str,
        start_year: int,
        end_year: int,
    ) -> list[Observation]:
        """
        Parse OECD SDMX-JSON response into Observation list.

        SDMX-JSON layout (dimensionAtObservation=allDimensions):
          dataSets[0].observations: {"0:1:2:...": [value, status]}
          structure.dimensions.observation[]: [{id, values: [{id, name}]}]
        """
        try:
            datasets = data.get("dataSets", [])
            structure = data.get("structure", {})
        except Exception:
            logger.error("OECD: unexpected response structure for %s", stat_name)
            return []

        if not datasets:
            logger.debug("OECD: empty dataSets for %s/%s", country_iso3, stat_name)
            return []

        observations_raw = datasets[0].get("observations", {})
        if not observations_raw:
            return []

        # Build time dimension index: position → year string
        obs_dims = structure.get("dimensions", {}).get("observation", [])
        time_dim_index = -1
        time_values: list[str] = []
        for i, dim in enumerate(obs_dims):
            if dim.get("id") in ("TIME_PERIOD", "YEAR", "TIME"):
                time_dim_index = i
                time_values = [v.get("id", "") for v in dim.get("values", [])]
                break

        if time_dim_index == -1 or not time_values:
            logger.warning("OECD: could not locate time dimension for %s", stat_name)
            return []

        now_iso = datetime.now(timezone.utc).isoformat()
        result: list[Observation] = []

        for key_str, obs_list in observations_raw.items():
            if not obs_list or obs_list[0] is None:
                continue
            try:
                value = float(obs_list[0])
            except (TypeError, ValueError):
                continue

            # Extract time position from key (e.g., "0:0:1:2:3")
            parts = key_str.split(":")
            if time_dim_index >= len(parts):
                continue
            try:
                time_idx = int(parts[time_dim_index])
                year_str = time_values[time_idx]
            except (ValueError, IndexError):
                continue

            try:
                year = int(year_str[:4])  # handles "2023" or "2023-01-01"
            except ValueError:
                continue
            if year < start_year or year > end_year:
                continue

            result.append(Observation(
                country=country_name,
                country_iso3=country_iso3,
                stat_name=stat_name,
                node_id=f"{country_iso3}_{stat_name}_{year_str}",
                category="",  # filled by pipeline
                value=value,
                period=str(year),
                unit="percent_gdp",
                source=f"OECD ({stat_name})",
                source_url=source_url,
                retrieved_at=now_iso,
            ))

        logger.info(
            "OECD: %s/%s → %d observations",
            country_iso3, stat_name, len(result),
        )
        return result
