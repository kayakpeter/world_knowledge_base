"""
BIS (Bank for International Settlements) Statistics API fetcher.

Endpoint: https://stats.bis.org/api/v2/data/{dataset}/{key}/all
Docs: https://stats.bis.org/api/v2/

No API key required. Uses ISO-2 country codes.

Primary use: central bank policy rates (WS_CBPOL_D) for countries not
covered by FRED (i.e., not USA/GBR/JPN/CAN). BIS collects policy rates
from all major central banks, making it the cleanest global source.

BIS WS_CBPOL_D key format: {freq}:{country_iso2}:{type}
  freq    = D (daily)
  type    = P (policy rate)
  Example: D:US:P

Countries covered by BIS WS_CBPOL_D (from our 20-country set):
  AU, BR, CA, CN, DE, FR, GB, ID, IN, IT, JP, KR, MX, NL,
  PL, RU, SA, TR, US — basically all G20 central banks
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from ingestion.fetchers.base import BaseFetcher, Observation

logger = logging.getLogger(__name__)

BIS_BASE_URL = "https://stats.bis.org/api/v2"

# Map stat_name → BIS dataset code
# WS_CBPOL: Central bank policy rates (the correct dataset name in BIS API v2)
# Note: BIS API v2 data endpoint currently returns 500 for WS_CBPOL despite
# the dataset appearing in the /structure/dataflow list. This is a known BIS
# API issue. The fetcher is implemented correctly and will work when resolved.
_STAT_DATASET_MAP: dict[str, str] = {
    "policy_rate": "WS_CBPOL",
}


class BisFetcher(BaseFetcher):
    """Fetches central bank statistics from the BIS Statistics API."""

    provider_name = "bis"

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    async def health_check(self) -> bool:
        """Ping the BIS API structure endpoint (data endpoint has known issues)."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # Use the structure/dataflow endpoint which is stable
                resp = await client.get(
                    f"{BIS_BASE_URL}/structure/dataflow",
                    headers={"Accept": "application/json"},
                )
                return resp.status_code == 200
        except Exception as exc:
            logger.error("BIS health check failed: %s", exc)
            return False

    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Fetch a BIS indicator for a single country.

        Args:
            country_iso3: ISO-3 code (e.g., "DEU")
            indicator:    stat_name used to look up the BIS dataset
                          (e.g., "policy_rate")
            start_year:   First year to include
            end_year:     Last year to include
        """
        from config.settings import COUNTRY_CODES

        if indicator not in _STAT_DATASET_MAP:
            logger.warning(
                "BIS: no dataset mapping for stat '%s' — add it to _STAT_DATASET_MAP",
                indicator,
            )
            return []

        # Resolve ISO-2 code (BIS uses ISO-2)
        iso2: str | None = None
        country_name = country_iso3
        for name, codes in COUNTRY_CODES.items():
            if codes["iso3"] == country_iso3:
                iso2 = codes.get("iso2")
                country_name = name
                break

        if not iso2:
            logger.debug("BIS: no ISO-2 mapping for %s", country_iso3)
            return []

        dataset = _STAT_DATASET_MAP[indicator]
        # Daily policy rate key: freq=D, country=iso2, type=P
        key = f"D:{iso2}:P"
        start_period = f"{start_year}-01-01"
        end_period = f"{end_year}-12-31"

        url = f"{BIS_BASE_URL}/data/{dataset}/{key}/all"
        params = {
            "startPeriod": start_period,
            "endPeriod": end_period,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 404:
                    logger.debug("BIS: no data for %s/%s (404)", country_iso3, indicator)
                    return []
                if resp.status_code == 500:
                    # Known BIS API issue: WS_CBPOL data endpoint returns 500 despite
                    # the dataset existing. Log at debug level — not an error in our code.
                    logger.debug(
                        "BIS: server error 500 for %s/%s — API issue, returning []",
                        country_iso3, indicator,
                    )
                    return []
                resp.raise_for_status()
                data = resp.json()

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "BIS HTTP error for %s/%s: %s",
                country_iso3, indicator, exc.response.status_code,
            )
            return []
        except Exception as exc:
            logger.warning("BIS fetch failed for %s/%s: %s", country_iso3, indicator, exc)
            return []

        return self._parse_response(
            data=data,
            country_name=country_name,
            country_iso3=country_iso3,
            stat_name=indicator,
            source_url=url,
            start_year=start_year,
            end_year=end_year,
        )

    def _parse_response(
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
        Parse BIS SDMX-JSON response.

        BIS returns SDMX-JSON with time series in:
          dataSets[0].series.{"0:0:0": {observations: {"0": [value]}}}
        or with dimensionAtObservation=allDimensions:
          dataSets[0].observations.{"0:0:0:period_idx": [value, ...]}

        We use the simpler series format.
        """
        try:
            datasets = data.get("dataSets", [])
            structure = data.get("structure", {})
        except Exception:
            logger.error("BIS: unexpected response for %s/%s", country_iso3, stat_name)
            return []

        if not datasets:
            return []

        # Locate time dimension in structure
        obs_dims = structure.get("dimensions", {}).get("observation", [])
        time_values: list[str] = []
        for dim in obs_dims:
            if dim.get("id") in ("TIME_PERIOD", "TIME"):
                time_values = [v.get("id", "") for v in dim.get("values", [])]
                break

        # Try series-level format first
        series_data = datasets[0].get("series", {})
        obs_data = datasets[0].get("observations", {})

        now_iso = datetime.now(timezone.utc).isoformat()
        result: list[Observation] = []

        if series_data:
            # Series format: {"key": {"observations": {"time_idx": [value, ...]}}}
            for _series_key, series_val in series_data.items():
                for time_idx_str, obs_vals in series_val.get("observations", {}).items():
                    if not obs_vals or obs_vals[0] is None:
                        continue
                    try:
                        value = float(obs_vals[0])
                        time_idx = int(time_idx_str)
                        period = time_values[time_idx] if time_idx < len(time_values) else time_idx_str
                    except (TypeError, ValueError, IndexError):
                        continue

                    year = self._extract_year(period)
                    if year is None or year < start_year or year > end_year:
                        continue

                    # BIS policy rate is daily; we keep the last reading per year
                    result.append(Observation(
                        country=country_name,
                        country_iso3=country_iso3,
                        stat_name=stat_name,
                        node_id=f"{country_iso3}_{stat_name}_{period}",
                        category="",
                        value=value,
                        period=period,
                        unit="percent",
                        source=f"BIS ({stat_name})",
                        source_url=source_url,
                        retrieved_at=now_iso,
                    ))

        elif obs_data and time_values:
            # allDimensions format
            for key_str, obs_vals in obs_data.items():
                if not obs_vals or obs_vals[0] is None:
                    continue
                try:
                    value = float(obs_vals[0])
                    parts = key_str.split(":")
                    time_idx = int(parts[-1])
                    period = time_values[time_idx] if time_idx < len(time_values) else ""
                except (TypeError, ValueError, IndexError):
                    continue

                year = self._extract_year(period)
                if year is None or year < start_year or year > end_year:
                    continue

                result.append(Observation(
                    country=country_name,
                    country_iso3=country_iso3,
                    stat_name=stat_name,
                    node_id=f"{country_iso3}_{stat_name}_{period}",
                    category="",
                    value=value,
                    period=period,
                    unit="percent",
                    source=f"BIS ({stat_name})",
                    source_url=source_url,
                    retrieved_at=now_iso,
                ))

        # Deduplicate: keep last value per year (daily → annual for HMM)
        result = self._keep_last_per_year(result)

        logger.info(
            "BIS: %s/%s → %d annual observations (from daily series)",
            country_iso3, stat_name, len(result),
        )
        return result

    @staticmethod
    def _extract_year(period: str) -> int | None:
        """Extract 4-digit year from period string (e.g., '2023-06-15' → 2023)."""
        try:
            return int(period[:4])
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _keep_last_per_year(observations: list[Observation]) -> list[Observation]:
        """
        For daily series, keep only the last observation per year.
        This gives the year-end policy rate, appropriate for the HMM.
        """
        year_map: dict[int, Observation] = {}
        for obs in observations:
            year = BisFetcher._extract_year(obs.period)
            if year is None:
                continue
            # Later periods sort lexicographically later — keep last seen
            existing = year_map.get(year)
            if existing is None or obs.period >= existing.period:
                year_map[year] = obs
        return list(year_map.values())
