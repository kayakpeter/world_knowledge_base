"""
Ingestion pipeline — parallel async data collection across all sources.

Runs all country fetches concurrently per API provider, with per-host
semaphores to respect rate limits. Uses a staleness cache to skip
re-fetching fresh data on daily runs.

Throughput:
  Before: ~8 hours (serial, 30s timeout × 1720 calls)
  After:  ~10-15 minutes (parallel, semaphore-limited, 15s timeout)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import polars as pl

from config.settings import (
    COUNTRIES,
    COUNTRY_CODES,
    STAT_REGISTRY,
    StatDefinition,
    get_api_stats,
    get_derived_stats,
    get_llm_stats,
    DEV_RAW_DIR,
)
from ingestion.fetchers.base import Observation
from ingestion.fetchers.world_bank import WorldBankFetcher
from ingestion.fetchers.fred import FredFetcher

logger = logging.getLogger(__name__)

# Per-host concurrency limits
_WB_CONCURRENCY = 3    # World Bank tolerates ~3 concurrent connections
_FRED_CONCURRENCY = 8  # FRED tolerates ~8 concurrent connections

# How many days before we re-fetch data from each provider type
_STALENESS_DAYS: dict[str, int] = {
    "world_bank": 30,  # Annual data — no point fetching daily
    "fred": 1,         # Monthly/daily series — always check
    "default": 7,
}


class IngestionPipeline:
    """
    Parallel async ingestion for all 20 countries × 50 macro stats.

    Usage:
        pipeline = IngestionPipeline(fred_api_key="your_key")
        observations_df = await pipeline.run()

        # Skip staleness cache — force re-fetch everything:
        observations_df = await pipeline.run(skip_cache=True)
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        output_dir: Optional[Path] = None,
        start_year: int = 2020,
        end_year: int = 2026,
    ):
        # 15s timeout — fail fast and log the error rather than blocking everything
        self._wb = WorldBankFetcher(timeout=15.0)
        self._fred = FredFetcher(api_key=fred_api_key, timeout=15.0)
        self._output_dir = output_dir or DEV_RAW_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._start_year = start_year
        self._end_year = end_year

        # Semaphores created in run() to bind to the correct event loop
        self._wb_sem: Optional[asyncio.Semaphore] = None
        self._fred_sem: Optional[asyncio.Semaphore] = None

        # Thread-safe accumulators — multiple coroutines write concurrently
        self._observations: list[dict] = []
        self._errors: list[dict] = []
        self._lock = asyncio.Lock()

        # Progress counter
        self._completed = 0
        self._total = 0

    async def run(self, skip_cache: bool = False) -> pl.DataFrame:
        """
        Execute the parallel ingestion pipeline.

        Args:
            skip_cache: Force re-fetch all data regardless of staleness.
                        Use True for initial Lambda build; False for daily runs.

        Returns:
            Polars DataFrame with all collected observations.
        """
        self._wb_sem = asyncio.Semaphore(_WB_CONCURRENCY)
        self._fred_sem = asyncio.Semaphore(_FRED_CONCURRENCY)

        staleness_cache = self._load_staleness_cache() if not skip_cache else {}

        api_stats = get_api_stats()
        all_tasks = []
        skipped = 0

        for stat in api_stats:
            for country_name in COUNTRIES:
                codes = COUNTRY_CODES.get(country_name)
                if not codes:
                    continue
                iso3 = codes["iso3"]
                cache_key = f"{iso3}_{stat.name}"

                if not skip_cache and self._is_fresh(cache_key, stat, staleness_cache):
                    skipped += 1
                    continue

                all_tasks.append(
                    asyncio.create_task(
                        self._fetch_with_semaphore(stat, country_name, iso3)
                    )
                )

        self._total = len(all_tasks)

        logger.info("=" * 70)
        logger.info("INGESTION PIPELINE START (parallel async)")
        logger.info(
            "Fetch tasks: %d launched, %d skipped (fresh cache)",
            self._total, skipped,
        )
        logger.info("Concurrency: WB=%d  FRED=%d  Timeout=15s", _WB_CONCURRENCY, _FRED_CONCURRENCY)
        logger.info("=" * 70)

        start_ts = datetime.now(timezone.utc)
        await asyncio.gather(*all_tasks, return_exceptions=True)
        elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()

        logger.info("Fetch phase complete in %.1f seconds", elapsed)

        # Phase 2: Derived stats (real_rates = policy_rate - core_cpi, etc.)
        derived_stats = get_derived_stats()
        logger.info("Computing %d derived stats...", len(derived_stats))
        self._compute_derived_stats(derived_stats)

        # Phase 3: LLM-inferred stats deferred to processing phase
        llm_stats = get_llm_stats()
        logger.info(
            "%d LLM-inferred stats deferred to processing phase",
            len(llm_stats),
        )

        if not self._observations:
            logger.warning("No observations collected — returning empty DataFrame")
            return pl.DataFrame()

        observations_df = pl.DataFrame(self._observations)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = self._output_dir / f"observations_{timestamp}.parquet"
        observations_df.write_parquet(output_path, compression="zstd")

        self._update_staleness_cache(observations_df)

        if self._errors:
            errors_df = pl.DataFrame(self._errors)
            error_path = self._output_dir / f"errors_{timestamp}.parquet"
            errors_df.write_parquet(error_path, compression="zstd")
            logger.warning(
                "%d fetch errors — see %s", len(self._errors), error_path
            )

        logger.info("=" * 70)
        logger.info(
            "INGESTION COMPLETE: %d observations, %d errors in %.1f seconds",
            len(self._observations), len(self._errors), elapsed,
        )
        logger.info("Output: %s", output_path)
        logger.info("=" * 70)

        return observations_df

    async def _fetch_with_semaphore(
        self,
        stat: StatDefinition,
        country_name: str,
        iso3: str,
    ) -> None:
        """Acquire the provider semaphore, fetch, append results under lock."""
        sem = self._fred_sem if stat.api_provider == "fred" else self._wb_sem

        async with sem:
            try:
                observations = await self._dispatch_fetch(stat, country_name, iso3)
                async with self._lock:
                    for obs in observations:
                        self._observations.append({
                            "country": obs.country,
                            "country_iso3": obs.country_iso3,
                            "stat_id": stat.stat_id,
                            "stat_name": stat.name,
                            "category": stat.category,
                            "value": obs.value,
                            "period": obs.period,
                            "unit": stat.unit,
                            "source": obs.source,
                            "source_url": obs.source_url,
                            "retrieved_at": obs.retrieved_at,
                            "node_id": f"{iso3}_{stat.name}",
                        })
                    self._completed += 1
                    if self._completed % 25 == 0 or self._completed == self._total:
                        pct = 100 * self._completed / max(self._total, 1)
                        logger.info(
                            "  Progress: %d/%d (%.0f%%) — %d observations so far",
                            self._completed, self._total, pct, len(self._observations),
                        )

            except Exception as exc:
                async with self._lock:
                    self._errors.append({
                        "country": country_name,
                        "stat_name": stat.name,
                        "provider": stat.api_provider or "unknown",
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    self._completed += 1
                logger.error("FAIL: %s / %s — %s", country_name, stat.name, exc)

    async def _dispatch_fetch(
        self,
        stat: StatDefinition,
        country_name: str,
        iso3: str,
    ) -> list[Observation]:
        """Route a stat fetch to the correct provider."""
        if stat.api_provider == "world_bank" and stat.wb_indicator:
            return await self._wb.fetch_indicator(
                country_iso3=iso3,
                indicator=stat.wb_indicator,
                start_year=self._start_year,
                end_year=self._end_year,
            )

        elif stat.api_provider == "fred" and stat.fred_series:
            series_id = stat.fred_series.get(iso3)
            if not series_id:
                # Many FRED series are US-only — not an error for other countries
                return []
            return await self._fred.fetch_indicator(
                country_iso3=iso3,
                indicator=series_id,
                start_year=self._start_year,
                end_year=self._end_year,
            )

        elif stat.api_provider in ("imf", "oecd", "bis", "eia"):
            # Fetchers not yet implemented — logged at debug level, not error
            logger.debug(
                "Pending fetcher: %s for %s/%s", stat.api_provider, iso3, stat.name
            )
            return []

        return []

    # ─── Staleness Cache ──────────────────────────────────────────────────────

    def _staleness_cache_path(self) -> Path:
        return self._output_dir / "staleness_cache.parquet"

    def _load_staleness_cache(self) -> dict[str, datetime]:
        """Load last-fetched timestamps. Returns {} if cache doesn't exist."""
        path = self._staleness_cache_path()
        if not path.exists():
            return {}
        try:
            df = pl.read_parquet(path)
            return {
                row["cache_key"]: datetime.fromisoformat(row["retrieved_at"])
                for row in df.to_dicts()
            }
        except Exception as exc:
            logger.warning("Could not load staleness cache: %s — will re-fetch all", exc)
            return {}

    def _is_fresh(
        self,
        cache_key: str,
        stat: StatDefinition,
        cache: dict[str, datetime],
    ) -> bool:
        """Return True if this stat is recent enough to skip fetching."""
        if cache_key not in cache:
            return False
        threshold_days = _STALENESS_DAYS.get(
            stat.api_provider or "default",
            _STALENESS_DAYS["default"],
        )
        last_fetched = cache[cache_key]
        # Ensure timezone-aware comparison
        if last_fetched.tzinfo is None:
            last_fetched = last_fetched.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_fetched
        return age < timedelta(days=threshold_days)

    def _update_staleness_cache(self, observations_df: pl.DataFrame) -> None:
        """Write the latest retrieved_at per (country_iso3, stat_name) to cache."""
        if observations_df.is_empty():
            return
        try:
            latest_df = (
                observations_df
                .with_columns(
                    pl.concat_str(["country_iso3", "stat_name"], separator="_")
                    .alias("cache_key")
                )
                .group_by("cache_key")
                .agg(pl.col("retrieved_at").last())
            )
            path = self._staleness_cache_path()
            if path.exists():
                existing = pl.read_parquet(path)
                latest_df = (
                    pl.concat([existing, latest_df])
                    .group_by("cache_key")
                    .agg(pl.col("retrieved_at").last())
                )
            latest_df.write_parquet(path, compression="zstd")
        except Exception as exc:
            logger.warning("Could not update staleness cache: %s", exc)

    # ─── Derived Stats ────────────────────────────────────────────────────────

    def _compute_derived_stats(self, stats: list[StatDefinition]) -> None:
        """Calculate stats that are derived from already-collected observations."""
        if not self._observations:
            logger.warning("No observations to derive from")
            return

        # Build lookup: (iso3, stat_name) → latest value
        latest: dict[tuple[str, str], float] = {}
        for obs in self._observations:
            key = (obs["country_iso3"], obs["stat_name"])
            if key not in latest:
                latest[key] = obs["value"]

        now_iso = datetime.now(timezone.utc).isoformat()

        for stat in stats:
            for country_name in COUNTRIES:
                codes = COUNTRY_CODES.get(country_name)
                if not codes:
                    continue
                iso3 = codes["iso3"]

                value: Optional[float] = None

                if stat.name == "real_rates":
                    policy = latest.get((iso3, "policy_rate"))
                    cpi = latest.get((iso3, "core_cpi"))
                    if policy is not None and cpi is not None:
                        value = policy - cpi

                elif stat.name in (
                    "velocity_of_money",
                    "currency_volatility",
                    "trade_concentration",
                    "refining_margin",
                ):
                    # Require additional data sources not yet ingested
                    logger.debug(
                        "%s derivation deferred — needs FX/trade data", stat.name
                    )
                    continue

                if value is not None:
                    self._observations.append({
                        "country": country_name,
                        "country_iso3": iso3,
                        "stat_id": stat.stat_id,
                        "stat_name": stat.name,
                        "category": stat.category,
                        "value": value,
                        "period": "derived",
                        "unit": stat.unit,
                        "source": f"Derived: {stat.description or stat.name}",
                        "source_url": "",
                        "retrieved_at": now_iso,
                        "node_id": f"{iso3}_{stat.name}",
                    })
