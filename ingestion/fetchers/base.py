"""
Base fetcher interface for all data source providers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """A single data point retrieved from an external source."""
    country: str
    country_iso3: str
    stat_name: str
    node_id: str
    category: str
    value: float
    period: str          # e.g. "2025" or "2025-Q4" or "2025-12"
    unit: str
    source: str
    source_url: str
    retrieved_at: str    # ISO timestamp


class BaseFetcher(ABC):
    """Abstract base for all data source fetchers."""

    provider_name: str = "base"

    @abstractmethod
    async def fetch_indicator(
        self,
        country_iso3: str,
        indicator: str,
        start_year: int = 2020,
        end_year: int = 2026,
    ) -> list[Observation]:
        """
        Fetch a single indicator for a single country.
        Returns a list of Observations (one per time period).
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the API is reachable and responding."""
        ...
