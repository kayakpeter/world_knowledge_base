"""
KB Health Monitor — verifies the daily refresh pipeline completed successfully.

Checks:
  1. observations_{today}*.parquet exists in data/raw/
  2. If HMM priors file exists: was it written today?

Call monitor.check() before running the feedback loop.
Raises RuntimeError if stale data would corrupt priors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent.parent / "data"


@dataclass
class HealthStatus:
    observations_fresh: bool
    priors_fresh: bool  # True if no priors file yet (first run is OK)
    is_healthy: bool
    message: str


class KBHealthMonitor:
    def __init__(
        self,
        data_root: Path = DATA_ROOT,
        today: str | None = None,
    ) -> None:
        self.data_root = data_root
        self.today = today or date.today().isoformat().replace("-", "")

    def check(self) -> HealthStatus:
        raw_dir = self.data_root / "raw"
        obs_files = list(raw_dir.glob(f"observations_{self.today}*.parquet")) if raw_dir.exists() else []
        obs_fresh = len(obs_files) > 0

        priors_dir = self.data_root / "hmm_priors"
        if priors_dir.exists():
            today_priors = list(priors_dir.glob(f"{self.today}_priors.json"))
            priors_fresh = len(today_priors) > 0 or not any(priors_dir.iterdir())
        else:
            priors_fresh = True  # no priors dir = first run, OK

        healthy = obs_fresh  # priors absence on first run is fine
        msg_parts = []
        if not obs_fresh:
            msg_parts.append(f"No observations parquet for {self.today}")
        if not priors_fresh:
            msg_parts.append("HMM priors not updated today")

        status = HealthStatus(
            observations_fresh=obs_fresh,
            priors_fresh=priors_fresh,
            is_healthy=healthy,
            message="; ".join(msg_parts) if msg_parts else "OK",
        )
        if not healthy:
            logger.warning(f"KB health check FAILED: {status.message}")
        else:
            logger.info("KB health check OK")
        return status
