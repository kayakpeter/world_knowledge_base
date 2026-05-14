"""
Scenario Archiver — move expired scenarios out of data/scenarios/active/.

A scenario is expired when its `expires_at` timestamp is in the past.
Scenarios without an `expires_at` are conservatively kept. Unparseable files
are left in place — never deleted, never moved blind.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE = Path(__file__).parent.parent / "data" / "scenarios"
ACTIVE_DIR = _BASE / "active"
ARCHIVE_DIR = _BASE / "archive"


def is_expired(scenario: dict, now: datetime) -> bool:
    """True if the scenario's expires_at is strictly before `now`."""
    expires = scenario.get("expires_at")
    if not expires:
        return False
    try:
        return datetime.fromisoformat(expires) < now
    except ValueError:
        logger.warning("Unparseable expires_at %r — treating as not expired", expires)
        return False


def archive_expired_scenarios(
    active_dir: Path = ACTIVE_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    now: datetime | None = None,
) -> list[str]:
    """
    Move expired scenario JSON files from active_dir to archive_dir.
    Returns the sorted list of archived filenames. Idempotent.
    """
    now = now or datetime.now(timezone.utc)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived: list[str] = []
    for f in sorted(active_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unparseable scenario file %s: %s", f.name, exc)
            continue
        if is_expired(data, now):
            f.rename(archive_dir / f.name)
            archived.append(f.name)
    if archived:
        logger.info("Archived %d expired scenario(s): %s", len(archived), archived)
    return archived
