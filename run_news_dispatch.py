"""
run_news_dispatch.py — Dispatch pending news parquets to agent inboxes.

Called from run_daily_ingestion.sh immediately after run_news_ingestion.py,
or independently for backfill / debugging.

Usage:
    python run_news_dispatch.py                       # dispatch all pending
    python run_news_dispatch.py --dry-run             # show what would be sent
    python run_news_dispatch.py --parquet <path>      # dispatch one specific file
    python run_news_dispatch.py --force <path>        # re-dispatch (ignores log)
"""
from __future__ import annotations

import argparse
import fcntl
import glob as _glob
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Process lock ─────────────────────────────────────────────────────────────
# Prevents concurrent dispatch runs (e.g. manual backfill + cron overlap)
# from double-dispatching the same parquets to agent inboxes.
_LOCK_PATH = Path("/tmp/news_dispatch.lock")


def _acquire_lock() -> "IO | None":
    """Try to acquire an exclusive lock. Returns the lock file handle, or None if
    another dispatch is already running (caller should exit cleanly)."""
    fh = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except BlockingIOError:
        fh.close()
        return None

# Load .env if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            import os
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from processing.news_dispatcher import NewsDispatcher

# ── KG ingestion (optional — skipped if Neo4j unavailable) ───────────────────
try:
    from processing.kg_ingest import ingest_parquet as _kg_ingest_parquet
    from processing.event_triggers import evaluate_triggers as _evaluate_triggers
    from processing.kg_to_openbrain import sync_all as _kg_sync_openbrain
    from knowledge_base.neo4j_client import Neo4jClient as _Neo4jClient
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("news_dispatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch news parquets to agent inboxes")
    parser.add_argument(
        "--parquet", metavar="PATH",
        help="Dispatch a single specific parquet file",
    )
    parser.add_argument(
        "--force", metavar="PATH",
        help="Re-dispatch a parquet even if already in the log",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be dispatched without sending anything",
    )
    args = parser.parse_args()

    # Single-instance guard — dry-run and single-file --parquet skip the lock
    # (they're read-only or scoped; only the full pending sweep needs it)
    lock_fh = None
    if not args.dry_run and not args.parquet:
        lock_fh = _acquire_lock()
        if lock_fh is None:
            logger.info(
                "Another dispatch process is already running (%s). Exiting.", _LOCK_PATH
            )
            sys.exit(0)

    dispatcher = NewsDispatcher(dry_run=args.dry_run)

    if args.force:
        path = Path(args.force)
        logger.info("FORCE dispatch of %s", path.name)
        n = dispatcher.dispatch_parquet(path)
    elif args.parquet:
        path = Path(args.parquet)
        n = dispatcher.dispatch_parquet(path)
    else:
        n = dispatcher.dispatch_pending()

    # ── KG ingestion ─────────────────────────────────────────────────────────
    if _KG_AVAILABLE and not args.dry_run:
        _neo4j = None
        try:
            _neo4j = _Neo4jClient()
            if args.force:
                _kg_ingest_parquet(Path(args.force), _neo4j)
            elif args.parquet:
                _kg_ingest_parquet(Path(args.parquet), _neo4j)
            else:
                interp_dir = Path(__file__).parent / "data" / "interpretations"
                for pq in sorted(_glob.glob(str(interp_dir / "interpretations_*.parquet")))[-5:]:
                    _kg_ingest_parquet(Path(pq), _neo4j)
            _evaluate_triggers(_neo4j)
            _kg_sync_openbrain(_neo4j)
        except Exception as _kg_exc:
            logger.warning("KG ingestion skipped (Neo4j unavailable?): %s", _kg_exc)
        finally:
            if _neo4j is not None:
                _neo4j.close()

    if n == 0:
        logger.info("Nothing dispatched.")
    else:
        logger.info("Dispatch complete — %d items sent to agents.", n)

    if lock_fh is not None:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()
        _LOCK_PATH.unlink(missing_ok=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
