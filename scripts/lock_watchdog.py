#!/usr/bin/env python3
"""lock_watchdog.py — detect & release stale /tmp/*.lock files in the KB pipeline.

Background (Hermes msg 20260511-131943-5a1c):
    4 stale-locks in 4 consecutive days. Structural pipeline reliability issue.
    Generalisation of the existing ollama_watchdog.sh pattern, with two
    improvements:
      1. Multi-lock registry (not just ollama).
      2. Heartbeat-aware staleness: if a worker writes `<lockfile>.heartbeat`
         while making progress, the effective lock age is the freshness of the
         heartbeat — not the lock-file mtime — so long but healthy LLM batches
         are not killed.

Action per lock (decision tree, per-lock policy may override):
    1. Lock file missing                   → no-op
    2. Lock age (effective) < max_age      → no-op
    3. Lock age ≥ warn_age but < max_age   → log warning, send notice
    4. Lock age ≥ max_age:
       a. No PID in lock file              → unlink (stale; rare for fcntl-based locks)
       b. PID present but process dead     → unlink (orphaned)
       c. PID alive                        → SIGTERM, 5s grace, SIGKILL, unlink

Audit:
    Append-only log at /home/peter/.claude/logs/lock_watchdog.log
    Per-release event also notifies Hermes via the agent message bus.

CLI:
    python lock_watchdog.py                 # scan all registered locks (cron)
    python lock_watchdog.py --dry-run       # report only; take no action
    python lock_watchdog.py --lock <path>   # check one specific lock file
    python lock_watchdog.py --list          # show registry
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import psutil

LOG_FILE = Path("/home/peter/.claude/logs/lock_watchdog.log")
SEND_PY = Path.home() / ".claude" / "agent-tools" / "send.py"
DEFAULT_PYTHON = Path("/home/peter/miniconda3/envs/rapids-26.04/bin/python3")


@dataclasses.dataclass(frozen=True)
class LockPolicy:
    """Per-lock policy for the watchdog.

    Attributes:
        path: lock-file path
        max_age_seconds: kill+release threshold
        warn_age_seconds: log-only warning threshold (must be < max)
        check_heartbeat: if True, consult `<path>.heartbeat` mtime in addition
            to the lock file's own mtime (effective age = min(...))
        kill_holder: if True, SIGTERM/SIGKILL the holding PID before removing
            the lock. If False, just unlink the lock file (use for bash-flock
            locks where killing the holder is too disruptive).
        owner_tag: short identifier for audit logs and Hermes notifications.
    """

    path: Path
    max_age_seconds: int = 1800        # 30 min default
    warn_age_seconds: int = 1200       # 20 min default
    check_heartbeat: bool = False
    kill_holder: bool = True
    owner_tag: str = "kb_pipeline"


# ── Registry ────────────────────────────────────────────────────────────────
# Add new locks here. To keep things conservative, every entry should be a
# specific lock file the watchdog is authorised to manage.

REGISTRY: tuple[LockPolicy, ...] = (
    # KB pipeline locks
    LockPolicy(
        path=Path("/tmp/kb_daily_ingestion.lock"),
        max_age_seconds=3600 * 4,      # daily ingestion can run hours
        warn_age_seconds=3600 * 3,
        check_heartbeat=False,
        kill_holder=False,             # bash flock — no PID in file; just remove
        owner_tag="kb_daily_ingestion",
    ),
    LockPolicy(
        path=Path("/tmp/news_dispatch.lock"),
        max_age_seconds=1800,          # 30 min for one dispatch sweep
        warn_age_seconds=1200,
        check_heartbeat=False,
        kill_holder=True,
        owner_tag="news_dispatch",
    ),
    LockPolicy(
        path=Path("/tmp/ollama_inference.lock"),
        max_age_seconds=900,           # 15 min — interpret_news.py heartbeats every ~70s/batch, so ~13x margin
        warn_age_seconds=600,          # 10 min — must stay < max_age
        check_heartbeat=True,          # interpret_news.py writes .heartbeat
        kill_holder=True,
        owner_tag="ollama_inference",
    ),
    # Long-running historical jobs — older than 24h is definitely stale
    LockPolicy(
        path=Path("/tmp/calibration_backfill.lock"),
        max_age_seconds=24 * 3600,
        warn_age_seconds=12 * 3600,
        check_heartbeat=False,
        kill_holder=False,
        owner_tag="calibration_backfill",
    ),
    LockPolicy(
        path=Path("/tmp/historical_label_join.lock"),
        max_age_seconds=24 * 3600,
        warn_age_seconds=12 * 3600,
        check_heartbeat=False,
        kill_holder=False,
        owner_tag="historical_label_join",
    ),
    LockPolicy(
        path=Path("/tmp/historical_recollection.lock"),
        max_age_seconds=24 * 3600,
        warn_age_seconds=12 * 3600,
        check_heartbeat=False,
        kill_holder=False,
        owner_tag="historical_recollection",
    ),
)


# ── Utilities ───────────────────────────────────────────────────────────────

def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lock_watchdog")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                       datefmt="%Y-%m-%dT%H:%M:%SZ"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # Also emit to stdout for cron capture
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def _read_pid_from_lock(path: Path) -> int | None:
    """Return the PID written to the lock file, or None if not parseable."""
    try:
        with open(path) as fh:
            for line in fh:
                token = line.strip().split()[0] if line.strip() else ""
                if token.isdigit():
                    return int(token)
    except (OSError, IndexError):
        return None
    return None


def _read_holder_tag_from_lock(path: Path) -> str:
    """Extract `agent:<name>` tag from lock file if present."""
    try:
        with open(path) as fh:
            text = fh.read()
        for line in text.splitlines():
            if line.startswith("agent:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _effective_lock_age_seconds(policy: LockPolicy) -> int:
    """Compute lock age, optionally using newer of lock mtime and heartbeat mtime."""
    if not policy.path.exists():
        return -1
    base_mtime = policy.path.stat().st_mtime
    if policy.check_heartbeat:
        hb = policy.path.with_name(policy.path.name + ".heartbeat")
        if hb.exists():
            hb_mtime = hb.stat().st_mtime
            base_mtime = max(base_mtime, hb_mtime)
    return int(time.time() - base_mtime)


def _pid_is_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _kill_pid(pid: int, logger: logging.Logger, grace_sec: float = 5.0) -> bool:
    """SIGTERM → wait → SIGKILL. Returns True if process is dead after.
    Silently no-ops if PID is invalid or already dead."""
    if pid <= 0:
        return True
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"  SIGTERM sent to pid={pid}")
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warning(f"  PermissionError sending SIGTERM to pid={pid}")
        return False

    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if not _pid_is_alive(pid):
            return True
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
        logger.info(f"  SIGKILL sent to pid={pid}")
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warning(f"  PermissionError sending SIGKILL to pid={pid}")
        return False

    time.sleep(0.5)
    return not _pid_is_alive(pid)


def _notify_hermes(subject: str, body: str, logger: logging.Logger,
                    enable: bool = True) -> None:
    """Best-effort Hermes notification. Failures are non-fatal."""
    if not enable or not SEND_PY.exists():
        return
    try:
        subprocess.run(
            [str(DEFAULT_PYTHON), str(SEND_PY),
             "--from", "watchdog", "--to", "hermes",
             "--type", "status", "--subject", subject, "--body", body],
            check=False, timeout=10,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        logger.warning(f"  hermes notify failed: {e}")


# ── Per-lock state machine ──────────────────────────────────────────────────

def process_lock(policy: LockPolicy, *, dry_run: bool, logger: logging.Logger,
                  notify: bool = True) -> dict:
    """Inspect one lock and take action per policy. Returns an audit dict.

    Audit dict fields:
        path, owner_tag, status, age_seconds, age_minutes, pid, action, dry_run
    """
    audit = {
        "path": str(policy.path),
        "owner_tag": policy.owner_tag,
        "status": "absent",
        "age_seconds": -1,
        "age_minutes": -1,
        "pid": None,
        "action": "none",
        "dry_run": dry_run,
        "ts": _now_utc_str(),
    }
    if not policy.path.exists():
        return audit

    age = _effective_lock_age_seconds(policy)
    audit["age_seconds"] = age
    audit["age_minutes"] = age // 60
    pid = _read_pid_from_lock(policy.path)
    audit["pid"] = pid
    holder = _read_holder_tag_from_lock(policy.path)

    if age < policy.warn_age_seconds:
        audit["status"] = "healthy"
        return audit

    if age < policy.max_age_seconds:
        audit["status"] = "warn"
        logger.warning(
            f"WARN {policy.owner_tag}: lock held {age // 60}m by pid={pid} "
            f"holder={holder} (warn threshold {policy.warn_age_seconds // 60}m)"
        )
        return audit

    # Over max_age — release
    audit["status"] = "stale"
    logger.warning(
        f"STALE {policy.owner_tag}: lock held {age // 60}m by pid={pid} "
        f"holder={holder} (max_age {policy.max_age_seconds // 60}m); acting"
    )

    pid_alive = pid is not None and _pid_is_alive(pid)

    if dry_run:
        audit["action"] = "dry_run_would_release"
        if pid_alive and policy.kill_holder:
            audit["action"] = "dry_run_would_kill_and_release"
        return audit

    # Determine action
    if not pid_alive:
        audit["action"] = "release_orphaned"
        _safe_unlink_lock(policy, logger)
        _notify_hermes(
            f"WATCHDOG: orphaned {policy.owner_tag} lock removed ({age // 60}m old)",
            f"Lock file: {policy.path}\nHolder: {holder}\nPID: {pid} (dead)\n"
            f"Age: {age // 60} minutes\nAction: removed lock file (and heartbeat if present)",
            logger, enable=notify,
        )
        return audit

    if policy.kill_holder:
        killed = _kill_pid(pid, logger)
        audit["action"] = "kill_and_release" if killed else "kill_failed_release_anyway"
        _safe_unlink_lock(policy, logger)
        _notify_hermes(
            f"WATCHDOG: {policy.owner_tag} lock force-released ({age // 60}m old)",
            f"Lock file: {policy.path}\nHolder agent: {holder}\nKilled PID: {pid}\n"
            f"Lock age at kill: {age // 60} minutes\n"
            f"Action: SIGTERM → 5s wait → SIGKILL → lock removed",
            logger, enable=notify,
        )
        return audit

    # kill_holder=False — just remove lock (bash flock cases where the
    # holder bash process is doing useful work but the lock file is leftover)
    audit["action"] = "release_lock_only"
    _safe_unlink_lock(policy, logger)
    _notify_hermes(
        f"WATCHDOG: stale {policy.owner_tag} lock removed ({age // 60}m old; holder kept alive)",
        f"Lock file: {policy.path}\nHolder agent: {holder}\nPID: {pid} (alive)\n"
        f"Age: {age // 60} minutes\nAction: removed lock file only; holder process not killed",
        logger, enable=notify,
    )
    return audit


def _safe_unlink_lock(policy: LockPolicy, logger: logging.Logger) -> None:
    try:
        policy.path.unlink()
        logger.info(f"  unlinked {policy.path}")
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.error(f"  failed to unlink {policy.path}: {e}")

    hb = policy.path.with_name(policy.path.name + ".heartbeat")
    if hb.exists():
        try:
            hb.unlink()
            logger.info(f"  unlinked heartbeat {hb}")
        except OSError as e:
            logger.error(f"  failed to unlink heartbeat {hb}: {e}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def _resolve_policies(
    registry: Iterable[LockPolicy], specific_path: str | None
) -> list[LockPolicy]:
    if specific_path is None:
        return list(registry)
    p = Path(specific_path)
    for policy in registry:
        if policy.path == p:
            return [policy]
    # Not in registry — synthesise a default policy
    return [LockPolicy(path=p)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KB-pipeline lock watchdog")
    parser.add_argument("--dry-run", action="store_true",
                         help="report only; do not kill/unlink")
    parser.add_argument("--lock", type=str,
                         help="check one specific lock file (defaults to scanning registry)")
    parser.add_argument("--list", action="store_true",
                         help="show registry and exit")
    parser.add_argument("--no-notify", action="store_true",
                         help="skip Hermes notifications (for tests/CI)")
    args = parser.parse_args(argv)

    if args.list:
        for p in REGISTRY:
            print(f"{p.path}  max={p.max_age_seconds}s  warn={p.warn_age_seconds}s  "
                  f"heartbeat={p.check_heartbeat}  kill={p.kill_holder}  tag={p.owner_tag}")
        return 0

    logger = _setup_logger()
    policies = _resolve_policies(REGISTRY, args.lock)
    audits = []
    n_acted = 0
    for policy in policies:
        audit = process_lock(policy, dry_run=args.dry_run, logger=logger,
                              notify=not args.no_notify)
        audits.append(audit)
        if audit["action"] not in ("none",):
            n_acted += 1

    # Concise summary to stdout (cron-friendly)
    summary = [a for a in audits if a["status"] != "absent"]
    if summary:
        for a in summary:
            print(json.dumps(a))
    if n_acted:
        logger.info(f"watchdog done: {n_acted} action(s) across {len(audits)} locks scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
