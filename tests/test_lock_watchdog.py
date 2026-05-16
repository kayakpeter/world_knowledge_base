"""Unit tests for scripts/lock_watchdog.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Make the script importable as a module
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import lock_watchdog as lw  # noqa: E402


@pytest.fixture
def tmp_lock(tmp_path: Path) -> Path:
    """A temporary lock-file path that doesn't exist yet."""
    return tmp_path / "test.lock"


def _write_lock(path: Path, pid: int | None = None, holder: str | None = None) -> None:
    parts = []
    if pid is not None:
        parts.append(str(pid))
    if holder is not None:
        parts.append(f"agent:{holder}")
    path.write_text("\n".join(parts))


def _make_old(path: Path, seconds_ago: int) -> None:
    """Set the file's mtime to N seconds ago."""
    now = time.time()
    os.utime(path, (now - seconds_ago, now - seconds_ago))


# ── _read_pid_from_lock ─────────────────────────────────────────────────────

def test_read_pid_from_lock_returns_int(tmp_lock):
    _write_lock(tmp_lock, pid=1234, holder="hephaestus")
    assert lw._read_pid_from_lock(tmp_lock) == 1234


def test_read_pid_from_lock_returns_none_when_empty(tmp_lock):
    tmp_lock.write_text("")
    assert lw._read_pid_from_lock(tmp_lock) is None


def test_read_pid_from_lock_ignores_non_numeric_first_token(tmp_lock):
    tmp_lock.write_text("agent:test\n")
    assert lw._read_pid_from_lock(tmp_lock) is None


def test_read_holder_tag(tmp_lock):
    _write_lock(tmp_lock, pid=999, holder="daedalus")
    assert lw._read_holder_tag_from_lock(tmp_lock) == "daedalus"


def test_read_holder_tag_returns_unknown_if_missing(tmp_lock):
    tmp_lock.write_text("12345\n")
    assert lw._read_holder_tag_from_lock(tmp_lock) == "unknown"


# ── effective lock age ──────────────────────────────────────────────────────

def test_effective_age_uses_lock_mtime_when_no_heartbeat(tmp_lock):
    _write_lock(tmp_lock, pid=1)
    _make_old(tmp_lock, 300)
    policy = lw.LockPolicy(path=tmp_lock, check_heartbeat=False)
    age = lw._effective_lock_age_seconds(policy)
    assert 290 < age < 320


def test_effective_age_returns_neg_one_when_lock_missing(tmp_lock):
    policy = lw.LockPolicy(path=tmp_lock)
    assert lw._effective_lock_age_seconds(policy) == -1


def test_effective_age_uses_heartbeat_when_newer(tmp_lock):
    """Lock mtime 600s ago, heartbeat 30s ago → effective age ≈ 30s."""
    _write_lock(tmp_lock, pid=1)
    _make_old(tmp_lock, 600)
    hb = tmp_lock.with_name(tmp_lock.name + ".heartbeat")
    hb.write_text("")
    _make_old(hb, 30)
    policy = lw.LockPolicy(path=tmp_lock, check_heartbeat=True)
    age = lw._effective_lock_age_seconds(policy)
    assert age < 60   # newer-of-two wins


def test_effective_age_ignores_heartbeat_when_disabled(tmp_lock):
    _write_lock(tmp_lock, pid=1)
    _make_old(tmp_lock, 600)
    hb = tmp_lock.with_name(tmp_lock.name + ".heartbeat")
    hb.write_text("")
    policy = lw.LockPolicy(path=tmp_lock, check_heartbeat=False)
    age = lw._effective_lock_age_seconds(policy)
    assert age >= 590


# ── process_lock — policy state machine ─────────────────────────────────────

def test_process_lock_absent_is_no_op(tmp_lock):
    policy = lw.LockPolicy(path=tmp_lock)
    audit = lw.process_lock(policy, dry_run=True, logger=lw._setup_logger(), notify=False)
    assert audit["status"] == "absent"
    assert audit["action"] == "none"


def test_process_lock_healthy_under_warn(tmp_lock):
    _write_lock(tmp_lock, pid=os.getpid(), holder="test")
    # Just-now lock — age ~0s
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
    )
    audit = lw.process_lock(policy, dry_run=True, logger=lw._setup_logger(), notify=False)
    assert audit["status"] == "healthy"


def test_process_lock_warn_between_thresholds(tmp_lock):
    _write_lock(tmp_lock, pid=os.getpid(), holder="test")
    _make_old(tmp_lock, 150)   # 150s ≥ warn(120), < max(300)
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
    )
    audit = lw.process_lock(policy, dry_run=True, logger=lw._setup_logger(), notify=False)
    assert audit["status"] == "warn"
    assert audit["action"] == "none"


def test_process_lock_dry_run_does_not_unlink(tmp_lock):
    _write_lock(tmp_lock, pid=os.getpid(), holder="test")
    _make_old(tmp_lock, 400)
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
        kill_holder=False,
    )
    audit = lw.process_lock(policy, dry_run=True, logger=lw._setup_logger(), notify=False)
    assert audit["status"] == "stale"
    assert "dry_run" in audit["action"]
    assert tmp_lock.exists(), "dry-run must not unlink"


def test_process_lock_release_orphaned_when_pid_dead(tmp_lock):
    """PID that doesn't exist → release (without trying to kill)."""
    # Use a PID very unlikely to be alive
    _write_lock(tmp_lock, pid=99_999_999, holder="ghost")
    _make_old(tmp_lock, 400)
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
        kill_holder=True,
    )
    audit = lw.process_lock(policy, dry_run=False, logger=lw._setup_logger(), notify=False)
    assert audit["status"] == "stale"
    assert audit["action"] == "release_orphaned"
    assert not tmp_lock.exists()


def test_process_lock_release_no_pid_when_lock_is_bash_flock(tmp_lock):
    """Empty-content lock (bash flock leaves no PID) → release_orphaned."""
    tmp_lock.write_text("")     # bash flock leaves no content
    _make_old(tmp_lock, 400)
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
        kill_holder=False,
    )
    audit = lw.process_lock(policy, dry_run=False, logger=lw._setup_logger(), notify=False)
    assert audit["status"] == "stale"
    # PID parses as None → orphaned path
    assert audit["action"] == "release_orphaned"
    assert not tmp_lock.exists()


def test_process_lock_release_only_when_kill_holder_false(tmp_lock):
    """Alive PID but policy says don't kill → release lock, keep process alive."""
    _write_lock(tmp_lock, pid=os.getpid(), holder="self")
    _make_old(tmp_lock, 400)
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
        kill_holder=False,
    )
    audit = lw.process_lock(policy, dry_run=False, logger=lw._setup_logger(), notify=False)
    assert audit["action"] == "release_lock_only"
    assert not tmp_lock.exists()
    # We're the holder PID; we must still be alive
    assert os.getpid() == os.getpid()


def test_process_lock_unlinks_heartbeat_too(tmp_lock):
    _write_lock(tmp_lock, pid=99_999_999, holder="ghost")
    _make_old(tmp_lock, 400)
    hb = tmp_lock.with_name(tmp_lock.name + ".heartbeat")
    hb.write_text("")
    _make_old(hb, 400)
    policy = lw.LockPolicy(
        path=tmp_lock, max_age_seconds=300, warn_age_seconds=120,
        check_heartbeat=True,
    )
    audit = lw.process_lock(policy, dry_run=False, logger=lw._setup_logger(), notify=False)
    assert audit["action"] == "release_orphaned"
    assert not tmp_lock.exists()
    assert not hb.exists()


# ── CLI surface ─────────────────────────────────────────────────────────────

def test_cli_list_shows_registry():
    res = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "lock_watchdog.py"), "--list"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "kb_daily_ingestion" in res.stdout
    assert "ollama_inference" in res.stdout


def test_cli_dry_run_exits_zero():
    res = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "lock_watchdog.py"),
         "--dry-run", "--no-notify"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0


def test_cli_one_lock_dry_run(tmp_path: Path):
    """--lock on a custom path should synth a default policy and report."""
    custom = tmp_path / "custom.lock"
    _write_lock(custom, pid=99_999_999, holder="x")
    _make_old(custom, 99_999)
    res = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "lock_watchdog.py"),
         "--lock", str(custom), "--dry-run", "--no-notify"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert "stale" in res.stdout
