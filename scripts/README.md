# scripts/

Operational scripts for the KB pipeline.

## lock_watchdog.py — stale-lock auto-release

Detects and releases stale `/tmp/*.lock` files from the KB pipeline. Generalisation of `agents/hephaestus/ollama_watchdog.sh`. See module docstring for the full state-machine.

### Registered locks

| lock path                            | max_age | warn   | heartbeat | kill_holder |
|--------------------------------------|--------:|-------:|----------:|-------------|
| `/tmp/kb_daily_ingestion.lock`       |    4 h  |  3 h   | no        | no (bash flock) |
| `/tmp/news_dispatch.lock`            |   30 m  | 20 m   | no        | yes (Python fcntl) |
| `/tmp/ollama_inference.lock`         |   30 m  | 20 m   | YES       | yes (Python fcntl) |
| `/tmp/calibration_backfill.lock`     |   24 h  | 12 h   | no        | no |
| `/tmp/historical_label_join.lock`    |   24 h  | 12 h   | no        | no |
| `/tmp/historical_recollection.lock`  |   24 h  | 12 h   | no        | no |

Edit the `REGISTRY` tuple in `lock_watchdog.py` to add or adjust policies. Audit log goes to `/home/peter/.claude/logs/lock_watchdog.log`. Each release event also sends a status message to Hermes (`agent-tools/send.py`).

### CLI

```bash
# Scan everything in the registry; take action on stale locks (default behaviour)
/home/peter/miniconda3/envs/rapids-26.04/bin/python3 scripts/lock_watchdog.py

# Dry-run: report what would happen without acting
scripts/lock_watchdog.py --dry-run

# Suppress Hermes notifications (useful for tests, local runs)
scripts/lock_watchdog.py --no-notify

# Inspect / debug one specific lock (registry policy applied if known, else default)
scripts/lock_watchdog.py --lock /tmp/some.lock --dry-run --no-notify

# Show the registry
scripts/lock_watchdog.py --list
```

### Cron install

Recommended: every 5 minutes, all hours. Bash hook so logs land in the watchdog audit:

```cron
*/5 * * * * /home/peter/miniconda3/envs/rapids-26.04/bin/python3 /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/scripts/lock_watchdog.py >> /home/peter/.claude/logs/lock_watchdog.cron.log 2>&1
```

The audit log (`/home/peter/.claude/logs/lock_watchdog.log`) is the SoT for release events; the `.cron.log` only captures stdout summary lines.

### Tests

```bash
cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb
/home/peter/miniconda3/envs/rapids-26.04/bin/python3 -m pytest tests/test_lock_watchdog.py -v
```

20 unit tests covering: lock-file parsing, heartbeat-aware age, dry-run safety, the four state-machine branches (healthy / warn / orphaned-release / kill-and-release), CLI surface.

### Failure modes the watchdog handles

| symptom | mechanism | resolution |
|---------|-----------|------------|
| process died, fcntl auto-released kernel lock, file persists | kernel already released; lock-file is just an empty file | `release_orphaned` — unlink |
| process died, lock file contains stale PID | PID is dead; flock-on-PID-death released kernel lock | `release_orphaned` — unlink |
| process alive but hung mid-LLM-call indefinitely | kernel still holds fcntl on its fd | `kill_and_release` — SIGTERM, 5s grace, SIGKILL, unlink |
| process alive, doing useful work, heartbeating | heartbeat fresh → effective age stays low | `healthy` — no-op (heartbeat-aware locks only) |
| bash flock without PID in file, multi-day stale | empty lock file leftover | `release_orphaned` — unlink |

### Failure modes the watchdog does NOT handle

- **Multi-process holders**: `lsof`-style inspection is not done. If a wrapper script forks a child that inherits the fd, only the holder PID written to file is killed.
- **Race with healthy startup**: if a worker is acquiring a fresh lock concurrently with the watchdog scan, the new lock file may be momentarily near-zero-age — the watchdog will correctly skip it (age < warn threshold).
- **Networked locks (NFS, etc.)**: not tested; lock files are assumed local `/tmp/`.

### Adding a new lock

1. Edit `REGISTRY` in `lock_watchdog.py`.
2. Add a `LockPolicy(...)` entry with `path`, `max_age_seconds`, `warn_age_seconds`, `check_heartbeat` (True iff acquiring code writes `<lockfile>.heartbeat` during long work), `kill_holder` (False for bash flock with no PID), and an `owner_tag`.
3. Add a test asserting the policy behaviour on the lock you registered.
4. Commit.
