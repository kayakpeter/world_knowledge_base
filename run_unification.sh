#!/usr/bin/env bash
# run_unification.sh — Concatenate per-agent interpretation parquets into the
# daily interpretations_unified_YYYYMMDD.parquet that geo_stress_scorer reads.
#
# Idempotent: re-running for the same --date overwrites the unified file with
# whatever per-agent files are currently on disk. This is intentional — late
# stragglers get picked up by the morning re-run pass.
#
# Cron-install: see crontab tag PROMETHEUS_UNIFY_DAILY
#   23:30 PDT  first-pass for today
#   06:15 PDT  re-unify yesterday (catches overnight stragglers; aligns with
#              morning_lambda_prep.sh design intent of pre-market unified parquet)
#
# Usage:
#   run_unification.sh                  # unify today (default)
#   run_unification.sh yesterday        # unify yesterday
#   run_unification.sh 2026-05-15       # unify a specific date

set -euo pipefail

PROJ="/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb"
PYTHON="/home/peter/miniconda3/envs/rapids-26.04/bin/python3"
LOGDIR="${PROJ}/data/logs"

mkdir -p "${LOGDIR}"

# Resolve target date (UTC, to match per-agent filename convention)
case "${1:-today}" in
    today)      DATE_STR=$(date -u +%F) ;;
    yesterday)  DATE_STR=$(date -u -d yesterday +%F) ;;
    *)          DATE_STR="$1" ;;
esac

LOG="${LOGDIR}/unification_${DATE_STR}.log"
LOCKFILE="/tmp/kb_unify_${DATE_STR}.lock"

# Per-date lock — different dates can run concurrently; same date cannot
exec 200>"${LOCKFILE}"
flock -n 200 || { echo "$(date): unify ${DATE_STR} already running, skipping" >> "${LOG}"; exit 0; }

{
    echo "======================================"
    echo "KB unification — date=${DATE_STR} — $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "======================================"
} >> "${LOG}"

cd "${PROJ}"

"${PYTHON}" processing/collect_interpretations.py --date "${DATE_STR}" >> "${LOG}" 2>&1
EXIT_CODE=$?

{
    echo "Exit code: ${EXIT_CODE}"
    echo "Done: $(date -u '+%Y-%m-%d %H:%M UTC')"
} >> "${LOG}"

# Rotate unification logs older than 30 days (matches run_daily_ingestion.sh policy)
find "${LOGDIR}" -name "unification_*.log" -mtime +30 -delete 2>/dev/null || true

exit ${EXIT_CODE}
