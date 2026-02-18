#!/usr/bin/env bash
# run_daily_ingestion.sh — Daily KB ingestion (API data only, no LLM)
#
# Runs Phase 1 (parallel API fetch) + graph construction + HMM training.
# LLM phases (edge weight re-inference) are left for the weekly Lambda run.
#
# Cron: 0 */6 * * * /path/to/run_daily_ingestion.sh   (every 6 hours)

set -euo pipefail

PROJ="/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb"
PYTHON="/home/peter/anaconda3/envs/rapids-25.10/bin/python"
LOGDIR="${PROJ}/data/logs"
LOCKFILE="/tmp/kb_daily_ingestion.lock"

mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/ingestion_$(date +%Y-%m-%d).log"

# Prevent concurrent runs
exec 200>"${LOCKFILE}"
flock -n 200 || { echo "$(date): already running, skipping" >> "${LOG}"; exit 0; }

echo "======================================" >> "${LOG}"
echo "KB daily ingestion — $(date -u '+%Y-%m-%d %H:%M UTC')" >> "${LOG}"
echo "======================================" >> "${LOG}"

cd "${PROJ}"

# Load API keys from environment file if present
if [ -f "${PROJ}/.env" ]; then
    set -a
    source "${PROJ}/.env"
    set +a
fi

# Run ingestion (API data only — no LLM calls)
"${PYTHON}" build_initial_kb.py --skip-llm >> "${LOG}" 2>&1
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}" >> "${LOG}"
echo "Done: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "${LOG}"

# Rotate logs older than 30 days
find "${LOGDIR}" -name "ingestion_*.log" -mtime +30 -delete 2>/dev/null || true

exit ${EXIT_CODE}
