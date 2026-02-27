#!/usr/bin/env bash
# run_daily_ingestion.sh — Daily KB ingestion (API data only, no LLM)
#
# Phase 0: News ingestion (NewsData.io + GNews + GDELT + RSS) → news_*.parquet
# Phase 1: Statistical API fetch (FRED, World Bank, IMF, OECD, BIS, EIA)
# Phase 2: Graph construction + HMM training
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

# Phase 0: News ingestion (runs first — no lock contention with statistical pipeline)
echo "--- Phase 0: News ingestion ---" >> "${LOG}"
"${PYTHON}" run_news_ingestion.py --hours 6 >> "${LOG}" 2>&1
NEWS_EXIT=$?
if [ "${NEWS_EXIT}" -ne 0 ]; then
    echo "WARNING: News ingestion exited with code ${NEWS_EXIT} — continuing" >> "${LOG}"
fi

# Phase 0.5: Dispatch new news parquets to agent inboxes
echo "--- Phase 0.5: News dispatch to agents ---" >> "${LOG}"
"${PYTHON}" run_news_dispatch.py >> "${LOG}" 2>&1
DISPATCH_EXIT=$?
if [ "${DISPATCH_EXIT}" -ne 0 ]; then
    echo "WARNING: News dispatch exited with code ${DISPATCH_EXIT} — continuing" >> "${LOG}"
fi

# Phase 0.7: Commodity ETF price fetch (incremental — skips if already up to date)
echo "--- Phase 0.7: Commodity ETF prices (yfinance) ---" >> "${LOG}"
"${PYTHON}" run_price_fetch.py >> "${LOG}" 2>&1
PRICE_EXIT=$?
if [ "${PRICE_EXIT}" -ne 0 ]; then
    echo "WARNING: Price fetch exited with code ${PRICE_EXIT} — continuing" >> "${LOG}"
fi

# Phase 1-2: Statistical ingestion + graph construction (API data only — no LLM calls)
echo "--- Phase 1: Statistical ingestion ---" >> "${LOG}"
"${PYTHON}" build_initial_kb.py --skip-llm >> "${LOG}" 2>&1
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}" >> "${LOG}"
echo "Done: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "${LOG}"

# Rotate logs older than 30 days
find "${LOGDIR}" -name "ingestion_*.log" -mtime +30 -delete 2>/dev/null || true

exit ${EXIT_CODE}
