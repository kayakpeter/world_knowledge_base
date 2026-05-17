#!/usr/bin/env bash
# run_kg_batch_ingest.sh — Ingest all recent interpretation parquets into Neo4j.
#
# Finds interpretation_*.parquet files modified in the last N hours (default 8)
# and runs kg_ingest on each, then evaluates triggers + syncs to Open Brain.
#
# Usage:
#   ./run_kg_batch_ingest.sh            # last 8 hours (default)
#   ./run_kg_batch_ingest.sh 24         # last 24 hours
#   ./run_kg_batch_ingest.sh --dry-run  # dry run (no Neo4j writes)
#
# Cron (45 min after each :00/:06/:12/:18 dispatch window):
#   45 0,6,12,18 * * * /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/run_kg_batch_ingest.sh >> /tmp/kg_batch_ingest.log 2>&1

set -euo pipefail

PROJ="/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb"
PYTHON="/home/peter/miniconda3/envs/rapids-26.04/bin/python"
INTERP_DIR="${PROJ}/data/interpretations"
HOURS="${1:-8}"
DRY_RUN=""

# Handle --dry-run flag
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    HOURS=8
fi

# Load env
if [ -f "${PROJ}/.env" ]; then
    set -a; source "${PROJ}/.env"; set +a
fi
# Also load global .env for Neo4j creds
if [ -f "/media/peter/fast-storage/.env" ]; then
    set -a; source /media/peter/fast-storage/.env; set +a
fi

echo "========================================"
echo "KG batch ingest — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "Looking for parquets modified in last ${HOURS}h"
echo "========================================"

cd "${PROJ}"

# Find all interpretation parquets modified in the last N hours
PARQUETS=$(find "${INTERP_DIR}" -name "interpretations_*.parquet" -mmin -$((HOURS * 60)) 2>/dev/null | sort)

if [ -z "${PARQUETS}" ]; then
    echo "No new interpretation parquets found in last ${HOURS}h — nothing to ingest."
    exit 0
fi

COUNT=$(echo "${PARQUETS}" | wc -l)
echo "Found ${COUNT} parquet(s) to ingest:"
echo "${PARQUETS}" | sed 's/^/  /'

INGESTED=0
FAILED=0

while IFS= read -r PQ; do
    FNAME=$(basename "${PQ}")
    echo ""
    echo "--- Ingesting: ${FNAME} ---"
    if "${PYTHON}" -m processing.kg_ingest --parquet "${PQ}" ${DRY_RUN} 2>&1; then
        INGESTED=$((INGESTED + 1))
    else
        echo "WARNING: kg_ingest failed for ${FNAME}"
        FAILED=$((FAILED + 1))
    fi
done <<< "${PARQUETS}"

echo ""
echo "--- Evaluating triggers ---"
"${PYTHON}" -c "
import sys; sys.path.insert(0, '.')
from knowledge_base.neo4j_client import Neo4jClient
from processing.event_triggers import evaluate_all_triggers
client = Neo4jClient()
try:
    evaluate_all_triggers(client)
    print('Triggers evaluated OK')
except Exception as e:
    print(f'Trigger evaluation failed: {e}')
finally:
    client.close()
" 2>&1 || echo "WARNING: trigger evaluation failed"

echo ""
echo "--- Syncing to Open Brain ---"
"${PYTHON}" -c "
import sys; sys.path.insert(0, '.')
from knowledge_base.neo4j_client import Neo4jClient
from processing.kg_to_openbrain import sync_all
client = Neo4jClient()
try:
    sync_all(client)
    print('Open Brain sync OK')
except Exception as e:
    print(f'Open Brain sync failed: {e}')
finally:
    client.close()
" 2>&1 || echo "WARNING: Open Brain sync failed"

echo ""
echo "========================================"
echo "KG batch ingest complete — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "Ingested: ${INGESTED} / Failed: ${FAILED}"
echo "========================================"
