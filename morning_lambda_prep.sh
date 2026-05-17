#!/usr/bin/env bash
# morning_lambda_prep.sh — 6 AM PT morning workflow for Lambda upload prep
#
# Run this after the 6 AM PST cron completes (approx 6:05 AM PT).
# The cron auto-handles: news collection + agent dispatch (Phases 0, 0.5).
# Agents auto-process their batches and write interpretation parquets.
#
# This script: collects + unifies interpretations, verifies coverage, packages.
#
# Usage:
#   bash morning_lambda_prep.sh [DATE]      # default: today's date
#   bash morning_lambda_prep.sh 2026-02-28

set -euo pipefail

PROJ="/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb"
PYTHON="/home/peter/miniconda3/envs/rapids-26.04/bin/python"
INTERP_DIR="${PROJ}/data/interpretations"

TARGET_DATE="${1:-$(date +%Y-%m-%d)}"
DATE_COMPACT=$(echo "${TARGET_DATE}" | tr -d '-')

echo "============================================================"
echo "  Morning Lambda Prep  |  date=${TARGET_DATE}"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"

# ── Step 1: Wait for 6 AM news batch to be dispatched ────────────────────────
echo ""
echo "[1/4] Checking news dispatch for ${TARGET_DATE}..."
OBS_FILE=$(find "${PROJ}/data/raw/" -name "observations_${DATE_COMPACT}_14*.parquet" 2>/dev/null | head -1 || true)
if [ -z "${OBS_FILE}" ]; then
    echo "  ⚠ 6AM batch observation file not yet found (observations_${DATE_COMPACT}_14*.parquet)"
    echo "  Cron runs at 6:00 AM PST — check back in a few minutes."
    echo "  To continue once it appears: bash ${BASH_SOURCE[0]} ${TARGET_DATE}"
    exit 1
fi
echo "  ✓ 6AM batch: $(basename ${OBS_FILE})"

# ── Step 2: Verify agent interpretation coverage ─────────────────────────────
echo ""
echo "[2/4] Checking agent parquets for ${DATE_COMPACT} (6AM batch = _14xxxx)..."

AGENTS=("apollo" "daedalus" "minerva" "prometheus")
ALL_PRESENT=true

for AGENT in "${AGENTS[@]}"; do
    AGENT_FILE=$(find "${INTERP_DIR}" -name "interpretations_${AGENT}_${DATE_COMPACT}_14*.parquet" 2>/dev/null | head -1 || true)
    if [ -n "${AGENT_FILE}" ]; then
        ROW_COUNT=$(${PYTHON} -c "import polars as pl; print(len(pl.read_parquet('${AGENT_FILE}')))" 2>/dev/null || echo "?")
        echo "  ✓ ${AGENT}: $(basename ${AGENT_FILE})  (${ROW_COUNT} rows)"
    else
        echo "  ⚠ ${AGENT}: no 6AM batch parquet yet"
        ALL_PRESENT=false
    fi
done

if [ "${ALL_PRESENT}" = false ]; then
    echo ""
    echo "  Some agents haven't processed yet. Options:"
    echo "  a) Wait ~10 minutes and re-run"
    echo "  b) Use all-day parquets: bash ${BASH_SOURCE[0]} ${TARGET_DATE} --any-time"
    echo "  c) Collect what's available: remove the ALL_PRESENT check below"
fi

# ── Step 3: Run collect_interpretations.py ────────────────────────────────────
echo ""
echo "[3/4] Collecting + normalising interpretations for ${TARGET_DATE}..."
cd "${PROJ}"
${PYTHON} processing/collect_interpretations.py --date "${TARGET_DATE}"
echo ""

UNIFIED_FILE="${INTERP_DIR}/interpretations_unified_${DATE_COMPACT}.parquet"
if [ -f "${UNIFIED_FILE}" ]; then
    SIZE=$(du -sh "${UNIFIED_FILE}" | cut -f1)
    echo "  ✓ Unified parquet: ${UNIFIED_FILE} (${SIZE})"
else
    echo "  ✗ Unified parquet not created — check output above"
    exit 1
fi

# ── Step 4: Print rsync command for Lambda upload ─────────────────────────────
echo ""
echo "[4/4] Lambda upload instructions:"
echo ""
echo "  1. Spin up Lambda instance: https://cloud.lambdalabs.com"
echo "     Recommended: gpu_1x_h100_sxm5 (\$1.99/hr, H100 80GB)"
echo "     Minimum:     gpu_1x_a6000 (\$0.80/hr, A6000 48GB)"
echo ""
echo "  2. Upload data + code (run from home machine):"
echo ""
echo "     LAMBDA_IP=<paste-ip-here>"
echo "     rsync -avz --progress \\"
echo "       --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \\"
echo "       --exclude='*.tar.gz' --exclude='.worktrees' \\"
echo "       ${PROJ}/ ubuntu@\${LAMBDA_IP}:~/global_financial_kb/"
echo ""
echo "  3. On Lambda:"
echo "     ssh ubuntu@\${LAMBDA_IP}"
echo "     bash ~/global_financial_kb/setup_lambda.sh    # install vLLM (~5 min)"
echo "     source ~/venv/bin/activate"
echo "     export FRED_API_KEY=<your-fred-key>"
echo "     export LLM_MODE=local"
echo "     cd ~/global_financial_kb"
echo "     python build_initial_kb.py --skip-ingestion"
echo ""
echo "  4. Download snapshot when done:"
echo "     rsync -avz ubuntu@\${LAMBDA_IP}:~/global_financial_kb/data/build/snapshots/ \\"
echo "       /media/peter/fast-storage/projects/world_knowledge_base/snapshots/"
echo ""
echo "  ✓ Phase 4.5 bug fixed: enrich_graph.py builder.G → builder.graph (2026-02-27)"
echo ""
echo "============================================================"
echo "  Prep complete — data ready for Lambda upload"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"
