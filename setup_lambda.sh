#!/usr/bin/env bash
# setup_lambda.sh — One-shot environment setup for Lambda Labs GPU instances
#
# TARGET INSTANCE: gpu_1x_h100_sxm5 (H100 80GB SXM5, 26vCPUs, 225 GiB RAM, 2.8 TiB SSD)
#   H100 SXM5 runs 70B AWQ-INT4 with ~38GB model weights + ample KV cache headroom.
#   Fallback for ≥48GB but <80GB instances: A6000 (48GB) — still runs 70B, less headroom.
#   DO NOT use: gpu_1x_a100 (A100 40GB) — cannot hold 70B model + KV cache
#
# Workflow for a pre-fetched data run (fastest, avoids Lambda IP throttling):
#
#   # Step 1 (home machine): fetch all API data
#   cd global_financial_kb
#   FRED_API_KEY=xxx python build_initial_kb.py --skip-llm
#
#   # Step 2: upload code + pre-fetched data
#   rsync -avz --exclude='*.pyc' --exclude='__pycache__' --exclude='*.tar.gz' \
#       --exclude='.git' \
#       /path/to/global_financial_kb/ ubuntu@{LAMBDA_IP}:~/global_financial_kb/
#
#   # Step 3 (Lambda): LLM phases only (~5-10 min, ~$0.10)
#   bash setup_lambda.sh
#   export FRED_API_KEY=xxx
#   export LLM_MODE=local
#   cd ~/global_financial_kb
#   python build_initial_kb.py --skip-ingestion

set -euo pipefail

echo "============================================================"
echo "  Global Financial KB — Lambda Environment Setup"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git rsync htop nvtop curl python3-venv

# ── 2. Check GPU VRAM (fail fast if wrong instance type) ─────────────────────
echo ""
echo "[2/6] Checking GPU..."
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "  GPU: ${GPU_NAME} (${GPU_MEM} MiB)"

if [ "${GPU_MEM:-0}" -ge 78000 ]; then
    # H100 80GB SXM5 (81920 MiB) — primary target; plenty of headroom for 70B + KV cache
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    echo "  ✓ H100 80GB detected — will use Llama-3.1-70B-Instruct AWQ-INT4"
    echo "  Note: ~37GB model weights, ~40GB KV cache headroom"
elif [ "${GPU_MEM:-0}" -ge 48000 ]; then
    # A6000 48GB (49152 MiB) — can run 70B AWQ-INT4, tighter on KV cache
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    echo "  ✓ GPU has ≥48GB VRAM — will use Llama-3.1-70B-Instruct AWQ-INT4"
    echo "  Note: ~37GB model weights, ~10GB KV cache headroom (tight)"
    echo "  Preferred: gpu_1x_h100_sxm5 (80GB) for full KV cache headroom"
else
    # A100 40GB (40960 MiB) — 70B does NOT fit; 70B AWQ needs ≥48GB
    MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    echo "  ✗ GPU has <48GB VRAM — falling back to Llama-3.1-8B-Instruct AWQ-INT4"
    echo "  For full 70B quality, request: gpu_1x_h100_sxm5 (80GB)"
fi

# ── 3. Python virtualenv (required — system TF conflicts with vLLM's numpy 2.x) ──
echo ""
echo "[3/6] Creating isolated Python virtualenv..."

# Lambda's system Python has TensorFlow compiled against numpy 1.x.
# vLLM requires numpy 2.x. Must isolate in a venv to avoid the conflict.
python3 -m venv ~/venv
source ~/venv/bin/activate

echo "  ✓ Virtualenv created at ~/venv"
echo "  ✓ Activated: $(which python)"

# ── 4. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing Python dependencies..."

pip install --quiet --upgrade pip

# Core dependencies first
pip install --quiet \
    polars \
    networkx \
    numpy \
    httpx \
    hmmlearn \
    anthropic \
    pyarrow \
    python-dotenv \
    scipy \
    scikit-learn

echo "  ✓ Core dependencies installed"

# vLLM: installs numpy 2.x, torch, transformers — isolated in venv from system TF
echo "  Installing vLLM (this takes ~3-5 min)..."
pip install --quiet vllm

echo "  ✓ vLLM installed"

# ── 5. Create data directories ────────────────────────────────────────────────
echo ""
echo "[5/6] Creating data directories..."
mkdir -p ~/global_financial_kb/data/{raw,processed,graph,build/{checkpoints,snapshots},briefings}
echo "  ✓ Directories created at ~/global_financial_kb/"

# ── 6. Start vLLM server (background) ────────────────────────────────────────
echo ""
echo "[6/6] Starting vLLM server: ${MODEL}"
echo "  First run downloads model weights (~37GB for 70B, ~5GB for 8B)"
echo "  Allow 10-20 min for download on first run"
echo ""

# H100 80GB can support 32k context with 70B AWQ-INT4 at 90% utilization.
# Increase max-model-len for richer reasoning on complex economic prompts.
# Launch with stdin from /dev/null so nohup doesn't hold the terminal
nohup ~/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --dtype auto \
    < /dev/null > ~/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"
echo "  Logs:     tail -f ~/vllm_server.log"

# Wait for server ready (health endpoint returns 200)
echo ""
echo "  Waiting for vLLM server (up to 25 min for model download + load)..."
MAX_WAIT=1500  # 25 minutes
ELAPSED=0
LAST_LOG_LINE=""

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "  ✓ vLLM server is ready! (${ELAPSED}s)"
        break
    fi

    # Print last log line every 30s so you can see what's happening
    if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
        CURRENT_LOG=$(tail -1 ~/vllm_server.log 2>/dev/null || echo "")
        if [ "${CURRENT_LOG}" != "${LAST_LOG_LINE}" ]; then
            echo "  [${ELAPSED}s] ${CURRENT_LOG}"
            LAST_LOG_LINE="${CURRENT_LOG}"
        fi
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "  ⚠ vLLM server not ready after ${MAX_WAIT}s"
    echo "  Check: tail -50 ~/vllm_server.log"
    echo "  You can still run with LLM_MODE=claude if server fails to start"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  Activate the venv in any new shell:"
echo "    source ~/venv/bin/activate"
echo ""
echo "  Set API keys:"
echo "    export FRED_API_KEY=<your_fred_key>"
echo "    export LLM_MODE=local"
echo "    # Only if using Claude API instead of local model:"
echo "    # export LLM_MODE=claude"
echo "    # export ANTHROPIC_API_KEY=<your_anthropic_key>"
echo ""
echo "  Run the build (pre-fetched data path — fastest):"
echo "    cd ~/global_financial_kb"
echo "    python build_initial_kb.py --skip-ingestion"
echo ""
echo "  Run the build (full ingestion on Lambda — slower due to IP throttling):"
echo "    python build_initial_kb.py"
echo ""
echo "  Or resume if interrupted:"
echo "    python build_initial_kb.py --skip-ingestion --resume"
echo ""
echo "  Download the snapshot when done:"
echo "    rsync -avz --progress \\"
echo "      ubuntu@<LAMBDA_IP>:~/global_financial_kb/data/build/snapshots/ \\"
echo "      /media/peter/fast-storage/projects/world_knowledge_base/snapshots/"
echo "============================================================"
