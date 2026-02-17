#!/usr/bin/env bash
# setup_lambda.sh — One-shot environment setup for Lambda Labs A6000/A100
#
# Run once after spinning up a new Lambda instance:
#   bash setup_lambda.sh
#
# Then set your API keys and run the build:
#   export FRED_API_KEY=your_key
#   export LLM_MODE=local           # use local Llama-70B (recommended on GPU)
#   python build_initial_kb.py
#
# Or to use Claude API instead of local model:
#   export LLM_MODE=claude
#   export ANTHROPIC_API_KEY=your_key
#   python build_initial_kb.py

set -euo pipefail

echo "============================================================"
echo "  Global Financial KB — Lambda Environment Setup"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git rsync htop nvtop curl

# ── 2. Python environment ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Setting up Python environment..."

# Lambda instances have Python 3.10+ and pip pre-installed
pip install --quiet --upgrade pip

pip install --quiet \
    polars \
    networkx \
    numpy \
    httpx \
    hmmlearn \
    anthropic \
    pyarrow \
    python-dotenv

echo "  ✓ Core dependencies installed"

# ── 3. vLLM for local inference (GPU mode) ────────────────────────────────────
echo ""
echo "[3/5] Installing vLLM for local model inference..."

# vLLM requires CUDA — pre-installed on Lambda instances
pip install --quiet vllm

echo "  ✓ vLLM installed"

# ── 4. Create data directories ────────────────────────────────────────────────
echo ""
echo "[4/5] Creating data directories..."

mkdir -p ~/global_financial_kb/data/{raw,processed,graph,build/{checkpoints,snapshots},briefings}

echo "  ✓ Directories created at ~/global_financial_kb/"

# ── 5. Start vLLM server (background) ────────────────────────────────────────
echo ""
echo "[5/5] Starting vLLM server with Llama-3.3-70B-Instruct (4-bit AWQ)..."
echo "  Note: First run downloads the model (~35GB) — allow 10-15 min"
echo ""

# Check available VRAM
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
echo "  GPU memory: ${GPU_MEM} MiB"

if [ "${GPU_MEM:-0}" -ge 40000 ]; then
    # A6000 48GB or A100 80GB — can run 70B in 4-bit
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    echo "  Using Llama-3.1-70B-Instruct (AWQ 4-bit) — fits in ${GPU_MEM}MiB"
else
    # Smaller GPU — fall back to Mistral 7B
    MODEL="mistralai/Mistral-7B-Instruct-v0.3"
    echo "  GPU < 40GB — falling back to Mistral-7B-Instruct"
fi

# Start vLLM in background, log to file
nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    > ~/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "  vLLM server starting (PID ${VLLM_PID})..."
echo "  Logs: ~/vllm_server.log"
echo "  Monitor: tail -f ~/vllm_server.log"

# Wait for server to be ready (model download + load)
echo ""
echo "  Waiting for vLLM server to be ready (this may take 10-15 min)..."
MAX_WAIT=900  # 15 minutes
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ vLLM server is ready!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    printf "  Waited ${ELAPSED}s...\r"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "  ⚠ vLLM server not ready after ${MAX_WAIT}s"
    echo "  Check logs: tail -50 ~/vllm_server.log"
    echo "  You can start the build with --skip-llm and run LLM phases separately"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Set your API keys:"
echo "     export FRED_API_KEY=b2745248eb3b33033a84db7c91f3a860"
echo "     export LLM_MODE=local"
echo ""
echo "  2. Run the build (from the project directory):"
echo "     cd ~/global_financial_kb"
echo "     python build_initial_kb.py"
echo ""
echo "  3. Or resume if interrupted:"
echo "     python build_initial_kb.py --resume"
echo ""
echo "  4. Download the snapshot when done:"
echo "     rsync -avz --progress \\"
echo "       lambda:~/global_financial_kb/data/build/snapshots/ \\"
echo "       /media/peter/fast-storage/projects/world_knowledge_base/snapshots/"
echo "============================================================"
