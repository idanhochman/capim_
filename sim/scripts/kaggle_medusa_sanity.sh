#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — one-shot setup + MEDUSA sanity run (Kaggle / Colab / any GPU box).
#
# Usage from a notebook cell, AFTER cloning the repo:
#     !git clone https://github.com/idanhochman/capim_.git
#     !bash capim_/sim/scripts/kaggle_medusa_sanity.sh
#
# You can also `source` it. No HuggingFace login needed -- all models are public.
#
# This script:
#   1. installs the python deps for trace collection
#   2. prints GPU + RAM so you can confirm the runtime
#   3. runs the MEDUSA sanity collection (20 built-in prompts, no dataset DL)
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/sim/scripts/) ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "==> repo root: $REPO_ROOT"

# --- python dependencies -----------------------------------------------------
echo "==> installing python deps (this takes 1-2 min) ..."
pip install bitsandbytes "transformers<5.0.0" accelerate datasets sentencepiece protobuf

# --- runtime info ------------------------------------------------------------
echo "==> GPU:";  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "  (no nvidia-smi / no GPU)"
echo "==> RAM:";  free -h

# --- MEDUSA sanity run -------------------------------------------------------
echo "==> running MEDUSA sanity collection (python -u, live output) ..."
python -u sim/scripts/collect_traces.py \
    --model-family vicuna7b --method medusa --sanity --load-in-4bit

echo ""
echo "==> DONE. Inspect the saved sanity trace under traces/ ."
echo "    If it loaded past quantization and has steps, run the 4 real collections:"
echo "      python -u sim/scripts/collect_traces.py --model-family vicuna7b --method eagle  --dataset alpaca --n-prompts 200 --load-in-4bit"
echo "      python -u sim/scripts/collect_traces.py --model-family vicuna7b --method eagle  --dataset gsm8k  --n-prompts 200 --load-in-4bit"
echo "      python -u sim/scripts/collect_traces.py --model-family vicuna7b --method medusa --dataset alpaca --n-prompts 200 --load-in-4bit"
echo "      python -u sim/scripts/collect_traces.py --model-family vicuna7b --method medusa --dataset gsm8k  --n-prompts 200 --load-in-4bit"
