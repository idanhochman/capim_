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
# IMPORTANT runtime requirements (learned the hard way):
#   * GPU must be >= sm_75 (Turing). A Kaggle P100 is sm_60 and is NOT supported
#     by Kaggle's PyTorch or by bitsandbytes 4-bit -> use "GPU T4 x2" instead.
#   * transformers MUST be pinned < 4.50. The vendored EAGLE/Medusa modeling code
#     is incompatible with the >=4.50 weight-loader rewrite (it throws
#     "AttributeError: `weight` is not an nn.Module" under 4-bit quantization).
#     4.36.2 is the Medusa/EAGLE-era known-good version; accelerate is pinned to
#     a matching release so the old transformers imports cleanly.
echo "==> installing python deps (this takes 1-2 min) ..."
pip install "transformers==4.36.2" "accelerate==0.25.0" bitsandbytes datasets sentencepiece protobuf

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
