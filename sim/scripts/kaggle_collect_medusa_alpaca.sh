#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — MEDUSA Alpaca trace collection (100 prompts, 8-bit).
#
# Usage from a notebook cell, AFTER cloning the repo (GPU runtime, >= sm_75 e.g.
# Kaggle T4):
#     !git clone https://github.com/idanhochman/capim_.git
#     !bash capim_/sim/scripts/kaggle_collect_medusa_alpaca.sh
#
# 8-bit (LLM.int8): base model int8, Medusa heads kept FP16 (so they load and
# produce real confidence scores). Output: traces/vicuna7b_medusa_alpaca.json
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/sim/scripts/) ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "==> repo root: $REPO_ROOT"

# --- python dependencies (pinned: see kaggle_medusa_sanity.sh for rationale) --
echo "==> installing python deps (this takes 1-2 min) ..."
pip install "transformers==4.36.2" "accelerate==0.25.0" bitsandbytes datasets sentencepiece protobuf

# --- runtime info ------------------------------------------------------------
echo "==> GPU:";  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "  (no nvidia-smi / no GPU)"
echo "==> RAM:";  free -h

# --- collection (100 Alpaca prompts, 8-bit) ----------------------------------
echo "==> MEDUSA on Alpaca (100 prompts, 8-bit) ..."
python -u sim/scripts/collect_traces.py \
    --model-family vicuna7b --method medusa --dataset alpaca --n-prompts 100 --load-in-8bit

echo ""
echo "==> DONE. Saved: traces/vicuna7b_medusa_alpaca.json"
echo "    Sanity-check: log_probs should VARY (not all -ln(vocab)) and"
echo "    mean_acceptance_rate should be non-trivial."
