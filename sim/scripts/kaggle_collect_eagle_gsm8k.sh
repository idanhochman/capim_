#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — EAGLE GSM8K trace collection (100 prompts, 8-bit).
#
# Usage from a notebook cell, AFTER cloning the repo (GPU runtime, >= sm_75 e.g.
# Kaggle T4):
#     !git clone https://github.com/idanhochman/capim_.git
#     !bash capim_/sim/scripts/kaggle_collect_eagle_gsm8k.sh
#
# 8-bit (LLM.int8): base int8, EAGLE draft head stays FP16 (EaModel loads it
# separately from the base, so it is NOT quantized). Same 100 GSM8K prompts as
# the MEDUSA run -> paired comparison. Output: traces/vicuna7b_eagle_gsm8k.json
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/sim/scripts/) ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "==> repo root: $REPO_ROOT"

# --- python dependencies -----------------------------------------------------
# EAGLE pins a DIFFERENT transformers than MEDUSA: the vendored EAGLE modeling
# imports transformers.modeling_rope_utils (added in 4.43), so 4.36.2 fails.
# Target [4.43, 4.49]: >=4.43 for modeling_rope_utils, <4.50 to avoid the loader
# rewrite. Per-method transformers does NOT affect fairness (same Vicuna weights,
# same 8-bit precision, same prompts; traces are independent JSON).
echo "==> installing python deps (this takes 1-2 min) ..."
pip install "transformers==4.46.3" "accelerate==1.0.1" bitsandbytes datasets sentencepiece protobuf

# --- runtime info ------------------------------------------------------------
echo "==> GPU:";  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "  (no nvidia-smi / no GPU)"
echo "==> RAM:";  free -h

# --- collection (100 GSM8K prompts, 8-bit) -----------------------------------
echo "==> EAGLE on GSM8K (100 prompts, 8-bit) ..."
python -u sim/scripts/collect_traces.py \
    --model-family vicuna7b --method eagle --dataset gsm8k --n-prompts 100 --load-in-8bit

echo ""
echo "==> DONE. Saved: traces/vicuna7b_eagle_gsm8k.json"
echo "    Sanity-check: log_probs VARY (not all -ln(vocab)); tree STRUCTURE varies"
echo "    step-to-step (dynamic); acceptance HIGHER than the Medusa GSM8K run and"
echo "    likely higher than EAGLE Alpaca (structured math = higher confidence)."
