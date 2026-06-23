#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CAPIM — one-shot setup + EAGLE sanity run (Kaggle / any GPU box >= sm_75).
#
# Usage from a notebook cell, AFTER cloning the repo:
#     !git clone https://github.com/idanhochman/capim_.git
#     !bash capim_/sim/scripts/kaggle_eagle_sanity.sh
#
# First-ever EAGLE run in this stack (transformers 4.36.2 + 8-bit base + single
# GPU). Make-or-break checks after it runs:
#   * len(steps) > 0
#   * log_probs VARY (not all -ln(vocab)) -> draft head loaded (it's FP16, not
#     quantized: EaModel loads it separately from the base)
#   * acceptance non-trivial, and HIGHER than Medusa with a DYNAMIC tree
#     (variable tree_size, unlike Medusa's constant 63)
#
# 8-bit (LLM.int8): base int8, EAGLE draft head stays FP16. No HF login needed.
# Output: traces/vicuna7b_eagle_sanity.json
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the repo root (this script lives in <repo>/sim/scripts/) ---------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "==> repo root: $REPO_ROOT"

# --- python dependencies -----------------------------------------------------
# NOTE: EAGLE pins a DIFFERENT transformers than MEDUSA. The vendored EAGLE
# modeling imports transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS, added
# in transformers 4.43 -> 4.36.2 (used for MEDUSA) fails with ModuleNotFoundError.
# We target the [4.43, 4.49] window: >=4.43 for modeling_rope_utils, <4.50 to
# avoid the loader rewrite that broke MEDUSA under quant. EAGLE and MEDUSA are
# separate collection runs producing independent JSON traces, so a per-method
# transformers version does NOT affect comparison fairness (same Vicuna weights,
# same 8-bit precision, same prompts).
echo "==> installing python deps (this takes 1-2 min) ..."
pip install "transformers==4.46.3" "accelerate==1.0.1" bitsandbytes datasets sentencepiece protobuf

# --- runtime info ------------------------------------------------------------
echo "==> GPU:";  nvidia-smi --query-gpu=name,memory.total --format=csv || echo "  (no nvidia-smi / no GPU)"
echo "==> RAM:";  free -h

# --- EAGLE sanity run --------------------------------------------------------
echo "==> running EAGLE sanity collection (python -u, live output) ..."
python -u sim/scripts/collect_traces.py \
    --model-family vicuna7b --method eagle --sanity --load-in-8bit

echo ""
echo "==> DONE. Inspect traces/vicuna7b_eagle_sanity.json :"
echo "    * steps > 0 and log_probs VARY (not all -ln(vocab))"
echo "    * acceptance non-trivial; tree_size should VARY (dynamic tree)"
echo "    If green, run the real collections (8-bit, same as Medusa):"
echo "      python -u sim/scripts/collect_traces.py --model-family vicuna7b --method eagle --dataset alpaca --n-prompts 100 --load-in-8bit"
echo "      python -u sim/scripts/collect_traces.py --model-family vicuna7b --method eagle --dataset gsm8k  --n-prompts 100 --load-in-8bit"
