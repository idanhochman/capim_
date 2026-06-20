# CAPIM Trace Collection

This directory contains two scripts:

| Script | Purpose |
|---|---|
| `collect_traces.py` | Run instrumented EAGLE-2 inference and save a `TraceDataset` JSON |
| `run_simulation.py` | Load a saved trace and run the CAPIM simulation against both baselines |

Traces are the bridge between the GPU-intensive inference step (which requires a real model) and the CPU-only simulation (which runs in seconds and can be swept over many parameter combinations).

---

## Prerequisites

### Hardware

| Configuration | VRAM | Notes |
|---|---|---|
| Full FP16 | ≥ 16 GB | RTX 3090/4090, A100, V100 |
| 8-bit quantization | ≥ 12 GB | Slightly slower; add `--load-in-8bit` |
| 4-bit quantization | ≥ 10 GB | Faster than 8-bit; add `--load-in-4bit` |

If you don't have a local GPU, options include KCL's HPC cluster, Google Colab Pro (A100), or a cloud GPU instance (RunPod, Lambda).

Disk space: ~20 GB for models, ~500 MB for traces.

### Python environment

```bash
pip install -r sim/scripts/requirements_collection.txt
```

If your GPU has less than 16 GB VRAM, also install bitsandbytes for quantization:

```bash
pip install bitsandbytes>=0.43.0
```

---

## Models

Both models are downloaded automatically from HuggingFace on first use. If you want to pre-download them manually:

```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
huggingface-cli download yuhuili/EAGLE-llama2-chat-7B
```

| Role | Model | Size |
|---|---|---|
| Target | `meta-llama/Llama-2-7b-chat-hf` | ~15 GB (FP16) |
| Draft (EAGLE-2) | `yuhuili/EAGLE-llama2-chat-7B` | ~1 GB |

If HuggingFace downloads are slow from your location:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Datasets

Two datasets are used, chosen to cover different confidence regimes and to match LP-Spec's evaluation conditions:

| Dataset | HuggingFace ID | Purpose |
|---|---|---|
| Alpaca | `tatsu-lab/alpaca` | General instruction-following; same dataset used by LP-Spec — ensures a fair baseline comparison |
| GSM8K | `openai/gsm8k` | Math word problems; structured reasoning with higher draft-model confidence |

Both datasets download automatically via the `datasets` library. No manual setup required.

> **Note:** An earlier version of this project listed "GDPval" as an evaluation dataset. This was an error — LP-Spec (our baseline) uses Alpaca, not GDPval. Using Alpaca ensures the comparison is on identical conditions.

---

## Step-by-step

### 1. Verify the pipeline (no GPU time wasted)

Run the dry-run before committing to a full collection. This loads the model, runs 2 prompts × 20 tokens, and saves a tiny trace file to confirm everything is wired up correctly:

```bash
cd /home/idanh/msc/capim
python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 3 --dry-run
```

Expected output:
```
=== Step 1: Loading alpaca prompts ===
Loaded 3 Alpaca prompts

=== Step 2: Loading EAGLE-2 model ===
Model loaded successfully

=== Step 3: Collecting traces (2 prompts) ===
  [1/2] Prompt: Give three tips for staying healthy...
    Collected 8 steps (total so far: 8)
  [2/2] Prompt: What are the three primary colors?...
    Collected 6 steps (total so far: 14)

=== Step 4: Saving trace ===
Saved 14 steps to traces/llama2_alpaca.json

Summary:
  Mean tree size        : 19.3 tokens
  Mean accepted/step    : 2.7 tokens
  Mean acceptance rate  : 14.0%
```

### 2. Collect Alpaca traces

```bash
python sim/scripts/collect_traces.py \
    --dataset alpaca \
    --n-prompts 200 \
    --max-new-tokens 200
```

With a 16 GB GPU this takes roughly 30–60 minutes. Output: `traces/llama2_alpaca.json`.

### 3. Collect GSM8K traces

```bash
python sim/scripts/collect_traces.py \
    --dataset gsm8k \
    --n-prompts 200 \
    --max-new-tokens 300
```

GSM8K answers are longer, so `--max-new-tokens 300` is recommended. Output: `traces/llama2_gsm8k.json`.

### 4. (Optional) Use quantization if VRAM is tight

```bash
# 4-bit: ~10 GB VRAM, slightly slower inference
python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 200 --load-in-4bit

# 8-bit: ~12 GB VRAM
python sim/scripts/collect_traces.py --dataset alpaca --n-prompts 200 --load-in-8bit
```

### 5. Use local model paths (if pre-downloaded)

```bash
python sim/scripts/collect_traces.py \
    --dataset alpaca \
    --base-model /path/to/Llama-2-7b-chat-hf \
    --ea-model   /path/to/EAGLE2-Llama-2-7b-chat-hf \
    --n-prompts 200
```

---

## Running the simulation

Once you have trace files, the simulation runs entirely on CPU:

```bash
# Single evaluation point
python sim/scripts/run_simulation.py --trace traces/llama2_alpaca.json

# With specific thresholds
python sim/scripts/run_simulation.py \
    --trace traces/llama2_alpaca.json \
    --sigma-th -2.0 \
    --mu-th 10

# Sweep σ_th to find the optimal pruning threshold
python sim/scripts/run_simulation.py --trace traces/llama2_alpaca.json --sweep sigma

# 2D grid search over (σ_th, μ_th)
python sim/scripts/run_simulation.py --trace traces/llama2_alpaca.json --sweep joint

# Save sensitivity plots (requires matplotlib)
python sim/scripts/run_simulation.py --trace traces/llama2_alpaca.json --sweep sigma --plot
```

Results are saved to `results/` as CSV files.

---

## Full argument reference

### `collect_traces.py`

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `alpaca` | `alpaca` or `gsm8k` |
| `--n-prompts` | `200` | Number of prompts to run |
| `--base-model` | `meta-llama/Llama-2-7b-chat-hf` | HuggingFace ID or local path |
| `--ea-model` | `yuhuili/EAGLE-llama2-chat-7B` | HuggingFace ID or local path |
| `--output-dir` | `traces` | Directory to save the JSON file |
| `--max-new-tokens` | `200` | Max tokens to generate per prompt |
| `--load-in-4bit` | off | 4-bit quantization (~10 GB VRAM) |
| `--load-in-8bit` | off | 8-bit quantization (~12 GB VRAM) |
| `--dry-run` | off | Run 2 prompts × 20 tokens to verify setup |

### `run_simulation.py`

| Argument | Default | Description |
|---|---|---|
| `--trace` | required | Path to trace JSON |
| `--sigma-th` | `-2.0` | Log-prob pruning threshold |
| `--mu-th` | `10` | Tree size routing threshold |
| `--sweep` | `none` | `none`, `sigma`, `mu`, or `joint` |
| `--output-dir` | `results` | Directory for CSV output |
| `--plot` | off | Save sensitivity plots (requires matplotlib) |

---

## Trace file format

Traces are saved as JSON and can be loaded with:

```python
from sim.trace.schema import TraceDataset
trace = TraceDataset.load("traces/llama2_alpaca.json")

print(f"{len(trace.steps)} decode steps")
print(f"Mean tree size: {trace.mean_tree_size:.1f} tokens")
print(f"Mean acceptance rate: {trace.mean_acceptance_rate:.1%}")

# Each step contains all draft nodes with confidence scores
step = trace.steps[0]
for node in step.nodes:
    print(f"  depth={node.depth} log_prob={node.log_prob:.3f} accepted={node.accepted}")
```

Each `TokenNode` stores:
- `depth` — position in the draft tree (0 = first speculative token)
- `log_prob` — per-token log-softmax probability (the signal used for σ_th pruning)
- `cumulative_log_prob` — sum of log_probs from root to this node
- `accepted` — whether the target model accepted this token
