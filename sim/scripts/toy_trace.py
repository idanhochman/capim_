"""
Toy trace: hand-crafted EAGLE-2 draft trees starting from the token "hi".

Purpose
-------
This script builds a small, human-readable TraceDataset that mirrors exactly
what the real collector.py would produce from running EAGLE-2 on Qwen2.5-7B.
Use it to:
  - Understand the trace format before real GPU traces arrive.
  - Step through the simulator by hand and verify the numbers make sense.
  - See how sigma_th pruning and mu_th routing interact on a concrete example.

Two decode steps are modelled:

  Step 0  context = "hi"  (1 token in KV-cache)
  Step 1  context = "hi there !"  (3 tokens after step 0's accepted tokens)

Run:
    python3 sim/scripts/toy_trace.py
"""

import json
import os
import sys

# Allow running from the project root or from sim/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sim.config.models import QWEN2_5_7B, QWEN2_5_0_5B
from sim.trace.schema import TokenNode, DecodeStepTrace, TraceDataset
from sim.simulation import simulate_capim
from sim.baselines.autoregressive import simulate_autoregressive_from_trace
from sim.baselines.lp_spec import simulate_lp_spec_from_trace
from sim.results import compare_results

# ---------------------------------------------------------------------------
# Token ID → human-readable string (Qwen2.5 tokenizer approximations)
# These IDs are illustrative; the simulation uses only log_prob and accepted.
# ---------------------------------------------------------------------------
TOKEN_STRINGS = {
    9945:  '"hi"',
    1052:  '" there"',
    0:     '"!"',
    11:    '","',
    2585:  '" How"',
    14436: '" everyone"',
    526:   '" are"',
    3555:  '" What"',
    358:   '" I"',
    481:   '" can"',
    265:   '" do"',
}


def tok(token_id: int) -> str:
    return TOKEN_STRINGS.get(token_id, f"<id={token_id}>")


# ---------------------------------------------------------------------------
# Step 0: context = "hi"
#
# The draft model (0.5B) proposes a tree of continuations after "hi".
# The target model (7B) accepts the prefix: " there" → "!"
# accepted_length = 2
#
# Tree layout (ASCII):
#
#   ROOT ("hi")
#   ├── d0·0  " there"    log=-0.40  cumul=-0.40  ← ACCEPTED
#   │   ├── d1·0  "!"     log=-0.60  cumul=-1.00  ← ACCEPTED
#   │   │   └── d2·0  " How"  log=-1.20  cumul=-2.20
#   │   └── d1·1  ","     log=-1.80  cumul=-2.20
#   ├── d0·1  "!"         log=-1.10  cumul=-1.10
#   │   └── d1·2  " How"  log=-1.90  cumul=-3.00
#   └── d0·2  " everyone" log=-2.40  cumul=-2.40   ← low confidence
#       └── d1·3  "!"     log=-2.90  cumul=-5.30
#
# parent_idx refers to the index of the parent within its own depth layer.
# Depth-0 nodes have parent_idx = -1 (parent is the last accepted token).
# ---------------------------------------------------------------------------

step0_nodes = [
    # ---- depth 0 ----
    TokenNode(depth=0, token_id=1052,  log_prob=-0.40, cumulative_log_prob=-0.40,
              parent_idx=-1, accepted=True,  layer_idx=0),  # " there"
    TokenNode(depth=0, token_id=0,     log_prob=-1.10, cumulative_log_prob=-1.10,
              parent_idx=-1, accepted=False, layer_idx=1),  # "!"
    TokenNode(depth=0, token_id=14436, log_prob=-2.40, cumulative_log_prob=-2.40,
              parent_idx=-1, accepted=False, layer_idx=2),  # " everyone"

    # ---- depth 1 ----
    #  children of d0·0 (" there")
    TokenNode(depth=1, token_id=0,    log_prob=-0.60, cumulative_log_prob=-1.00,
              parent_idx=0, accepted=True,  layer_idx=0),  # "!"
    TokenNode(depth=1, token_id=11,   log_prob=-1.80, cumulative_log_prob=-2.20,
              parent_idx=0, accepted=False, layer_idx=1),  # ","
    #  child of d0·1 ("!")
    TokenNode(depth=1, token_id=2585, log_prob=-1.90, cumulative_log_prob=-3.00,
              parent_idx=1, accepted=False, layer_idx=2),  # " How"
    #  child of d0·2 (" everyone")
    TokenNode(depth=1, token_id=0,    log_prob=-2.90, cumulative_log_prob=-5.30,
              parent_idx=2, accepted=False, layer_idx=3),  # "!"

    # ---- depth 2 ----
    #  child of d1·0 ("!" child of " there")
    TokenNode(depth=2, token_id=2585, log_prob=-1.20, cumulative_log_prob=-2.20,
              parent_idx=0, accepted=False, layer_idx=0),  # " How"
]

step0 = DecodeStepTrace(
    step_id=0,
    context_length=1,          # "hi" = 1 token in KV-cache
    nodes=step0_nodes,
    accepted_length=2,          # target accepted " there" + "!"
    dataset="toy",
    prompt_id=0,
)

# ---------------------------------------------------------------------------
# Step 1: context = "hi there !" (3 tokens)
#
# The accepted tokens from step 0 (" there", "!") are now in the KV-cache.
# The draft model now proposes continuations after "!".
# The target model accepts " How" → " are"
# accepted_length = 2
#
# Tree layout:
#
#   ROOT ("!")
#   ├── d0·0  " How"    log=-0.70  cumul=-0.70  ← ACCEPTED
#   │   ├── d1·0  " are"  log=-0.90  cumul=-1.60  ← ACCEPTED
#   │   └── d1·1  " can"  log=-1.50  cumul=-2.20
#   ├── d0·1  " What"   log=-1.30  cumul=-1.30
#   │   └── d1·2  " do"  log=-1.80  cumul=-3.10
#   └── d0·2  " I"      log=-2.60  cumul=-2.60   ← low confidence
# ---------------------------------------------------------------------------

step1_nodes = [
    # ---- depth 0 ----
    TokenNode(depth=0, token_id=2585, log_prob=-0.70, cumulative_log_prob=-0.70,
              parent_idx=-1, accepted=True,  layer_idx=0),  # " How"
    TokenNode(depth=0, token_id=3555, log_prob=-1.30, cumulative_log_prob=-1.30,
              parent_idx=-1, accepted=False, layer_idx=1),  # " What"
    TokenNode(depth=0, token_id=358,  log_prob=-2.60, cumulative_log_prob=-2.60,
              parent_idx=-1, accepted=False, layer_idx=2),  # " I"

    # ---- depth 1 ----
    #  children of d0·0 (" How")
    TokenNode(depth=1, token_id=526,  log_prob=-0.90, cumulative_log_prob=-1.60,
              parent_idx=0, accepted=True,  layer_idx=0),  # " are"
    TokenNode(depth=1, token_id=481,  log_prob=-1.50, cumulative_log_prob=-2.20,
              parent_idx=0, accepted=False, layer_idx=1),  # " can"
    #  child of d0·1 (" What")
    TokenNode(depth=1, token_id=265,  log_prob=-1.80, cumulative_log_prob=-3.10,
              parent_idx=1, accepted=False, layer_idx=2),  # " do"
    # d0·2 (" I") has no depth-1 children — draft stopped here (low confidence)
]

step1 = DecodeStepTrace(
    step_id=1,
    context_length=3,          # "hi there !" = 3 tokens
    nodes=step1_nodes,
    accepted_length=2,          # target accepted " How" + " are"
    dataset="toy",
    prompt_id=0,
)

# ---------------------------------------------------------------------------
# Assemble the TraceDataset
# ---------------------------------------------------------------------------

trace = TraceDataset(
    steps=[step0, step1],
    model_target="Qwen2.5-7B",
    model_draft="Qwen2.5-0.5B",
    metadata={"synthetic": False, "note": "hand-crafted toy trace from 'hi'"},
)
trace.compute_summary()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_tree(step: DecodeStepTrace, sigma_th: float = float("-inf")) -> None:
    """Print the draft tree with pruning annotations."""
    from sim.scheduler import prune_tree
    pruned = {id(n) for n in prune_tree(step, sigma_th)}

    # Group nodes by depth
    max_d = step.max_depth
    by_depth: dict[int, list[TokenNode]] = {d: [] for d in range(max_d + 1)}
    for n in step.nodes:
        by_depth[n.depth].append(n)
    for d in by_depth:
        by_depth[d].sort(key=lambda n: n.layer_idx)

    def node_label(n: TokenNode) -> str:
        kept    = "✓ kept  " if id(n) in pruned else "✗ pruned"
        acc     = " [ACCEPTED]" if n.accepted else ""
        return (f"d{n.depth}·{n.layer_idx}  {tok(n.token_id):<14}"
                f"  log={n.log_prob:+.2f}  cumul={n.cumulative_log_prob:+.2f}"
                f"  {kept}{acc}")

    print(f"\n  Step {step.step_id}  |  context_length={step.context_length}"
          f"  |  tree_size={step.tree_size}  |  accepted_length={step.accepted_length}")
    print(f"  σ_th={sigma_th}  →  "
          f"{sum(1 for n in step.nodes if id(n) in pruned)} nodes survive pruning\n")

    # Print each depth level
    for d in range(max_d + 1):
        indent = "  " + "    " * d + ("└── " if d > 0 else "")
        for n in by_depth[d]:
            prefix = "  " + "    " * d
            connector = "├── " if n.layer_idx < len(by_depth[d]) - 1 else "└── "
            print(f"  {prefix}{connector}{node_label(n)}")
    print()


def print_separator(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# 1. Show the full trees (no pruning)
# ---------------------------------------------------------------------------

print_separator("TOY TRACE — Full draft trees (no pruning, σ_th = -∞)")
for step in trace.steps:
    print_tree(step, sigma_th=float("-inf"))

print(f"  Summary: {trace.mean_tree_size:.1f} nodes/step avg, "
      f"{trace.mean_accepted_length:.1f} accepted tokens/step avg\n")


# ---------------------------------------------------------------------------
# 2. Show the effect of σ_th pruning on each step
# ---------------------------------------------------------------------------

print_separator("PRUNING DEMO — same trees at different σ_th thresholds")
for sigma in [-2.0, -1.5, -1.0]:
    print(f"\n  ── σ_th = {sigma} ──")
    for step in trace.steps:
        print_tree(step, sigma_th=sigma)


# ---------------------------------------------------------------------------
# 3. Run the full CAPIM simulation at several (σ_th, μ_th) combinations
# ---------------------------------------------------------------------------

print_separator("SIMULATION — CAPIM at different (σ_th, μ_th) settings")

ar  = simulate_autoregressive_from_trace(QWEN2_5_7B, trace)
lp  = simulate_lp_spec_from_trace(QWEN2_5_7B, trace)

configs = [
    ("no pruning,  PIM-heavy", float("-inf"), 8),
    ("no pruning,  NPU-heavy", float("-inf"), 2),
    ("σ_th=-2.0,   μ_th=4",   -2.0,          4),
    ("σ_th=-1.5,   μ_th=4",   -1.5,          4),
    ("σ_th=-1.0,   μ_th=4",   -1.0,          4),
]

rows = []
for label, sigma, mu in configs:
    r = simulate_capim(
        trace=trace,
        target_model=QWEN2_5_7B,
        draft_model=QWEN2_5_0_5B,
        sigma_th=sigma,
        mu_th=mu,
        scenario="toy",
        ar_latency_per_token=ar.latency_per_token_s,
        lp_latency_per_token=lp.latency_per_token_s,
        lp_energy_per_token=lp.energy_per_token_j,
        store_steps=True,
    )
    rows.append((label, r))

# Header
hdr = f"  {'Config':<30} {'tok/s':>7} {'mJ/tok':>8} {'speedup_AR':>10} {'PIM%':>6} {'μ avg':>6}"
print("\n" + hdr)
print("  " + "-" * (len(hdr) - 2))
for label, r in rows:
    print(f"  {label:<30} {r.tokens_per_second:>7.2f} "
          f"{r.energy_per_token_j*1e3:>8.3f} "
          f"{r.speedup_vs_ar:>10.2f}x "
          f"{r.pim_fraction*100:>5.0f}% "
          f"{r.mean_pruned_tree_size:>6.1f}")

print(f"\n  {'AR baseline':<30} {ar.tokens_per_second:>7.2f} {ar.energy_per_token_j*1e3:>8.3f}")
print(f"  {'LP-Spec baseline':<30} {lp.tokens_per_second:>7.2f} {lp.energy_per_token_j*1e3:>8.3f}")


# ---------------------------------------------------------------------------
# 4. Step-by-step breakdown for one config
# ---------------------------------------------------------------------------

print_separator("STEP-BY-STEP BREAKDOWN — σ_th=-2.0, μ_th=4")

_, r_detail = next((x for x in rows if x[0] == "σ_th=-2.0,   μ_th=4"), rows[2])
for sr in r_detail.steps:
    print(f"\n  Step {sr.step_id}:")
    print(f"    original_tree_size = {sr.original_tree_size}  →  pruned μ = {sr.pruned_tree_size}")
    print(f"    destination        = {sr.destination}")
    print(f"    t_draft            = {sr.t_draft_s*1e3:.2f} ms")
    print(f"    t_verify           = {sr.t_verify_s*1e3:.2f} ms"
          f"  (NPU_FC={sr.t_npu_fc_s*1e3:.2f}ms, PIM_attn={sr.t_pim_attn_s*1e3:.2f}ms)")
    print(f"    t_total            = {sr.t_total_s*1e3:.2f} ms")
    print(f"    e_total            = {sr.e_total_j*1e3:.3f} mJ")
    print(f"    accepted_tokens    = {sr.accepted_tokens}")
    print(f"    false_negatives    = {sr.false_negatives}  "
          f"(accepted nodes wrongly pruned)")


# ---------------------------------------------------------------------------
# 5. Save the trace to JSON
# ---------------------------------------------------------------------------

out_dir = os.path.join(os.path.dirname(__file__), "../../traces")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "toy_hi.json")
trace.save(out_path)

print_separator("SAVED")
print(f"  Trace written to: {out_path}")
print(f"  Load it with:     TraceDataset.load('{out_path}')\n")
