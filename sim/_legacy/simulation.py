"""
CAPIM end-to-end simulation loop.

Runs the full CAPIM pipeline on a TraceDataset and returns per-step and
aggregate metrics.

Memory layout
-------------
All weights (7B target + 0.5B draft) and KV-cache live in LPDDR5-PIM banks.
No duplication is needed.  The same physical bank is accessed in two modes:
  HOST mode — NPU reads weights over the external bus (51.2 GB/s)
  PIM mode  — near-bank compute units process locally (409.6 GOPS, compute-bound)

Pipeline per decode step
------------------------
  1. Simulate live confidence-gated pruning  (scheduler.prune_tree)
  2. Drafting phase in PIM mode (0.5B draft model; compute-bound)
  3. Route based on pruned tree size μ vs μ_th  (scheduler.route)
  4a. PIM path (μ < μ_th):
        PIM mode runs FC layers (compute-bound: ~32 ms × μ for 7B model)
        PIM mode runs attention on KV-cache (compute-bound, negligible)
        NPU is NOT fully idle — it still runs the nonlinear glue (t_nl).
  4b. NPU+PIM path (μ ≥ μ_th), SEQUENTIAL at batch=1 (no concurrency):
        NPU reads FC weights from PIM banks in HOST mode (127 ms, BW-bound)
        PIM reads KV-cache for attention in PIM mode (≈4 ms, compute-bound)
        t_verify = t_npu_fc + t_pim_attn + t_nl   (ADDITIVE — see AUDIT.md #1)
        No explicit transfer step — NPU reads directly from PIM banks.
        NB: the LP-Spec baseline IS concurrent (max); CAPIM is not.
  5. Count accepted tokens from the pruned tree
  6. Accumulate step results

Crossover point
---------------
PIM-only is faster than NPU+PIM when:
  32 ms × μ  <  127 ms   →   μ < ~4
Optimal μ_th ≈ 3–4.

Accepted tokens from the pruned tree
--------------------------------------
EAGLE-2 accept_length counts the length of the longest accepted prefix
along the best candidate path.  In the pruned tree, we re-compute this
conservatively: any node that was accepted in the original trace AND
survived pruning counts as accepted.  If pruning removed an accepted node,
we do NOT count it (the pruner discarded a path the target would have taken,
so we lose that speedup).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import sim.hw.npu as npu
import sim.hw.pim as pim
from sim.config.models import ModelConfig
from sim.scheduler import prune_tree, route
from sim.trace.schema import DecodeStepTrace, TokenNode, TraceDataset


@dataclass
class StepResult:
    """Per-step simulation output."""

    step_id: int
    original_tree_size: int
    pruned_tree_size: int         # μ after confidence-gated pruning
    destination: str              # "PIM" or "NPU"

    # Latencies (seconds)
    t_draft_s: float
    t_verify_s: float             # total verification time
    t_total_s: float
    # Verification breakdown (both populated; for PIM path t_npu_fc_s = 0)
    t_npu_fc_s: float             # NPU FC latency (NPU path only; 0 for PIM path)
    t_pim_attn_s: float           # PIM attention latency (both paths)

    # Energies (joules)
    e_draft_j: float
    e_verify_j: float             # total verification energy
    e_total_j: float

    accepted_tokens: int          # tokens accepted from pruned tree (≥ 1 always: the bonus token)
    false_negatives: int          # accepted nodes wrongly pruned

    context_length: int


@dataclass
class SimResult:
    """Aggregate simulation result for a full TraceDataset."""

    # --- Primary metrics ---
    tokens_per_second: float
    energy_per_token_j: float       # Joules/token
    acceptance_rate: float           # fraction of draft tokens (in pruned tree) accepted
    mean_accepted_per_step: float    # avg accepted tokens per decode step
    speedup_vs_ar: float             # vs autoregressive baseline
    energy_reduction_vs_lp: float    # fractional reduction: 1 - (CAPIM/LP-Spec)
    latency_speedup_vs_lp: float     # LP-Spec_latency / CAPIM_latency (>1 is better)

    # --- Routing statistics ---
    pim_fraction: float             # fraction of steps routed to PIM
    npu_fraction: float             # fraction of steps routed to NPU
    mean_pruned_tree_size: float
    mean_original_tree_size: float
    mean_pruning_ratio: float

    # --- Totals ---
    total_latency_s: float
    total_energy_j: float
    total_accepted_tokens: int
    total_steps: int

    # --- Quality ---
    mean_false_neg_rate: float       # fraction of accepted nodes wrongly pruned

    # --- Parameters used ---
    sigma_th: float
    mu_th: int
    scenario: str

    # --- Per-step detail (optional, for analysis) ---
    steps: List[StepResult] = field(default_factory=list)


def simulate_capim(
    trace: TraceDataset,
    target_model: ModelConfig,
    draft_model: ModelConfig,
    sigma_th: float = float("-inf"),
    mu_th: int = 10,
    scenario: str = "default",
    ar_latency_per_token: Optional[float] = None,
    lp_latency_per_token: Optional[float] = None,
    lp_energy_per_token: Optional[float] = None,
    store_steps: bool = False,
) -> SimResult:
    """
    Run the CAPIM simulation on a TraceDataset.

    Args:
        trace: Collected EAGLE-2 trace (TraceDataset).
        target_model: Target model config (e.g. LLAMA2_7B).
        draft_model: Draft model config (e.g. EAGLE_HEAD_LLAMA2_7B).
        sigma_th: Log-probability pruning threshold.
                  float('-inf') = no pruning (EAGLE-2 baseline).
                  Typical range: −5.0 to −0.5.
        mu_th: Tree size threshold for PIM/NPU routing.
               Steps with pruned_tree_size < mu_th go to PIM.
               Steps with pruned_tree_size ≥ mu_th go to NPU.
        scenario: Label for this run (e.g. "gdpval", "gsm8k", "low_power").
        ar_latency_per_token: Pre-computed autoregressive baseline latency.
                              If None, computed from hardware model.
        lp_latency_per_token: Pre-computed LP-Spec baseline latency.
        lp_energy_per_token: Pre-computed LP-Spec baseline energy.
        store_steps: If True, include per-step StepResult list in output.

    Returns:
        SimResult with full aggregate and per-step metrics.
    """
    step_results: List[StepResult] = []

    # Accumulators
    total_latency = 0.0
    total_energy = 0.0
    total_accepted = 0
    total_original_size = 0
    total_pruned_size = 0
    total_draft_nodes = 0
    total_accepted_draft_nodes = 0
    pim_steps = 0
    npu_steps = 0
    total_false_neg = 0
    total_accepted_in_orig = 0

    for step in trace.steps:
        # ---------------------------------------------------------------
        # 1. Simulate live confidence-gated pruning
        # ---------------------------------------------------------------
        pruned_nodes = prune_tree(step, sigma_th)
        mu = len(pruned_nodes)

        original_size = step.tree_size
        total_original_size += original_size

        if mu == 0:
            # All branches pruned — fall back to bonus token (single AR step)
            mu = 1

        total_pruned_size += mu

        # ---------------------------------------------------------------
        # 2. Drafting phase (always in PIM)
        # ---------------------------------------------------------------
        t_draft = pim.draft_latency(draft_model, mu)
        e_draft = pim.draft_energy(draft_model, mu)

        # ---------------------------------------------------------------
        # 3. Routing decision
        # ---------------------------------------------------------------
        dest = route(mu, mu_th)

        # ---------------------------------------------------------------
        # 4. Verification
        # ---------------------------------------------------------------
        if dest == "PIM":
            # PIM-only path: near-bank compute units run full target model.
            # FC layers (compute-bound: ~32 ms × μ) + attention (negligible).
            # NPU stays idle.
            pim_steps += 1
            t_npu_fc = 0.0
            t_pim_fc = pim.verify_latency(target_model, mu)
            t_pim_attn = pim.attn_latency(target_model, mu, step.context_length)
            # FC and attention are sequential within each transformer layer.
            t_verify = t_pim_fc + t_pim_attn
            e_verify = (
                pim.verify_energy(target_model, mu)
                + pim.attn_energy(target_model, step.context_length)
            )
        else:
            # NPU+PIM path, SEQUENTIAL (batch=1, no concurrency — see AUDIT.md #1):
            #   NPU reads FC weights from PIM banks via external bus (HOST mode)
            #   PIM reads KV-cache for attention in PIM mode
            #   No explicit transfer — NPU accesses PIM banks directly.
            # Verify latency is ADDITIVE: t_fc + t_attn + t_nl. This differs from
            # the LP-Spec baseline, which IS concurrent (max). t_nl is the NPU
            # nonlinear cost (softmax/RMSNorm), additive in both routes — modeled
            # as 0 here pending the per-crossing-latency decision (AUDIT.md #2).
            npu_steps += 1
            t_npu_fc = npu.verify_latency(target_model, mu, step.context_length)
            e_npu_fc = npu.verify_energy(target_model, mu, step.context_length)
            t_pim_attn = pim.attn_latency(target_model, mu, step.context_length)
            e_pim_attn = pim.attn_energy(target_model, step.context_length)
            t_nl = 0.0  # TODO(#2): NPU nonlinear cost, additive
            t_verify = t_npu_fc + t_pim_attn + t_nl
            e_verify = e_npu_fc + e_pim_attn

        t_step = t_draft + t_verify
        e_step = e_draft + e_verify

        # ---------------------------------------------------------------
        # 5. Count accepted tokens
        #    Use the original step's accepted_length as the ground truth for
        #    how many tokens were accepted in the full tree.  For the pruned
        #    tree, we conservatively use the number of accepted nodes that
        #    survived pruning, capped at the original accepted_length.
        # ---------------------------------------------------------------
        pruned_accepted = sum(1 for n in pruned_nodes if n.accepted)
        accepted_in_step = min(pruned_accepted, step.accepted_length)
        # Always count at least 1 (the bonus token from the target model)
        accepted_in_step = max(1, accepted_in_step)

        # False negatives: accepted nodes that were pruned
        pruned_ids = set(id(n) for n in pruned_nodes)
        fn = sum(1 for n in step.nodes if n.accepted and id(n) not in pruned_ids)

        total_latency += t_step
        total_energy += e_step
        total_accepted += accepted_in_step
        total_false_neg += fn
        total_accepted_in_orig += sum(1 for n in step.nodes if n.accepted)
        total_draft_nodes += mu
        total_accepted_draft_nodes += pruned_accepted

        sr = StepResult(
            step_id=step.step_id,
            original_tree_size=original_size,
            pruned_tree_size=mu,
            destination=dest,
            t_draft_s=t_draft,
            t_verify_s=t_verify,
            t_total_s=t_step,
            t_npu_fc_s=t_npu_fc if dest == "NPU" else 0.0,
            t_pim_attn_s=t_pim_attn,
            e_draft_j=e_draft,
            e_verify_j=e_verify,
            e_total_j=e_step,
            accepted_tokens=accepted_in_step,
            false_negatives=fn,
            context_length=step.context_length,
        )
        if store_steps:
            step_results.append(sr)

    # ---------------------------------------------------------------
    # 6. Aggregate metrics
    # ---------------------------------------------------------------
    n_steps = len(trace.steps)
    if n_steps == 0:
        raise ValueError("Empty trace — nothing to simulate.")

    mean_accepted = total_accepted / n_steps
    acceptance_rate = (
        total_accepted_draft_nodes / total_draft_nodes
        if total_draft_nodes > 0
        else 0.0
    )

    # Latency per token = total_latency / total_accepted_tokens
    latency_per_token = total_latency / total_accepted if total_accepted > 0 else float("inf")
    energy_per_token = total_energy / total_accepted if total_accepted > 0 else float("inf")
    tps = 1.0 / latency_per_token if latency_per_token > 0 else 0.0

    # Compute AR baseline if not supplied
    if ar_latency_per_token is None:
        mean_ctx = sum(s.context_length for s in trace.steps) / n_steps
        ar_latency_per_token = npu.ar_token_latency(target_model, int(mean_ctx))

    speedup_vs_ar = ar_latency_per_token / latency_per_token if latency_per_token > 0 else 1.0

    # LP-Spec comparison
    if lp_latency_per_token is not None and latency_per_token > 0:
        latency_speedup_vs_lp = lp_latency_per_token / latency_per_token
    else:
        latency_speedup_vs_lp = float("nan")

    if lp_energy_per_token is not None and energy_per_token > 0:
        energy_reduction_vs_lp = 1.0 - (energy_per_token / lp_energy_per_token)
    else:
        energy_reduction_vs_lp = float("nan")

    mean_false_neg_rate = (
        total_false_neg / total_accepted_in_orig
        if total_accepted_in_orig > 0
        else 0.0
    )

    return SimResult(
        tokens_per_second=tps,
        energy_per_token_j=energy_per_token,
        acceptance_rate=acceptance_rate,
        mean_accepted_per_step=mean_accepted,
        speedup_vs_ar=speedup_vs_ar,
        energy_reduction_vs_lp=energy_reduction_vs_lp,
        latency_speedup_vs_lp=latency_speedup_vs_lp,
        pim_fraction=pim_steps / n_steps,
        npu_fraction=npu_steps / n_steps,
        mean_pruned_tree_size=total_pruned_size / n_steps,
        mean_original_tree_size=total_original_size / n_steps,
        mean_pruning_ratio=(
            1.0 - (total_pruned_size / total_original_size)
            if total_original_size > 0
            else 0.0
        ),
        total_latency_s=total_latency,
        total_energy_j=total_energy,
        total_accepted_tokens=total_accepted,
        total_steps=n_steps,
        mean_false_neg_rate=mean_false_neg_rate,
        sigma_th=sigma_th,
        mu_th=mu_th,
        scenario=scenario,
        steps=step_results,
    )
