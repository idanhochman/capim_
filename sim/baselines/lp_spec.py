"""
Baseline 2: LP-Spec Analytical Model.

LP-Spec (Peking University, 2025) is the current state-of-the-art mobile PIM
architecture for LLM inference.  It uses:
  - MEDUSA for speculative decoding (static draft trees)
  - Draft Token Pruner (DTP) based on retrospective acceptance statistics
  - Dynamic scheduling between LPDDR5-PIM and mobile NPU

We model LP-Spec analytically following the same methodology it used itself:
published performance figures + hardware energy constants.

Key published figures from LP-Spec (Table III / Section V):
  - MEDUSA candidate tree size: 63 tokens (mc_sim_7b_63 configuration)
  - After DTP pruning: effective tree size ≈ 40% of original (prunes ~60%)
  - Average acceptance length: ~2.5 tokens/step on Alpaca dataset
  - PIM drafting: MEDUSA heads run as GEMV (batch=1) in PIM

HOW LP-Spec VERIFICATION ACTUALLY WORKS (Section V.B / Fig. 7-8):

LP-Spec does NOT route verification entirely to the NPU.  Instead, it splits
the transformer forward pass across two hardware units that run concurrently:

  1. NPU path — FC / linear-projection layers:
       Reads weight matrices from off-chip DRAM.  Performs the GEMM for Wq,
       Wk, Wv, Wo, and FFN (gate/up/down) across all effective_tree tokens.
       Bottleneck: DRAM bandwidth or NPU compute throughput (whichever is
       higher), following the standard roofline model.

  2. PIM path — attention computation:
       KV-cache is stored in PIM banks (it accumulates there during prefill).
       PIM reads KV internally and computes attention scores in-DRAM, keeping
       data movement off the external bus entirely.
       Bottleneck: PIM internal bandwidth (51.2 TB/s).

These two paths execute SIMULTANEOUSLY.  LP-Spec's DAU (Dynamic Allocation
Unit) adjusts the FC/attention partition to keep T_NPU_FC ≈ T_PIM_attn.

Total verification latency = max(T_NPU_FC, T_PIM_attn)

There is NO explicit activation-transfer step: FC weights live in DRAM
(never in PIM), and the KV-cache stays in PIM throughout.  The two units
operate on disjoint memory regions in parallel.

This is LP-Spec's primary hardware contribution — it differs from naive
"draft in PIM, verify in NPU" models.  Modelling LP-Spec as sequential
NPU-only verification would overstate CAPIM's energy advantage.
"""

from dataclasses import dataclass
from typing import Optional

import sim.hw.npu as npu
import sim.hw.pim as pim
from sim.config.hardware import (
    NPU_ENERGY_PJ_PER_INT8_OP,
    NPU_INT8_TOPS,
    NPU_OFFCHIP_BW,
    OFFCHIP_ENERGY_PJ_PER_BIT,
    PIM_ENERGY_PJ_PER_BIT,
    PIM_INT8_GOPS,
    PIM_INTERNAL_BW,
    pj_to_j,
)
from sim.config.models import ModelConfig
from sim.trace.schema import TraceDataset


# ---------------------------------------------------------------------------
# LP-Spec published constants (from paper, Table III and Section V)
# ---------------------------------------------------------------------------

# MEDUSA candidate tree: mc_sim_7b_63 = 63 draft tokens
MEDUSA_TREE_SIZE: int = 63

# DTP reduces the effective tree size to ~40% of original (conservative)
# LP-Spec reports DTP prunes ~60% of MEDUSA candidates on average
DTP_PRUNING_RATIO: float = 0.60
MEDUSA_EFFECTIVE_TREE_SIZE: int = int(MEDUSA_TREE_SIZE * (1 - DTP_PRUNING_RATIO))  # ~25

# Average acceptance length from LP-Spec (Alpaca dataset, Vicuna-7B)
# We use this as our representative value; actual Qwen2.5 may differ slightly
MEDUSA_ACCEPTANCE_LENGTH: float = 2.5   # draft tokens accepted per step

# MEDUSA heads are extremely lightweight (small linear layers appended to target)
# Their weight footprint is negligible vs the draft model; model as 5% overhead
MEDUSA_HEAD_OVERHEAD_RATIO: float = 0.05  # 5% extra weight over target model per head


@dataclass
class LPSpecResult:
    """Results from LP-Spec baseline simulation."""

    latency_per_step_s: float        # wall-clock time per decode step (seconds)
    energy_per_step_j: float         # energy per decode step (joules)

    latency_per_token_s: float       # latency / acceptance_length
    energy_per_token_j: float        # energy / acceptance_length

    tokens_per_second: float
    acceptance_length: float         # avg tokens generated per step

    # Sub-component latencies
    t_draft_pim_s: float             # MEDUSA drafting in PIM
    t_npu_fc_s: float                # NPU FC path (bottleneck: DRAM BW or NPU compute)
    t_pim_attn_s: float              # PIM attention path (bottleneck: PIM internal BW)
    t_verify_concurrent_s: float     # = max(t_npu_fc_s, t_pim_attn_s)

    # Sub-component energies
    e_draft_pim_j: float             # MEDUSA drafting energy
    e_npu_fc_j: float                # NPU FC path energy (compute + DRAM access)
    e_pim_attn_j: float              # PIM attention energy (internal KV-cache access)


def _concurrent_verify(
    target_model: ModelConfig,
    effective_tree: int,
    seq_len: int,
) -> tuple:
    """
    Compute LP-Spec's concurrent PIM+NPU verification cost.

    NPU path: reads all FC/projection weights from off-chip DRAM, runs matrix
    multiplications for effective_tree tokens.  Roofline bottleneck is
    max(DRAM bandwidth, NPU compute throughput).

    PIM path: reads KV-cache from PIM banks (internally), runs attention for
    effective_tree query tokens.  Bottleneck is PIM internal bandwidth.

    Both paths run simultaneously; total time = max(T_NPU_FC, T_PIM_attn).

    Returns:
        (t_npu_fc, t_pim_attn, t_verify, e_npu_fc, e_pim_attn)
    """
    # ------------------------------------------------------------------
    # NPU FC path: weight matrix reads from DRAM + matrix multiplications
    # ------------------------------------------------------------------
    fc_weight_bytes = target_model.weight_bytes()
    fc_params = fc_weight_bytes / target_model.word_size_bytes
    fc_flops = 2 * fc_params * effective_tree   # each weight × effective_tree queries

    t_npu_bw = fc_weight_bytes / NPU_OFFCHIP_BW
    t_npu_compute = fc_flops / NPU_INT8_TOPS
    t_npu_fc = max(t_npu_bw, t_npu_compute)    # roofline

    e_npu_compute_pj = fc_flops * NPU_ENERGY_PJ_PER_INT8_OP
    e_npu_mem_pj = fc_weight_bytes * 8 * OFFCHIP_ENERGY_PJ_PER_BIT
    e_npu_fc = pj_to_j(e_npu_compute_pj + e_npu_mem_pj)

    # ------------------------------------------------------------------
    # PIM attention path: KV-cache accessed in PIM mode (compute-bound)
    # Arithmetic intensity = n_heads × batch / n_kv_heads >> PIM ridge.
    # ------------------------------------------------------------------
    t_pim_attn = pim.attn_latency(target_model, effective_tree, seq_len)
    e_pim_attn = pim.attn_energy(target_model, seq_len)

    # ------------------------------------------------------------------
    # Concurrent execution: total time = max of the two parallel paths
    # Energy: both units consume power simultaneously → sum the energies
    # ------------------------------------------------------------------
    t_verify = max(t_npu_fc, t_pim_attn)

    return t_npu_fc, t_pim_attn, t_verify, e_npu_fc, e_pim_attn


def simulate_lp_spec(
    target_model: ModelConfig,
    mean_context_length: float = 512.0,
    medusa_tree_size: int = MEDUSA_TREE_SIZE,
    dtp_pruning_ratio: float = DTP_PRUNING_RATIO,
    acceptance_length: float = MEDUSA_ACCEPTANCE_LENGTH,
) -> LPSpecResult:
    """
    Simulate LP-Spec's per-step latency and energy analytically.

    LP-Spec pipeline per decode step:
      1. Draft in PIM: MEDUSA heads generate medusa_tree_size candidates (batch=1
         GEMV, entirely in PIM).  Each candidate requires one MEDUSA-head pass.
      2. DTP pruning: retrospective, negligible latency (folded into drafting).
      3. Concurrent verification (the key LP-Spec contribution):
           - NPU: FC/projection layers over effective_tree tokens, reading model
             weights from off-chip DRAM.
           - PIM: attention computation over effective_tree queries, reading
             KV-cache from PIM banks.
           - Both run simultaneously; t_verify = max(T_NPU_FC, T_PIM_attn).

    Args:
        target_model: Target model (typically QWEN2_5_7B).
        mean_context_length: Representative KV-cache length.
        medusa_tree_size: MEDUSA candidate count before DTP (default 63).
        dtp_pruning_ratio: Fraction pruned by DTP (default 0.60).
        acceptance_length: Average tokens accepted per step.

    Returns:
        LPSpecResult with full breakdown.
    """
    effective_tree = max(1, int(medusa_tree_size * (1 - dtp_pruning_ratio)))

    # 1. Draft in PIM: MEDUSA heads (≈5% of target weight, batch=1 per candidate)
    #    Roofline: compute-bound (same analysis as EAGLE-2 draft model).
    medusa_weight_bytes = target_model.weight_bytes() * MEDUSA_HEAD_OVERHEAD_RATIO
    medusa_params = medusa_weight_bytes / target_model.word_size_bytes
    bw_lat = (medusa_weight_bytes * medusa_tree_size) / PIM_INTERNAL_BW
    compute_lat = (2.0 * medusa_params * medusa_tree_size) / PIM_INT8_GOPS
    t_draft = max(bw_lat, compute_lat)
    e_draft_pj = medusa_weight_bytes * medusa_tree_size * PIM_ENERGY_PJ_PER_BIT / 8.0
    e_draft = pj_to_j(e_draft_pj)

    # 2. Concurrent verification: NPU FC + PIM attention (LP-Spec Section V.B)
    t_npu_fc, t_pim_attn, t_verify, e_npu_fc, e_pim_attn = _concurrent_verify(
        target_model,
        effective_tree=effective_tree,
        seq_len=int(mean_context_length),
    )

    # Step totals (draft is sequential with verification)
    t_step = t_draft + t_verify
    e_step = e_draft + e_npu_fc + e_pim_attn

    al = max(1.0, acceptance_length)
    t_token = t_step / al
    e_token = e_step / al

    return LPSpecResult(
        latency_per_step_s=t_step,
        energy_per_step_j=e_step,
        latency_per_token_s=t_token,
        energy_per_token_j=e_token,
        tokens_per_second=1.0 / t_token if t_token > 0 else float("inf"),
        acceptance_length=al,
        t_draft_pim_s=t_draft,
        t_npu_fc_s=t_npu_fc,
        t_pim_attn_s=t_pim_attn,
        t_verify_concurrent_s=t_verify,
        e_draft_pim_j=e_draft,
        e_npu_fc_j=e_npu_fc,
        e_pim_attn_j=e_pim_attn,
    )


def simulate_lp_spec_from_trace(
    target_model: ModelConfig,
    trace: TraceDataset,
    medusa_tree_size: int = MEDUSA_TREE_SIZE,
    dtp_pruning_ratio: float = DTP_PRUNING_RATIO,
    acceptance_length: float = MEDUSA_ACCEPTANCE_LENGTH,
) -> LPSpecResult:
    """
    Simulate LP-Spec using the same context-length distribution as the trace.

    Ensures a fair comparison with CAPIM (identical evaluation conditions).
    """
    if not trace.steps:
        raise ValueError("Empty trace dataset.")

    mean_ctx = sum(s.context_length for s in trace.steps) / len(trace.steps)

    return simulate_lp_spec(
        target_model,
        mean_context_length=mean_ctx,
        medusa_tree_size=medusa_tree_size,
        dtp_pruning_ratio=dtp_pruning_ratio,
        acceptance_length=acceptance_length,
    )
