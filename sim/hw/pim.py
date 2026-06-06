"""
LPDDR5-PIM analytical latency and energy model.

Memory layout
-------------
All model weights (7B target + 0.5B draft) and the KV-cache are stored in
PIM banks.  The same physical bank can be accessed in two modes:

  HOST mode  — the NPU (or CPU) reads data over the external I/O bus
               (PIM_EXTERNAL_BW = 51.2 GB/s).  Identical to ordinary DRAM.
  PIM mode   — near-bank compute units process data locally, without moving
               anything across the external bus.  Uses internal bandwidth
               (PIM_INTERNAL_BW = 51.2 TB/s) but is limited by compute
               throughput (PIM_INT8_GOPS = 409.6 GOPS).

No weight duplication is required.  When CAPIM routes a step to the NPU,
the NPU reads weights from PIM banks in HOST mode.  When CAPIM routes to
PIM-only, the near-bank compute units process those same weights in PIM mode.

Roofline for PIM operations
----------------------------
Ridge point = PIM_INT8_GOPS / PIM_INTERNAL_BW
            = 409.6e9 / 51.2e12  ≈  0.008 ops/byte

GEMV arithmetic intensity = 2 FLOPs/byte (INT8)  >>  0.008

Therefore ALL GEMV/GEMM operations on PIM are COMPUTE-BOUND, not
bandwidth-bound.  Latency = max(bw_time, compute_time), where compute_time
dominates for any practical matrix size.

For attention (KV-cache reads on PIM):
  Intensity = n_heads × batch_size / n_kv_heads  >>  0.008  →  compute-bound.

Energy
------
Energy is dominated by DRAM cell reads, not the near-bank ALU (which is
extremely efficient by design).  Both latency formulas use compute-bound
timing; energy formulas use memory-access cost (PIM_ENERGY_PJ_PER_BIT).
"""

from math import ceil

from sim.config.hardware import (
    OFFCHIP_ENERGY_PJ_PER_BIT,
    PIM_ENERGY_PJ_PER_BIT,
    PIM_EXTERNAL_BW,
    PIM_INT8_GOPS,
    PIM_INTERNAL_BW,
    PIM_NALU,
    pj_to_j,
)
from sim.config.models import ModelConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _roofline(bw_bytes: float, flops: float) -> float:
    """Return roofline latency: max(bandwidth time, compute time)."""
    bw_lat = bw_bytes / PIM_INTERNAL_BW
    compute_lat = flops / PIM_INT8_GOPS
    return max(bw_lat, compute_lat)


# ---------------------------------------------------------------------------
# Drafting phase — draft model runs entirely in PIM mode
# ---------------------------------------------------------------------------

def draft_latency(model: ModelConfig, tree_size: int) -> float:
    """
    Latency (seconds) for the drafting phase in PIM mode.

    The draft model generates tree_size tokens.  Each token requires one
    full forward pass (GEMV at batch=1 per depth level; depth-parallel
    execution means the dominant cost is compute, not memory).

    Roofline: compute-bound for any practical model size.
      bw_time     = weight_bytes × tree_size / PIM_INTERNAL_BW
      compute_time = 2 × weight_params × tree_size / PIM_INT8_GOPS
      latency      = max(bw_time, compute_time)   ← compute_time dominates

    Args:
        model: Draft model configuration (e.g. QWEN2_5_0_5B).
        tree_size: Number of tokens in the draft tree to be generated.

    Returns:
        Latency in seconds.
    """
    if tree_size <= 0:
        return 0.0
    w = model.weight_bytes()
    params = w / model.bytes_per_param
    return _roofline(
        bw_bytes=w * tree_size,
        flops=2.0 * params * tree_size,
    )


def draft_energy(model: ModelConfig, tree_size: int) -> float:
    """
    Energy (joules) for the drafting phase in PIM mode.

    Energy is dominated by DRAM cell reads (PIM_ENERGY_PJ_PER_BIT).
    Near-bank ALU compute energy is negligible by design.

    Args:
        model: Draft model configuration.
        tree_size: Number of draft tokens.

    Returns:
        Energy in joules.
    """
    if tree_size <= 0:
        return 0.0
    w = model.weight_bytes()
    energy_pj = w * tree_size * PIM_ENERGY_PJ_PER_BIT / 8.0
    return pj_to_j(energy_pj)


# ---------------------------------------------------------------------------
# PIM-only verification — target model FC layers run in PIM mode
# ---------------------------------------------------------------------------

def verify_latency(model: ModelConfig, batch_size: int) -> float:
    """
    Latency (seconds) for FC/linear-layer verification inside PIM mode.

    Used when μ < μ_th (PIM-only path).  Runs the target model's weight
    matrix multiplications (Wq, Wk, Wv, Wo, FFN) on PIM compute units.
    Attention is NOT included here — use attn_latency() separately.

    Roofline: compute-bound (2 ops/byte >> PIM ridge of 0.008 ops/byte).
      bw_time     = weight_bytes × passes / PIM_INTERNAL_BW
      compute_time = 2 × weight_params × batch_size / PIM_INT8_GOPS
      latency      = max(bw_time, compute_time)   ← compute_time dominates

    Args:
        model: Target model configuration (e.g. QWEN2_5_7B).
        batch_size: Number of tokens to verify (μ after pruning).

    Returns:
        Latency in seconds.
    """
    if batch_size <= 0:
        return 0.0
    w = model.weight_bytes()
    params = w / model.bytes_per_param
    passes = ceil(batch_size / PIM_NALU)
    return _roofline(
        bw_bytes=w * passes,
        flops=2.0 * params * batch_size,
    )


def verify_energy(model: ModelConfig, batch_size: int) -> float:
    """
    Energy (joules) for FC/linear-layer verification inside PIM mode.

    Args:
        model: Target model configuration.
        batch_size: Number of tokens to verify.

    Returns:
        Energy in joules.
    """
    if batch_size <= 0:
        return 0.0
    w = model.weight_bytes()
    passes = ceil(batch_size / PIM_NALU)
    energy_pj = w * passes * PIM_ENERGY_PJ_PER_BIT / 8.0
    return pj_to_j(energy_pj)


# ---------------------------------------------------------------------------
# Attention on PIM — KV-cache accessed in PIM mode (both paths)
# ---------------------------------------------------------------------------

def attn_latency(model: ModelConfig, batch_size: int, seq_len: int) -> float:
    """
    Latency (seconds) for attention computation on PIM (KV-cache in PIM mode).

    Used in both the PIM-only path and the concurrent NPU+PIM path.
    PIM reads K and V from its own banks (internal bandwidth) and computes
    attention scores for batch_size query tokens.

    Arithmetic intensity = n_heads × batch_size / n_kv_heads >> PIM ridge
    → compute-bound.

    bw_time     = kv_cache_bytes(seq_len) / PIM_INTERNAL_BW
    compute_time = 2 × n_layers × n_heads × seq_len × head_dim × batch_size
                   / PIM_INT8_GOPS
    latency      = max(bw_time, compute_time)

    Args:
        model: Target model configuration.
        batch_size: Number of query tokens (μ for verification, 1 for draft).
        seq_len: Current KV-cache length.

    Returns:
        Latency in seconds.
    """
    if batch_size <= 0 or seq_len <= 0:
        return 0.0
    kv_bytes = model.kv_cache_bytes(seq_len, batch_size=1)  # KV shared across batch
    attn_flops = (
        2 * model.n_layers * model.n_heads * seq_len * model.head_dim * batch_size
    )
    return _roofline(bw_bytes=kv_bytes, flops=attn_flops)


def attn_energy(model: ModelConfig, seq_len: int) -> float:
    """
    Energy (joules) for attention computation on PIM.

    Energy is dominated by reading the KV-cache from PIM banks.
    The KV-cache is read once regardless of batch size (shared K/V).

    Args:
        model: Target model configuration.
        seq_len: Current KV-cache length.

    Returns:
        Energy in joules.
    """
    if seq_len <= 0:
        return 0.0
    kv_bytes = model.kv_cache_bytes(seq_len, batch_size=1)
    energy_pj = kv_bytes * 8 * PIM_ENERGY_PJ_PER_BIT
    return pj_to_j(energy_pj)


# ---------------------------------------------------------------------------
# PIM → NPU transfer (used in legacy paths; no longer needed for CAPIM)
# ---------------------------------------------------------------------------

def transfer_latency(payload_bytes: float) -> float:
    """
    Latency (seconds) to transfer data from PIM to NPU over external I/O.

    Note: CAPIM's NPU path does not use an explicit transfer — the NPU reads
    weights directly from PIM banks in HOST mode over the external bus.
    This function is retained for reference.

    Args:
        payload_bytes: Bytes to transfer.

    Returns:
        Transfer latency in seconds.
    """
    if payload_bytes <= 0:
        return 0.0
    return payload_bytes / PIM_EXTERNAL_BW


def transfer_energy(payload_bytes: float) -> float:
    """
    Energy (joules) for PIM → NPU data transfer over external I/O.

    Args:
        payload_bytes: Bytes transferred.

    Returns:
        Energy in joules.
    """
    if payload_bytes <= 0:
        return 0.0
    bits = payload_bytes * 8
    energy_pj = bits * OFFCHIP_ENERGY_PJ_PER_BIT
    return pj_to_j(energy_pj)
