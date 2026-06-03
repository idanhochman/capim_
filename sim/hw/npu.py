"""
Mobile NPU analytical latency and energy model.

The NPU model implements a roofline analysis for transformer forward passes
in the autoregressive decoding regime (batch ≥ 1, KV-cache already loaded).

Roofline principle:
  - If the operation is compute-bound:  latency = FLOPs / peak_TOPS
  - If the operation is bandwidth-bound: latency = bytes_accessed / peak_BW
  - latency = max(compute_latency, bandwidth_latency)   [roofline]

For INT8 matrix multiplications at batch=1 (GEMV), mobile NPUs are almost
always bandwidth-bound because the arithmetic intensity (ops/byte) is low.
As batch_size grows, the operation shifts toward compute-bound.

Note on LLMCompass integration:
  LLMCompass's TransformerBlockAutoRegressionTP.roofline_model() is designed
  for GPU/TPU hardware (A100, TPUv3, MI210).  Its Device abstraction does not
  directly support mobile NPU specs, and its overhead constants are GPU-specific.

  Instead, we implement the same roofline methodology analytically using the
  LP-Spec mobile NPU parameters from hardware.py.  The arithmetic matches what
  LLMCompass does internally; we simply bypass the GPU-specific overhead terms.
"""

from math import ceil
from typing import Optional

from sim.config.hardware import (
    NPU_ENERGY_PJ_PER_INT8_OP,
    NPU_INT8_TOPS,
    NPU_OFFCHIP_BW,
    OFFCHIP_ENERGY_PJ_PER_BIT,
    pj_to_j,
)
from sim.config.models import ModelConfig


def _compute_flops_verify(model: ModelConfig, batch_size: int, seq_len: int) -> float:
    """
    FLOPs for one target-model forward pass over `batch_size` tokens with
    KV-cache of length `seq_len`.

    Dominant terms:
      - Weight GEMMs: 2 × weight_params × batch_size  (each param participates in
        batch_size MACs, each MAC = 2 FLOPs)
      - Attention: 2 × n_layers × n_heads × seq_len × head_dim × batch_size
        (reading KV-cache for each query token)
    """
    w = model.weight_bytes() / model.word_size_bytes   # parameter count
    linear_flops = 2 * w * batch_size

    attn_flops = (
        2
        * model.n_layers
        * model.n_heads
        * seq_len
        * model.head_dim
        * batch_size
    )
    return linear_flops + attn_flops


def _bytes_accessed_verify(
    model: ModelConfig, batch_size: int, seq_len: int
) -> float:
    """
    Bytes the NPU reads from PIM banks via the external bus during verification.

    The NPU handles only the FC/linear layers (Wq, Wk, Wv, Wo, FFN).
    It reads weight matrices from PIM banks in HOST mode over the 51.2 GB/s
    external bus.

    The KV-cache is NOT included here: it remains in PIM banks and is accessed
    by PIM compute units in PIM mode (see pim.attn_latency / pim.attn_energy).
    There is no off-chip transfer of KV data to the NPU.
    """
    return model.weight_bytes()


def verify_latency(
    model: ModelConfig,
    batch_size: int,
    seq_len: int,
) -> float:
    """
    Roofline latency (seconds) for the NPU's FC/linear-layer computation.

    This models only the FC portion of verification (Wq, Wk, Wv, Wo, FFN).
    The NPU reads weight matrices from PIM banks via the external bus (HOST
    mode, 51.2 GB/s).  The KV-cache stays in PIM and is handled separately
    by pim.attn_latency().

    In the concurrent NPU+PIM model, total verification latency is:
        t_verify = max(npu.verify_latency(...), pim.attn_latency(...))

    Args:
        model: Target model configuration.
        batch_size: Number of draft tokens to verify (μ).
        seq_len: Current KV-cache length (used only for FLOP count).

    Returns:
        Latency in seconds (FC portion only).
    """
    if batch_size <= 0:
        return 0.0

    flops = _compute_flops_verify(model, batch_size, seq_len)
    bytes_acc = _bytes_accessed_verify(model, batch_size, seq_len)

    compute_latency = flops / NPU_INT8_TOPS
    bandwidth_latency = bytes_acc / NPU_OFFCHIP_BW

    return max(compute_latency, bandwidth_latency)


def verify_energy(model: ModelConfig, batch_size: int, seq_len: int = 512) -> float:
    """
    Energy (joules) for the NPU's FC/linear-layer computation.

    Includes compute energy (FLOPs × pJ/op) and memory energy (weight bytes
    streamed from PIM banks over external bus × OFFCHIP_ENERGY_PJ_PER_BIT).
    KV-cache energy is NOT included — handled by pim.attn_energy().

    Args:
        model: Target model configuration.
        batch_size: Number of draft tokens to verify.
        seq_len: Context length (for FLOP count only, not memory).

    Returns:
        Energy in joules (FC portion only).
    """
    if batch_size <= 0:
        return 0.0

    flops = _compute_flops_verify(model, batch_size, seq_len)
    bytes_acc = _bytes_accessed_verify(model, batch_size, seq_len)

    compute_energy_pj = flops * NPU_ENERGY_PJ_PER_INT8_OP
    mem_energy_pj = bytes_acc * 8 * OFFCHIP_ENERGY_PJ_PER_BIT

    return pj_to_j(compute_energy_pj + mem_energy_pj)


def ar_token_latency(model: ModelConfig, seq_len: int) -> float:
    """
    Latency (seconds) for a single autoregressive decode step on the NPU.

    Equivalent to verify_latency with batch_size=1.

    Args:
        model: Target model configuration.
        seq_len: Current context length.

    Returns:
        Latency in seconds.
    """
    return verify_latency(model, batch_size=1, seq_len=seq_len)


def ar_token_energy(model: ModelConfig, seq_len: int = 512) -> float:
    """
    Energy (joules) for a single autoregressive decode step on the NPU.

    Args:
        model: Target model configuration.
        seq_len: Current context length.

    Returns:
        Energy in joules.
    """
    return verify_energy(model, batch_size=1, seq_len=seq_len)
