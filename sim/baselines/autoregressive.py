"""
Baseline 1: Standard Autoregressive Decoding (no speculative decoding).

The target model runs one forward pass per output token on the NPU.
No draft model is used; no PIM computation (other than memory serving weights).

Modelling decisions:
  - All computation is on the NPU (roofline model from hw/npu.py).
  - Weights are served from LPDDR5-PIM over the external I/O channel
    (51.2 GB/s).  The NPU is bandwidth-bound at batch=1.
  - Energy includes both NPU compute and DRAM access (off-chip transfer).
  - KV-cache grows with each generated token; we use the mean context length
    from the trace (or a provided fixed value) as the representative seq_len.

This baseline establishes the lower bound: CAPIM must beat this to be useful.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import sim.hw.npu as npu
from sim.config.models import ModelConfig
from sim.trace.schema import TraceDataset


@dataclass
class ARResult:
    """Results from autoregressive baseline simulation."""

    latency_per_token_s: float       # seconds per output token
    energy_per_token_j: float        # joules per output token
    tokens_per_second: float         # throughput
    n_tokens: int                    # total tokens generated
    mean_context_length: float       # mean KV-cache length used


def simulate_autoregressive(
    target_model: ModelConfig,
    n_tokens: int,
    mean_context_length: float = 512.0,
) -> ARResult:
    """
    Simulate pure autoregressive decoding for `n_tokens` output tokens.

    Args:
        target_model: Target model configuration (e.g. QWEN2_5_7B).
        n_tokens: Total output tokens to generate (used for energy sum).
        mean_context_length: Representative KV-cache length for the roofline
                             calculation.  Use mean context from the trace.

    Returns:
        ARResult with per-token and aggregate metrics.
    """
    lat = npu.ar_token_latency(target_model, seq_len=int(mean_context_length))
    eng = npu.ar_token_energy(target_model, seq_len=int(mean_context_length))

    return ARResult(
        latency_per_token_s=lat,
        energy_per_token_j=eng,
        tokens_per_second=1.0 / lat if lat > 0 else float("inf"),
        n_tokens=n_tokens,
        mean_context_length=mean_context_length,
    )


def simulate_autoregressive_from_trace(
    target_model: ModelConfig,
    trace: TraceDataset,
) -> ARResult:
    """
    Simulate autoregressive baseline using context lengths from the trace.

    This matches the evaluation conditions of CAPIM: same prompts, same
    context growth pattern.

    Args:
        target_model: Target model configuration.
        trace: Collected trace (used for context lengths only).

    Returns:
        ARResult reflecting the trace's context length distribution.
    """
    if not trace.steps:
        raise ValueError("Empty trace dataset.")

    mean_ctx = sum(s.context_length for s in trace.steps) / len(trace.steps)

    # Each decode step generates (accepted_length + 1) tokens in CAPIM,
    # but AR generates exactly 1 token per step.
    # For a fair comparison, we count the same number of forward passes
    # as CAPIM decode steps, and report per-token metrics.
    n_steps = len(trace.steps)

    return simulate_autoregressive(target_model, n_tokens=n_steps, mean_context_length=mean_ctx)
