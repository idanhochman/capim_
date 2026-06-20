"""
Model architecture parameters for LLaMA-2-7B-Chat and its EAGLE draft head.

These are used to compute weight tensor sizes for bandwidth calculations,
FLOP counts for energy/latency models, and KV-cache footprints.

Sources:
  LLaMA-2 paper / meta-llama/Llama-2-7b-chat-hf config.json.
  Target model chosen to align with the LP-Spec baseline (which evaluates LLaMA-2).
  bytes_per_param = 1 for INT8 quantization (assumed for mobile deployment).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    d_model: int            # hidden dimension
    n_heads: int            # number of attention heads
    n_kv_heads: int         # number of KV heads (GQA; == n_heads for MHA)
    n_layers: int           # transformer layers
    intermediate_size: int  # FFN intermediate dimension
    vocab_size: int         # vocabulary size
    bytes_per_param: int    # bytes per parameter: 1=INT8, 2=FP16

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


    def weight_bytes(self) -> float:
        """
        Approximate total model weight in bytes.

        Counts the dominant linear layers only (attention projections + FFN).
        Embedding table excluded (not streamed during decoding).

        Attention per layer:
          Wq: d_model × d_model
          Wk: d_model × (n_kv_heads × head_dim)  [GQA]
          Wv: d_model × (n_kv_heads × head_dim)  [GQA]
          Wo: d_model × d_model
        FFN per layer (SwiGLU → 3 matrices):
          W_gate: d_model × intermediate_size
          W_up:   d_model × intermediate_size
          W_down: intermediate_size × d_model
        """
        d = self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        ffn = self.intermediate_size

        # Per-layer weight count
        attn_params = d * d + d * kv_dim + d * kv_dim + d * d
        ffn_params = d * ffn + d * ffn + ffn * d  # SwiGLU: gate+up+down

        total_params = self.n_layers * (attn_params + ffn_params)
        return total_params * self.bytes_per_param

    def kv_cache_bytes(self, seq_len: int, batch_size: int = 1) -> float:
        """
        KV-cache footprint for a given sequence length.

        Shape per layer: 2 × batch × n_kv_heads × seq_len × head_dim
        """
        per_layer = (
            2 * batch_size * self.n_kv_heads * seq_len * self.head_dim
        )
        return self.n_layers * per_layer * self.bytes_per_param

    def flops_per_token(self, seq_len: int) -> float:
        """
        Approximate FLOPs for a single autoregressive decode step (batch=1).

        Follows the standard 2×params heuristic for linear layers,
        plus attention FLOPs for reading KV-cache.
        """
        # Linear layer FLOPs ≈ 2 × weight_bytes / bytes_per_param
        linear_flops = 2 * self.weight_bytes() / self.bytes_per_param

        # Attention: for each layer, for each head, q·K (seq_len multiplications)
        # Each head: 2 × seq_len × head_dim (qK + aV)
        attn_flops = (
            self.n_layers * self.n_heads * 2 * seq_len * self.head_dim
        )
        return linear_flops + attn_flops


# ---------------------------------------------------------------------------
# Pre-defined model configurations
# ---------------------------------------------------------------------------

# Target model: meta-llama/Llama-2-7b-chat-hf.
# LLaMA-2-7B uses standard multi-head attention (no GQA → n_kv_heads == n_heads),
# RMSNorm, SwiGLU (SiLU) FFN, and RoPE. Chosen to align with the LP-Spec baseline.
LLAMA2_7B = ModelConfig(
    name="LLaMA-2-7B-Chat",
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,       # MHA (no GQA on the 7B model)
    n_layers=32,
    intermediate_size=11008,
    vocab_size=32000,
    bytes_per_param=1,   # INT8 quantization
)

# EAGLE draft head for LLaMA-2-7B-Chat (yuhuili/EAGLE-llama2-chat-7B).
# The EAGLE "draft model" is a lightweight head — a single decoder layer at the
# target's dimensions plus a fusion FC — trained to predict the target's hidden
# states. It is inseparable from the target (must be trained against the same
# weights). Only the FC/attention weights matter for the compute roofline.
# This is the correct draft-model config for CAPIM.
EAGLE_HEAD_LLAMA2_7B = ModelConfig(
    name="EAGLE-Head-LLaMA-2-7B",
    d_model=4096,
    n_heads=32,
    n_kv_heads=32,       # full attention (matches the target's MHA)
    n_layers=1,
    intermediate_size=11008,
    vocab_size=32000,
    bytes_per_param=1,   # INT8 quantization for mobile deployment
)
