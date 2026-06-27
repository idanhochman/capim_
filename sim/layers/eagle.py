"""
EAGLE-2 draft-head layer builder.

Grounded in EAGLE/eagle/model/cnets1.py (the EAGLE-1/2 head for our target):
  - self.fc = Linear(2*hidden, hidden)            # fusion FC (concat emb + feature)
  - self.layers = [LlamaDecoderLayer] * 1         # exactly ONE decoder layer
  - shared embed_tokens / norm / lm_head          # vocab projection reused from target
  - topK_genrate runs `for i in range(depth)`     # D sequential draft sub-steps

One draft SUB-STEP over `width` parallel tree nodes at context `ctx`:
  fusion FC -> 1 decoder layer -> lm_head -> sampling softmax.
The driver calls this once per tree depth, with `width` = nodes at that depth
(from the trace), so the emitted work matches the real (pre-prune) tree shape.

Nonlinear ops (norms, attention softmax, sampling softmax) are emitted as normal
NL layers; the driver routes them to the NPU and inserts the PIM<->NPU COMM
crossings that make drafting a ping-pong (the dominant drafting cost).
"""

from __future__ import annotations

from typing import List

from sim.config.models import ModelConfig
from sim.kernel.layer import Layer, LayerType
from sim.layers.target import build_decoder_layer, build_lm_head


def build_eagle_draft_step(model: ModelConfig, width: int, ctx: int) -> List[Layer]:
    """One EAGLE draft sub-step over `width` nodes at KV context `ctx`."""
    d = model.d_model
    db = model.bytes_per_param

    layers: List[Layer] = [
        # fusion FC: concat(token_embedding, previous_feature) -> hidden  (k = 2*d)
        Layer("fusion_fc", LayerType.FC, m=width, n=d, k=2 * d, dbyte=db),
    ]
    # exactly one decoder layer (EAGLE head depth = 1); eagle_draft drops the
    # input_layernorm (head layer is index 0, cnets1.py:399) -> one RMSNorm, and
    # the head reads the layer output directly (no final norm before lm_head).
    layers += build_decoder_layer(model, m=width, ctx=ctx, eagle_draft=True)
    # vocab projection (shared target lm_head) + sampling softmax -> confidence
    layers.append(build_lm_head(model, m=width))
    layers.append(Layer("sample_softmax", LayerType.SOFTMAX, m=width,
                        n=model.vocab_size, dbyte=db))
    return layers
