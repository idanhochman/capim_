"""
Target-model layer builders (LLaMA-2-7B), ported from PAPI's Transformer.build
at tensor-parallel degree 1 (mobile, batch=1).

A builder emits ONE decoder layer's typed Layer list; the driver multiplies the
per-layer cost by n_layers and tags each layer with a device.  Layers are emitted
device-agnostic; routing is the driver's job.

Decoder layer (LLaMA-2): qkv -> score -> softmax -> context -> proj -> norm1 ->
(SwiGLU FFN: ff1 gate, ff2 up, glu, ff3 down) -> norm2.  Layer names follow PAPI's
LLAMA branch (model.py:206-217).
"""

from __future__ import annotations

from typing import List

from sim.config.models import ModelConfig
from sim.kernel.layer import Layer, LayerType


def build_decoder_layer(model: ModelConfig, m: int, ctx: int) -> List[Layer]:
    """One target decoder layer over `m` query tokens at KV context length `ctx`.

    `m` = number of tokens processed this pass (prefill: lin; verify: tree size mu).
    `ctx` = KV-cache length the attention attends over.
    """
    d = model.d_model            # hidden / residual-stream width
    h = model.n_heads            # number of query heads (attention numOp)
    dh = model.head_dim          # per-head dim (= d // h); attention contraction
    ff = model.intermediate_size # SwiGLU FFN hidden width
    db = model.bytes_per_param   # bytes/elem (1=INT8); scales traffic, not FLOPs

    # Fused q,k,v output width.  We follow PAPI's decoder cost model: PAPI uses
    # 3*hdim/tp_dense (PAPI model.py:173, the gen 'qkv' layer), where its `hdim`
    # is the hidden dim (== our d_model) and tp_dense is the tensor-parallel
    # degree.  At mobile batch=1 tp_dense=1, so this reduces to 3*d.
    qkv_n = 3 * d

    return [
        Layer("qkv", LayerType.FC, m=m, n=qkv_n, k=d, dbyte=db),
        Layer("score", LayerType.MATMUL, m=m, n=ctx, k=dh, numOp=h, dbyte=db),
        Layer("softmax", LayerType.SOFTMAX, m=m, n=ctx, numOp=h, dbyte=db),
        Layer("context", LayerType.MATMUL, m=m, n=dh, k=ctx, numOp=h, dbyte=db),
        Layer("proj", LayerType.FC, m=m, n=d, k=d, dbyte=db),
        Layer("norm1", LayerType.NORM, m=m, n=d, dbyte=db),
        Layer("ff1", LayerType.FC, m=m, n=ff, k=d, dbyte=db),
        Layer("ff2", LayerType.FC, m=m, n=ff, k=d, dbyte=db),
        Layer("glu", LayerType.ACT, m=m, n=ff, dbyte=db),
        Layer("ff3", LayerType.FC, m=m, n=d, k=ff, dbyte=db),
        Layer("norm2", LayerType.NORM, m=m, n=d, dbyte=db),
    ]


def build_lm_head(model: ModelConfig, m: int) -> Layer:
    """Vocabulary projection over `m` tokens (one big FC: d_model -> vocab).

    Not modeled in PAPI. Modeled as FC because the LLaMA head
    is nn.Linear(hidden_size, vocab_size, bias=False) -- a plain GEMM
    (EAGLE modeling_llama_kv.py:1212; dims from llama_2_chat_7B_config.json).
    """
    return Layer("lm_head", LayerType.FC, m=m, n=model.vocab_size,
                 k=model.d_model, dbyte=model.bytes_per_param)


def build_prefill(model: ModelConfig, lin: int) -> List[Layer]:
    """Full prefill: n_layers decoder passes over the prompt, then one lm_head.

    Returns a flat list for ONE decoder layer; the driver scales by n_layers and
    appends the lm_head once.  (Attention context == lin for the causal prompt.)
    """
    return build_decoder_layer(model, m=lin, ctx=lin)
