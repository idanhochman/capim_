"""
MEDUSA draft-head layer builder (for the LP-Spec baseline).

Grounded in Medusa/medusa/model/medusa_model.py:
  - ResBlock(x) = x + SiLU(Linear(hidden, hidden))          # one FC + SiLU + residual
  - medusa_head = [ Sequential(ResBlock * medusa_num_layers,
                               Linear(hidden, vocab, bias=False)) ] * medusa_num_heads
  - defaults: medusa_num_heads = 5, medusa_num_layers = 1
  - forward: every head applied to the SAME hidden state (parallel, one shot;
    no autoregression, no attention, no KV) -> the "free tail".

Unlike EAGLE, each head has its OWN lm_head (K independent vocab projections),
so we emit K separate lm_head FCs (matters for PIM residency + energy breakdown,
not for the FLOP total).

The candidate tree is STATIC: `mc_sim_7b_63` (63 nodes), copied below.
"""

from __future__ import annotations

from typing import List

from sim.config.models import ModelConfig
from sim.kernel.layer import Layer, LayerType

# Static MEDUSA sparse tree for 7B models (Medusa/medusa/model/medusa_choices.py:
# mc_sim_7b_63).  63 candidate nodes.
MC_SIM_7B_63 = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4],
    [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0],
    [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0],
    [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5],
    [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7],
    [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2],
    [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1],
    [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0],
]
MEDUSA_TREE_SIZE = len(MC_SIM_7B_63)   # 63

# Measured Medusa-2 Vicuna-7B tree (Medusa/medusa/model/medusa_choices.py:
# vicuna_7b_stage2).  This is the tree our LP-Spec baseline actually uses, since
# CAPIM and LP-Spec share the Vicuna-7B-v1.3 backbone.  Stored as
# lists (the file uses tuples) to match MC_SIM_7B_63's format.
VICUNA_7B_STAGE2 = [
    [0], [0, 0], [1], [0, 1], [0, 0, 0], [1, 0], [2], [0, 2], [0, 0, 1], [0, 3],
    [3], [0, 1, 0], [2, 0], [4], [0, 0, 2], [0, 4], [1, 1], [1, 0, 0],
    [0, 0, 0, 0], [5], [0, 0, 3], [0, 5], [0, 2, 0], [3, 0], [0, 1, 1], [0, 6],
    [6], [0, 7], [0, 0, 4], [4, 0], [1, 2], [0, 8], [7], [0, 3, 0], [0, 0, 0, 1],
    [0, 0, 5], [2, 1], [0, 0, 6], [1, 0, 1], [0, 0, 1, 0], [2, 0, 0], [5, 0],
    [0, 9], [0, 1, 2], [8], [0, 4, 0], [0, 2, 1], [1, 3], [0, 0, 7],
    [0, 0, 0, 2], [0, 0, 8], [1, 1, 0], [0, 1, 0, 0], [6, 0], [9], [0, 1, 3],
    [0, 0, 0, 3], [1, 0, 2], [0, 5, 0], [3, 1], [0, 0, 2, 0], [7, 0], [1, 4],
]
VICUNA_7B_STAGE2_SIZE = len(VICUNA_7B_STAGE2)   # 63

# Registry so drivers/collectors can name a tree.
MEDUSA_TREES = {
    "mc_sim_7b_63": MC_SIM_7B_63,
    "vicuna_7b_stage2": VICUNA_7B_STAGE2,
}


def tree_topology(choices):
    """Resolve a MEDUSA `medusa_choices` list into the per-node structure the DTP
    driver and tests reason about.

    Mirrors the schema's static-tree layout: paths are ordered by (len, lex) so
    same-parent siblings are contiguous and ascending in prediction rank.  Returns
    a list of dicts (one per node, in node order) with:
      path, depth, layer_idx, parent_idx (within the previous depth layer; -1 at
      depth 0), and k_pred (= path[-1] = rank among same-parent siblings).
    """
    paths = sorted([list(p) for p in choices], key=lambda p: (len(p), p))
    max_d = max(len(p) for p in paths) - 1
    layer_paths = [[] for _ in range(max_d + 1)]
    for p in paths:
        layer_paths[len(p) - 1].append(p)
    path_to_layeridx = {}
    for d in range(max_d + 1):
        for li, p in enumerate(layer_paths[d]):
            path_to_layeridx[tuple(p)] = li
    nodes = []
    for d in range(max_d + 1):
        for li, p in enumerate(layer_paths[d]):
            parent_idx = -1 if d == 0 else path_to_layeridx[tuple(p[:-1])]
            nodes.append({
                "path": p, "depth": d, "layer_idx": li,
                "parent_idx": parent_idx, "k_pred": p[-1],
            })
    return nodes


def build_medusa_draft(
    model: ModelConfig,
    medusa_num_heads: int = 5,
    medusa_num_layers: int = 1,
) -> List[Layer]:
    """K MEDUSA heads from one hidden state (parallel, batch=1)."""
    d = model.d_model
    db = model.bytes_per_param
    layers: List[Layer] = []
    for h in range(medusa_num_heads):
        for _ in range(medusa_num_layers):
            layers.append(Layer(f"medusa{h}_resblock", LayerType.FC, m=1, n=d, k=d, dbyte=db))
            layers.append(Layer(f"medusa{h}_silu", LayerType.ACT, m=1, n=d, dbyte=db))
        layers.append(Layer(f"medusa{h}_lmhead", LayerType.FC, m=1,
                            n=model.vocab_size, k=d, dbyte=db))
    # one fused sampling softmax over all K heads' vocab logits (top-k for the tree)
    layers.append(Layer("medusa_softmax", LayerType.SOFTMAX, m=1,
                        n=model.vocab_size, numOp=medusa_num_heads, dbyte=db))
    return layers
