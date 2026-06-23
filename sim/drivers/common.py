"""
Shared driver helpers: routers + a full-target-forward cost (decoder x n_layers
+ lm_head), reused by the verify path (CAPIM / EAGLE-NPU) and the AR baseline.
"""

from __future__ import annotations

from sim.config.models import ModelConfig
from sim.drivers.base import Composed, compose_concurrent, compose_sequential, tag
from sim.kernel.layer import Device as Dev
from sim.kernel.layer import Layer, LayerType
from sim.kernel.npu import MobileNPU
from sim.kernel.pim import LPDDR5PIM
from sim.layers.target import build_decoder_layer, build_lm_head


def router_capim_verify(fc_device: Dev):
    """FC -> routed device; attention MATMUL -> PIM; nonlinear -> NPU."""
    def _r(layer: Layer) -> Dev:
        if layer.type == LayerType.FC:
            return fc_device
        if layer.type == LayerType.MATMUL:
            return Dev.PIM
        return Dev.NPU  # SOFTMAX / ACT / NORM / COMM
    return _r


def router_all_npu(layer: Layer) -> Dev:
    return Dev.NPU


def router_eagle_draft(layer: Layer) -> Dev:
    """EAGLE draft on PIM: weights/attention pinned to PIM, nonlinear -> NPU."""
    if layer.type in (LayerType.FC, LayerType.MATMUL):
        return Dev.PIM
    return Dev.NPU


def cost_target_pass(
    model: ModelConfig,
    m: int,
    ctx: int,
    fc_device: Dev,
    npu: MobileNPU,
    pim: LPDDR5PIM,
    all_npu: bool = False,
    concurrent: bool = False,
) -> Composed:
    """Cost one full target forward over `m` tokens at context `ctx`.

    n_layers identical decoder layers (costed once, scaled) + one lm_head.
    - all_npu=True  -> AR / EAGLE-NPU: everything on NPU, no crossings.
    - concurrent=True -> LP-Spec makespan composition (FC||attn).
    - else          -> CAPIM sequential/additive with PIM<->NPU crossings.
    """
    decoder = build_decoder_layer(model, m=m, ctx=ctx)
    if all_npu:
        tag(decoder, router_all_npu)
    else:
        tag(decoder, router_capim_verify(fc_device))

    if concurrent:
        one = compose_concurrent(decoder, npu, pim)
    else:
        one = compose_sequential(decoder, npu, pim, count_crossings=not all_npu)

    # n_layers identical blocks (same shapes, same ctx) -> cost one, multiply
    total = one.scale(model.n_layers)

    # Inter-layer crossings (PIM route only): compose counts crossings within a block,
    # not the boundary norm2(NPU)->next qkv(PIM) -> add (n_layers-1) bus hops.
    if not all_npu and not concurrent and fc_device == Dev.PIM:
        # boundary hidden state = m tokens x d_model (was m=1: undercounted by m)
        hop = Layer("interlayer", LayerType.COMM, m=m, n=model.d_model, dbyte=model.bytes_per_param)
        r = pim.cost(hop)
        total.time_s += r.time_s * (model.n_layers - 1)
        for i in range(len(total.energy_j)):
            total.energy_j[i] += r.energy_j[i] * (model.n_layers - 1)
        total.crossings += (model.n_layers - 1)

    # lm_head (once)
    head = build_lm_head(model, m=m)
    head.device = Dev.NPU if all_npu else fc_device
    if concurrent:
        head_c = compose_concurrent([head], npu, pim)
    else:
        head_c = compose_sequential([head], npu, pim, count_crossings=False)
    total.merge(head_c)
    return total
