"""
LPDDR5-PIM cost model (near-bank compute).

Replaces PAPI's Ramulator-backed PIM timing with an analytical roofline:
  FC / MATMUL:  time = max(flops / PIM_INT8_GOPS, traffic / PIM_INTERNAL_BW)
Only FC / MATMUL / COMM run on PIM; nonlinear ops route to the NPU (cross-bank
reduction wall, see nonlinear-placement-resolved) and are asserted out in cost().
Per the LPDDR5-PIM ridge point (409.6 GOPS / 51.2 TB/s = 0.008 ops/byte), GEMV
(intensity 2 ops/byte >> ridge) is COMPUTE-bound on PIM, so this normally reduces
to flops / GOPS — but we keep the full max() so the bound tag is computed, not
assumed.  PAPI's "GEMM = reuse(=m) x GEMV" is implicit here: flops already
carries the m factor, so no separate reuse multiply is needed.

COMM (the PIM<->NPU handoff) is costed here, on the external bus (51.2 GB/s):
  time = bytes / PIM_EXTERNAL_BW + FIXED_CROSSING_LATENCY_S
The fixed per-crossing latency is the load-bearing parameter flagged in the
project notes (draft-nl-cost-is-latency): the NL data itself is tiny, so the
crossing cost is dominated by fixed control/launch latency, not bandwidth.

Energy: internal-bank traffic x PIM_ENERGY  +  MACs x INT8_OP energy.
External-bus traffic (COMM) is charged at the off-chip energy rate.
"""

from __future__ import annotations

from sim.config.hardware import (
    MEM_INTERNAL_PJ_PER_BIT,
    MEM_OFFCHIP_PJ_PER_BIT,
    PIM_MAC_PJ_PER_OP,
    PIM_EXTERNAL_BW,
    PIM_INT8_GOPS,
    PIM_INTERNAL_BW,
    pj_to_j,
)
from sim.kernel.device import (
    MAX_COMPUTE_UTIL,
    MAX_MEM_UTIL,
    CostResult,
    Device,
    zero_energy,
)
from sim.kernel.layer import Layer, LayerType

# Load-bearing: fixed latency charged once per PIM<->NPU crossing (control/launch
# overhead, not bandwidth).  Default is a placeholder to be validated against the
# literature (FlightLLM / Samsung commercial-DRAM-PIM); see project notes.
FIXED_CROSSING_LATENCY_S: float = 1.0e-6   # 1 us per crossing (TODO: validate)


class LPDDR5PIM(Device):
    name = "PIM"

    def __init__(
        self,
        int8_gops: float = PIM_INT8_GOPS,
        internal_bw: float = PIM_INTERNAL_BW,
        external_bw: float = PIM_EXTERNAL_BW,
        crossing_latency_s: float = FIXED_CROSSING_LATENCY_S,
    ):
        self.int8_gops = int8_gops
        self.internal_bw = internal_bw
        self.external_bw = external_bw
        self.crossing_latency_s = crossing_latency_s

    def cost(self, layer: Layer) -> CostResult:
        if layer.type == LayerType.COMM:
            return self._comm_cost(layer)

        # NL ops never run on PIM
        assert layer.type in (LayerType.FC, LayerType.MATMUL), (
            f"PIM.cost got {layer.type.name}; NL must route to NPU, not PIM"
        )

        flops = layer.get_flops()
        in1, in2, out = layer.get_size()
        # Internal traffic: the stationary operand dominates (weights for FC,
        # KV-cache for attention MATMUL); count all operands touched in-bank.
        traffic = in1 + in2 + out

        compute_t = flops / (self.int8_gops * MAX_COMPUTE_UTIL)
        mem_t = traffic / (self.internal_bw * MAX_MEM_UTIL)

        if compute_t >= mem_t:
            time_s, bound = compute_t, "compute"
        else:
            time_s, bound = mem_t, "memory"

        e = zero_energy()
        e[0] = pj_to_j(traffic * 8 * MEM_INTERNAL_PJ_PER_BIT)            # internal bank access
        e[2] = pj_to_j((flops / 2.0) * PIM_MAC_PJ_PER_OP)                # near-bank MAC (20nm DRAM)
        layer.bound = bound
        layer.time_s = time_s
        layer.energy = e
        return CostResult(time_s, e, bound)

    def _comm_cost(self, layer: Layer) -> CostResult:
        in1, _, _ = layer.get_size()
        bytes_moved = in1
        time_s = self.crossing_latency_s + (
            bytes_moved / (self.external_bw * MAX_MEM_UTIL) if bytes_moved > 0 else 0.0
        )
        e = zero_energy()
        e[3] = pj_to_j(bytes_moved * 8 * MEM_OFFCHIP_PJ_PER_BIT)         # external-bus energy
        layer.bound = "comm"
        layer.time_s = time_s
        layer.energy = e
        return CostResult(time_s, e, "comm")
