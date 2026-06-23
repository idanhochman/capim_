"""
Device base class + the cost-result container.

Every device implements `cost(layer) -> CostResult` with an identical signature,
so the three drivers (AR / LP-Spec / CAPIM) share one per-op cost model and the
comparison is fair.  A device decides time via a per-layer roofline
`max(compute, mem)` and energy as a 4-component vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sim.kernel.layer import Layer

# Utilisation derates (PAPI SCALING_FACTOR).
MAX_COMPUTE_UTIL = 0.8
MAX_MEM_UTIL = 0.85

# Energy-vector slot names (4 slots, mobile).  Structure ported from PAPI's
# 6-vector [off_mem, L2, L1, reg, flop, comm] (src/devices.py:_get_energy),
# trimmed for a mobile NPU/PIM with no cache hierarchy:
#   off_mem : DRAM/memory-access energy -- off-chip bus on the NPU (5.4 pJ/bit),
#             cheap internal near-bank access on PIM (0.8 pJ/bit).
#   on_chip : PAPI's L2+L1+reg tiling, collapsed to one slot; unused (=0) here
#             (no mobile cache hierarchy; second-order at batch=1 GEMV).
#   alu     : arithmetic-datapath energy = (flops/2) * pJ_per_INT8_MAC.  PAPI
#             names this slot 'flop' but keys its coefficient 'alu'; we use 'alu'
#             since this is INT8 (W8A8), not floating point.
#   comm    : PIM<->NPU crossing over the external bus (charged on the PIM side).
# Magnitudes are mobile (LP-Spec Table II + cited figures), NOT PAPI's HBM/GPU.
E_OFF, E_ONCHIP, E_ALU, E_COMM = 0, 1, 2, 3
ENERGY_SLOTS = ("off_mem", "on_chip", "alu", "comm")


@dataclass
class CostResult:
    time_s: float
    energy_j: List[float]   # [off_mem, on_chip, alu, comm]
    bound: str              # "compute" | "memory" | "comm"

    @property
    def total_energy_j(self) -> float:
        return sum(self.energy_j)


def zero_energy() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0]


class Device:
    """Abstract device.  Subclasses implement `cost`."""

    name: str = "device"

    def cost(self, layer: Layer) -> CostResult:  # pragma: no cover - interface
        raise NotImplementedError

    # convenience: cost a list of layers, return (time, energy_vec) summed
    def cost_sequential(self, layers: List[Layer]):
        t = 0.0
        e = zero_energy()
        for layer in layers:
            r = self.cost(layer)
            t += r.time_s
            for i in range(len(e)):
                e[i] += r.energy_j[i]
        return t, e
