"""
Driver scaffolding: composition rules + per-step records.

A driver emits typed layers (tagged with a device by its router) and composes
their per-op costs per its execution model:
  - compose_sequential : AR & CAPIM (batch=1, no device concurrency) -> additive,
    with an explicit PIM<->NPU COMM crossing inserted at every device switch.
  - compose_concurrent : LP-Spec -> makespan, FC (NPU) || attention (PIM) overlap
    with nonlinear glue additive on the NPU (the DAU tensor-parallel model).

All three share the identical Device.cost(), so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from sim.kernel.device import Device, zero_energy
from sim.kernel.layer import Layer, LayerType, Device as Dev
from sim.kernel.pim import LPDDR5PIM
from sim.kernel.npu import MobileNPU


def _add(dst: List[float], src: List[float]) -> None:
    for i in range(len(dst)):
        dst[i] += src[i]


@dataclass
class Composed:
    """Result of costing a layer list."""
    time_s: float = 0.0
    energy_j: List[float] = field(default_factory=zero_energy)
    time_by_device: Dict[str, float] = field(default_factory=lambda: {"NPU": 0.0, "PIM": 0.0})
    time_by_type: Dict[str, float] = field(default_factory=dict)
    crossings: int = 0

    def scale(self, factor: float) -> "Composed":
        c = Composed(
            time_s=self.time_s * factor,
            energy_j=[e * factor for e in self.energy_j],
            time_by_device={k: v * factor for k, v in self.time_by_device.items()},
            time_by_type={k: v * factor for k, v in self.time_by_type.items()},
            crossings=int(self.crossings * factor),
        )
        return c

    def merge(self, other: "Composed") -> None:
        self.time_s += other.time_s
        _add(self.energy_j, other.energy_j)
        for k, v in other.time_by_device.items():
            self.time_by_device[k] = self.time_by_device.get(k, 0.0) + v
        for k, v in other.time_by_type.items():
            self.time_by_type[k] = self.time_by_type.get(k, 0.0) + v
        self.crossings += other.crossings


def _dev_obj(dev: Dev, npu: MobileNPU, pim: LPDDR5PIM) -> Device:
    return pim if dev == Dev.PIM else npu


def compose_sequential(layers: List[Layer], npu: MobileNPU, pim: LPDDR5PIM,
                       count_crossings: bool = True) -> Composed:
    """Additive composition with a COMM crossing at each device switch."""
    out = Composed()
    prev: Dev = None
    for layer in layers:
        dev = layer.device
        if count_crossings and prev is not None and dev != prev:
            # data must hop the external bus: cost the boundary activation as COMM.
            # in1 is already in BYTES (get_size folds in dbyte), so encode it as a
            # flat 1 x in1 block with dbyte=1 (m and dbyte set to 1 so _comm_cost's
            # get_size returns exactly in1; avoids double-counting dbyte).
            in1, _, _ = layer.get_size()
            crossing = Layer("xfer", LayerType.COMM, m=1, n=max(1, in1), dbyte=1)
            r = pim.cost(crossing)
            out.time_s += r.time_s
            _add(out.energy_j, r.energy_j)
            out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + r.time_s
            out.time_by_type["COMM"] = out.time_by_type.get("COMM", 0.0) + r.time_s
            out.crossings += 1
        r = _dev_obj(dev, npu, pim).cost(layer)
        out.time_s += r.time_s
        _add(out.energy_j, r.energy_j)
        dname = "PIM" if dev == Dev.PIM else "NPU"
        out.time_by_device[dname] = out.time_by_device.get(dname, 0.0) + r.time_s
        out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + r.time_s
        prev = dev
    return out


def compose_concurrent(layers: List[Layer], npu: MobileNPU, pim: LPDDR5PIM) -> Composed:
    """LP-Spec makespan via the DAU's balanced column-wise split.

    LP-Spec does NOT route whole kernels (FC->NPU, attn->PIM).  It partitions
    *every* GEMM kernel -- both FC weights and the attention KV-cache -- column-
    wise across NPU and PIM in a ratio r the DAU picks so the two devices finish
    together (lp_spec_inferenece_flow.md S V-B).  For one kernel with full-device
    times t_n (all-NPU) and t_p (all-PIM), the balanced NPU fraction is
    f = t_p / (t_n + t_p), giving the parallel-combination makespan

        t_kernel = f * t_n = t_n * t_p / (t_n + t_p)

    and a blended energy E = f*E_n + (1-f)*E_p: the NPU share reads its weights
    over the external bus (expensive off-chip energy) while the PIM share reads
    them in-bank (cheap) -- the asymmetry CAPIM exploits by keeping small trees
    PIM-only.  Nonlinear ops are never split -> additive on the NPU.

        t_total = sum_kernels t_kernel  +  sum_NL t_nl_npu
    """
    out = Composed()
    t_compute = 0.0
    t_nl = 0.0
    for layer in layers:
        if layer.type in (LayerType.FC, LayerType.MATMUL):
            rn = npu.cost(layer)
            rp = pim.cost(layer)
            t_n, t_p = rn.time_s, rp.time_s
            denom = t_n + t_p
            if denom <= 0:
                continue
            f = t_p / denom                         # NPU's balanced column fraction
            t_kernel = t_n * t_p / denom            # = f*t_n = (1-f)*t_p
            t_compute += t_kernel
            for i in range(len(out.energy_j)):
                out.energy_j[i] += f * rn.energy_j[i] + (1.0 - f) * rp.energy_j[i]
            # both devices are busy for t_kernel (balanced), concurrently
            out.time_by_device["NPU"] = out.time_by_device.get("NPU", 0.0) + t_kernel
            out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + t_kernel
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + t_kernel
        else:  # SOFTMAX / ACT / NORM / COMM -> NPU, additive
            r = npu.cost(layer)
            t_nl += r.time_s
            _add(out.energy_j, r.energy_j)
            out.time_by_device["NPU"] = out.time_by_device.get("NPU", 0.0) + r.time_s
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + r.time_s
    out.time_s = t_compute + t_nl
    return out


@dataclass
class StepRecord:
    prompt_id: int
    dataset: str
    step_id: int
    tokens_emitted: float
    time_s: float
    energy_j: float
    time_by_device: Dict[str, float] = field(default_factory=dict)
    energy_by_component: Dict[str, float] = field(default_factory=dict)
    time_by_type: Dict[str, float] = field(default_factory=dict)


@dataclass
class DriverResult:
    driver: str
    model: str
    steps: List[StepRecord] = field(default_factory=list)


# router type: (Layer) -> Device enum
Router = Callable[[Layer], Dev]


def tag(layers: List[Layer], router: Router) -> List[Layer]:
    for layer in layers:
        layer.device = router(layer)
    return layers
