"""
Driver scaffolding: composition rules + per-step records.

A driver emits typed layers (tagged with a device by its router) and composes
their per-op costs per its execution model:
  - compose_sequential : AR & CAPIM (batch=1, no device concurrency) -> additive,
    with an explicit PIM<->NPU COMM crossing inserted at every device switch.
  - compose_concurrent : LP-Spec -> makespan, every GEMM column-split NPU||PIM at
    the DAU ratio r, the (1-r) output slice gathered over the bus, nonlinear glue
    additive on the NPU (the DAU tensor-parallel model).

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
    wise across NPU and PIM in a ratio the DAU picks so the two devices finish
    together.

    Provenance of the balance (be precise -- the paper does NOT box T_NPU=T_PIM):
      [STATED, prose] S V-B: "To synchronize the execution time of both NPU and
        PIM devices, the partition ratio of model parameters across PIM and DRAM
        ranks needs to correlate with the computation throughput of NPU and PIM
        devices."  This is the equality goal, in words, not as an equation.
      [STATED, eqns] S V-A prints only the per-share device times and a total:
        T_NPU   = N_params,DRAM / BW_Off-chip
        T_PIM   = N_params,PIM  / BW_PIM x ceil(L_spec / N_ALU)
        T_total = min(T_NPU, T_PIM)
        (subscripts DRAM/PIM => these are each device's ALLOCATED-share time, run
        concurrently -- the column split we model.)
      [INFERRED] the closed form below is OURS, derived from the "synchronize"
        prose: the partition that makes the two share-times equal is
            r = t_p / (t_n + t_p)        # NPU/DRAM column share; (1-r) -> PIM
        (t_n, t_p = the all-NPU / all-PIM time of THIS kernel), giving the
        parallel-combination (harmonic) makespan
            t_kernel = r * t_n = t_n * t_p / (t_n + t_p).
      [INFERRED] the printed T_total = min(.) reduces to this: at the synchronized
        point T_NPU = T_PIM, so min = max = t_kernel.  (Off-balance the makespan is
        the max; the paper's min coincides only because the DAU equalizes them --
        the printed `min` reads like a loose/typo'd `max`.)

    Energy is blended E = r*E_n + (1-r)*E_p: the NPU share reads its weights
    over the external bus (expensive off-chip energy) while the PIM share reads
    them in-bank (cheap) -- the asymmetry CAPIM exploits by keeping small trees
    PIM-only.  Nonlinear ops are never split -> additive on the NPU.

    Per-kernel vs the paper's table: we derive r per kernel, whereas LP-Spec looks
    up ONE r per L_spec from a precomputed "model partition table" (Fig 7).  These
    coincide here because

        t_p/t_n = (8 * ceil(m/N_ALU) * BW_Off-chip) / GOPS

    is kernel-INDEPENDENT at fixed tree size m -- the n,k of an FC cancel -- so our
    per-kernel r IS the single ratio the table would store for that L_spec.  With
    the N_ALU=4 token-batching (kernel/pim.py) r is now BINNED in m, jumping at
    m=4,8,12 exactly like the table's 1~4 / 5~8 rows (e.g. r~0.515 at m<=4, 0.679 at
    m<=8, 0.808 at m=16; attention MATMULs ~0.78).  We do NOT reproduce Fig 7's
    ratio VALUES (5:1, 4:1): the paper gives no derivation for them ("illustrative"),
    and makespan balance on the real Table-II GOPS/BW gives an NPU-heavier split.
    Two disclosed, conservative gaps: (i) we use the exact balance, not the table's
    quantized lookup (per-kernel-optimal <= single-ratio makespan -> slightly FAVORS
    LP-Spec); (ii) we do NOT charge the inter-iteration weight-MIGRATION cost (Fig 8)
    incurred when L_spec changes the ratio -> also favors the LP-Spec baseline.

    Output gather: a column split leaves each device holding a slice of the
    output, so PIM's (1-r) slice must cross the external bus to the NPU before the
    next op can read the full activation.  Charged per split kernel -- every FC AND
    attention MATMUL, since LP-Spec column-splits attention too -- through the
    shared COMM cost (one fixed crossing + bandwidth), sized to THAT kernel's
    output, so it scales with r and catches the wide FFN / lm_head gathers.  (The
    attention gather/scatter around the NPU-softmax is approximated by one output
    gather per MATMUL, and the symmetric NPU->PIM input broadcast for the next
    split kernel is not modelled -- both are ~us, well below the noise floor.)

        t_total = sum_kernels t_kernel  +  sum_gathers t_comm  +  sum_NL t_nl_npu
    """
    out = Composed()
    t_compute = 0.0
    t_comm = 0.0
    t_nl = 0.0
    for layer in layers:
        if layer.type in (LayerType.FC, LayerType.MATMUL):
            c_n = npu.cost(layer)
            c_p = pim.cost(layer)
            t_n, t_p = c_n.time_s, c_p.time_s
            denom = t_n + t_p
            if denom <= 0:
                continue
            r = t_p / denom                         # NPU/DRAM column share (paper's r)
            t_kernel = t_n * t_p / denom            # = r*t_n = (1-r)*t_p
            t_compute += t_kernel
            for i in range(len(out.energy_j)):
                out.energy_j[i] += r * c_n.energy_j[i] + (1.0 - r) * c_p.energy_j[i]
            # both devices are busy for t_kernel (balanced), concurrently
            out.time_by_device["NPU"] = out.time_by_device.get("NPU", 0.0) + t_kernel
            out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + t_kernel
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + t_kernel

            # DAU output gather: PIM's (1-r) output slice crosses to the NPU over
            # the external bus so the full activation is reassembled for the next
            # op.  Charged for EVERY split kernel (FC and attention MATMUL) -- LP-Spec
            # column-splits attention too, so its output slices are gathered as well.
            # get_size's `out` already folds numOp (the head count) for a MATMUL.
            _, _, out_bytes = layer.get_size()
            gather_bytes = (1.0 - r) * out_bytes
            if gather_bytes > 0:
                hop = Layer("dau_gather", LayerType.COMM, m=1, n=gather_bytes, dbyte=1)
                g = pim.cost(hop)
                t_comm += g.time_s
                _add(out.energy_j, g.energy_j)
                out.time_by_device["PIM"] = out.time_by_device.get("PIM", 0.0) + g.time_s
                out.time_by_type["COMM"] = out.time_by_type.get("COMM", 0.0) + g.time_s
                out.crossings += 1
        else:  # SOFTMAX / ACT / NORM / COMM -> NPU, additive
            c = npu.cost(layer)
            t_nl += c.time_s
            _add(out.energy_j, c.energy_j)
            out.time_by_device["NPU"] = out.time_by_device.get("NPU", 0.0) + c.time_s
            out.time_by_type[layer.type.name] = out.time_by_type.get(layer.type.name, 0.0) + c.time_s
    out.time_s = t_compute + t_comm + t_nl
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
