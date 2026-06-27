"""
The typed-layer atom of the cost kernel.

A workload (a target decoder layer, an EAGLE draft step, a MEDUSA head, ...) is
just an ordered list of `Layer` objects.  Cost is a pure function of a Layer's
shape `(m, n, k, numOp, dbyte)` and the device it runs on (see kernel.device).

`get_flops` / `get_size` are ported verbatim from PAPI's `src/model.py:Layer`
(they are model-agnostic); only the layer-type set is trimmed to what mobile
batch=1 inference needs (PAPI's G2G/X2G all-reduce/comm collapse to a single
COMM type used for the PIM<->NPU handoff).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LayerType(Enum):
    FC = 0       # weight-stationary GEMM/GEMV: y = W x   (qkv, proj, ffn, lm_head, fusion, resblock)
    MATMUL = 1   # weightless batched matmul: attention score / context
    SOFTMAX = 2  # attention softmax or sampling softmax  (nonlinear)
    ACT = 3      # activation: SiLU / GELU / SwiGLU        (nonlinear)
    NORM = 4     # RMSNorm / LayerNorm                     (nonlinear)
    COMM = 5     # data movement: PIM<->NPU handoff over the external bus


# Layer types the kernel treats as "nonlinear glue" (always NPU in CAPIM/LP-Spec).
NONLINEAR = {LayerType.SOFTMAX, LayerType.ACT, LayerType.NORM}


class Device(Enum):
    NPU = 0
    PIM = 1


@dataclass
class Layer:
    """One typed operator.

    Shape semantics (match PAPI):
      FC/MATMUL:  m x k  times  k x n  ->  m x n, repeated `numOp` times.
      NL/COMM:    operate elementwise on an m x n activation, `numOp` times.

    `device` is assigned by the driver's router, not by the builder.
    `bound`, `time_s`, `energy` are filled in by Device.cost().
    """

    name: str
    type: LayerType
    m: int                               # rows of the output
    n: int                               # columns of the output
    k: int = 1                           # contraction dim
    numOp: int = 1                       # how many times the op repeats
    dbyte: int = 1                       # 1 = INT8 (W8A8), 2 = FP16
    device: Optional[Device] = None

    # filled by Device.cost()
    bound: str = ""                      # "compute" | "memory"
    time_s: float = 0.0
    energy: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    def get_infos(self):
        return self.m, self.n, self.k, self.numOp, self.dbyte

    def get_flops(self) -> float:
        t = self.type
        if t == LayerType.SOFTMAX:
            return 5 * self.m * self.n * self.numOp
        if t == LayerType.NORM:
            return 5 * self.m * self.n * self.numOp
        if t == LayerType.ACT:
            if "relu" in self.name:
                return 1 * self.m * self.n * self.numOp
            if "glu" in self.name:            # SwiGLU/GeGLU: gate*up + activation
                return (8 + 1) * self.m * self.n * self.numOp
            return 8 * self.m * self.n * self.numOp
        if t in (LayerType.FC, LayerType.MATMUL):
            return 2 * self.m * self.n * self.k * self.numOp
        if t == LayerType.COMM:
            return 0
        raise ValueError(f"get_flops: unsupported layer type {t}")

    def get_size(self):
        """Return (in1, in2, out) traffic in BYTES.

        FC/MATMUL: in1 = activation, in2 = weight/second-operand, out = result.
        NL/COMM:   in1 = out = the activation, in2 = 0 (glu reads two inputs).
        NORM:      reads activation twice (x and the reduction), writes once.
        """
        in1 = self.numOp * self.m * self.k * self.dbyte
        in2 = self.numOp * self.n * self.k * self.dbyte
        out = self.numOp * self.m * self.n * self.dbyte

        if self.type in (LayerType.SOFTMAX, LayerType.ACT, LayerType.COMM):
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = self.numOp * self.m * self.n * self.dbyte if "glu" in self.name else 0
            out = self.numOp * self.m * self.n * self.dbyte
        elif self.type == LayerType.NORM:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = in1
            out = in1
        return in1, in2, out

    def weight_bytes(self) -> float:
        """Bytes of the stationary operand (the weight matrix for FC)."""
        if self.type == LayerType.FC:
            return self.numOp * self.n * self.k * self.dbyte
        return 0.0
