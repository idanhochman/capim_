"""
EAGLE-2 / NPU-only baseline.

Same algorithm + same traces as CAPIM, but everything on the NPU with the
confidence gate disabled and no PIM route.  Isolates exactly CAPIM's hardware +
gating contribution.  It is literally the CAPIM driver with all_npu=True,
sigma_th=-inf, mu_th=+inf -> no behaviour of its own beyond config.
"""

from __future__ import annotations

from sim.config.models import ModelConfig
from sim.drivers.capim import CapimConfig, simulate as _simulate
from sim.kernel.npu import MobileNPU
from sim.kernel.pim import LPDDR5PIM
from sim.trace.schema import TraceDataset


def simulate(model: ModelConfig, trace: TraceDataset,
             npu: MobileNPU = None, pim: LPDDR5PIM = None):
    config = CapimConfig(
        sigma_th=float("-inf"),
        mu_th=10 ** 9,
        all_npu=True,
        name="EAGLE-2/NPU",
    )
    return _simulate(model, trace, config, npu=npu, pim=pim)
