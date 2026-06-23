"""
Autoregressive baseline (lower bound / normalization anchor).

No speculation: one full target forward per emitted token, on the NPU, at the
token's REAL context length.  Replayed from a trace so it generates the same
number of tokens at the same contexts as the SD runs (fixes the old baseline's
KV omission + step-mean context -> now token-weighted per-token contexts).
"""

from __future__ import annotations

from sim.config.models import ModelConfig
from sim.drivers.base import DriverResult, StepRecord
from sim.drivers.common import cost_target_pass
from sim.kernel.layer import Device as Dev
from sim.kernel.npu import MobileNPU
from sim.kernel.pim import LPDDR5PIM
from sim.trace.schema import TraceDataset


def simulate(model: ModelConfig, trace: TraceDataset,
             npu: MobileNPU = None, pim: LPDDR5PIM = None) -> DriverResult:
    npu = npu or MobileNPU()
    pim = pim or LPDDR5PIM()
    result = DriverResult(driver="Autoregressive", model=model.name)

    for step in trace.steps:
        # AR emits the same #tokens this step as SD committed (accepted + bonus),
        # one forward each, at growing context.
        n_tokens = step.accepted_length + 1
        time_s = 0.0
        energy = [0.0, 0.0, 0.0, 0.0]
        tdev = {"NPU": 0.0, "PIM": 0.0}
        ttype = {}
        for j in range(n_tokens):
            c = cost_target_pass(
                model, m=1, ctx=step.context_length + j, fc_device=Dev.NPU,
                npu=npu, pim=pim, all_npu=True, concurrent=False,
            )
            time_s += c.time_s
            for i in range(4):
                energy[i] += c.energy_j[i]
            for k, v in c.time_by_device.items():
                tdev[k] = tdev.get(k, 0.0) + v
            for k, v in c.time_by_type.items():
                ttype[k] = ttype.get(k, 0.0) + v

        result.steps.append(StepRecord(
            prompt_id=step.prompt_id,
            dataset=step.dataset,
            step_id=step.step_id,
            tokens_emitted=n_tokens,
            time_s=time_s,
            energy_j=sum(energy),
            time_by_device=tdev,
            energy_by_component={
                "off_mem": energy[0], "on_chip": energy[1],
                "alu": energy[2], "comm": energy[3],
            },
            time_by_type=ttype,
        ))
    return result
