"""
LP-Spec baseline driver — MEDUSA + retrospective DTP (trace-replay) + concurrent
NPU‖PIM verification.

Per decode step, replayed from a MEDUSA trace (handover.md §1–5):
  1. DRAFT  : K=5 MEDUSA heads, one parallel shot on the NPU ("free tail").
  2. SELECT : the DTP picks which nodes to verify from a retrospective per-(head, k)
              acceptance histogram (`sim.lp_spec_dtp`).  This is the content-blind
              counterpart to CAPIM's live σ_th gate — same greedy ∏ p construction,
              but the accuracies come from PAST verification history, not this step.
  3. VERIFY : ONE target forward over the kept tree (m = |kept|), composed
              CONCURRENTLY (FC on NPU ‖ attention on PIM, nonlinear additive on NPU)
              — the DAU tensor-parallel makespan.
  4. ACCEPT : the measured accepted path truncated to the kept tree, + 1 bonus.

`L_spec` (LP-Spec's verified tree size) is NOT derived from a reconstructed
hardware estimator here; it is the swept knob `L` (handover §3).  Sweep it with
`sim.sweeps.sweep_lp_spec_L` and report LP-Spec as a band over L plus its post-hoc
objective optimum.

Histogram causality: at step t the selection uses history from steps < t only;
step t's observations are folded in AFTER costing it.  Step 0 is a cold start that
verifies the full static tree.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim import lp_spec_dtp
from sim.config.models import ModelConfig
from sim.drivers.base import (
    DriverResult,
    StepRecord,
    compose_sequential,
    tag,
)
from sim.drivers.common import cost_target_pass, router_all_npu
from sim.kernel.layer import Device as Dev, Layer, LayerType
from sim.kernel.npu import MobileNPU
from sim.kernel.pim import LPDDR5PIM
from sim.layers.medusa import MEDUSA_TREE_SIZE, build_medusa_draft
from sim.trace.schema import TraceDataset


@dataclass
class LPSpecConfig:
    L: int = 16                                # verified tree size (swept; LP-Spec's L_spec)
    selection: str = "greedy_headk"            # see lp_spec_dtp.select_kept
    medusa_num_heads: int = 5
    charge_gather: bool = True                 # charge the column-split output gather
    name: str = "LP-Spec"


def _granularity(selection: str) -> str:
    return "node" if selection == "greedy_node" else "headk"


def _gather_cost(model: ModelConfig, m: int, pim: LPDDR5PIM):
    """Per-layer output-gather comm for the DAU column-wise split: the NPU and PIM
    output column-slices of each decoder layer must be concatenated on the bus.
    PAPI models this as an X2G/COMM layer; `compose_concurrent` omits it (which
    would slightly favour LP-Spec), so charge ~m×d_model per decoder layer here.
    """
    hop = Layer("dau_gather", LayerType.COMM, m=m, n=model.d_model, dbyte=model.bytes_per_param)
    r = pim.cost(hop)
    return r.time_s, r.energy_j


def simulate(model: ModelConfig, trace: TraceDataset, config: LPSpecConfig = None,
             npu: MobileNPU = None, pim: LPDDR5PIM = None) -> DriverResult:
    config = config or LPSpecConfig()
    npu = npu or MobileNPU()
    pim = pim or LPDDR5PIM()
    result = DriverResult(driver=config.name, model=model.name)

    hist = lp_spec_dtp.DTPHist(granularity=_granularity(config.selection))

    for t, step in enumerate(trace.steps):
        kp = lp_spec_dtp.k_pred_map(step)
        pp = lp_spec_dtp.parent_pos_map(step)

        # 1. MEDUSA draft: K parallel heads on the NPU (free tail, no crossings)
        heads = build_medusa_draft(model, medusa_num_heads=config.medusa_num_heads)
        tag(heads, router_all_npu)
        draft = compose_sequential(heads, npu, pim, count_crossings=False)

        # 2. DTP select (causal: history < t only; step 0 = full-tree cold start)
        kept = lp_spec_dtp.select_kept(step, t, config.L, config.selection, hist, kp, pp)
        m = max(1, len(kept))

        # 3. concurrent verify over the kept tree
        verify = cost_target_pass(
            model, m=m, ctx=step.context_length, fc_device=Dev.NPU,
            npu=npu, pim=pim, all_npu=False, concurrent=True,
        )

        time_s = draft.time_s + verify.time_s
        energy = [draft.energy_j[i] + verify.energy_j[i] for i in range(4)]
        tdev = {k: draft.time_by_device.get(k, 0.0) + verify.time_by_device.get(k, 0.0)
                for k in ("NPU", "PIM")}
        ttype = {}
        for d in (draft.time_by_type, verify.time_by_type):
            for k, v in d.items():
                ttype[k] = ttype.get(k, 0.0) + v

        # 3b. output-gather comm for the column-split (n_layers hops)
        if config.charge_gather:
            g_t, g_e = _gather_cost(model, m, pim)
            time_s += g_t * model.n_layers
            for i in range(4):
                energy[i] += g_e[i] * model.n_layers
            tdev["PIM"] = tdev.get("PIM", 0.0) + g_t * model.n_layers
            ttype["COMM"] = ttype.get("COMM", 0.0) + g_t * model.n_layers

        # 4. accept: measured accepted path truncated to the kept tree, + bonus
        tokens = lp_spec_dtp.effective_accept(step, kept) + 1

        result.steps.append(StepRecord(
            prompt_id=step.prompt_id,
            dataset=step.dataset,
            step_id=step.step_id,
            tokens_emitted=tokens,
            time_s=time_s,
            energy_j=sum(energy),
            time_by_device=tdev,
            energy_by_component={
                "off_mem": energy[0], "on_chip": energy[1],
                "alu": energy[2], "comm": energy[3],
            },
            time_by_type=ttype,
        ))

        # 5. fold step t into the histogram (AFTER costing -> strict causality)
        hist.update(step, kp, pp)

    return result
