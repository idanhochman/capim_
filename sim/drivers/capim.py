"""
CAPIM driver (also serves the EAGLE-2/NPU ablation via config).

Per decode step, replayed from an EAGLE-2 trace:
  1. DRAFT   - EAGLE head, D depths; the live σ_th gate terminates a branch
               before drafting its descendants, so the drafted width at each
               depth is the number of nodes that survive the gate.  Weights on
               PIM, nonlinear on NPU -> the PIM<->NPU ping-pong.
  2. PRUNE   - μ = surviving tree size  (scheduler.prune_tree on cumulative_log_prob).
  3. ROUTE   - binary: FC -> PIM if μ < μ_th else NPU; attention always PIM (scheduler.route).
  4. VERIFY  - one target forward over μ tokens, sequential/additive.
  5. ACCEPT  - committed accepted prefix that survives the gate, + 1 bonus token.

Config flags reproduce the EAGLE-2/NPU baseline:
  all_npu=True, sigma_th=-inf, mu_th=+inf  -> full tree drafted+verified on NPU,
  no gate, no PIM route, no crossings.
"""

from __future__ import annotations

from dataclasses import dataclass

from sim.config.models import ModelConfig
from sim.drivers.base import (
    DriverResult,
    StepRecord,
    compose_sequential,
    tag,
)
from sim.drivers.common import (
    cost_target_pass,
    router_all_npu,
    router_eagle_draft,
)
from sim.kernel.layer import Device as Dev
from sim.kernel.npu import MobileNPU
from sim.kernel.pim import LPDDR5PIM
from sim.layers.eagle import build_eagle_draft_step
from sim.scheduler import route
from sim.trace.schema import DecodeStepTrace, TraceDataset


@dataclass
class CapimConfig:
    sigma_th: float = float("-inf")   # cumulative-log-prob gate
    mu_th: int = 4                    # binary route threshold
    all_npu: bool = False             # True -> EAGLE-2/NPU ablation
    name: str = "CAPIM"


def _generated_by_depth(step: DecodeStepTrace, sigma_th: float):
    """Map depth -> count of nodes CAPIM actually *generates* under the gate.

    A node must be generated (drafted once) before its cumulative_log_prob is
    known and the gate can judge it.  The gate then decides whether to *expand*
    it (draft its children).  So the cost-relevant set is "nodes whose parent
    was expanded" = "nodes whose parent survived the gate" (a depth-0 node hangs
    off the always-present root).  By monotonicity of cumulative_log_prob,
    "parent survived" implies the whole ancestor path survived, so this single
    check suffices.  Crucially this set *includes* the boundary nodes that fail
    the gate themselves but were still drafted once to be scored before their
    branch is killed -- the generation cost the proactive gate cannot avoid.
    """
    if sigma_th == float("-inf"):
        out = {}
        for n in step.nodes:
            out[n.depth] = out.get(n.depth, 0) + 1
        return out
    survived = [n.cumulative_log_prob >= sigma_th for n in step.nodes]
    out = {}
    for n in step.nodes:
        if n.depth == 0:
            generated = True                       # parent = root, always expanded
        else:
            p = n.parent_idx
            generated = 0 <= p < len(step.nodes) and survived[p]
        if generated:
            out[n.depth] = out.get(n.depth, 0) + 1
    return out


def _draft_cost(model, step, sigma_th, all_npu, npu, pim):
    gen = _generated_by_depth(step, sigma_th)
    from sim.drivers.base import Composed
    total = Composed()
    ctx0 = step.context_length
    for depth in sorted(gen):
        width = gen[depth]
        if width <= 0:
            continue
        layers = build_eagle_draft_step(model, width=width, ctx=ctx0 + depth)
        tag(layers, router_all_npu if all_npu else router_eagle_draft)
        total.merge(compose_sequential(layers, npu, pim, count_crossings=not all_npu))
    return total


def _effective_accept(step, sigma_th):
    surviving_accepted = sum(
        1 for n in step.nodes
        if n.accepted and (sigma_th == float("-inf") or n.cumulative_log_prob >= sigma_th)
    )
    return min(step.accepted_length, surviving_accepted)


def simulate(
    model: ModelConfig,
    trace: TraceDataset,
    config: CapimConfig,
    npu: MobileNPU = None,
    pim: LPDDR5PIM = None,
) -> DriverResult:
    npu = npu or MobileNPU()
    pim = pim or LPDDR5PIM()
    result = DriverResult(driver=config.name, model=model.name)

    for step in trace.steps:
        # 1-2. draft (gated) + prune
        draft = _draft_cost(model, step, config.sigma_th, config.all_npu, npu, pim)
        # mu = VERIFY tree size = STRICT survivors (nodes that themselves pass the
        # gate).  This deliberately differs from the DRAFT set (_generated_by_depth),
        # which also includes the boundary nodes that were drafted+scored but failed
        # the gate -- you must generate a node to score it, but a pruned node is not
        # verified.  Monotonicity of cumulative_log_prob guarantees the survivor set
        # is ancestor-closed, so this count is a valid connected pruned tree.
        #
        # TODO(verify-frontier): evaluate a variant that verifies ALL generated nodes
        # (survivors + boundary), since they are already drafted+scored.  Rationale:
        # PIM verify cost is quantized to ceil(m/N_ALU) passes (kernel/pim.py), so
        # adding boundary nodes up to the next multiple of N_ALU is ~FREE, and any
        # ACCEPTED boundary node (a gate false-negative) would be recovered ->
        # longer accept, higher acceptance rate at ~zero extra PIM cost.  Open
        # interactions to resolve before adopting: (a) the gate's role shifts from
        # "save verification" to "save draft expansion" (narrative change); (b) mu is
        # also the routing signal, so a larger verify-mu could tip a step over mu_th
        # onto the costly NPU path -- likely want to ROUTE on survivor-mu but
        # COST-verify on generated-mu (two mus); (c) only free within a pass -- if the
        # frontier straddles a multiple of N_ALU it costs a full extra pass.
        if config.sigma_th == float("-inf"):
            mu = step.tree_size
        else:
            mu = sum(1 for n in step.nodes if n.cumulative_log_prob >= config.sigma_th)
        mu = max(1, mu)

        # 3. route
        if config.all_npu:
            fc_dev = Dev.NPU
        else:
            fc_dev = Dev.PIM if route(mu, config.mu_th) == "PIM" else Dev.NPU

        # 4. verify
        verify = cost_target_pass(
            model, m=mu, ctx=step.context_length, fc_device=fc_dev,
            npu=npu, pim=pim, all_npu=config.all_npu, concurrent=False,
        )

        # 5. accept
        tokens = _effective_accept(step, config.sigma_th) + 1  # + bonus token

        time_s = draft.time_s + verify.time_s
        energy = [draft.energy_j[i] + verify.energy_j[i] for i in range(4)]
        tdev = {k: draft.time_by_device.get(k, 0.0) + verify.time_by_device.get(k, 0.0)
                for k in ("NPU", "PIM")}
        ttype = {}
        for d in (draft.time_by_type, verify.time_by_type):
            for k, v in d.items():
                ttype[k] = ttype.get(k, 0.0) + v

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
    return result
