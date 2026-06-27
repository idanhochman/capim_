"""Smoke tests for the rewritten kernel + drivers (synthetic trace, no GPU)."""

import math

from sim.config.models import LLAMA2_7B
from sim.kernel.layer import Layer, LayerType, Device as Dev
from sim.kernel.npu import MobileNPU
from sim.kernel.pim import LPDDR5PIM
from sim.layers.target import build_decoder_layer, build_prefill
from sim.layers.eagle import build_eagle_draft_step
from sim.layers.medusa import build_medusa_draft, MEDUSA_TREE_SIZE
from sim.drivers import autoregressive, capim, eagle_npu, lp_spec
from sim.report import summarize, comparison_table
from sim.sweeps import sweep_sigma_th, sweep_lp_spec_L, trace_percentiles
from sim.trace.schema import (
    make_synthetic_trace,
    make_synthetic_medusa_trace,
    DecodeStepTrace,
    TokenNode,
)
from sim import lp_spec_dtp
from sim.layers.medusa import VICUNA_7B_STAGE2, tree_topology


def test_layer_flops_size():
    fc = Layer("fc", LayerType.FC, m=2, n=4096, k=4096)
    assert fc.get_flops() == 2 * 2 * 4096 * 4096
    sm = Layer("softmax", LayerType.SOFTMAX, m=2, n=128, numOp=32)
    assert sm.get_flops() == 5 * 2 * 128 * 32


def test_npu_pim_cost_sane():
    npu, pim = MobileNPU(), LPDDR5PIM()
    fc = Layer("ff", LayerType.FC, m=1, n=11008, k=4096, dbyte=1)
    rn = npu.cost(fc)
    rp = pim.cost(fc)
    assert rn.time_s > 0 and rp.time_s > 0
    assert rn.total_energy_j > 0 and rp.total_energy_j > 0
    # GEMV is compute-bound on PIM (ridge 0.008)
    assert rp.bound == "compute"


def test_builders():
    dec = build_decoder_layer(LLAMA2_7B, m=4, ctx=256)
    assert any(l.type == LayerType.MATMUL for l in dec)
    assert any(l.type == LayerType.SOFTMAX for l in dec)
    eagle = build_eagle_draft_step(LLAMA2_7B, width=8, ctx=256)
    assert eagle[0].name == "fusion_fc"
    med = build_medusa_draft(LLAMA2_7B)
    assert MEDUSA_TREE_SIZE == 63
    # K=5 heads -> 5 lm_heads
    assert sum(1 for l in med if "lmhead" in l.name) == 5


def test_eagle_draft_is_small_vs_verify():
    """The EAGLE draft (1 layer) should be a small fraction of a 32-layer verify."""
    draft = build_eagle_draft_step(LLAMA2_7B, width=8, ctx=256)
    draft_flops = sum(l.get_flops() for l in draft)
    verify_layer = build_decoder_layer(LLAMA2_7B, m=8, ctx=256)
    verify_flops = sum(l.get_flops() for l in verify_layer) * LLAMA2_7B.n_layers
    # draft has a vocab head so it's not negligible, but well under the full verify
    assert draft_flops < 0.2 * verify_flops


def test_eagle_draft_has_single_norm():
    """EAGLE's draft head layer is index 0, so input_layernorm is never built
    (cnets1.py:399), and the head reads the layer output directly (no final norm).
    Net: ONE RMSNorm in the draft block vs TWO in a full target layer.  Pins the
    eagle_draft=True fix so build_decoder_layer can't silently re-inflate it."""
    target = build_decoder_layer(LLAMA2_7B, m=8, ctx=256)
    eagle_layer = build_decoder_layer(LLAMA2_7B, m=8, ctx=256, eagle_draft=True)
    assert sum(1 for l in target if l.type == LayerType.NORM) == 2
    assert sum(1 for l in eagle_layer if l.type == LayerType.NORM) == 1
    # the kept norm is the post-attention one (norm1), not the trailing norm2
    assert [l.name for l in eagle_layer if l.type == LayerType.NORM] == ["norm1"]
    # full draft step: exactly one NORM, and ff3 -> lm_head is contiguous (no norm
    # between the layer output and the head -> no spurious PIM<->NPU round-trip)
    draft = build_eagle_draft_step(LLAMA2_7B, width=8, ctx=256)
    assert sum(1 for l in draft if l.type == LayerType.NORM) == 1
    names = [l.name for l in draft]
    assert names[names.index("ff3") + 1] == "lm_head"


def test_concurrent_gather_per_split_kernel():
    """compose_concurrent charges one DAU output-gather COMM per split kernel
    (every FC and attention MATMUL), r-weighted, in the comm energy slot only."""
    from sim.drivers.base import compose_concurrent
    npu, pim = MobileNPU(), LPDDR5PIM()
    dec = build_decoder_layer(LLAMA2_7B, m=16, ctx=256)
    n_split = sum(1 for l in dec if l.type in (LayerType.FC, LayerType.MATMUL))
    c = compose_concurrent(dec, npu, pim)
    # one crossing per split kernel (FC + attention MATMUL)
    assert c.crossings == n_split
    assert c.time_by_type.get("COMM", 0.0) > 0.0
    # gather lands on PIM (external bus) and only in the comm energy slot
    assert c.energy_j[3] > 0.0
    # an NL-only list draws no gather (nothing is column-split)
    nls = [l for l in dec if l.type in (LayerType.SOFTMAX, LayerType.ACT, LayerType.NORM)]
    assert compose_concurrent(nls, npu, pim).crossings == 0


def test_concurrent_gather_scales_with_pim_share():
    """A stronger PIM takes a larger column share -> larger (1-r) output slice ->
    larger gather (the gather moves PIM's produced slice to the NPU)."""
    from sim.drivers.base import compose_concurrent
    fc = [Layer("ff", LayerType.FC, m=16, n=11008, k=4096, dbyte=1)]
    strong = compose_concurrent(fc, MobileNPU(), LPDDR5PIM())                  # default PIM
    weak = compose_concurrent(fc, MobileNPU(), LPDDR5PIM(int8_gops=40.96e9))   # 10x weaker PIM
    assert strong.time_by_type["COMM"] > weak.time_by_type["COMM"]


def test_drivers_run_and_order():
    trace = make_synthetic_trace(n_steps=40, tree_size=24, acceptance_rate=0.35)
    mtrace = make_synthetic_medusa_trace(n_steps=40, tree_choices=VICUNA_7B_STAGE2)
    ar = autoregressive.simulate(LLAMA2_7B, trace)
    en = eagle_npu.simulate(LLAMA2_7B, trace)
    cap = capim.simulate(LLAMA2_7B, trace, capim.CapimConfig(sigma_th=-4.0, mu_th=4))
    lp = lp_spec.simulate(LLAMA2_7B, mtrace, lp_spec.LPSpecConfig(L_spec=16))
    for r in (ar, en, cap, lp):
        assert len(r.steps) == 40
        s = summarize(r)
        assert s.token_per_s_mean > 0
        assert s.token_per_j_mean > 0
    # SD should beat AR on throughput
    assert summarize(en).token_per_s_mean > summarize(ar).token_per_s_mean
    print("\n" + comparison_table([ar, en, lp, cap]))


# --------------------------------------------------------------------------
# LP-Spec DTP trace-replay model (handover §5 item 7)
# --------------------------------------------------------------------------

def _two_head_step():
    """A hand-built 2-head tree to exercise the reachable denominator.

    depth0: n0 (accepted), n1 (rejected)
    depth1: children of n0 = c00 (accepted), c01 (rejected)
            children of n1 = c10, c11  -> NOT reachable (parent n1 rejected)
    """
    nodes = [
        TokenNode(depth=0, token_id=1, log_prob=-0.1, cumulative_log_prob=-0.1,
                  parent_idx=-1, accepted=True, layer_idx=0),
        TokenNode(depth=0, token_id=2, log_prob=-0.5, cumulative_log_prob=-0.5,
                  parent_idx=-1, accepted=False, layer_idx=1),
        TokenNode(depth=1, token_id=3, log_prob=-0.2, cumulative_log_prob=-0.3,
                  parent_idx=0, accepted=True, layer_idx=0),
        TokenNode(depth=1, token_id=4, log_prob=-0.7, cumulative_log_prob=-0.8,
                  parent_idx=0, accepted=False, layer_idx=1),
        TokenNode(depth=1, token_id=5, log_prob=-0.3, cumulative_log_prob=-0.8,
                  parent_idx=1, accepted=False, layer_idx=2),
        TokenNode(depth=1, token_id=6, log_prob=-0.9, cumulative_log_prob=-1.4,
                  parent_idx=1, accepted=False, layer_idx=3),
    ]
    return DecodeStepTrace(step_id=0, context_length=10, nodes=nodes,
                           accepted_length=2, dataset="synthetic", prompt_id=0)


def test_dtp_kpred_is_sibling_rank():
    step = _two_head_step()
    lp_spec_dtp.assert_sibling_rank_order(step)
    kp = lp_spec_dtp.k_pred_map(step)
    # depth-0 nodes: ranks 0,1 (single parent = root)
    assert kp[(0, 0)] == 0 and kp[(0, 1)] == 1
    # children of n0 (layer_idx 0,1) -> ranks 0,1
    assert kp[(1, 0)] == 0 and kp[(1, 1)] == 1
    # children of n1 (layer_idx 2,3) -> ranks 0,1 again (per-parent, not per-layer)
    assert kp[(1, 2)] == 0 and kp[(1, 3)] == 1
    # k_pred derived from a real MEDUSA tree must equal path[-1]
    for nd in tree_topology(VICUNA_7B_STAGE2):
        assert nd["k_pred"] == nd["path"][-1]


def test_dtp_reachable_denominator():
    step = _two_head_step()
    hist = lp_spec_dtp.DTPHist(granularity="headk")
    hist.update(step)
    # (0,0) accepted, (0,1) reachable-not-accepted
    assert hist.counts[(0, 0)] == [1, 1]
    assert hist.counts[(0, 1)] == [0, 1]
    # depth-1 reachable observations come ONLY from n0's children (n1 rejected)
    assert hist.counts[(1, 0)] == [1, 1]   # c00 accepted
    assert hist.counts[(1, 1)] == [0, 1]   # c01 rejected, but reachable
    # n1's children must NOT inflate the denominator: total depth-1 reachable == 2
    depth1_reach = sum(v[1] for k, v in hist.counts.items() if k[0] == 1)
    assert depth1_reach == 2


def test_dtp_histogram_causality():
    trace = make_synthetic_medusa_trace(n_steps=10, branching=2, max_depth=3, seed=1)
    hist = lp_spec_dtp.DTPHist()
    # fold only the first 3 steps
    for s in trace.steps[:3]:
        hist.update(s)
    after3 = {k: list(v) for k, v in hist.counts.items()}
    # a 4th update must change totals -> proves updates are incremental/causal
    hist.update(trace.steps[3])
    total_before = sum(v[1] for v in after3.values())
    total_after = sum(v[1] for v in hist.counts.values())
    assert total_after > total_before


def test_dtp_topL_ancestor_closed():
    trace = make_synthetic_medusa_trace(n_steps=20, tree_choices=VICUNA_7B_STAGE2, seed=2)
    hist = lp_spec_dtp.DTPHist()
    for s in trace.steps[:5]:
        hist.update(s)
    step = trace.steps[5]
    ranked, _ = lp_spec_dtp.score_nodes(step, hist)
    by_pos = {(n.depth, n.layer_idx): n for n in step.nodes}
    for L in range(1, len(ranked) + 1):
        kept = {(n.depth, n.layer_idx) for n in ranked[:L]}
        for n in ranked[:L]:
            if n.depth > 0:
                par = step.nodes[n.parent_idx]   # parent_idx is a GLOBAL index
                assert (par.depth, par.layer_idx) in kept, \
                    f"L={L}: parent of {(n.depth, n.layer_idx)} not kept"


def test_dtp_effective_accept_monotone_and_bounded():
    trace = make_synthetic_medusa_trace(n_steps=60, tree_choices=VICUNA_7B_STAGE2, seed=3)
    tree_size = trace.metadata["tree_size"]
    Ls = [1, 2, 4, 8, 16, 32, tree_size]
    mean_tokens = []
    for L in Ls:
        res = lp_spec.simulate(LLAMA2_7B, trace, lp_spec.LPSpecConfig(L_spec=L))
        mean_tokens.append(summarize(res).mean_tokens)
    # realised accept is monotone non-decreasing in L
    assert mean_tokens == sorted(mean_tokens), mean_tokens
    # and full-tree (L = tree_size) accept is the max
    assert mean_tokens[-1] == max(mean_tokens)


def test_dtp_cold_start_full_tree():
    step = make_synthetic_medusa_trace(n_steps=1, tree_choices=VICUNA_7B_STAGE2).steps[0]
    hist = lp_spec_dtp.DTPHist()
    all_pos = {(n.depth, n.layer_idx) for n in step.nodes}
    # t == 0 ignores L and verifies the whole static tree
    assert lp_spec_dtp.select_kept(step, 0, 1, "greedy_headk", hist) == all_pos


def test_dtp_oracle_full_bracket_greedy():
    trace = make_synthetic_medusa_trace(n_steps=60, tree_choices=VICUNA_7B_STAGE2, seed=4)
    L = 8
    def toks(sel):
        return summarize(lp_spec.simulate(
            LLAMA2_7B, trace, lp_spec.LPSpecConfig(L_spec=L, selection=sel))).mean_tokens
    greedy = toks("greedy_headk")
    oracle = toks("oracle")
    full = toks("full")
    # oracle keeps the accepted chain first -> upper-bounds greedy at the same L;
    # full verifies everything -> at least as many accepted tokens as greedy.
    assert oracle >= greedy - 1e-9
    assert full >= greedy - 1e-9


def test_lp_spec_L_sweep_optimum():
    trace = make_synthetic_medusa_trace(n_steps=40, tree_choices=VICUNA_7B_STAGE2, seed=5)
    sw = sweep_lp_spec_L(LLAMA2_7B, trace)
    assert sw.tree_size == 63
    assert sw.best_L_throughput in sw.by_L
    assert sw.best_L_energy in sw.by_L
    # every swept point produced positive metrics
    for s in sw.by_L.values():
        assert s.token_per_s_mean > 0 and s.token_per_j_mean > 0


def test_sigma_sweep_monotone():
    trace = make_synthetic_trace(n_steps=30, tree_size=30, acceptance_rate=0.3)
    pcts = trace_percentiles(trace)
    sigmas = [float("-inf"), pcts[10], pcts[50], pcts[90], 0.0]
    pts = sweep_sigma_th(trace, sigmas)
    # pruning ratio should be non-decreasing as sigma_th rises
    ratios = [p.pruning_ratio for p in pts]
    assert ratios == sorted(ratios)
