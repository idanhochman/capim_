"""
Parameter sweeps for calibration:
  - sigma_th sweep: pruning ratio vs false-negative rate (the σ_th knee) -- pure
    trace analysis via scheduler.prune_stats, no hardware model needed.
  - driver sweep: run CAPIM across (sigma_th, mu_th) -> token/s, token/J, EDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sim.config.models import ModelConfig
from sim.drivers import capim as capim_driver
from sim.drivers import lp_spec as lp_spec_driver
from sim.report import Summary, summarize
from sim.scheduler import prune_stats
from sim.trace.schema import TraceDataset


@dataclass
class SigmaPoint:
    sigma_th: float
    pruning_ratio: float     # fraction of nodes removed (mean over steps)
    false_neg_rate: float    # fraction of accepted nodes wrongly pruned (mean over steps)
    mean_mu: float           # mean surviving tree size


def sweep_sigma_th(trace: TraceDataset, sigmas: List[float]) -> List[SigmaPoint]:
    points = []
    for sig in sigmas:
        prs, fnr, mus = [], [], []
        for step in trace.steps:
            st = prune_stats(step, sig)
            prs.append(st["pruning_ratio"])
            fnr.append(st["false_neg_rate"])
            mus.append(st["pruned_size"])
        n = max(1, len(prs))
        points.append(SigmaPoint(
            sigma_th=sig,
            pruning_ratio=sum(prs) / n,
            false_neg_rate=sum(fnr) / n,
            mean_mu=sum(mus) / n,
        ))
    return points


def sweep_capim(model: ModelConfig, trace: TraceDataset,
                sigmas: List[float], mu_ths: List[int]):
    """Return list of (sigma, mu_th, Summary) for a joint sweep."""
    out = []
    for sig in sigmas:
        for mu_th in mu_ths:
            cfg = capim_driver.CapimConfig(sigma_th=sig, mu_th=mu_th, name="CAPIM")
            res = capim_driver.simulate(model, trace, cfg)
            out.append((sig, mu_th, summarize(res)))
    return out


@dataclass
class LPSpecSweep:
    by_L: dict                 # L -> Summary
    best_L_throughput: int     # argmax token/s
    best_L_energy: int         # argmax token/J
    tree_size: int


def _trace_tree_size(trace: TraceDataset) -> int:
    ts = trace.metadata.get("tree_size")
    if ts:
        return int(ts)
    return trace.steps[0].tree_size if trace.steps else 0


def sweep_lp_spec_L(model: ModelConfig, trace: TraceDataset,
                    L_values: List[int] = None, selection: str = "greedy_headk",
                    npu=None, pim=None) -> LPSpecSweep:
    """Sweep LP-Spec's verified tree size L and read off the objective optima.

    Range defaults to 1 … tree_size//2 (handover §5).  If an optimum lands on the
    upper edge the range is extended once toward tree_size and a warning printed
    (the boundary-optimum guard).
    """
    tree_size = _trace_tree_size(trace)
    if L_values is None:
        hi = max(2, tree_size // 2)
        L_values = list(range(1, hi + 1))

    def run(Ls):
        out = {}
        for L in Ls:
            cfg = lp_spec_driver.LPSpecConfig(L_spec=L, selection=selection)
            res = lp_spec_driver.simulate(model, trace, cfg, npu=npu, pim=pim)
            out[L] = summarize(res)
        return out

    by_L = run(L_values)

    def argmax(metric):
        return max(by_L, key=lambda L: getattr(by_L[L], metric))

    best_tps = argmax("token_per_s_mean")
    best_tpj = argmax("token_per_j_mean")

    # Boundary-optimum guard: if either optimum sits on the current upper edge and
    # there is headroom toward the full tree, extend the sweep once.
    upper = max(L_values)
    if (best_tps == upper or best_tpj == upper) and upper < tree_size:
        ext = list(range(upper + 1, tree_size + 1))
        print(f"[sweep_lp_spec_L] optimum at upper edge L={upper}; "
              f"extending to L={tree_size} (boundary-optimum guard).")
        by_L.update(run(ext))
        best_tps = argmax("token_per_s_mean")
        best_tpj = argmax("token_per_j_mean")

    return LPSpecSweep(by_L=by_L, best_L_throughput=best_tps,
                       best_L_energy=best_tpj, tree_size=tree_size)


def trace_percentiles(trace: TraceDataset, ps=(10, 50, 90)) -> dict:
    """Percentiles of node cumulative_log_prob -- to seed the σ_th sweep grid."""
    vals = sorted(n.cumulative_log_prob for s in trace.steps for n in s.nodes)
    if not vals:
        return {p: 0.0 for p in ps}
    out = {}
    for p in ps:
        idx = min(len(vals) - 1, int(p / 100 * len(vals)))
        out[p] = vals[idx]
    return out
