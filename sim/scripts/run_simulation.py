"""
Run the CAPIM simulation: all four drivers over collected traces, with the LP-Spec
metrics (token/s, token/J, EDP), plus optional σ_th and LP-Spec-L sweeps.

Two traces, one per SD method, both for the SAME dataset (handover §5 step 4):
  - --eagle-trace  : consumed by AR, EAGLE-2/NPU, and CAPIM (EAGLE confidence).
  - --medusa-trace : consumed by the LP-Spec baseline (static MEDUSA tree + DTP).
If --medusa-trace is omitted, LP-Spec is skipped with a warning.

Usage:
  python3 -m sim.scripts.run_simulation \
      --eagle-trace traces/vicuna7b_eagle_alpaca.json \
      --medusa-trace traces/vicuna7b_medusa_alpaca.json
  ... --sweep-sigma            # σ_th pruning vs false-neg knee (CAPIM)
  ... --sweep-lp-L             # LP-Spec L band + objective-optimal L
"""

import argparse

from sim.config.models import VICUNA_7B
from sim.drivers import autoregressive, capim, eagle_npu, lp_spec
from sim.report import comparison_table, export_csv
from sim.sweeps import sweep_lp_spec_L, sweep_sigma_th, trace_percentiles
from sim.trace.schema import TraceDataset


def _limit_prompts(trace: TraceDataset, max_prompts: int) -> TraceDataset:
    if max_prompts <= 0:
        return trace
    keep = set()
    steps = []
    for s in trace.steps:
        if s.prompt_id not in keep and len(keep) >= max_prompts:
            continue
        keep.add(s.prompt_id)
        steps.append(s)
    trace.steps = steps
    return trace


def _dataset(trace: TraceDataset) -> str:
    # A trace covers exactly one dataset; the collector records it in metadata, and
    # individual steps also carry it.
    return trace.metadata.get("dataset") or (trace.steps[0].dataset if trace.steps else "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle-trace", required=True,
                    help="EAGLE trace (AR / EAGLE-2-NPU / CAPIM)")
    ap.add_argument("--medusa-trace", default=None,
                    help="MEDUSA trace (LP-Spec baseline); LP-Spec skipped if absent")
    ap.add_argument("--sigma-th", type=float, default=-4.0)
    ap.add_argument("--mu-th", type=int, default=4)
    ap.add_argument("--lp-L", type=int, default=16, help="LP-Spec verified tree size")
    ap.add_argument("--lp-selection", default="greedy_headk")
    ap.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--sweep-sigma", action="store_true")
    ap.add_argument("--sweep-lp-L", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    model = VICUNA_7B

    print(f"Loading EAGLE trace {args.eagle_trace} ...")
    etrace = _limit_prompts(TraceDataset.load(args.eagle_trace), args.max_prompts)
    dataset = _dataset(etrace)
    print(f"  {len(etrace.steps)} steps, dataset={dataset}")

    mtrace = None
    if args.medusa_trace:
        print(f"Loading MEDUSA trace {args.medusa_trace} ...")
        mtrace = _limit_prompts(TraceDataset.load(args.medusa_trace), args.max_prompts)
        mds = _dataset(mtrace)
        print(f"  {len(mtrace.steps)} steps, dataset={mds}")
        assert mds == dataset, (
            f"EAGLE/MEDUSA traces cover different datasets: {dataset} vs {mds}")
    else:
        print("  (no --medusa-trace -> LP-Spec baseline skipped)")

    pcts = trace_percentiles(etrace)
    print(f"  cumulative_log_prob percentiles p10/p50/p90 = "
          f"{pcts[10]:.2f} / {pcts[50]:.2f} / {pcts[90]:.2f}")

    print("\nRunning drivers ...")
    ar = autoregressive.simulate(model, etrace)
    en = eagle_npu.simulate(model, etrace)
    cap = capim.simulate(model, etrace,
                         capim.CapimConfig(sigma_th=args.sigma_th, mu_th=args.mu_th))
    results = [ar, en, cap]
    if mtrace is not None:
        lp = lp_spec.simulate(model, mtrace,
                              lp_spec.LPSpecConfig(L=args.lp_L, selection=args.lp_selection))
        results.insert(2, lp)

    print(f"\n=== {dataset} ===")
    base = "LP-Spec" if mtrace is not None else "EAGLE-2/NPU"
    print(comparison_table(results, baseline_driver=base))

    if args.sweep_sigma:
        print("\n=== σ_th sweep (pruning vs false-neg) ===")
        sigmas = [float("-inf"), pcts[90], pcts[50], pcts[10], -2.5]
        for p in sweep_sigma_th(etrace, sigmas):
            print(f"  σ={p.sigma_th:>7.2f}  prune={p.pruning_ratio:5.1%}  "
                  f"false_neg={p.false_neg_rate:5.1%}  mean_μ={p.mean_mu:5.1f}")

    if args.sweep_lp_L and mtrace is not None:
        print("\n=== LP-Spec L sweep ===")
        sw = sweep_lp_spec_L(model, mtrace, selection=args.lp_selection)
        for L in sorted(sw.by_L):
            s = sw.by_L[L]
            print(f"  L={L:>3}  token/s={s.token_per_s_mean:7.2f}  "
                  f"token/J={s.token_per_j_mean:7.2f}  EDP={s.edp_mean:8.3g}")
        print(f"  best L (throughput) = {sw.best_L_throughput}, "
              f"best L (energy) = {sw.best_L_energy}, tree_size = {sw.tree_size}")

    if args.csv:
        export_csv(results, args.csv)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
