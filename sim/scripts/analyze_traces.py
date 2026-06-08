"""
CAPIM Trace Analyser

Validates the core CAPIM premise: draft tokens with higher confidence
(less negative log_prob) are accepted by the target model at higher rates.

This is the primary empirical validation of the sigma_th pruner — if the
monotone relationship holds, low-confidence branches can be pruned without
significantly impacting acceptance quality.

Usage:
    # Correlation analysis on a trace file
    python sim/scripts/analyze_traces.py --trace traces/qwen25_sanity.json

    # With plot saved to disk (requires matplotlib)
    python sim/scripts/analyze_traces.py --trace traces/qwen25_sanity.json --plot

    # Compare two traces side by side (e.g. alpaca vs gsm8k)
    python sim/scripts/analyze_traces.py \\
        --trace traces/qwen25_alpaca.json \\
        --trace2 traces/qwen25_gsm8k.json
"""

import argparse
import math
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
capim_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, capim_dir)

from sim.trace.schema import TraceDataset


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

BUCKETS = [
    (-100, -10),
    (-10,  -5),
    (-5,   -3),
    (-3,   -2),
    (-2,   -1),
    (-1,    0),
]

# Buckets for cumulative_log_prob (path probability from root to node).
# Wider range than per-node: deep nodes accumulate multiple negative log_probs.
# Calibrated from Alpaca trace: p10=-5.6, p50=-4.1, p90=-2.2, min≈-30.
CUMULATIVE_BUCKETS = [
    (-100, -6),
    (-6,   -5),
    (-5,   -4),
    (-4,   -3),
    (-3,   -2),
    (-2,   -1),
    (-1,    0),
]


def correlation_table(trace: TraceDataset, label: str = "") -> list[dict]:
    """
    Group all nodes by log_prob bucket and compute acceptance rate per bucket.
    Returns a list of dicts with keys: lo, hi, n, acceptance_rate.
    """
    results = []
    all_nodes = [n for s in trace.steps for n in s.nodes]

    for lo, hi in BUCKETS:
        bucket = [
            n for n in all_nodes
            if not math.isnan(n.log_prob) and lo <= n.log_prob < hi
        ]
        if not bucket:
            continue
        acc = sum(1 for n in bucket if n.accepted) / len(bucket)
        results.append({"lo": lo, "hi": hi, "n": len(bucket), "acceptance_rate": acc})

    return results


def print_correlation_table(results: list[dict], label: str = ""):
    if label:
        print(f"\n{'='*55}")
        print(f"  {label}")
    print(f"{'='*55}")
    print(f"  {'log_prob range':<20} {'n nodes':>8}  {'acceptance':>10}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  [{r['lo']:4d}, {r['hi']:3d})          {r['n']:>8,}  {r['acceptance_rate']:>9.1%}")
    print(f"{'='*55}")


def per_depth_correlation(trace: TraceDataset) -> dict[int, list[dict]]:
    """
    Correlation table broken down by depth level.
    Returns {depth: [bucket_results]} so we can see whether the
    log_prob-acceptance relationship holds within each depth independently.
    """
    from collections import defaultdict
    depth_nodes: dict[int, list] = defaultdict(list)
    for s in trace.steps:
        for n in s.nodes:
            if not math.isnan(n.log_prob):
                depth_nodes[n.depth].append(n)

    result = {}
    for depth in sorted(depth_nodes):
        nodes = depth_nodes[depth]
        buckets = []
        for lo, hi in BUCKETS:
            bucket = [n for n in nodes if lo <= n.log_prob < hi]
            if not bucket:
                continue
            acc = sum(1 for n in bucket if n.accepted) / len(bucket)
            buckets.append({"lo": lo, "hi": hi, "n": len(bucket), "acceptance_rate": acc})
        result[depth] = buckets
    return result


def print_per_depth_correlation(data: dict[int, list[dict]], label: str = ""):
    width = 60
    if label:
        print(f"\n{'='*width}")
        print(f"  {label} — confidence vs acceptance per depth")
    print(f"{'='*width}")
    for depth, buckets in data.items():
        if not buckets:
            continue
        print(f"\n  depth {depth}:")
        print(f"  {'log_prob range':<20} {'n nodes':>8}  {'acceptance':>10}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*10}")
        for r in buckets:
            print(f"  [{r['lo']:4d}, {r['hi']:3d})          {r['n']:>8,}  {r['acceptance_rate']:>9.1%}")
    print(f"\n{'='*width}")


def cumulative_correlation_table(trace: TraceDataset) -> list[dict]:
    """
    Group all nodes by cumulative_log_prob bucket and compute acceptance rate.

    This is the calibration table for CAPIM's σ_th pruner, which compares
    cumulative_log_prob against σ_th. Unlike per-node log_prob, cumulative
    encodes both confidence and depth, making it the correct pruning signal.
    """
    all_nodes = [n for s in trace.steps for n in s.nodes]
    results = []
    for lo, hi in CUMULATIVE_BUCKETS:
        bucket = [
            n for n in all_nodes
            if not math.isnan(n.cumulative_log_prob) and lo <= n.cumulative_log_prob < hi
        ]
        if not bucket:
            continue
        acc = sum(1 for n in bucket if n.accepted) / len(bucket)
        results.append({"lo": lo, "hi": hi, "n": len(bucket), "acceptance_rate": acc})
    return results


def print_cumulative_correlation_table(results: list[dict], label: str = ""):
    width = 55
    if label:
        print(f"\n{'='*width}")
        print(f"  {label}")
    print(f"{'='*width}")
    print(f"  {'cumulative log_prob range':<26} {'n nodes':>8}  {'acceptance':>10}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*10}")
    for r in results:
        print(f"  [{r['lo']:4d}, {r['hi']:3d})                    {r['n']:>8,}  {r['acceptance_rate']:>9.1%}")
    print(f"{'='*width}")


def depth_breakdown(trace: TraceDataset) -> dict:
    """Acceptance rate per depth level."""
    from collections import defaultdict
    depth_accepted = defaultdict(int)
    depth_total = defaultdict(int)
    for s in trace.steps:
        for n in s.nodes:
            depth_total[n.depth] += 1
            if n.accepted:
                depth_accepted[n.depth] += 1
    return {
        d: {"n": depth_total[d], "acceptance_rate": depth_accepted[d] / depth_total[d]}
        for d in sorted(depth_total)
    }


def print_depth_breakdown(breakdown: dict, label: str = ""):
    if label:
        print(f"\n{'='*45}")
        print(f"  {label} — acceptance by depth")
    print(f"{'='*45}")
    print(f"  {'depth':>5}  {'n nodes':>8}  {'acceptance':>10}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}")
    for depth, stats in breakdown.items():
        print(f"  {depth:>5}  {stats['n']:>8,}  {stats['acceptance_rate']:>9.1%}")
    print(f"{'='*45}")


def print_summary(trace: TraceDataset, label: str = ""):
    nan_count = sum(
        1 for s in trace.steps for n in s.nodes if math.isnan(n.log_prob)
    )
    total_nodes = sum(len(s.nodes) for s in trace.steps)
    print(f"\nTrace summary ({label or 'unnamed'}):")
    print(f"  Steps                      : {len(trace.steps)}")
    print(f"  Total nodes                : {total_nodes:,}")
    print(f"  NaN log_probs              : {nan_count:,} ({100*nan_count/total_nodes:.1f}%)")
    print(f"  Mean tree size             : {trace.mean_tree_size:.1f}")
    print(f"  Mean accepted (draft only) : {trace.mean_accepted_length:.2f} tokens/step")
    print(f"  Mean tokens generated/step : {trace.mean_accepted_length + 1:.2f}  (= speedup vs autoregressive)")
    print(f"  Acceptance rate            : {trace.mean_acceptance_rate*100:.1f}% (per node)")


def plot_correlation(results: list[dict], label: str, output_path: str):
    import matplotlib.pyplot as plt

    labels = [f"[{r['lo']},{r['hi']})" for r in results]
    rates = [r["acceptance_rate"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, rates, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("log_prob bucket")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title(f"Confidence–Acceptance Correlation\n{label}")
    ax.set_ylim(0, max(rates) * 1.2)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_cumulative_correlation(results: list[dict], label: str, output_path: str):
    """
    Bar chart of cumulative_log_prob bucket vs acceptance rate.
    X-axis uses the same log-prob bucket notation as plot_correlation.
    """
    import matplotlib.pyplot as plt

    x_labels = [f"[{r['lo']}, {r['hi']})" for r in results]
    rates = [r["acceptance_rate"] * 100 for r in results]
    counts = [r["n"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x_labels, rates, color="steelblue", edgecolor="black", linewidth=0.6)

    for bar, rate, count in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"n={count//1000}K", ha="center", va="center", fontsize=8, color="white")

    ax.set_xlabel("Cumulative log_prob bucket  (low confidence → high confidence)", fontsize=11)
    ax.set_ylabel("Acceptance rate (%)", fontsize=11)
    ax.set_title(
        f"Cumulative Path Confidence vs Acceptance Rate\n{label}",
        fontsize=12,
    )
    ax.set_ylim(0, max(rates) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse CAPIM traces: validate confidence-acceptance correlation"
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to primary trace JSON file",
    )
    parser.add_argument(
        "--trace2",
        default=None,
        help="Optional second trace for side-by-side comparison",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save correlation bar chart (requires matplotlib)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for plot output (default: results/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load primary trace
    trace1 = TraceDataset.load(args.trace)
    label1 = os.path.splitext(os.path.basename(args.trace))[0]

    print_summary(trace1, label1)
    results1 = correlation_table(trace1, label1)
    print_correlation_table(results1, f"{label1} — per-node confidence vs acceptance")
    cum_results1 = cumulative_correlation_table(trace1)
    print_cumulative_correlation_table(cum_results1, f"{label1} — cumulative confidence vs acceptance (σ_th signal)")
    print_depth_breakdown(depth_breakdown(trace1), label1)
    print_per_depth_correlation(per_depth_correlation(trace1), label1)

    if args.plot:
        plot_path = os.path.join(args.output_dir, f"{label1}_correlation.png")
        plot_correlation(results1, label1, plot_path)
        cum_plot_path = os.path.join(args.output_dir, f"{label1}_cumulative_correlation.png")
        plot_cumulative_correlation(cum_results1, label1, cum_plot_path)

    # Optional second trace
    if args.trace2:
        trace2 = TraceDataset.load(args.trace2)
        label2 = os.path.splitext(os.path.basename(args.trace2))[0]
        print_summary(trace2, label2)
        results2 = correlation_table(trace2, label2)
        print_correlation_table(results2, f"{label2} — per-node confidence vs acceptance")
        cum_results2 = cumulative_correlation_table(trace2)
        print_cumulative_correlation_table(cum_results2, f"{label2} — cumulative confidence vs acceptance (σ_th signal)")
        print_depth_breakdown(depth_breakdown(trace2), label2)
        print_per_depth_correlation(per_depth_correlation(trace2), label2)
        if args.plot:
            plot_path = os.path.join(args.output_dir, f"{label2}_correlation.png")
            plot_correlation(results2, label2, plot_path)
            cum_plot_path = os.path.join(args.output_dir, f"{label2}_cumulative_correlation.png")
            plot_cumulative_correlation(cum_results2, label2, cum_plot_path)


if __name__ == "__main__":
    main()
