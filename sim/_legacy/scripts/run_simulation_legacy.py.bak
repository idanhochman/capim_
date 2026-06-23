"""
CAPIM Simulation Runner

Loads a collected TraceDataset and runs the full CAPIM simulation against
both baselines. Prints a comparison table and exports CSV results.

Usage:
    # Basic run with defaults
    python sim/scripts/run_simulation.py --trace traces/llama2_alpaca.json

    # Custom thresholds
    python sim/scripts/run_simulation.py \
        --trace traces/llama2_alpaca.json \
        --sigma-th -2.0 --mu-th 10

    # Sweep sigma_th and save results
    python sim/scripts/run_simulation.py \
        --trace traces/llama2_alpaca.json \
        --sweep sigma

    # Sweep both (2D grid search)
    python sim/scripts/run_simulation.py \
        --trace traces/llama2_alpaca.json \
        --sweep joint
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
capim_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, capim_dir)


def main():
    parser = argparse.ArgumentParser(description="Run CAPIM simulation")
    parser.add_argument("--trace", type=str, required=True,
                        help="Path to TraceDataset JSON (from collect_traces.py)")
    parser.add_argument("--sigma-th", type=float, default=-4.0,
                        help="Cumulative log-prob pruning threshold (default: -4.0; "
                             "LLaMA-2 median ≈ -4.2)")
    parser.add_argument("--mu-th", type=int, default=4,
                        help="Tree size PIM/NPU routing threshold (default: 4; "
                             "PAPI's α at RLP=1, crossover μ≈4)")
    parser.add_argument("--sweep", choices=["none", "sigma", "mu", "joint"],
                        default="none",
                        help="Run a sensitivity sweep instead of a single point")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for CSV/plot output")
    parser.add_argument("--plot", action="store_true",
                        help="Generate sensitivity plots (requires matplotlib)")
    args = parser.parse_args()

    from sim.trace.schema import TraceDataset
    from sim.config.models import LLAMA2_7B, EAGLE_HEAD_LLAMA2_7B
    from sim.baselines.autoregressive import simulate_autoregressive_from_trace
    from sim.baselines.lp_spec import simulate_lp_spec_from_trace
    from sim.simulation import simulate_capim
    from sim.results import (
        compare_results, sigma_sweep, mu_sweep, joint_sweep,
        export_csv, plot_sensitivity,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load trace
    print(f"Loading trace: {args.trace}")
    trace = TraceDataset.load(args.trace)
    scenario = os.path.splitext(os.path.basename(args.trace))[0]
    print(f"  {len(trace.steps)} steps, mean tree size {trace.mean_tree_size:.1f}")

    # Compute baselines (once)
    print("\nComputing baselines...")
    ar = simulate_autoregressive_from_trace(LLAMA2_7B, trace)
    lp = simulate_lp_spec_from_trace(LLAMA2_7B, trace)
    print(f"  AR:      {ar.tokens_per_second:.1f} tok/s, {ar.energy_per_token_j*1000:.2f} mJ/tok")
    print(f"  LP-Spec: {lp.tokens_per_second:.1f} tok/s, {lp.energy_per_token_j*1000:.2f} mJ/tok")

    baseline_kwargs = dict(
        ar_latency_per_token=ar.latency_per_token_s,
        lp_latency_per_token=lp.latency_per_token_s,
        lp_energy_per_token=lp.energy_per_token_j,
    )

    if args.sweep == "none":
        # Single-point evaluation
        capim = simulate_capim(
            trace, LLAMA2_7B, EAGLE_HEAD_LLAMA2_7B,
            sigma_th=args.sigma_th, mu_th=args.mu_th,
            scenario=scenario, **baseline_kwargs,
        )
        print(compare_results(
            capim,
            ar_energy_per_token=ar.energy_per_token_j,
            ar_latency_per_token=ar.latency_per_token_s,
            lp_energy_per_token=lp.energy_per_token_j,
            lp_latency_per_token=lp.latency_per_token_s,
        ))
        csv_path = os.path.join(args.output_dir, f"{scenario}_single.csv")
        export_csv([capim], csv_path)
        print(f"Saved to {csv_path}")

    elif args.sweep == "sigma":
        import math
        # Calibrated from the LLaMA-2 cumulative_log_prob distribution (AUDIT.md #7):
        #   Alpaca: p10=-5.95 p50=-4.23 p90=-1.98
        #   GSM8K:  p10=-7.93 p50=-4.42 p90=-1.52  (heavier tail)
        # Grid spans no-pruning to aggressive, covering both datasets' ranges.
        sigma_values = [float("-inf"), -10.0, -8.0, -6.0, -5.0, -4.5, -4.0,
                        -3.5, -3.0, -2.5, -2.0, -1.5]
        print(f"\nSweeping σ_th over {sigma_values}...")
        results = sigma_sweep(
            trace, LLAMA2_7B, EAGLE_HEAD_LLAMA2_7B,
            sigma_values=sigma_values, mu_th=args.mu_th,
            scenario=scenario, **baseline_kwargs,
        )
        csv_path = os.path.join(args.output_dir, f"{scenario}_sigma_sweep.csv")
        export_csv(results, csv_path)
        print(f"Saved {len(results)} rows to {csv_path}")
        _print_sweep_table(results, "sigma_th")
        if args.plot:
            plot_path = os.path.join(args.output_dir, f"{scenario}_sigma_sweep.png")
            plot_sensitivity(results, "sigma_th", save_path=plot_path)

    elif args.sweep == "mu":
        mu_values = [1, 3, 5, 8, 10, 15, 20, 30, 50, 100]
        print(f"\nSweeping μ_th over {mu_values}...")
        results = mu_sweep(
            trace, LLAMA2_7B, EAGLE_HEAD_LLAMA2_7B,
            mu_values=mu_values, sigma_th=args.sigma_th,
            scenario=scenario, **baseline_kwargs,
        )
        csv_path = os.path.join(args.output_dir, f"{scenario}_mu_sweep.csv")
        export_csv(results, csv_path)
        print(f"Saved {len(results)} rows to {csv_path}")
        _print_sweep_table(results, "mu_th")
        if args.plot:
            plot_path = os.path.join(args.output_dir, f"{scenario}_mu_sweep.png")
            plot_sensitivity(results, "mu_th", save_path=plot_path)

    elif args.sweep == "joint":
        import math
        sigma_values = [float("-inf"), -6.0, -4.5, -3.0, -2.0]
        mu_values = [2, 4, 8, 16]  # centered on the μ≈4 PIM/NPU crossover
        print(f"\n2D sweep: σ_th × μ_th ({len(sigma_values)}×{len(mu_values)} grid)...")
        grid = joint_sweep(
            trace, LLAMA2_7B, EAGLE_HEAD_LLAMA2_7B,
            sigma_values=sigma_values, mu_values=mu_values,
            scenario=scenario, **baseline_kwargs,
        )
        results = list(grid.values())
        csv_path = os.path.join(args.output_dir, f"{scenario}_joint_sweep.csv")
        export_csv(results, csv_path)
        print(f"Saved {len(results)} rows to {csv_path}")
        # Print best point by energy
        best = min(results, key=lambda r: r.energy_per_token_j)
        print(f"\nBest energy point: σ_th={best.sigma_th}, μ_th={best.mu_th}")
        print(f"  {best.energy_per_token_j*1000:.3f} mJ/tok, "
              f"{best.tokens_per_second:.1f} tok/s, "
              f"speedup_vs_ar={best.speedup_vs_ar:.2f}x")


def _print_sweep_table(results, param_name):
    print(f"\n{'─'*75}")
    print(f"{'σ_th' if param_name=='sigma_th' else 'μ_th':>8}  "
          f"{'tok/s':>8}  {'mJ/tok':>8}  "
          f"{'spdup_AR':>10}  {'spdup_LP':>10}  "
          f"{'e_red_LP':>10}  {'PIM%':>6}  {'FN%':>6}")
    print(f"{'─'*75}")
    for r in results:
        val = getattr(r, param_name)
        val_str = f"{val:.1f}" if val != float("-inf") else "-inf"
        e_red = r.energy_reduction_vs_lp
        e_red_str = f"{e_red*100:.1f}%" if e_red == e_red else "n/a"
        ls = r.latency_speedup_vs_lp
        ls_str = f"{ls:.2f}x" if ls == ls else "n/a"
        print(f"{val_str:>8}  {r.tokens_per_second:>8.1f}  "
              f"{r.energy_per_token_j*1000:>8.3f}  "
              f"{r.speedup_vs_ar:>10.2f}x  {ls_str:>10}  "
              f"{e_red_str:>10}  {r.pim_fraction*100:>6.0f}%  "
              f"{r.mean_false_neg_rate*100:>6.2f}%")
    print(f"{'─'*75}")


if __name__ == "__main__":
    main()
