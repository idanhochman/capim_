"""
Results aggregation, comparison tables, and sensitivity analysis.

This module provides:
  1. compare_results() — print a formatted comparison table (CAPIM vs baselines)
  2. sigma_sweep()     — sweep σ_th across a range, return list of SimResults
  3. mu_sweep()        — sweep μ_th across a range, return list of SimResults
  4. joint_sweep()     — 2D grid search over (σ_th, μ_th)
  5. plot_sensitivity() — matplotlib plot of σ_th or μ_th sensitivity
  6. export_csv()       — export sweep results to CSV

Design: all sweep functions call simulate_capim() internally and return raw
SimResult lists.  Plotting and CSV export are separated so that simulation
results can be reused without re-running the hardware model.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from sim.config.models import ModelConfig
from sim.simulation import SimResult, simulate_capim
from sim.trace.schema import TraceDataset


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def compare_results(
    capim: SimResult,
    ar_energy_per_token: float,
    ar_latency_per_token: float,
    lp_energy_per_token: float,
    lp_latency_per_token: float,
) -> str:
    """
    Return a formatted ASCII comparison table.

    Args:
        capim: CAPIM simulation result.
        ar_energy_per_token: Autoregressive baseline J/token.
        ar_latency_per_token: Autoregressive baseline s/token.
        lp_energy_per_token: LP-Spec baseline J/token.
        lp_latency_per_token: LP-Spec baseline s/token.

    Returns:
        Multi-line string with the comparison table.
    """
    lp_speedup_vs_ar = ar_latency_per_token / lp_latency_per_token if lp_latency_per_token > 0 else float("nan")
    lp_e_reduction = 1.0 - (lp_energy_per_token / ar_energy_per_token) if ar_energy_per_token > 0 else float("nan")

    rows = [
        ("System", "J/token", "tokens/s", "Speedup vs AR", "E reduction vs AR"),
        ("─" * 12, "─" * 12, "─" * 12, "─" * 14, "─" * 18),
        (
            "Autoregressive",
            f"{ar_energy_per_token:.4e}",
            f"{1.0/ar_latency_per_token:.2f}",
            "1.00×",
            "0.0%",
        ),
        (
            "LP-Spec",
            f"{lp_energy_per_token:.4e}",
            f"{1.0/lp_latency_per_token:.2f}",
            f"{lp_speedup_vs_ar:.2f}×",
            f"{lp_e_reduction*100:.1f}%",
        ),
        (
            "CAPIM",
            f"{capim.energy_per_token_j:.4e}",
            f"{capim.tokens_per_second:.2f}",
            f"{capim.speedup_vs_ar:.2f}×",
            f"{(1.0 - capim.energy_per_token_j / ar_energy_per_token)*100:.1f}%",
        ),
    ]

    # Compute column widths
    col_widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    sep = "  "
    lines = []
    for row in rows:
        lines.append(sep.join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)))

    header = (
        f"\nCAPIM vs Baselines  [σ_th={capim.sigma_th:.2f}, μ_th={capim.mu_th},"
        f" scenario={capim.scenario}]\n"
    )
    divider = "=" * sum(col_widths + [len(sep)] * (len(col_widths) - 1))

    extra = (
        f"\nCAPIM details:\n"
        f"  Mean pruned tree size  : {capim.mean_pruned_tree_size:.1f} tokens"
        f"  (original: {capim.mean_original_tree_size:.1f})\n"
        f"  Mean pruning ratio     : {capim.mean_pruning_ratio*100:.1f}%\n"
        f"  PIM routing fraction   : {capim.pim_fraction*100:.1f}%\n"
        f"  Acceptance rate        : {capim.acceptance_rate*100:.1f}%\n"
        f"  Mean accepted/step     : {capim.mean_accepted_per_step:.2f} tokens\n"
        f"  Mean false-neg rate    : {capim.mean_false_neg_rate*100:.2f}%\n"
        f"  Latency speedup vs LP  : {capim.latency_speedup_vs_lp:.2f}×\n"
        f"  Energy reduction vs LP : {capim.energy_reduction_vs_lp*100:.1f}%\n"
    )

    return header + divider + "\n" + "\n".join(lines) + "\n" + divider + extra


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------


def sigma_sweep(
    trace: TraceDataset,
    target_model: ModelConfig,
    draft_model: ModelConfig,
    sigma_values: List[float],
    mu_th: int = 10,
    scenario: str = "sweep",
    ar_latency_per_token: Optional[float] = None,
    lp_latency_per_token: Optional[float] = None,
    lp_energy_per_token: Optional[float] = None,
) -> List[SimResult]:
    """
    Sweep σ_th over a list of values and return corresponding SimResults.

    Args:
        trace: TraceDataset.
        target_model: Target model config.
        draft_model: Draft model config.
        sigma_values: List of σ_th values to evaluate (log-prob, e.g. [-5, -3, -2, -1]).
        mu_th: Fixed μ_th during the sweep.
        scenario: Scenario label.

    Returns:
        List of SimResult, one per sigma value, in the same order.
    """
    results = []
    for sigma in sigma_values:
        r = simulate_capim(
            trace=trace,
            target_model=target_model,
            draft_model=draft_model,
            sigma_th=sigma,
            mu_th=mu_th,
            scenario=scenario,
            ar_latency_per_token=ar_latency_per_token,
            lp_latency_per_token=lp_latency_per_token,
            lp_energy_per_token=lp_energy_per_token,
        )
        results.append(r)
    return results


def mu_sweep(
    trace: TraceDataset,
    target_model: ModelConfig,
    draft_model: ModelConfig,
    mu_values: List[int],
    sigma_th: float = float("-inf"),
    scenario: str = "sweep",
    ar_latency_per_token: Optional[float] = None,
    lp_latency_per_token: Optional[float] = None,
    lp_energy_per_token: Optional[float] = None,
) -> List[SimResult]:
    """
    Sweep μ_th over a list of values and return corresponding SimResults.

    Args:
        mu_values: List of μ_th values to evaluate (e.g. [1, 5, 10, 20, 50]).
        sigma_th: Fixed σ_th during the sweep.

    Returns:
        List of SimResult, one per mu value.
    """
    results = []
    for mu in mu_values:
        r = simulate_capim(
            trace=trace,
            target_model=target_model,
            draft_model=draft_model,
            sigma_th=sigma_th,
            mu_th=mu,
            scenario=scenario,
            ar_latency_per_token=ar_latency_per_token,
            lp_latency_per_token=lp_latency_per_token,
            lp_energy_per_token=lp_energy_per_token,
        )
        results.append(r)
    return results


def joint_sweep(
    trace: TraceDataset,
    target_model: ModelConfig,
    draft_model: ModelConfig,
    sigma_values: List[float],
    mu_values: List[int],
    scenario: str = "joint_sweep",
    ar_latency_per_token: Optional[float] = None,
    lp_latency_per_token: Optional[float] = None,
    lp_energy_per_token: Optional[float] = None,
) -> Dict[Tuple[float, int], SimResult]:
    """
    2D grid search over (σ_th, μ_th) combinations.

    Returns:
        Dict mapping (sigma_th, mu_th) tuples to SimResult.
    """
    grid: Dict[Tuple[float, int], SimResult] = {}
    for sigma in sigma_values:
        for mu in mu_values:
            r = simulate_capim(
                trace=trace,
                target_model=target_model,
                draft_model=draft_model,
                sigma_th=sigma,
                mu_th=mu,
                scenario=scenario,
                ar_latency_per_token=ar_latency_per_token,
                lp_latency_per_token=lp_latency_per_token,
                lp_energy_per_token=lp_energy_per_token,
            )
            grid[(sigma, mu)] = r
    return grid


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_csv(results: List[SimResult], path: str) -> None:
    """
    Export a list of SimResults to CSV.

    Args:
        results: List of SimResult objects.
        path: Output CSV file path.
    """
    if not results:
        return

    # Fields to export (exclude the heavy steps list)
    keys = [
        "sigma_th", "mu_th", "scenario",
        "tokens_per_second", "energy_per_token_j",
        "acceptance_rate", "mean_accepted_per_step",
        "speedup_vs_ar", "energy_reduction_vs_lp", "latency_speedup_vs_lp",
        "pim_fraction", "npu_fraction",
        "mean_pruned_tree_size", "mean_original_tree_size", "mean_pruning_ratio",
        "total_latency_s", "total_energy_j",
        "total_accepted_tokens", "total_steps",
        "mean_false_neg_rate",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in keys}
            writer.writerow(row)


def plot_sensitivity(
    sweep_results: List[SimResult],
    sweep_param: str = "sigma_th",
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot sensitivity of key metrics across a σ_th or μ_th sweep.

    Args:
        sweep_results: List of SimResult from sigma_sweep() or mu_sweep().
        sweep_param: "sigma_th" or "mu_th" — the x-axis variable.
        metrics: List of SimResult field names to plot.  Defaults to
                 ["energy_per_token_j", "tokens_per_second", "acceptance_rate",
                  "pim_fraction", "mean_false_neg_rate"].
        save_path: If provided, save the figure to this path instead of showing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    if metrics is None:
        metrics = [
            "energy_per_token_j",
            "tokens_per_second",
            "acceptance_rate",
            "pim_fraction",
            "mean_false_neg_rate",
        ]

    x_vals = [getattr(r, sweep_param) for r in sweep_results]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    labels = {
        "energy_per_token_j": "Energy (J/token)",
        "tokens_per_second": "Throughput (tokens/s)",
        "acceptance_rate": "Acceptance rate",
        "pim_fraction": "PIM routing fraction",
        "mean_false_neg_rate": "False-negative rate",
        "speedup_vs_ar": "Speedup vs AR",
        "energy_reduction_vs_lp": "Energy reduction vs LP-Spec",
        "latency_speedup_vs_lp": "Latency speedup vs LP-Spec",
        "mean_pruning_ratio": "Pruning ratio",
    }

    for ax, metric in zip(axes, metrics):
        y_vals = [getattr(r, metric) for r in sweep_results]
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_ylabel(labels.get(metric, metric))
        ax.grid(True, alpha=0.3)

    x_label = "σ_th (log-prob threshold)" if sweep_param == "sigma_th" else "μ_th (tree size threshold)"
    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"CAPIM sensitivity: {sweep_param} sweep", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
