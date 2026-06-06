"""
CAPIM Scheduler: confidence-aware pruner and PIM/NPU router.

Two responsibilities:
  1. prune_tree(step, sigma_th) — simulate live confidence-gated pruning.
     Terminates branches where cumulative_log_prob < sigma_th.

  2. route(mu, mu_th) — decide whether verification runs in PIM or NPU.
     PIM if μ < μ_th (small, low-confidence tree).
     NPU if μ ≥ μ_th (large, high-confidence tree worth parallel verification).

Both functions operate on the trace schema types (TokenNode, DecodeStepTrace)
and have no hardware-model dependencies — they are pure algorithmic logic.

Design notes:
- sigma_th is a cumulative log-probability threshold (negative float, e.g. −4.0).
  A node is pruned if its cumulative_log_prob < sigma_th, where
  cumulative_log_prob is the sum of log_probs from root to this node —
  equivalent to the log joint probability of the entire path.
  At sigma_th = −∞ (float('-inf')), no pruning occurs → baseline behaviour.
  At sigma_th = 0.0, only paths with probability 1.0 survive (prunes everything).

- No parent propagation needed: cumulative_log_prob is strictly monotonically
  decreasing with depth (every step adds a negative log_prob). A node that
  fails the threshold guarantees all its descendants will also fail, so each
  node can be checked independently in a single pass.

- Using cumulative rather than per-node log_prob avoids the "collateral damage"
  of per-node pruning, where a single uncertain intermediate step would kill an
  otherwise high-quality deep path. Cumulative evaluates path quality as a whole.

- The pruner returns a flat list of surviving TokenNode objects.
  The tree size μ = len(surviving_nodes).

- Acceptance counting: we count accepted tokens in the pruned tree as the
  number of surviving nodes with accepted=True. We also track how many
  accepted nodes were incorrectly pruned (false negatives).
"""

from typing import List, Literal

from sim.trace.schema import DecodeStepTrace, TokenNode


def prune_tree(
    step: DecodeStepTrace,
    sigma_th: float,
) -> List[TokenNode]:
    """
    Simulate live confidence-gated pruning on a draft tree.

    A node is eliminated if its cumulative_log_prob < sigma_th, where
    cumulative_log_prob is the sum of log_probs from the root to this node.

    No parent propagation is required: cumulative_log_prob is strictly
    monotonically decreasing with depth, so a failing node guarantees all
    its descendants also fail. Each node is checked independently.

    Args:
        step: A single decode step trace containing the full draft tree.
        sigma_th: Cumulative log-probability threshold. Nodes whose path
                  probability falls below this are pruned.
                  Use float('-inf') to disable pruning (σ_th = −∞).

    Returns:
        List of surviving TokenNode objects (order matches original tree order).
    """
    if not step.nodes:
        return []

    if sigma_th == float("-inf"):
        return list(step.nodes)

    return [n for n in step.nodes if n.cumulative_log_prob >= sigma_th]


def route(mu: int, mu_th: int) -> Literal["PIM", "NPU"]:
    """
    Route verification to PIM or NPU based on pruned tree size.

    Args:
        mu: Number of surviving draft tokens after pruning (pruned tree size).
        mu_th: Tree size threshold.  If μ < μ_th, route to PIM; else to NPU.

    Returns:
        "PIM" or "NPU" as a string literal.
    """
    return "PIM" if mu < mu_th else "NPU"


def count_false_negatives(
    step: DecodeStepTrace,
    pruned_nodes: List[TokenNode],
) -> int:
    """
    Count how many accepted nodes were incorrectly pruned.

    A false negative is a node that:
      - was accepted by the target model (accepted=True)
      - but was removed by the pruner (not in pruned_nodes)

    This metric quantifies the quality-loss from pruning.

    Args:
        step: Original (un-pruned) decode step.
        pruned_nodes: Surviving nodes after pruning.

    Returns:
        Number of pruned accepted nodes (false negatives).
    """
    pruned_set = set(id(n) for n in pruned_nodes)
    false_neg = sum(
        1 for n in step.nodes if n.accepted and id(n) not in pruned_set
    )
    return false_neg


def prune_stats(
    step: DecodeStepTrace,
    sigma_th: float,
) -> dict:
    """
    Return a summary dict of pruning statistics for a single step.

    Useful for sweeping σ_th to find the knee of the pruning curve.

    Returns:
        {
            "original_size": int,
            "pruned_size": int,
            "pruning_ratio": float,      # fraction removed
            "false_negatives": int,       # accepted nodes wrongly pruned
            "false_neg_rate": float,      # false_negatives / total_accepted
        }
    """
    pruned = prune_tree(step, sigma_th)
    original_size = step.tree_size
    pruned_size = len(pruned)
    total_accepted = sum(1 for n in step.nodes if n.accepted)
    false_neg = count_false_negatives(step, pruned)

    return {
        "original_size": original_size,
        "pruned_size": pruned_size,
        "pruning_ratio": 1.0 - (pruned_size / original_size) if original_size > 0 else 0.0,
        "false_negatives": false_neg,
        "false_neg_rate": (false_neg / total_accepted) if total_accepted > 0 else 0.0,
    }
