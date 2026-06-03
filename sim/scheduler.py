"""
CAPIM Scheduler: confidence-aware pruner and PIM/NPU router.

Two responsibilities:
  1. prune_tree(step, sigma_th) — simulate live confidence-gated pruning.
     Walks the draft tree and terminates branches where log_prob < sigma_th.
     Children of pruned nodes are automatically removed.

  2. route(mu, mu_th) — decide whether verification runs in PIM or NPU.
     PIM if μ < μ_th (small, low-confidence tree).
     NPU if μ ≥ μ_th (large, high-confidence tree worth parallel verification).

Both functions operate on the trace schema types (TokenNode, DecodeStepTrace)
and have no hardware-model dependencies — they are pure algorithmic logic.

Design notes:
- sigma_th is a log-probability threshold (negative float, e.g. −2.3).
  A node is pruned if its per-token log_prob < sigma_th.
  At sigma_th = −∞ (float('-inf')), no pruning occurs → baseline behaviour.
  At sigma_th = 0.0 (log(1.0)), all non-certain tokens are pruned.

- Parent propagation: if a node at depth d is pruned, all of its children
  (depth d+1, d+2, …) that trace their lineage through that node are also
  pruned.  We implement this by tracking which parent indices survive at each
  depth level.

- The pruner returns a flat list of surviving TokenNode objects.  The tree
  size μ = len(surviving_nodes).

- Acceptance counting: we count accepted tokens in the pruned tree as the
  number of surviving nodes with accepted=True.  This is an optimistic but
  consistent metric — a pruned (un-accepted) node would never have been
  accepted anyway (per EAGLE-2's correlation analysis), but we also track
  how many accepted nodes we prune incorrectly (false negatives).
"""

from typing import Dict, List, Literal, Set, Tuple

from sim.trace.schema import DecodeStepTrace, TokenNode


def prune_tree(
    step: DecodeStepTrace,
    sigma_th: float,
) -> List[TokenNode]:
    """
    Simulate live confidence-gated pruning on a draft tree.

    A node is eliminated if:
      (a) its own log_prob < sigma_th  [direct threshold violation], OR
      (b) its parent was already eliminated  [propagated pruning]

    Args:
        step: A single decode step trace containing the full draft tree.
        sigma_th: Log-probability threshold.  Any node with log_prob < sigma_th
                  is pruned along with all its descendants.
                  Use float('-inf') to disable pruning (σ_th = −∞).

    Returns:
        List of surviving TokenNode objects (order matches original tree order).
    """
    if not step.nodes:
        return []

    # Fast path: no pruning
    if sigma_th == float("-inf"):
        return list(step.nodes)

    # Group nodes by depth for efficient traversal
    max_depth = step.max_depth
    layers: Dict[int, List[TokenNode]] = {d: [] for d in range(max_depth + 1)}
    for node in step.nodes:
        layers[node.depth].append(node)

    # Sort each layer by layer_idx for deterministic ordering
    for d in layers:
        layers[d].sort(key=lambda n: n.layer_idx)

    # surviving_layer_indices[d] = set of layer_idx values that survived at depth d
    # Depth -1 (root) always survives
    surviving_at_depth: Dict[int, Set[int]] = {-1: {-1}}  # root is always alive

    surviving_nodes: List[TokenNode] = []

    for d in range(max_depth + 1):
        alive_parents = surviving_at_depth.get(d - 1, set())
        alive_at_d: Set[int] = set()

        for node in layers[d]:
            # Check 1: parent must have survived
            parent_key = node.parent_idx if d > 0 else -1
            parent_alive = parent_key in alive_parents

            # Check 2: this node's own confidence must clear the threshold
            node_passes = node.log_prob >= sigma_th

            if parent_alive and node_passes:
                surviving_nodes.append(node)
                alive_at_d.add(node.layer_idx)

        surviving_at_depth[d] = alive_at_d

    return surviving_nodes


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
