"""
LP-Spec Draft Token Pruner (DTP) — trace-replay model (pure, no hardware).

This is the retrospective, content-blind selector that is the LP-Spec baseline's
counterpart to CAPIM's live σ_th gate.  It is grounded in two papers (verified in
handover.md §1):

  - MEDUSA (arXiv:2401.10774v3): a path's expected accept length = ∏_j p_j^{k_j}
    along the path under an explicit per-position independence assumption; the tree
    is grown greedily, repeatedly adding the frontier node with the highest path
    product.  MEDUSA measures p_i^k OFFLINE on a calibration set (a fixed tree).
  - LP-Spec (arXiv:2508.07227 §V-A): the SAME greedy construction, but p_i^k is
    measured AT RUNTIME from verification history ("we track the speculation
    accuracy p_i^k … after each decoding step based on previous verification
    results").  The verified tree size L_spec emerges from a hardware perf/energy
    estimator's stop rule; we replace that stop with a sweep over L (handover §3).

Terminology (papers' wording, handover §4): p_i^k = "accuracy of the k-th (top)
prediction at the i-th decode head".  `i` = head = tree depth; `k` = which of that
head's top-k candidates = "rank among same-parent siblings".  `k` is NOT stored in
the trace — it is DERIVED from `parent_idx` here (siblings are stored contiguously
in rank order; see schema.make_synthetic_medusa_trace / the real collector).

Key modelling choices (handover §4):
  - Reachable denominator [INFERRED — not specified by either paper]: a node counts
    toward its (head, k) statistic only if its PARENT was accepted (i.e. the node was
    actually reached).  LP-Spec §V-A says only that p_i^k is tracked "based on
    previous verification results" — it gives no denominator.  We chose `accepted /
    reachable` because p_i^k is meant to be head i's per-position prediction accuracy
    (MEDUSA §2.3.3, a_j^{(i_j)}), and at runtime you can only OBSERVE that accuracy
    when verification actually reaches position i (the prefix was accepted).  The
    alternative `accepted / steps` would count never-reached steps as failures,
    conflating head i's accuracy with an upstream prefix failing — biasing p_i^k down,
    worst for deep nodes.  Offline this issue is invisible (MEDUSA's calibration has
    ground truth at every position, so all positions are "reached"); it is purely a
    runtime correction.  Bonus: with the conditional p_i^k, the path product telescopes
    by the chain rule to P(path accepted) EXACTLY — no independence assumption — so it
    is at least as principled as MEDUSA's marginal product.
  - Unseen (head, k) → prior p = 1.0 ("full tree": keep it).  After step 0 (which
    verifies the full static tree) every reachable (head, k) at shallow depths is
    populated; deep (head, k) whose parents are rarely accepted stay at the prior,
    so the DTP never prunes a branch it has not yet observed (conservative).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from sim.trace.schema import DecodeStepTrace, TokenNode

# A node's stable identity across steps = its position in the static tree.
Pos = Tuple[int, int]        # (depth, layer_idx)
Key = Tuple[int, int]        # histogram key: (depth, k_pred) or (depth, layer_idx)


def k_pred_map(step: DecodeStepTrace) -> Dict[Pos, int]:
    """Derive k_pred (rank among same-parent siblings) for every node from
    `parent_idx` alone — no schema field.

    A node's siblings are the nodes sharing its (depth, parent_idx).  MEDUSA stores
    siblings in ascending prediction-rank order (the (len, lex) sort), so a node's
    k_pred is just its position among its siblings ordered by layer_idx.
    """
    # 1. Bucket nodes into sibling groups (same depth, same parent).
    siblings_by_parent: Dict[Tuple[int, int], List[TokenNode]] = defaultdict(list)
    for node in step.nodes:
        siblings_by_parent[(node.depth, node.parent_idx)].append(node)

    # 2. Within each group, k_pred = the node's position ordered by layer_idx.
    k_pred: Dict[Pos, int] = {}
    for siblings in siblings_by_parent.values():
        siblings.sort(key=lambda s: s.layer_idx)
        for rank, node in enumerate(siblings):
            k_pred[(node.depth, node.layer_idx)] = rank
    return k_pred


def parent_pos_map(step: DecodeStepTrace) -> Dict[Pos, Pos]:
    """Map each node's Pos -> its parent's Pos, resolving `parent_idx` (a GLOBAL
    index into `step.nodes`; see schema.TokenNode) through the actual parent node.

    This is the ONLY place that interprets the raw `parent_idx` integer, so the
    rest of the DTP is convention-independent: it works whatever `parent_idx`
    indexes, as long as `nodes[parent_idx]` is the parent.  Depth-0 nodes have no
    entry (their parent is the always-accepted root).
    """
    pos: Dict[Pos, Pos] = {}
    for n in step.nodes:
        if n.depth > 0:
            par = step.nodes[n.parent_idx]
            pos[(n.depth, n.layer_idx)] = (par.depth, par.layer_idx)
    return pos


def assert_sibling_rank_order(step: DecodeStepTrace) -> None:
    """Assert siblings appear in ascending layer_idx == ascending derived rank,
    so deriving k_pred from parent_idx is valid (handover §5 disclosed assumption 5).
    """
    groups: Dict[Tuple[int, int], List[int]] = {}
    for n in step.nodes:
        groups.setdefault((n.depth, n.parent_idx), []).append(n.layer_idx)
    for (depth, parent), idxs in groups.items():
        assert idxs == sorted(idxs), (
            f"siblings not in rank order at depth={depth} parent={parent}: {idxs}"
        )


def _keyfn(granularity: str) -> Callable[[TokenNode, Dict[Pos, int]], Key]:
    if granularity == "headk":
        return lambda n, kp: (n.depth, kp[(n.depth, n.layer_idx)])
    if granularity == "node":
        return lambda n, kp: (n.depth, n.layer_idx)
    raise ValueError(f"unknown granularity {granularity!r}")


class DTPHist:
    """Retrospective per-(head, k) acceptance histogram with reachable denominator.

    `granularity="headk"` (default) = LP-Spec's per-(head, k) statistic.
    `granularity="node"` = per-node empirical acceptance (drops the head-sharing
    assumption; an assumption-free cross-check, handover §5).  This node mode is a
    validation-only knob (see select_kept) — a candidate for removal once the real
    traces confirm it tracks the per-(head, k) baseline.
    """

    def __init__(self, granularity: str = "headk"):
        self.granularity = granularity
        self._keyfn = _keyfn(granularity)
        self.counts: Dict[Key, List[int]] = {}   # key -> [n_accepted, n_reachable]

    def p(self, key: Key) -> float:
        c = self.counts.get(key)
        if c is None or c[1] == 0:
            return 1.0                            # unseen -> prior = full tree (keep)
        return c[0] / c[1] # n_accepted/n_reachable

    def update(self, step: DecodeStepTrace, kp: Dict[Pos, int] = None,
               pp: Dict[Pos, Pos] = None) -> None:
        """Fold one step's observations in, counting a node only if reachable
        (its parent was accepted; depth-0 nodes hang off the always-accepted root).
        """
        if kp is None:
            kp = k_pred_map(step)
        if pp is None:
            pp = parent_pos_map(step)
        accepted_at: Dict[Pos, bool] = {(n.depth, n.layer_idx): n.accepted for n in step.nodes}
        for n in step.nodes:
            if n.depth == 0:
                reachable = True
            else: # a node counts only if its parent was accepted
                reachable = accepted_at.get(pp[(n.depth, n.layer_idx)], False)
            if not reachable:
                continue
            key = self._keyfn(n, kp)
            c = self.counts.setdefault(key, [0, 0]) # c = [n_accepted, n_reachable]
            c[1] += 1 # n_reachable++
            if n.accepted:
                c[0] += 1 # n_accepted++


def score_nodes(step: DecodeStepTrace, hist: DTPHist,
                kp: Dict[Pos, int] = None,
                pp: Dict[Pos, Pos] = None) -> Tuple[List[TokenNode], Dict[Pos, float]]:
    """Score every node by the path-product ∏ p along root→node, then return the
    nodes sorted for the greedy top-L selection.

    Sort key = (score desc, depth asc, layer_idx asc).  Because p ≤ 1 a parent's
    score ≥ its child's, and the depth-asc tiebreak keeps a parent strictly before
    its child, so EVERY prefix of the sorted list is ancestor-closed — the greedy
    "grow a connected tree" construction falls out for free, for all L at once.
    """
    if kp is None:
        kp = k_pred_map(step)
    if pp is None:
        pp = parent_pos_map(step)
    keyfn = hist._keyfn
    by_pos: Dict[Pos, TokenNode] = {(n.depth, n.layer_idx): n for n in step.nodes}
    score: Dict[Pos, float] = {}
    for n in sorted(step.nodes, key=lambda n: (n.depth, n.layer_idx)):
        p = hist.p(keyfn(n, kp))
        if n.depth == 0:
            score[(n.depth, n.layer_idx)] = p
        else:
            parent_score = score[pp[(n.depth, n.layer_idx)]]
            score[(n.depth, n.layer_idx)] = parent_score * p
    ranked = sorted(step.nodes,
                    key=lambda n: (-score[(n.depth, n.layer_idx)], n.depth, n.layer_idx))
    return ranked, score


def effective_accept(step: DecodeStepTrace, kept: set) -> int:
    """Realised accept length = the measured accepted path truncated at the first
    accepted node NOT in the verified tree `kept` (a set of Pos).

    The accepted nodes form a connected chain from the root (MEDUSA commits one
    path).  Since `kept` is ancestor-closed, the realised accept is the longest
    connected prefix of that chain whose nodes are all kept.  Monotone
    non-decreasing in L (kept sets are nested) and ≤ the measured full-tree accept.
    Mirror of CAPIM's `_effective_accept`.
    """
    chain = sorted((n for n in step.nodes if n.accepted), key=lambda n: n.depth)
    count = 0
    for n in chain:
        if (n.depth, n.layer_idx) in kept:
            count += 1
        else:
            break
    return count


def select_kept(step: DecodeStepTrace, t: int, L: int, selection: str,
                hist: DTPHist, kp: Dict[Pos, int] = None,
                pp: Dict[Pos, Pos] = None) -> set:
    """Return the set of Pos verified at step `t` under `selection` at tree size L.

    Cold start: step 0 always verifies the full static tree (no history yet).
    Selections (handover §5):
      greedy_headk : LP-Spec — greedy ∏ p_i^k from the (head, k) histogram.
      greedy_node  : assumption-free cross-check — greedy ∏ p over per-node stats.
                     Validates that LP-Spec's per-(head, k) pooling doesn't distort
                     the baseline (see DTPHist).  NOTE: a validation-only instrument,
                     not part of the headline comparison; consider removing it once
                     the real-trace check confirms it tracks greedy_headk.
      first_l      : MEDUSA generation order (offline accuracies) — sanity check.
      full         : verify the whole static tree every step.
      oracle       : keep the accepted chain first — upper-bounds realised accept.
    """
    all_pos = {(n.depth, n.layer_idx) for n in step.nodes}
    if t == 0 or selection == "full":
        return all_pos
    if selection in ("greedy_headk", "greedy_node"):
        ranked, _ = score_nodes(step, hist, kp, pp)
    elif selection == "first_l":
        ranked = sorted(step.nodes, key=lambda n: (n.depth, n.layer_idx))
    elif selection == "oracle":
        ranked = sorted(step.nodes, key=lambda n: (0 if n.accepted else 1, n.depth, n.layer_idx))
    else:
        raise ValueError(f"unknown selection {selection!r}")
    return {(n.depth, n.layer_idx) for n in ranked[:L]}
