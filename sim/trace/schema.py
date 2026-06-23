"""
Trace data structures for CAPIM simulation.

A trace is collected by running instrumented EAGLE-2 on LLaMA-2 with Alpaca
or GSM8K datasets.  The schema is designed to be trace-agnostic: simulation
modules only consume TraceDataset objects and do not depend on the EAGLE-2
implementation details.

Key design decisions:
- log_prob: per-token log-softmax probability at the token's depth in the
  draft tree.  Negative float.  Used for σ_th comparison in the pruner.
- cumulative_log_prob: sum of log_probs from the root to this node.  Used
  for path-level scoring (matches EAGLE-2's cu_scores).
- accepted: whether the target model accepted this exact token in this step.
  Populated by the collector from evaluate_posterior output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TokenNode:
    """Represents one draft token in the draft tree for a single decode step."""

    depth: int              # 0 = first layer (direct children of root/last accepted token)
    token_id: int           # vocabulary index of this draft token
    log_prob: float         # per-token log-softmax probability at this depth
    cumulative_log_prob: float  # sum of log_probs along the path root → this node
    parent_idx: int         # index of this node's parent in the previous layer's node list
                            # (−1 for depth-0 nodes whose parent is the true accepted token)
    accepted: bool          # True if the target model accepted this exact token

    # Position within its layer (used by scheduler to reconstruct tree structure)
    layer_idx: int = 0      # index of this node within its depth layer
    token_str: str = ""     # human-readable decoded token (e.g. " Paris", "\n")


@dataclass
class DecodeStepTrace:
    """
    All information captured for a single speculative decoding step.

    A step corresponds to one call to EAGLE-2's topK_genrate followed by
    one call to evaluate_posterior.
    """

    step_id: int                    # sequential decode step index (0-based)
    context_length: int             # number of tokens in the KV-cache at step start
    nodes: List[TokenNode]          # all nodes in the FULL (un-pruned) draft tree
    accepted_length: int            # number of tokens the target accepted (0 = only the
                                    # bonus token from autoregressive fallback)
    dataset: str                    # "alpaca" or "gsm8k"
    prompt_id: int                  # index of the source prompt within the dataset
    sample_token_id: int = 0        # vocabulary index of the root token (last accepted token)
    sample_token_str: str = ""      # human-readable root token (e.g. " The")

    @property
    def tree_size(self) -> int:
        """Total number of nodes in this step's draft tree."""
        return len(self.nodes)

    def nodes_at_depth(self, depth: int) -> List[TokenNode]:
        """Return all nodes at a given depth."""
        return [n for n in self.nodes if n.depth == depth]

    @property
    def max_depth(self) -> int:
        if not self.nodes:
            return -1
        return max(n.depth for n in self.nodes)


@dataclass
class TraceDataset:
    """
    Collection of decoded steps with metadata.

    Saved as a JSON file so it can be re-used across simulation runs without
    re-running the GPU-intensive EAGLE-2 inference.
    """

    steps: List[DecodeStepTrace]
    model_target: str           # e.g. "LLaMA-2-7B-Chat"
    model_draft: str            # e.g. "EAGLE-llama2-chat-7B"
    metadata: Dict              # free-form metadata (sigma_th used, dataset split, etc.)

    # Summary statistics (populated after collection)
    mean_tree_size: float = 0.0
    mean_accepted_length: float = 0.0
    mean_acceptance_rate: float = 0.0  # accepted / tree_size per step, averaged

    def compute_summary(self) -> None:
        """Populate summary fields from collected steps."""
        if not self.steps:
            return
        n = len(self.steps)
        self.mean_tree_size = sum(s.tree_size for s in self.steps) / n
        self.mean_accepted_length = sum(s.accepted_length for s in self.steps) / n
        rates = []
        for s in self.steps:
            if s.tree_size > 0:
                accepted_nodes = sum(1 for nd in s.nodes if nd.accepted)
                rates.append(accepted_nodes / s.tree_size)
        self.mean_acceptance_rate = sum(rates) / len(rates) if rates else 0.0

    def save(self, path: str) -> None:
        """Serialize to JSON."""
        data = {
            "model_target": self.model_target,
            "model_draft": self.model_draft,
            "metadata": self.metadata,
            "mean_tree_size": self.mean_tree_size,
            "mean_accepted_length": self.mean_accepted_length,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "steps": [
                {
                    "step_id": s.step_id,
                    "context_length": s.context_length,
                    "accepted_length": s.accepted_length,
                    "dataset": s.dataset,
                    "prompt_id": s.prompt_id,
                    "sample_token_id": s.sample_token_id,
                    "sample_token_str": s.sample_token_str,
                    "nodes": [
                        {
                            "depth": n.depth,
                            "token_id": n.token_id,
                            "token_str": n.token_str,
                            "log_prob": n.log_prob,
                            "cumulative_log_prob": n.cumulative_log_prob,
                            "parent_idx": n.parent_idx,
                            "accepted": n.accepted,
                            "layer_idx": n.layer_idx,
                        }
                        for n in s.nodes
                    ],
                }
                for s in self.steps
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> "TraceDataset":
        """Deserialize from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        steps = []
        for sd in data["steps"]:
            nodes = [
                TokenNode(
                    depth=nd["depth"],
                    token_id=nd["token_id"],
                    token_str=nd.get("token_str", ""),
                    log_prob=nd["log_prob"],
                    cumulative_log_prob=nd["cumulative_log_prob"],
                    parent_idx=nd["parent_idx"],
                    accepted=nd["accepted"],
                    layer_idx=nd.get("layer_idx", 0),
                )
                for nd in sd["nodes"]
            ]
            steps.append(
                DecodeStepTrace(
                    step_id=sd["step_id"],
                    context_length=sd["context_length"],
                    nodes=nodes,
                    accepted_length=sd["accepted_length"],
                    dataset=sd["dataset"],
                    prompt_id=sd["prompt_id"],
                    sample_token_id=sd.get("sample_token_id", 0),
                    sample_token_str=sd.get("sample_token_str", ""),
                )
            )

        return TraceDataset(
            steps=steps,
            model_target=data["model_target"],
            model_draft=data["model_draft"],
            metadata=data["metadata"],
            mean_tree_size=data.get("mean_tree_size", 0.0),
            mean_accepted_length=data.get("mean_accepted_length", 0.0),
            mean_acceptance_rate=data.get("mean_acceptance_rate", 0.0),
        )


def make_synthetic_trace(
    n_steps: int = 100,
    tree_size: int = 20,
    acceptance_rate: float = 0.4,
    max_depth: int = 4,
    dataset: str = "synthetic",
    seed: int = 42,
) -> TraceDataset:
    """
    Generate a synthetic TraceDataset for unit-testing simulation modules
    before real EAGLE-2 traces are available.

    Nodes are distributed uniformly across depths.  log_prob values are
    sampled from a realistic distribution (log-normal centered around −2.0).
    acceptance_rate fraction of nodes are marked accepted.
    """
    import random
    import math

    rng = random.Random(seed)
    steps = []
    nodes_per_depth = max(1, tree_size // max_depth)

    for i in range(n_steps):
        nodes = []
        global_idx = 0
        for d in range(max_depth):
            n_nodes = nodes_per_depth if d < max_depth - 1 else tree_size - global_idx
            n_nodes = max(1, n_nodes)
            layer_size_prev = nodes_per_depth if d > 0 else 1
            for j in range(n_nodes):
                # Sample a per-token log probability (log-softmax: negative)
                lp = -abs(rng.gauss(2.0, 1.5))
                # Cumulative log prob: sum from root
                cum_lp = lp * (d + 1) + rng.gauss(0, 0.2)  # approximate
                parent = rng.randint(0, layer_size_prev - 1) if d > 0 else -1
                accepted = rng.random() < acceptance_rate
                nodes.append(
                    TokenNode(
                        depth=d,
                        token_id=rng.randint(0, 151935),
                        log_prob=lp,
                        cumulative_log_prob=cum_lp,
                        parent_idx=parent,
                        accepted=accepted,
                        layer_idx=j,
                    )
                )
                global_idx += 1

        accepted_len = max(1, int(rng.gauss(2.5, 1.0)))
        steps.append(
            DecodeStepTrace(
                step_id=i,
                context_length=128 + i,
                nodes=nodes,
                accepted_length=accepted_len,
                dataset=dataset,
                prompt_id=0,
            )
        )

    td = TraceDataset(
        steps=steps,
        model_target="LLaMA-2-7B-Chat",
        model_draft="EAGLE-llama2-chat-7B",
        metadata={"synthetic": True, "seed": seed},
    )
    td.compute_summary()
    return td


# ---------------------------------------------------------------------------
# Synthetic MEDUSA trace (for the LP-Spec DTP driver — GPU-free testing)
# ---------------------------------------------------------------------------
#
# A MEDUSA trace differs from an EAGLE trace in three ways the LP-Spec driver
# relies on:
#   1. The draft tree is STATIC: every step verifies the same full tree shape.
#   2. Confidence (log_prob) is not the control signal — LP-Spec's DTP scores
#      nodes from a per-(head, k) acceptance histogram measured at runtime.
#      `k` = the rank of a head's prediction = "rank among same-parent siblings",
#      which the driver derives from `parent_idx` (NOT a stored field).
#   3. Acceptance is a single connected path from the root (the longest accepted
#      prefix), exactly as MEDUSA's tree-attention verification commits one path.
#
# For the derived-k_pred trick to work, same-parent siblings must be stored
# contiguously and in ascending-k order.  We guarantee this by ordering each
# depth layer lexicographically by path (mirrors MEDUSA's (len, lex) sort).


def balanced_medusa_choices(branching: int = 2, max_depth: int = 3) -> List[List[int]]:
    """MEDUSA-format choices for a balanced tree: every internal node has
    `branching` children with k = 0..branching-1, down to `max_depth` levels.

    Each entry is a path of prediction ranks, e.g. [0, 1] = take head-0's
    top-0 prediction, then head-1's top-1 prediction.  This is the same format
    as `sim.layers.medusa.MC_SIM_7B_63`.
    """
    choices: List[List[int]] = []

    def grow(prefix: List[int], depth: int) -> None:
        if depth >= max_depth:
            return
        for k in range(branching):
            path = prefix + [k]
            choices.append(path)
            grow(path, depth + 1)

    grow([], 0)
    return choices


def make_synthetic_medusa_trace(
    n_steps: int = 100,
    tree_choices: Optional[List[List[int]]] = None,
    branching: int = 2,
    max_depth: int = 3,
    base_accept_prob: float = 0.7,
    depth_decay: float = 0.85,
    rank_decay: float = 0.55,
    dataset: str = "synthetic",
    seed: int = 42,
) -> TraceDataset:
    """Generate a synthetic MEDUSA TraceDataset for the LP-Spec DTP driver.

    The tree shape is identical every step (static MEDUSA tree).  Per step we
    sample one connected accepted path from the root: at each level there is a
    correct child with probability `base_accept_prob * depth_decay**depth`, and
    when there is one, its rank is drawn from a stationary categorical favouring
    low k (weight `rank_decay**k`).  Stationarity is what lets the DTP histogram
    converge — mirroring real MEDUSA behaviour and making the trace meaningful
    for the retrospective per-(head, k) estimator.

    Args:
        tree_choices: MEDUSA-format paths.  Defaults to a balanced tree built
            from `branching`/`max_depth`.  Pass `MC_SIM_7B_63` for realism.
    """
    import random
    import math

    rng = random.Random(seed)

    if tree_choices is None:
        tree_choices = balanced_medusa_choices(branching, max_depth)

    # Order each depth layer lexicographically so same-parent siblings are
    # contiguous and ascending in k (the property the driver's k_pred relies on).
    paths = sorted(tree_choices, key=lambda p: (len(p), p))
    max_d = max(len(p) for p in paths) - 1

    # layer_paths[d] = lex-sorted paths at depth d; index within = layer_idx.
    layer_paths: List[List[List[int]]] = [[] for _ in range(max_d + 1)]
    for p in paths:
        layer_paths[len(p) - 1].append(p)

    # Build a static node template (structure is identical across steps).
    # template[i] = (depth, layer_idx, parent_idx, k_pred)
    template = []
    path_to_layeridx: Dict[tuple, int] = {}
    for d in range(max_d + 1):
        for li, p in enumerate(layer_paths[d]):
            path_to_layeridx[tuple(p)] = li
    for d in range(max_d + 1):
        for li, p in enumerate(layer_paths[d]):
            if d == 0:
                parent_idx = -1
            else:
                parent_idx = path_to_layeridx[tuple(p[:-1])]
            template.append((d, li, parent_idx, p[-1]))

    # children[(depth, parent_idx)] = list of (global_idx, k_pred, layer_idx)
    children: Dict[tuple, List[tuple]] = {}
    for gidx, (d, li, parent_idx, k) in enumerate(template):
        children.setdefault((d, parent_idx), []).append((gidx, k, li))

    steps = []
    for i in range(n_steps):
        # Decide the accepted path (connected chain from the root).
        accepted_ids = set()
        parent_layer_idx = -1
        for d in range(max_d + 1):
            sibs = children.get((d, parent_layer_idx))
            if not sibs:
                break
            if rng.random() > base_accept_prob * (depth_decay ** d):
                break  # no correct child at this level -> path stops
            weights = [rank_decay ** k for (_, k, _) in sibs]
            total = sum(weights)
            r = rng.random() * total
            chosen = sibs[-1]
            acc = 0.0
            for s, w in zip(sibs, weights):
                acc += w
                if r <= acc:
                    chosen = s
                    break
            gidx, _, li = chosen
            accepted_ids.add(gidx)
            parent_layer_idx = li

        # Materialise nodes with plausible (unused-by-DTP) confidence values.
        nodes: List[TokenNode] = []
        cum_by_gidx: Dict[int, float] = {}
        for gidx, (d, li, parent_idx, k) in enumerate(template):
            lp = -0.3 - 0.7 * k + rng.gauss(0, 0.1)
            lp = min(lp, -1e-4)
            if d == 0:
                cum = lp
            else:
                # parent global idx = the depth-(d-1) node with layer_idx==parent_idx
                cum = cum_by_gidx[_parent_global_idx(template, d, parent_idx)] + lp
            cum_by_gidx[gidx] = cum
            nodes.append(
                TokenNode(
                    depth=d,
                    token_id=rng.randint(0, 31999),
                    log_prob=lp,
                    cumulative_log_prob=cum,
                    parent_idx=parent_idx,
                    accepted=(gidx in accepted_ids),
                    layer_idx=li,
                )
            )

        steps.append(
            DecodeStepTrace(
                step_id=i,
                context_length=128 + i,
                nodes=nodes,
                accepted_length=len(accepted_ids),
                dataset=dataset,
                prompt_id=0,
            )
        )

    td = TraceDataset(
        steps=steps,
        model_target="Vicuna-7B-v1.3",
        model_draft="medusa-vicuna-7b-v1.3",
        metadata={
            "synthetic": True,
            "method": "medusa",
            "static_tree": True,
            "tree_size": len(template),
            "seed": seed,
        },
    )
    td.compute_summary()
    return td


def _parent_global_idx(template, depth: int, parent_layer_idx: int) -> int:
    """Global node index of the depth-(depth-1) node with the given layer_idx.

    Nodes in `template` are grouped by depth ascending, so we scan the previous
    depth layer for the matching layer_idx.
    """
    for gidx, (d, li, _parent, _k) in enumerate(template):
        if d == depth - 1 and li == parent_layer_idx:
            return gidx
    raise ValueError(f"no parent at depth {depth - 1} with layer_idx {parent_layer_idx}")
