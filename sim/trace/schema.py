"""
Trace data structures for CAPIM simulation.

A trace is collected by running instrumented EAGLE-2 on Qwen2.5 with Alpaca
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
    model_target: str           # e.g. "Qwen2.5-7B"
    model_draft: str            # e.g. "Qwen2.5-0.5B"
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
        model_target="Qwen2.5-7B",
        model_draft="Qwen2.5-0.5B",
        metadata={"synthetic": True, "seed": seed},
    )
    td.compute_summary()
    return td
