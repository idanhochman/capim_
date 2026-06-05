"""
EAGLE-2 trace collector for CAPIM.

Instruments EAGLE's inference code to log per-token confidence data needed
for CAPIM's confidence-gated pruner calibration.

Design principles:
  - Minimal modification to EAGLE: one line added to cnets1.py saves
    cumulative scores to ea_layer._capim_scores before they are deleted.
  - Uses Python monkey-patching at runtime for everything else.
  - Captures: token IDs, cumulative log_probs, per-token log_probs
    (derived as node_cumulative - parent_cumulative), parent indices,
    and acceptance flags.

What we patch:
  1. ea_layer.topK_genrate (cnets1.Model.topK_genrate)
     — reads ea_layer._capim_scores after the call to get cumulative scores.
  2. eagle.model.ea_model.evaluate_posterior
     — captures best_candidate and accept_length to mark accepted nodes.

How to use:
    from sim.trace.collector import Collector
    from eagle.model.ea_model import EaModel

    collector = Collector(dataset="gsm8k", prompt_id=0)
    collector.attach(ea_model)
    output = ea_model.eagenerate(...)
    trace = collector.detach()
    trace.save("traces/gsm8k_run1.json")

Thread safety: not guaranteed. Use one Collector per inference thread.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import torch

from sim.trace.schema import DecodeStepTrace, TokenNode, TraceDataset


class Collector:
    """Attaches to an EaModel instance to log per-token trace data."""

    def __init__(
        self,
        dataset: str = "unknown",
        prompt_id: int = 0,
        model_target: str = "Qwen2.5-7B",
        model_draft: str = "Qwen2.5-0.5B",
    ):
        self.dataset = dataset
        self.prompt_id = prompt_id
        self.model_target = model_target
        self.model_draft = model_draft

        # Internal state
        self._steps: List[DecodeStepTrace] = []
        self._step_counter: int = 0
        self._pending_nodes: Optional[List[TokenNode]] = None
        self._pending_retrieve_indices: Optional[Any] = None
        self._pending_context_len: int = 0
        self._pending_sample_token_id: int = 0

        # Saved originals (for detach)
        self._original_topk: Optional[Any] = None
        self._original_eval_posterior: Optional[Any] = None
        self._attached_ea_layer: Optional[Any] = None
        self._ea_model_mod: Optional[Any] = None

    def attach(self, ea_model: Any) -> None:
        ea_layer = ea_model.ea_layer
        self._attached_ea_layer = ea_layer

        self._original_topk = ea_layer.topK_genrate
        ea_layer.topK_genrate = self._make_topk_wrapper(ea_layer.topK_genrate)

        # Patch evaluate_posterior in ea_model's namespace, not utils —
        # ea_model.py does `from .utils import *` creating its own reference.
        try:
            import eagle.model.ea_model as ea_model_mod
            self._ea_model_mod = ea_model_mod
            self._original_eval_posterior = ea_model_mod.evaluate_posterior
            ea_model_mod.evaluate_posterior = self._make_eval_wrapper(
                ea_model_mod.evaluate_posterior
            )
        except (ImportError, AttributeError):
            print(
                "[Collector] Warning: could not patch evaluate_posterior. "
                "Acceptance flags will not be set."
            )

    def detach(self) -> TraceDataset:
        if self._attached_ea_layer is not None and self._original_topk is not None:
            self._attached_ea_layer.topK_genrate = self._original_topk
        if self._ea_model_mod is not None and self._original_eval_posterior is not None:
            self._ea_model_mod.evaluate_posterior = self._original_eval_posterior

        td = TraceDataset(
            steps=list(self._steps),
            model_target=self.model_target,
            model_draft=self.model_draft,
            metadata={
                "dataset": self.dataset,
                "prompt_id": self.prompt_id,
                "total_steps": self._step_counter,
            },
        )
        td.compute_summary()
        return td

    # ------------------------------------------------------------------
    # Internal wrappers
    # ------------------------------------------------------------------

    def _make_topk_wrapper(self, original_fn):
        collector = self

        @functools.wraps(original_fn)
        def wrapper(hidden_states, input_ids, head, logits_processor):
            context_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 0
            collector._pending_context_len = int(context_len)
            collector._pending_sample_token_id = int(input_ids[0, -1].item()) if hasattr(input_ids, "shape") else 0

            result = original_fn(hidden_states, input_ids, head, logits_processor)

            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result
            collector._pending_retrieve_indices = retrieve_indices.cpu()

            # Token IDs for the 60 draft nodes (exclude index 0 = sample token)
            tokens = draft_tokens[0, 1:].cpu().tolist()
            n = len(tokens)

            # Depths: tree_position_ids are 1-indexed (sample token = 0 excluded).
            # Subtract 1 to make depth-0 the first draft layer.
            depths = [d - 1 for d in tree_position_ids[1:].cpu().tolist()]

            # Cumulative scores saved by cnets1.py before the del statement.
            # Shape: [n] — one value per node in tree-position order.
            ea_layer = collector._attached_ea_layer
            cum_scores = ea_layer._capim_scores.tolist() if hasattr(ea_layer, "_capim_scores") else [float("nan")] * n

            # Recover parent indices from tree_mask.
            # tree_mask[0,0,i,j]=True means j is an ancestor of i (1-indexed).
            # The direct parent is the second-largest True index in row i
            # (largest is always self, second-largest is direct parent).
            parent_idx_map: Dict[int, int] = {}
            mask = tree_mask[0, 0].bool()  # strip batch+head dims [1,1,n+1,n+1] → [n+1,n+1]
            # mask is (n+1) x (n+1): index 0 = sample token (root), indices 1..n = draft tokens.
            # We skip index 0 and convert to 0-indexed tokens_flat space via i-1.
            # nonzero returns ascending indices: last = self (i), second-to-last = direct parent.
            # Parents always have smaller indices than children (tree built in depth order).
            for i in range(1, n + 1):
                ancestors = torch.nonzero(mask[i], as_tuple=False).squeeze(1)
                # ancestors[-1]=self(i), ancestors[-2]=direct parent (or root=0 for depth-0 nodes)
                direct_parent = int(ancestors[-2]) - 1  # convert to 0-indexed tokens space
                parent_idx_map[i - 1] = direct_parent

            # Build nodes.
            # log_prob = cumulative score of this node minus parent's cumulative score.
            # For depth-0 nodes (no parent in tree) log_prob == cumulative_log_prob.
            nodes = []
            depth_layer_counter: Dict[int, int] = {}
            for i, (token_id, depth) in enumerate(zip(tokens, depths)):
                cum = cum_scores[i]
                parent = parent_idx_map.get(i, -1)
                parent_cum = cum_scores[parent] if parent >= 0 else 0.0
                log_prob = cum - parent_cum

                layer_idx = depth_layer_counter.get(depth, 0)
                depth_layer_counter[depth] = layer_idx + 1

                nodes.append(TokenNode(
                    depth=depth,
                    token_id=int(token_id),
                    log_prob=log_prob,
                    cumulative_log_prob=cum,
                    parent_idx=parent,
                    accepted=False,
                    layer_idx=layer_idx,
                ))

            collector._pending_nodes = nodes
            return result

        return wrapper

    def _make_eval_wrapper(self, original_fn):
        collector = self

        @functools.wraps(original_fn)
        def wrapper(logits, candidates, logits_processor):
            best_candidate, accept_length, sample_p = original_fn(
                logits, candidates, logits_processor
            )

            if collector._pending_nodes is not None:
                al = int(accept_length)
                best_cand_int = int(best_candidate)
                ri = collector._pending_retrieve_indices

                try:
                    path = ri[best_cand_int].tolist()
                    accepted_node_indices = set()
                    # retrieve_indices[0] = sample_token (index 0, always skipped).
                    # Real draft tokens start at index 1, so iterate al+1 entries.
                    for j in range(al + 1):
                        if j >= len(path):
                            break
                        idx = path[j]
                        if idx > 0:  # exclude sample_token (0) and padding (-1)
                            accepted_node_indices.add(idx - 1)
                    for k, node in enumerate(collector._pending_nodes):
                        if k in accepted_node_indices:
                            node.accepted = True
                except Exception as e:
                    print(f"[Collector] Warning: could not mark accepted nodes at step "
                          f"{collector._step_counter}: {e}. Step skipped.")
                    collector._pending_nodes = None
                    return best_candidate, accept_length, sample_p

                step = DecodeStepTrace(
                    step_id=collector._step_counter,
                    context_length=collector._pending_context_len,
                    nodes=list(collector._pending_nodes),
                    accepted_length=al,
                    dataset=collector.dataset,
                    prompt_id=collector.prompt_id,
                    sample_token_id=collector._pending_sample_token_id,
                )
                collector._steps.append(step)
                collector._step_counter += 1
                collector._pending_nodes = None

            return best_candidate, accept_length, sample_p

        return wrapper


# ---------------------------------------------------------------------------
# High-level collection runner
# ---------------------------------------------------------------------------


def collect_traces(
    ea_model: Any,
    tokenizer: Any,
    prompts: List[str],
    dataset: str = "unknown",
    max_new_tokens: int = 200,
    output_path: Optional[str] = None,
) -> TraceDataset:
    """
    Run EAGLE-2 inference on a list of prompts and collect a TraceDataset.

    Args:
        ea_model: EaModel instance with ea_layer already loaded.
        tokenizer: Tokenizer for the base model.
        prompts: List of input prompt strings.
        dataset: Dataset label for the trace metadata.
        max_new_tokens: Maximum tokens to generate per prompt.
        output_path: If provided, save the trace JSON to this path.

    Returns:
        TraceDataset with all collected steps.
    """
    import torch

    all_steps = []

    for prompt_id, prompt in enumerate(prompts):
        collector = Collector(
            dataset=dataset,
            prompt_id=prompt_id,
            model_target=ea_model.base_model_name_or_path,
            model_draft=str(ea_model.ea_layer.__class__.__name__),
        )
        collector.attach(ea_model)

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(ea_model.base_model.device)
            with torch.no_grad():
                _ = ea_model.eagenerate(
                    inputs["input_ids"],
                    is_llama3=False,
                    temperature=0,
                    top_p=0,
                    top_k=0,
                    max_new_tokens=max_new_tokens,
                )
        except Exception as e:
            print(f"[Collector] Warning: inference failed for prompt {prompt_id}: {e}")
        finally:
            partial = collector.detach()
            all_steps.extend(partial.steps)

    td = TraceDataset(
        steps=all_steps,
        model_target=ea_model.base_model_name_or_path,
        model_draft="EAGLE-2",
        metadata={
            "dataset": dataset,
            "n_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
        },
    )
    td.compute_summary()

    if output_path:
        td.save(output_path)
        print(f"[Collector] Saved {len(all_steps)} steps to {output_path}")

    return td
