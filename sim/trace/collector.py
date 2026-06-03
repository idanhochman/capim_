"""
EAGLE-2 trace collector for CAPIM.

Instruments EAGLE's inference code to log per-token confidence data needed
for CAPIM's confidence-gated pruner calibration.

Design principles:
  - Does NOT modify any EAGLE source files.
  - Uses Python monkey-patching at runtime.
  - Captures the minimum needed data: token IDs, per-token log_probs,
    cumulative log_probs, parent indices, and acceptance flags.
  - Saves results as TraceDataset JSON for offline simulation.

What we patch:
  1. ea_model.Model.ea_layer.topK_genrate  (cnets1.Model.topK_genrate)
     — captures scores_list, ss_token, parents_list before the `del` call.
  2. eagle.model.utils.evaluate_posterior
     — captures (best_candidate, accept_length) for each step.

How to use:
    from sim.trace.collector import Collector
    from eagle.model.ea_model import EaModel

    collector = Collector(dataset="gsm8k", prompt_id=0)
    collector.attach(ea_model)           # ea_model is an EaModel instance

    # Run inference normally...
    output = ea_model.eagenerate(...)

    trace = collector.detach()           # returns TraceDataset
    trace.save("traces/gsm8k_run1.json")

Instrumentation notes:
  - We patch the instance's ea_layer.topK_genrate (not the class),
    so multiple EaModel instances can be used simultaneously without conflict.
  - The `evaluate_posterior` patch is applied at module level (eagle.model.utils)
    since it is a standalone function.  The collector maintains a step counter
    to match topK_genrate calls to evaluate_posterior calls.
  - Thread safety: not guaranteed.  Use one Collector per inference thread.

EAGLE-2 tree structure recap:
  - depth 0: top_k tokens, scores = topk_p[0]  (log-probs for each of top_k)
  - depth d: top_k tokens selected from top_k*top_k candidates using cu_scores
  - scores_list[0].shape = [1, top_k]  (depth 0)
  - scores_list[d].shape = [top_k, top_k]  (depth d, before flattening)
  - After the topK_genrate loop: all scores are indexed by the selected top_k subset.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sim.trace.schema import DecodeStepTrace, TokenNode, TraceDataset


class _LogSoftmaxCapture(nn.Module):
    """Wraps an nn.LogSoftmax to intercept outputs into a list for trace collection."""

    def __init__(self, original: nn.Module, sink: list):
        super().__init__()
        self.original = original
        self.sink = sink

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original(x)
        self.sink.append(result.detach().cpu())
        return result


class Collector:
    """
    Attaches to an EaModel instance to log per-token trace data.

    Args:
        dataset: Dataset label ("alpaca" or "gsm8k").
        prompt_id: Index of the current prompt in the dataset.
        model_target: Name of the target model.
        model_draft: Name of the draft model.
    """

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
        self._pending_nodes: Optional[List[TokenNode]] = None  # set by topK hook
        self._pending_retrieve_indices: Optional[Any] = None   # set by topK hook
        self._pending_context_len: int = 0

        # Saved original methods (for detach)
        self._original_topk: Optional[Any] = None
        self._original_eval_posterior: Optional[Any] = None
        self._attached_ea_layer: Optional[Any] = None
        self._ea_model_mod: Optional[Any] = None

    def attach(self, ea_model: Any) -> None:
        """
        Monkey-patch ea_model.ea_layer.topK_genrate and
        eagle.model.utils.evaluate_posterior.

        Args:
            ea_model: An EaModel instance (from eagle.model.ea_model).
        """
        ea_layer = ea_model.ea_layer
        self._attached_ea_layer = ea_layer

        # Patch topK_genrate on the instance (not the class)
        self._original_topk = ea_layer.topK_genrate
        ea_layer.topK_genrate = self._make_topk_wrapper(ea_layer.topK_genrate)

        # Patch evaluate_posterior in eagle.model.ea_model — that module does
        # `from .utils import *`, so it holds its own reference to the function.
        # Patching utils alone has no effect; we must patch the caller's namespace.
        try:
            import eagle.model.ea_model as ea_model_mod
            self._ea_model_mod = ea_model_mod
            self._original_eval_posterior = ea_model_mod.evaluate_posterior
            ea_model_mod.evaluate_posterior = self._make_eval_wrapper(ea_model_mod.evaluate_posterior)
        except (ImportError, AttributeError):
            print(
                "[Collector] Warning: could not patch evaluate_posterior in "
                "eagle.model.ea_model. Acceptance flags will not be set."
            )

    def detach(self) -> TraceDataset:
        """
        Remove patches and return the collected TraceDataset.

        Returns:
            TraceDataset with all collected steps.
        """
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
        """
        Return a wrapped version of topK_genrate that logs node data before
        the `del` statement destroys the intermediate tensors.
        """
        collector = self

        @functools.wraps(original_fn)
        def wrapper(hidden_states, input_ids, head, logits_processor):
            # We re-implement the logging by calling the original function
            # but intercepting internal variables.  Since we can't inject
            # into the middle of the function without modifying it, we call
            # a lightweight re-run of just the scoring logic.
            #
            # Strategy: call the original function normally, then reconstruct
            # node metadata from the returned draft_tokens and the ea_layer's
            # attributes that are accessible post-call.
            #
            # For the per-token log_prob, we use a pre-call hook that
            # captures the scores by temporarily replacing logsoftmax.

            # Context length = current input_ids length (used for KV-cache sizing)
            context_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 0
            collector._pending_context_len = int(context_len)

            # Install a logsoftmax interceptor
            original_logsoftmax = None
            ea_layer = collector._attached_ea_layer

            # Intercept logsoftmax to capture raw log-prob distributions
            depth_scores: List[torch.Tensor] = []      # [top_k] per-token log_probs per depth
            depth_tokens: List[torch.Tensor] = []      # [top_k] token indices per depth
            depth_cu_scores: List[torch.Tensor] = []   # cumulative scores per depth

            if hasattr(ea_layer, "logsoftmax"):
                original_logsoftmax = ea_layer.logsoftmax
                depth_raw: List[torch.Tensor] = []
                ea_layer.logsoftmax = _LogSoftmaxCapture(original_logsoftmax, depth_raw)

            # Call the original function
            result = original_fn(hidden_states, input_ids, head, logits_processor)

            # Restore logsoftmax
            if original_logsoftmax is not None:
                ea_layer.logsoftmax = original_logsoftmax

            # Reconstruct node structure from the returned draft_tokens.
            # draft_tokens[0] is the sample token (root); draft_tokens[1:] are the
            # proposed tokens in tree order.
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result
            collector._pending_retrieve_indices = retrieve_indices.cpu()
            tokens_flat = draft_tokens[0, 1:].cpu().tolist()  # exclude sample token

            # tree_position_ids[0] = 0 (sample token), tree_position_ids[1:] = depths
            if tree_position_ids is not None:
                depths = tree_position_ids[1:].cpu().tolist()
            else:
                depths = [0] * len(tokens_flat)

            # Build tree_mask-based parent mapping
            # tree_mask shape: [1, 1, total_tokens+1, total_tokens+1]
            # mask[i][j]=1 means token j is an ancestor of token i
            nodes = collector._build_nodes_from_tree(
                tokens=tokens_flat,
                depths=depths,
                depth_raw=depth_raw if original_logsoftmax is not None else [],
                tree_mask=tree_mask,
            )

            collector._pending_nodes = nodes
            return result

        return wrapper

    def _build_nodes_from_tree(
        self,
        tokens: List[int],
        depths: List[int],
        depth_raw: List[torch.Tensor],
        tree_mask: Optional[torch.Tensor],
    ) -> List[TokenNode]:
        """
        Build TokenNode list from draft tree structure.

        Args:
            tokens: Flat list of draft token IDs (len = total_tokens).
            depths: Depth of each token in the tree (0-based).
            depth_raw: Raw log-softmax outputs from each logsoftmax call.
                       depth_raw[0] = depth-0 distribution (shape [1, vocab])
                       depth_raw[1:] = deeper distributions (shape [top_k, vocab])
            tree_mask: Boolean tree mask from topK_genrate,
                       shape [1, 1, total+1, total+1].

        Returns:
            List of TokenNode objects.
        """
        n = len(tokens)
        if n == 0:
            return []

        nodes = []

        # Recover parent indices from tree_mask
        # tree_mask[0,0,i,j] = True means token j is an ancestor of i
        # The direct parent is the last ancestor below i in the mask
        # (i.e., the one with the highest position/depth that is still set)
        parent_idx_map: Dict[int, int] = {}
        if tree_mask is not None:
            mask = tree_mask[0, 0].bool()  # [total+1, total+1]
            for i in range(1, n + 1):  # skip index 0 (sample token)
                # ancestors of token i = indices where mask[i, j] is True
                ancestor_mask = mask[i]  # [total+1]
                # The direct parent is the nearest ancestor (excluding self, excluding root=0)
                # In EAGLE's mask: index 0 = sample token (always True), index i = self (True)
                # Direct parent = max index j < i where mask[i, j] = True (excluding i)
                ancestors = torch.nonzero(ancestor_mask, as_tuple=False).squeeze(1)
                # Remove self (index i) and root (index 0 = sample token is index 0 in +1 space)
                # In +1 indexed space: token 0 = sample_token, tokens 1..n = draft tokens
                direct_parent = -1
                for j in sorted(ancestors.tolist(), reverse=True):
                    if j != i:
                        direct_parent = j - 1  # convert back to 0-indexed draft token space
                        break
                parent_idx_map[i - 1] = direct_parent

        # Assign log_prob and cumulative_log_prob from intercepted distributions
        # depth_raw[call_idx] = logsoftmax output at a given inference call
        # We have one call per depth level (0 to depth)
        # shape of depth_raw[0]: [1, vocab] (first token)
        # shape of depth_raw[d] for d>0: [top_k_prev, vocab]

        # Group tokens by depth
        depth_to_token_indices: Dict[int, List[int]] = {}
        for i, d in enumerate(depths):
            depth_to_token_indices.setdefault(d, []).append(i)

        max_depth = max(depths) if depths else 0

        # We'll store log_prob and cum_log_prob per token
        log_probs = [float("nan")] * n
        cum_log_probs = [float("nan")] * n

        # Recover from depth_raw when available
        # depth_raw has one entry per logsoftmax call:
        #   call 0 → depth 0 distribution (the initial token from hidden_states)
        #   calls 1..depth → distributions for each tree expansion level
        # Note: each call may produce top_k or top_k*top_k outputs

        for idx, (token_id, depth) in enumerate(zip(tokens, depths)):
            raw_call_idx = depth  # logsoftmax call index matches depth
            if raw_call_idx < len(depth_raw):
                dist = depth_raw[raw_call_idx]  # [*, vocab]
                # dist may be [1, vocab] or [top_k, vocab] — pick the relevant row
                # For depth 0: all top_k tokens came from dist[0]
                # For depth d: each top_k parent produced top_k children
                # Use the token's position within its depth layer
                layer_pos = depth_to_token_indices[depth].index(idx)
                if dist.dim() == 2:
                    row = min(layer_pos // max(1, (n // (max_depth + 1))), dist.shape[0] - 1)
                    lp = float(dist[row, token_id])
                elif dist.dim() == 1:
                    lp = float(dist[token_id])
                else:
                    lp = float("nan")
                log_probs[idx] = lp
            else:
                log_probs[idx] = float("nan")

            # Cumulative log_prob: sum from root
            parent = parent_idx_map.get(idx, -1)
            if parent >= 0 and not math.isnan(log_probs[idx]):
                cum_log_probs[idx] = log_probs[idx] + (
                    cum_log_probs[parent] if not math.isnan(cum_log_probs[parent]) else 0.0
                )
            else:
                cum_log_probs[idx] = log_probs[idx]

        for i, (token_id, depth) in enumerate(zip(tokens, depths)):
            layer_idx = depth_to_token_indices[depth].index(i)
            nodes.append(
                TokenNode(
                    depth=depth,
                    token_id=int(token_id),
                    log_prob=log_probs[i],
                    cumulative_log_prob=cum_log_probs[i],
                    parent_idx=parent_idx_map.get(i, -1),
                    accepted=False,  # filled in by evaluate_posterior hook
                    layer_idx=layer_idx,
                )
            )

        return nodes

    def _make_eval_wrapper(self, original_fn):
        """
        Return a wrapped evaluate_posterior that records acceptance flags.
        """
        collector = self

        @functools.wraps(original_fn)
        def wrapper(logits, candidates, logits_processor):
            best_candidate, accept_length, sample_p = original_fn(
                logits, candidates, logits_processor
            )

            # Mark accepted nodes in the pending tree
            if collector._pending_nodes is not None:
                al = int(accept_length)
                best_cand_int = int(best_candidate)

                # retrieve_indices[best_candidate] = path of draft_tokens[0] indices
                # along the accepted candidate, padded with -1.  Values are 1-indexed
                # into draft_tokens[0] (0 = sample_token), so subtract 1 to get
                # 0-indexed positions in tokens_flat (= pending_nodes order).
                ri = collector._pending_retrieve_indices
                if ri is not None and best_cand_int < ri.shape[0]:
                    path = ri[best_cand_int].tolist()
                    accepted_node_indices = set()
                    for j in range(al):
                        idx = path[j]
                        if idx > 0:  # >0 excludes sample_token (0) and padding (-1)
                            accepted_node_indices.add(idx - 1)
                    for k, node in enumerate(collector._pending_nodes):
                        if k in accepted_node_indices:
                            node.accepted = True
                else:
                    # Fallback if retrieve_indices unavailable
                    for node in collector._pending_nodes:
                        if node.depth < al:
                            node.accepted = True

                # Record the step
                step = DecodeStepTrace(
                    step_id=collector._step_counter,
                    context_length=collector._pending_context_len,
                    nodes=list(collector._pending_nodes),
                    accepted_length=al,
                    dataset=collector.dataset,
                    prompt_id=collector.prompt_id,
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

    This is the top-level convenience function for data collection.

    Args:
        ea_model: EaModel instance with ea_layer already loaded.
        tokenizer: Tokenizer for the base model.
        prompts: List of input prompt strings.
        dataset: Dataset label for the trace metadata.
        max_new_tokens: Maximum tokens to generate per prompt.
        output_path: If provided, save the trace JSON to this path.

    Returns:
        TraceDataset with all collected steps.

    Example:
        from transformers import AutoTokenizer
        from eagle.model.ea_model import EaModel

        # ... load ea_model ...

        trace = collect_traces(
            ea_model=ea_model,
            tokenizer=tokenizer,
            prompts=gsm8k_prompts,
            dataset="gsm8k",
            output_path="traces/qwen25_gsm8k.json",
        )
    """
    import torch

    all_steps = []
    total_prompt_id = 0

    for prompt_id, prompt in enumerate(prompts):
        collector = Collector(
            dataset=dataset,
            prompt_id=prompt_id,
            model_target=ea_model.base_model_name_or_path,
            model_draft=str(ea_model.ea_layer.__class__.__name__),
        )
        collector.attach(ea_model)

        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(ea_model.base_model.device)
            input_ids = inputs["input_ids"]

            # Run EAGLE-2 generation
            # eagenerate is the standard interface
            with torch.no_grad():
                _ = ea_model.eagenerate(
                    input_ids,
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

    # Build combined dataset
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
