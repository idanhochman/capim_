"""
MEDUSA trace collector for CAPIM (LP-Spec baseline).  *** SKETCH / DRAFT ***

Mirror of sim/trace/eagle_collector.py so the LP-Spec baseline is built from
traces collected the SAME way, on the SAME backbone (Vicuna-7B-v1.3), as the
EAGLE-2 traces -- replacing the current published/assumed scalars in
drivers/lp_spec.py (acceptance_length, dtp_pruning_ratio) with measured per-step
values.  Emits the IDENTICAL schema (DecodeStepTrace / TokenNode) so the
simulator drivers consume EAGLE and MEDUSA traces uniformly.

STATUS: not yet run.  Hook points and field names are grounded in the vendored
repo (releted-repos/Medusa/medusa/model/{utils,medusa_model}.py), but tensor
SHAPES (esp. medusa_attn_mask dims, retrieve_indices padding, medusa_logits
layout) must be confirmed on a real Colab run -- see the TODO/CONFIRM markers.

------------------------------------------------------------------------------
EAGLE  ->  MEDUSA  mapping
------------------------------------------------------------------------------
EAGLE topK_genrate (returns the draft tree)   ->  MEDUSA's tree is STATIC:
    its shape (depth, parent, layer_idx) comes ONCE from model.medusa_buffers
    (generate_medusa_buffers), and the per-step TOKENS come from
    generate_candidates' `tree_candidates`.
EAGLE evaluate_posterior (returns 3-tuple)    ->  MEDUSA evaluate_posterior
    returns a 2-tuple (best_candidate, accept_length); acceptance marking via
    retrieve_indices is otherwise identical.
context_length                                ->  taken from tree_decoding's
    `input_ids` (generate_candidates does not receive it).

What we hook (all in the medusa.model.medusa_model namespace, where the
medusa_generate loop calls them as bare names -- same reason the EAGLE collector
patches ea_model's namespace, not utils):
  1. generate_candidates  -> capture per-node tokens (+ best-effort confidence).
  2. tree_decoding        -> capture context_length (input_ids.shape[1]).
  3. evaluate_posterior   -> capture acceptance, assemble + emit the step.

NOTE on confidence: MEDUSA per-node log_prob (from the heads' logits) is NOT
load-bearing for the LP-Spec driver -- that driver needs tree_size +
accepted_length + per-position acceptance (for the retrospective DTP), none of
which need confidence.  We capture log_prob best-effort for schema parity and
possible ablations; it falls back to NaN if the head->node mapping can't be
resolved.  (CAPIM's sigma_th gate is EAGLE-only and never reads these.)
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional

import torch

from sim.trace.schema import DecodeStepTrace, TokenNode, TraceDataset


class MedusaCollector:
    """Attaches to a MedusaModel instance to log per-step trace data."""

    def __init__(
        self,
        dataset: str = "unknown",
        prompt_id: int = 0,
        model_target: str = "Vicuna-7B-v1.3",
        model_draft: str = "medusa-vicuna-7b-v1.3",
        precision: str = "fp16",
    ):
        self.dataset = dataset
        self.prompt_id = prompt_id
        self.model_target = model_target
        self.model_draft = model_draft
        self.precision = precision

        # Collected steps
        self._steps: List[DecodeStepTrace] = []
        self._step_counter: int = 0

        # Per-step pending state (filled across the 3 hooks, flushed in eval)
        self._pending_tokens: Optional[List[int]] = None      # N draft node tokens
        self._pending_logprob: Optional[List[float]] = None   # best-effort per-node
        self._pending_context_len: int = 0
        self._pending_sample_token_id: int = 0

        # Static tree structure (cached: same for every step at fixed medusa_choices)
        self._static_depths: Optional[List[int]] = None       # 0-indexed draft depth
        self._static_parents: Optional[List[int]] = None       # 0-indexed parent in node space
        self._static_layer_idx: Optional[List[int]] = None
        self._retrieve_indices: Optional[torch.Tensor] = None

        # Patch bookkeeping
        self._model: Optional[Any] = None
        self._mod: Optional[Any] = None
        self._orig: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def attach(self, medusa_model: Any) -> None:
        self._model = medusa_model
        import medusa.model.medusa_model as mod  # the loop's namespace
        self._mod = mod

        # CONFIRM: medusa_model.py does `from .utils import *`, so these names
        # live in `mod`.  If not, patch medusa.model.utils instead.
        self._orig["generate_candidates"] = mod.generate_candidates
        self._orig["tree_decoding"] = mod.tree_decoding
        self._orig["evaluate_posterior"] = mod.evaluate_posterior

        mod.generate_candidates = self._wrap_candidates(mod.generate_candidates)
        mod.tree_decoding = self._wrap_tree_decoding(mod.tree_decoding)
        mod.evaluate_posterior = self._wrap_eval(mod.evaluate_posterior)

    def detach(self) -> TraceDataset:
        if self._mod is not None:
            for name, fn in self._orig.items():
                setattr(self._mod, name, fn)
        td = TraceDataset(
            steps=list(self._steps),
            model_target=self.model_target,
            model_draft=self.model_draft,
            metadata={
                "dataset": self.dataset,
                "prompt_id": self.prompt_id,
                "total_steps": self._step_counter,
                "precision": self.precision,
                "sd_method": "medusa",
            },
        )
        td.compute_summary()
        return td

    # ------------------------------------------------------------------
    # Static structure: derived once from model.medusa_buffers.
    # medusa_attn_mask: [1,1,L,L] ancestry (row i attends ancestors, [:,0]=root).
    # medusa_position_ids: [L] depth (0=root, i+1 = i-th head layer).
    # Index 0 = root/sample token; draft nodes are indices 1..N (N=L-1).
    # ------------------------------------------------------------------
    def _ensure_static(self) -> None:
        if self._static_depths is not None:
            return
        buffers = self._model.medusa_buffers
        pos = buffers["medusa_position_ids"].cpu().tolist()           # [L]
        mask = buffers["medusa_attn_mask"][0, 0].bool().cpu()         # [L, L]  CONFIRM dims
        self._retrieve_indices = buffers["retrieve_indices"].cpu()    # [n_paths, max_len]
        L = len(pos)

        depths, parents, layer_idx = [], [], []
        depth_counter: Dict[int, int] = {}
        for i in range(1, L):                       # skip root (index 0)
            d = int(pos[i]) - 1                      # 0-indexed draft depth
            # parent = second-largest True ancestor in row i (largest is self=i)
            anc = torch.nonzero(mask[i], as_tuple=False).squeeze(1)
            parent_node = int(anc[-2]) - 1 if anc.numel() >= 2 else -1   # -1 => child of root
            depths.append(d)
            parents.append(parent_node)
            layer_idx.append(depth_counter.get(d, 0))
            depth_counter[d] = depth_counter.get(d, 0) + 1
        self._static_depths = depths
        self._static_parents = parents
        self._static_layer_idx = layer_idx

    # ------------------------------------------------------------------
    # Hook 1: generate_candidates -> per-node tokens (+ best-effort confidence)
    # ------------------------------------------------------------------
    def _wrap_candidates(self, original_fn):
        collector = self

        @functools.wraps(original_fn)
        def wrapper(medusa_logits, logits, tree_indices, retrieve_indices, *args, **kwargs):
            cart_candidates, tree_candidates = original_fn(
                medusa_logits, logits, tree_indices, retrieve_indices, *args, **kwargs
            )
            # tree_candidates: [1, L]; index 0 = root/sample token, 1.. = draft nodes
            flat = tree_candidates[0].cpu().tolist()
            collector._pending_sample_token_id = int(flat[0])
            collector._pending_tokens = [int(t) for t in flat[1:]]
            collector._pending_logprob = collector._best_effort_logprob(
                medusa_logits, tree_indices, n_nodes=len(flat) - 1
            )
            return cart_candidates, tree_candidates

        return wrapper

    def _best_effort_logprob(self, medusa_logits, tree_indices, n_nodes) -> List[float]:
        """Per-node draft log-prob from the heads' logits.  Optional / NOT used
        by the LP-Spec driver.  Node at draft depth d came from head d; its
        log_prob = log_softmax(medusa_logits[d, 0, -1])[token].  Returns NaN on
        any shape mismatch (mapping is fiddly; validate on a real run)."""
        try:
            self._ensure_static()
            out = []
            for k in range(n_nodes):
                d = self._static_depths[k]                # head index = draft depth
                head_logits = medusa_logits[d, 0, -1]     # CONFIRM layout [H,1,S,V]
                lp = torch.log_softmax(head_logits.float(), dim=-1)
                out.append(float(lp[self._pending_tokens[k]].item()))
            return out
        except Exception:
            return [float("nan")] * n_nodes

    # ------------------------------------------------------------------
    # Hook 2: tree_decoding -> context_length (it receives input_ids)
    # ------------------------------------------------------------------
    def _wrap_tree_decoding(self, original_fn):
        collector = self

        @functools.wraps(original_fn)
        def wrapper(model, tree_candidates, past_key_values, medusa_position_ids,
                    input_ids, retrieve_indices, *args, **kwargs):
            collector._pending_context_len = int(input_ids.shape[1])
            return original_fn(model, tree_candidates, past_key_values,
                               medusa_position_ids, input_ids, retrieve_indices,
                               *args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Hook 3: evaluate_posterior -> acceptance; assemble + emit the step
    # MEDUSA returns a 2-tuple (best_candidate, accept_length).
    # ------------------------------------------------------------------
    def _wrap_eval(self, original_fn):
        collector = self

        @functools.wraps(original_fn)
        def wrapper(logits, candidates, *args, **kwargs):
            best_candidate, accept_length = original_fn(logits, candidates, *args, **kwargs)

            if collector._pending_tokens is not None:
                try:
                    collector._ensure_static()
                    nodes = collector._build_nodes()
                    collector._mark_accepted(nodes, int(best_candidate), int(accept_length))
                    collector._steps.append(DecodeStepTrace(
                        step_id=collector._step_counter,
                        context_length=collector._pending_context_len,
                        nodes=nodes,
                        accepted_length=int(accept_length),
                        dataset=collector.dataset,
                        prompt_id=collector.prompt_id,
                        sample_token_id=collector._pending_sample_token_id,
                    ))
                    collector._step_counter += 1
                except Exception as e:
                    print(f"[MedusaCollector] step {collector._step_counter} skipped: {e}")
                finally:
                    collector._pending_tokens = None

            return best_candidate, accept_length

        return wrapper

    def _build_nodes(self) -> List[TokenNode]:
        tokens = self._pending_tokens
        lps = self._pending_logprob or [float("nan")] * len(tokens)
        depths, parents, lidx = self._static_depths, self._static_parents, self._static_layer_idx
        # cumulative_log_prob = parent cumulative + this node's log_prob (path sum)
        cum: List[float] = [0.0] * len(tokens)
        nodes: List[TokenNode] = []
        for k, tok in enumerate(tokens):
            p = parents[k]
            parent_cum = cum[p] if (0 <= p < len(cum)) else 0.0
            lp = lps[k]
            cum[k] = (parent_cum + lp) if not math.isnan(lp) else float("nan")
            nodes.append(TokenNode(
                depth=depths[k], token_id=tok, log_prob=lp,
                cumulative_log_prob=cum[k], parent_idx=p,
                accepted=False, layer_idx=lidx[k],
            ))
        return nodes

    def _mark_accepted(self, nodes: List[TokenNode], best_candidate: int, accept_length: int) -> None:
        # retrieve_indices[best_candidate] = tree-position path; 0=root, j>0 => node j-1.
        # Accept accept_length tokens along the chosen path (+root offset), as in EAGLE.
        path = self._retrieve_indices[best_candidate].tolist()
        for j in range(accept_length + 1):
            if j >= len(path):
                break
            idx = path[j]
            if idx > 0:                              # skip root (0) and padding (-1)
                node_k = idx - 1
                if 0 <= node_k < len(nodes):
                    nodes[node_k].accepted = True
