"""
Reporting + aggregation.

Aggregation unit = per prompt (group StepRecords by prompt_id).  We sum per-prompt
TOTALS (time, energy, tokens) and then average those totals across prompts, so the
comparison ratios come out token-weighted automatically (CLAUDE.md §5 / handover §2).

Metrics match LP-Spec Table III:
  - throughput        token/s   = tokens / time
  - energy efficiency token/J   = tokens / energy
  - EDP (s*mJ)                  = time * energy_mJ      (lower = better)
Reported as mean +/- std across prompts.  A trace covers one dataset (Alpaca or
GSM8K); run the two datasets as separate traces and compare their reports.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List

from sim.drivers.base import DriverResult, StepRecord


@dataclass
class PromptAgg:
    prompt_id: int
    dataset: str
    tokens: float
    time_s: float
    energy_j: float

    @property
    def token_per_s(self) -> float:
        return self.tokens / self.time_s if self.time_s > 0 else 0.0

    @property
    def token_per_j(self) -> float:
        return self.tokens / self.energy_j if self.energy_j > 0 else 0.0

    @property
    def edp_s_mj(self) -> float:
        # Per-token energy-delay product so it is comparable across drivers that
        # emit different token counts: (s/token) * (mJ/token).
        if self.tokens <= 0 or self.time_s <= 0:
            return 0.0
        lat_per_token = self.time_s / self.tokens
        e_per_token_mj = (self.energy_j * 1e3) / self.tokens
        return lat_per_token * e_per_token_mj


def aggregate_by_prompt(result: DriverResult) -> List[PromptAgg]:
    # A trace covers exactly one dataset, so prompt_id is unique within it and we
    # bucket on prompt_id alone (no dataset slicing needed).
    buckets: Dict[int, PromptAgg] = {}
    for s in result.steps:
        a = buckets.get(s.prompt_id)
        if a is None:
            buckets[s.prompt_id] = PromptAgg(s.prompt_id, s.dataset, s.tokens_emitted,
                                             s.time_s, s.energy_j)
        else:
            a.tokens += s.tokens_emitted
            a.time_s += s.time_s
            a.energy_j += s.energy_j
    return list(buckets.values())


def _mean_std(xs: List[float]):
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


@dataclass
class Summary:
    driver: str
    dataset: str
    n_prompts: int
    token_per_s_mean: float
    token_per_s_std: float
    token_per_j_mean: float
    token_per_j_std: float
    edp_mean: float
    edp_std: float
    # token-weighted totals (averaged per-prompt totals)
    mean_time_s: float
    mean_energy_j: float
    mean_tokens: float


def summarize(result: DriverResult) -> Summary:
    aggs = aggregate_by_prompt(result)
    tps_m, tps_s = _mean_std([a.token_per_s for a in aggs])
    tpj_m, tpj_s = _mean_std([a.token_per_j for a in aggs])
    edp_m, edp_s = _mean_std([a.edp_s_mj for a in aggs])
    t_m, _ = _mean_std([a.time_s for a in aggs])
    e_m, _ = _mean_std([a.energy_j for a in aggs])
    tok_m, _ = _mean_std([a.tokens for a in aggs])
    ds = aggs[0].dataset if aggs else "unknown"
    return Summary(
        driver=result.driver, dataset=ds, n_prompts=len(aggs),
        token_per_s_mean=tps_m, token_per_s_std=tps_s,
        token_per_j_mean=tpj_m, token_per_j_std=tpj_s,
        edp_mean=edp_m, edp_std=edp_s,
        mean_time_s=t_m, mean_energy_j=e_m, mean_tokens=tok_m,
    )


def comparison_table(results: List[DriverResult],
                     baseline_driver: str = "LP-Spec") -> str:
    """Format a comparison table with speedup/efficiency vs a baseline driver."""
    summaries = [summarize(r) for r in results]
    base = next((s for s in summaries if s.driver == baseline_driver), summaries[0])

    rows = []
    header = f"{'Driver':<16} {'token/s':>14} {'token/J':>14} {'EDP(s·mJ)':>14} {'vs '+base.driver:>22}"
    rows.append(header)
    rows.append("-" * len(header))
    for s in summaries:
        spd = s.token_per_s_mean / base.token_per_s_mean if base.token_per_s_mean else 0.0
        eff = s.token_per_j_mean / base.token_per_j_mean if base.token_per_j_mean else 0.0
        edp_imp = base.edp_mean / s.edp_mean if s.edp_mean else 0.0
        rows.append(
            f"{s.driver:<16} "
            f"{s.token_per_s_mean:>10.1f}±{s.token_per_s_std:<3.0f} "
            f"{s.token_per_j_mean:>10.1f}±{s.token_per_j_std:<3.0f} "
            f"{s.edp_mean:>14.3g} "
            f"{spd:>6.2f}x sp {eff:>5.2f}x eff {edp_imp:>5.2f}x edp"
        )
    return "\n".join(rows)


def export_csv(results: List[DriverResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["driver", "dataset", "n_prompts", "token_per_s", "token_per_s_std",
                    "token_per_j", "token_per_j_std", "edp_s_mj", "edp_std",
                    "mean_time_s", "mean_energy_j", "mean_tokens"])
        for r in results:
            s = summarize(r)
            w.writerow([s.driver, s.dataset, s.n_prompts,
                        s.token_per_s_mean, s.token_per_s_std,
                        s.token_per_j_mean, s.token_per_j_std,
                        s.edp_mean, s.edp_std, s.mean_time_s, s.mean_energy_j, s.mean_tokens])
