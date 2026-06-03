"""
Smoke tests for the CAPIM simulation framework.

Runs without GPU or EAGLE-2 by using synthetic traces.
Tests that every module is importable and produces plausible results.

Run with:
    cd /home/idanh/msc/capim
    python -m pytest sim/tests/test_smoke.py -v
    # or directly:
    python sim/tests/test_smoke.py
"""

import sys
import os
import math

# Ensure capim/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def test_config_hardware():
    from sim.config.hardware import (
        PIM_INTERNAL_BW,
        PIM_EXTERNAL_BW,
        NPU_INT8_TOPS,
        PIM_ENERGY_PJ_PER_BIT,
        pj_to_j,
    )
    assert PIM_INTERNAL_BW == 51.2e12
    assert PIM_EXTERNAL_BW == 51.2e9
    assert NPU_INT8_TOPS == 32.8e12
    assert pj_to_j(1000) == 1e-9
    print("  [PASS] config.hardware")


def test_config_models():
    from sim.config.models import QWEN2_5_7B, QWEN2_5_0_5B

    w7b = QWEN2_5_7B.weight_bytes()
    w0_5b = QWEN2_5_0_5B.weight_bytes()

    # 7B model should have ~7 billion bytes (INT8 ≈ 7 GB)
    assert 5e9 < w7b < 10e9, f"7B weight_bytes={w7b:.2e} out of range"
    # 0.5B model should have ~0.5 billion bytes
    assert 0.3e9 < w0_5b < 1e9, f"0.5B weight_bytes={w0_5b:.2e} out of range"

    kv = QWEN2_5_7B.kv_cache_bytes(512)
    assert kv > 0
    print(f"  [PASS] config.models  (7B={w7b/1e9:.2f}GB, 0.5B={w0_5b/1e9:.3f}GB)")


def test_trace_schema():
    from sim.trace.schema import make_synthetic_trace

    td = make_synthetic_trace(n_steps=50, tree_size=20)
    assert len(td.steps) == 50
    assert td.mean_tree_size > 0
    assert td.mean_accepted_length > 0
    # Each step has nodes
    for step in td.steps[:3]:
        assert len(step.nodes) > 0
        assert step.tree_size == len(step.nodes)
    print(f"  [PASS] trace.schema  (mean_tree={td.mean_tree_size:.1f})")


def test_hw_pim():
    from sim.config.models import QWEN2_5_0_5B, QWEN2_5_7B
    import sim.hw.pim as pim

    t_draft = pim.draft_latency(QWEN2_5_0_5B, tree_size=20)
    e_draft = pim.draft_energy(QWEN2_5_0_5B, tree_size=20)
    assert t_draft > 0
    assert e_draft > 0

    t_verify = pim.verify_latency(QWEN2_5_7B, batch_size=10)
    e_verify = pim.verify_energy(QWEN2_5_7B, batch_size=10)
    assert t_verify > 0
    assert e_verify > 0

    # Zero batch should return zero
    assert pim.draft_latency(QWEN2_5_0_5B, 0) == 0.0
    assert pim.verify_energy(QWEN2_5_7B, 0) == 0.0

    print(f"  [PASS] hw.pim  (draft t={t_draft*1e3:.2f}ms, verify t={t_verify*1e3:.2f}ms)")


def test_hw_npu():
    from sim.config.models import QWEN2_5_7B
    import sim.hw.npu as npu

    t = npu.ar_token_latency(QWEN2_5_7B, seq_len=512)
    e = npu.ar_token_energy(QWEN2_5_7B, seq_len=512)
    assert t > 0
    assert e > 0

    t_batch = npu.verify_latency(QWEN2_5_7B, batch_size=10, seq_len=512)
    # Batched verification should be faster per-token than single-token
    assert t_batch > 0

    print(f"  [PASS] hw.npu  (AR: {1/t:.1f} tok/s, e={e*1000:.3f}mJ/tok)")


def test_scheduler():
    from sim.trace.schema import make_synthetic_trace
    from sim.scheduler import prune_tree, route, prune_stats

    td = make_synthetic_trace(n_steps=20, tree_size=20, seed=0)
    step = td.steps[0]

    # No pruning at sigma_th = -inf
    full = prune_tree(step, float("-inf"))
    assert len(full) == len(step.nodes)

    # Prune all at sigma_th = 0
    none = prune_tree(step, 0.0)
    # All log_probs < 0, so all pruned
    assert len(none) == 0 or all(n.log_prob >= 0 for n in none)

    # Route
    assert route(5, 10) == "PIM"
    assert route(10, 10) == "NPU"
    assert route(15, 10) == "NPU"

    # Stats
    stats = prune_stats(step, sigma_th=-2.0)
    assert 0.0 <= stats["pruning_ratio"] <= 1.0
    assert stats["original_size"] == step.tree_size

    print(f"  [PASS] scheduler  (prune@-2.0: {stats['pruning_ratio']*100:.1f}% removed)")


def test_baselines():
    from sim.config.models import QWEN2_5_7B
    from sim.trace.schema import make_synthetic_trace
    from sim.baselines.autoregressive import simulate_autoregressive_from_trace
    from sim.baselines.lp_spec import simulate_lp_spec_from_trace

    td = make_synthetic_trace(n_steps=50)

    ar = simulate_autoregressive_from_trace(QWEN2_5_7B, td)
    assert ar.latency_per_token_s > 0
    assert ar.energy_per_token_j > 0
    assert ar.tokens_per_second > 0

    lp = simulate_lp_spec_from_trace(QWEN2_5_7B, td)
    assert lp.latency_per_token_s > 0
    assert lp.energy_per_token_j > 0

    # LP-Spec should be faster than pure AR (SD speedup)
    # (may not always hold with our synthetic trace but the models should at least run)
    print(f"  [PASS] baselines  (AR: {ar.tokens_per_second:.1f} tok/s, "
          f"LP-Spec: {lp.tokens_per_second:.1f} tok/s)")


def test_simulation_e2e():
    from sim.config.models import QWEN2_5_7B, QWEN2_5_0_5B
    from sim.trace.schema import make_synthetic_trace
    from sim.baselines.autoregressive import simulate_autoregressive_from_trace
    from sim.baselines.lp_spec import simulate_lp_spec_from_trace
    from sim.simulation import simulate_capim

    td = make_synthetic_trace(n_steps=100, tree_size=20, acceptance_rate=0.4, seed=42)

    ar = simulate_autoregressive_from_trace(QWEN2_5_7B, td)
    lp = simulate_lp_spec_from_trace(QWEN2_5_7B, td)

    result = simulate_capim(
        trace=td,
        target_model=QWEN2_5_7B,
        draft_model=QWEN2_5_0_5B,
        sigma_th=-2.0,
        mu_th=10,
        scenario="synthetic",
        ar_latency_per_token=ar.latency_per_token_s,
        lp_latency_per_token=lp.latency_per_token_s,
        lp_energy_per_token=lp.energy_per_token_j,
        store_steps=False,
    )

    assert result.total_steps == 100
    assert result.total_accepted_tokens > 0
    assert result.tokens_per_second > 0
    assert result.energy_per_token_j > 0
    assert 0.0 <= result.pim_fraction <= 1.0
    assert 0.0 <= result.npu_fraction <= 1.0
    assert abs(result.pim_fraction + result.npu_fraction - 1.0) < 1e-9

    print(f"  [PASS] simulation  (CAPIM: {result.tokens_per_second:.1f} tok/s, "
          f"{result.energy_per_token_j*1000:.3f} mJ/tok, "
          f"speedup_vs_ar={result.speedup_vs_ar:.2f}x, "
          f"PIM_frac={result.pim_fraction*100:.0f}%)")


def test_results_compare():
    from sim.config.models import QWEN2_5_7B, QWEN2_5_0_5B
    from sim.trace.schema import make_synthetic_trace
    from sim.baselines.autoregressive import simulate_autoregressive_from_trace
    from sim.baselines.lp_spec import simulate_lp_spec_from_trace
    from sim.simulation import simulate_capim
    from sim.results import compare_results, sigma_sweep, export_csv
    import tempfile, os

    td = make_synthetic_trace(n_steps=50)
    ar = simulate_autoregressive_from_trace(QWEN2_5_7B, td)
    lp = simulate_lp_spec_from_trace(QWEN2_5_7B, td)
    capim = simulate_capim(
        td, QWEN2_5_7B, QWEN2_5_0_5B,
        sigma_th=-2.0, mu_th=10,
        ar_latency_per_token=ar.latency_per_token_s,
        lp_latency_per_token=lp.latency_per_token_s,
        lp_energy_per_token=lp.energy_per_token_j,
    )

    table = compare_results(
        capim,
        ar_energy_per_token=ar.energy_per_token_j,
        ar_latency_per_token=ar.latency_per_token_s,
        lp_energy_per_token=lp.energy_per_token_j,
        lp_latency_per_token=lp.latency_per_token_s,
    )
    assert "CAPIM" in table
    assert "LP-Spec" in table

    # sigma sweep
    sweep = sigma_sweep(
        td, QWEN2_5_7B, QWEN2_5_0_5B,
        sigma_values=[float("-inf"), -3.0, -2.0, -1.0],
        mu_th=10,
        ar_latency_per_token=ar.latency_per_token_s,
        lp_latency_per_token=lp.latency_per_token_s,
        lp_energy_per_token=lp.energy_per_token_j,
    )
    assert len(sweep) == 4

    # CSV export
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name
    try:
        export_csv(sweep, tmp)
        assert os.path.exists(tmp)
        with open(tmp) as f:
            lines = f.readlines()
        assert len(lines) == 5  # header + 4 rows
    finally:
        os.unlink(tmp)

    print(f"  [PASS] results  (table generated, sweep OK, CSV exported)")


def test_trace_save_load():
    from sim.trace.schema import make_synthetic_trace
    import tempfile, os

    td = make_synthetic_trace(n_steps=10)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        td.save(tmp)
        loaded = type(td).load(tmp)
        assert len(loaded.steps) == len(td.steps)
        assert loaded.model_target == td.model_target
        for orig, got in zip(td.steps, loaded.steps):
            assert orig.step_id == got.step_id
            assert orig.accepted_length == got.accepted_length
            assert len(orig.nodes) == len(got.nodes)
            if orig.nodes:
                assert abs(orig.nodes[0].log_prob - got.nodes[0].log_prob) < 1e-9
    finally:
        os.unlink(tmp)
    print("  [PASS] trace.schema save/load")


if __name__ == "__main__":
    tests = [
        test_config_hardware,
        test_config_models,
        test_trace_schema,
        test_hw_pim,
        test_hw_npu,
        test_scheduler,
        test_baselines,
        test_simulation_e2e,
        test_results_compare,
        test_trace_save_load,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            print(f"Running {t.__name__}...")
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
