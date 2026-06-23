# Archived — superseded simulation (do not use)

These files are the **old "toy roofline" simulator**, replaced 2026-06-20 by the
PAPI-structured cost kernel ("one device kernel, three drivers"). They are kept
only for reference and are **not** on any live import path; their internal imports
(e.g. `results.py` → `sim.simulation`, `hw/` cost models) no longer resolve from
`sim/` and will break if run. Safe to hard-delete once the rewrite is fully trusted.

Replaced by:

| Archived (old) | Live replacement |
|---|---|
| `hw/{npu,pim}.py` | `sim/kernel/{device,layer,npu,pim}.py` |
| `baselines/{autoregressive,lp_spec}.py` | `sim/drivers/{autoregressive,lp_spec}.py` |
| `simulation.py` | `sim/drivers/` + `sim/scripts/run_simulation.py` |
| `results.py` | `sim/report.py` + `sim/sweeps.py` |
| `scripts/run_simulation_legacy.py.bak`, `scripts/toy_trace.py` | `sim/scripts/run_simulation.py` |
| `tests/test_smoke.py` | `sim/tests/test_kernel_v2.py` |

Run the live simulation with:

    python3 -m sim.scripts.run_simulation --trace traces/llama2_alpaca.json
