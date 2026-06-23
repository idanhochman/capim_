"""
CAPIM shared cost kernel ("one device kernel, three drivers").

Structured after PAPI/AttAcc's src/devices.py:
  - typed-layer decomposition (FC, MATMUL, SOFTMAX, ACT, NORM, COMM)
  - per-layer roofline  time = max(compute, mem)  + a `bound` tag
  - nonlinear ops first-class (supplies the `t_nl` the old toy roofline lacked)
  - energy = traffic x published J/op, as a per-component vector

Differences from PAPI (deliberate, mobile-appropriate):
  - no Ramulator: PIM GEMV timing is analytical (max(compute, mem)).
  - no L1/L2 tiling optimizer: the mobile NPU has no datacenter cache hierarchy,
    so the energy vector is 4 slots [off_mem, on_chip, alu, comm] not PAPI's 6.
  - configured for LPDDR5-PIM + a mobile NPU (LP-Spec Table II), not HBM/GPU.
"""
