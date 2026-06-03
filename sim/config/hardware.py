"""
Hardware constants for CAPIM simulation.

All values sourced from LP-Spec Table II and cited works:
  [24] AttAcc (PIM energy figures)
  [26] McDRAM (DRAM internal energy)
  LP-Spec Table II (bandwidth, compute, capacity)
"""

# ---------------------------------------------------------------------------
# LPDDR5-PIM (4-die configuration, Samsung LPDDR5-PIM)
# Source: LP-Spec Table II
# ---------------------------------------------------------------------------

# Compute throughput: 4 dies × 102.4 GOPS each = 409.6 GOPS INT8
PIM_INT8_GOPS: float = 409.6e9          # ops/s INT8

# Internal bandwidth: total across all banks (4 dies × 12.8 TB/s per die?)
# LP-Spec Table II states 51.2 TB/s internal bank bandwidth for 4-die config.
PIM_INTERNAL_BW: float = 51.2e12        # bytes/s (51.2 TB/s)

# External I/O bandwidth (off-chip, shared with NPU)
# LP-Spec Table II: 51.2 GB/s external I/O
PIM_EXTERNAL_BW: float = 51.2e9         # bytes/s (51.2 GB/s)

# Operating frequency
PIM_FREQ_HZ: float = 200e6              # Hz

# Capacity: 3 PIM ranks + 1 DRAM rank × 4 GB each = 16 GB
PIM_CAPACITY_BYTES: float = 16e9        # bytes

# Number of ALU units (NALU) per die; used for batch rounding in verification
PIM_NALU: int = 1                        # treat as 1 for GEMV baseline; refine later

# ---------------------------------------------------------------------------
# Energy constants
# Source: AttAcc [24] and McDRAM [26] as cited by LP-Spec
# ---------------------------------------------------------------------------

# Internal DRAM access energy (within PIM banks)
PIM_ENERGY_PJ_PER_BIT: float = 0.8     # pJ/bit   (McDRAM [26])

# Off-chip transfer energy (PIM → NPU over external I/O)
OFFCHIP_ENERGY_PJ_PER_BIT: float = 5.4 # pJ/bit   (AttAcc [24])

# INT8 MAC energy on the NPU
NPU_ENERGY_PJ_PER_INT8_OP: float = 0.5 # pJ/op    (AttAcc [24])

# ---------------------------------------------------------------------------
# Mobile NPU
# Source: LP-Spec Table II
# ---------------------------------------------------------------------------

# Matrix unit peak throughput
NPU_INT8_TOPS: float = 32.8e12          # ops/s INT8 (32.8 TOPS)

# Vector unit throughput
NPU_VECTOR_TOPS: float = 8.2e12         # ops/s

# Clock frequency
NPU_FREQ_HZ: float = 1e9               # Hz (1 GHz)

# Number of NPU compute cores
NPU_CORES: int = 16

# Local (per-core) scratchpad buffer
NPU_LOCAL_BUFFER_BYTES: float = 256e3  # bytes (256 KB per core)

# Total NPU on-chip scratchpad
NPU_SCRATCHPAD_BYTES: float = 8e6      # bytes (8 MB)

# Off-chip bandwidth (shared channel with PIM external I/O)
NPU_OFFCHIP_BW: float = 51.2e9         # bytes/s (51.2 GB/s)

# ---------------------------------------------------------------------------
# Derived / convenience
# ---------------------------------------------------------------------------

def pj_to_j(pj: float) -> float:
    """Convert picojoules to joules."""
    return pj * 1e-12


def bits_to_bytes(bits: int) -> float:
    return bits / 8
