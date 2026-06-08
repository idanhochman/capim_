"""
Hardware constants for CAPIM simulation.

Primary source: LP-Spec Table II (bandwidth, compute, capacity).
Energy values are derived from ratios stated in LP-Spec and anchored to
SpecPIM's measured figures, as follows:

  Off-chip transfer energy (5.4 pJ/bit):
    SpecPIM (ASPLOS 2024) Table 2 states 5.47 pJ/bit for external HBM
    bandwidth. Used as the off-chip anchor for LPDDR5-PIM.

  Internal DRAM access energy (0.8 pJ/bit):
    LP-Spec Section II-A states "data transfers within DRAM consume only
    15% of the energy required for off-DRAM transfers [23]."
    Derived: 0.15 × 5.4 pJ/bit = 0.81 pJ/bit → rounded to 0.8 pJ/bit.

  NPU INT8 MAC energy (0.5 pJ/op):
    LP-Spec Section IV-B states "an INT8 MAC unit ... consumes 63.6% of
    the energy of a FP16 MAC in 20 nm DRAM process [32]."
    [32] = Samsung ISCA 2021 (Lee et al., "Hardware Architecture and
    Software Stack for PIM Based on Commercial DRAM Technology").
    Table I of that paper gives values normalised to INT16 MAC (=1.0):
    INT8(32-bit Acc.)=0.77, FP16=1.21 → ratio 0.77/1.21 = 63.6%. VERIFIED.
    0.5 pJ/op is an approximation consistent with LP-Spec's methodology.

# TODO (energy verification):
#   VERIFIED:
#     - 5.47 pJ/bit off-chip: SpecPIM (ASPLOS 2024) Table 2, explicit value.
#     - 15% internal/off-chip ratio: LP-Spec Sec. II-A [23], explicit quote.
#     - 63.6% INT8/FP16 ratio: Samsung ISCA 2021 Table I (normalised to INT16).
#
#   NOT VERIFIED (approximations):
#     - 5.4 vs 5.47 pJ/bit: SpecPIM value is for HBM, not LPDDR5. No LPDDR5
#       specific off-chip energy figure found in any cited paper.
#     - 0.5 pJ/op INT8 MAC: requires absolute INT16 MAC energy at 20 nm as
#       baseline. Samsung ISCA 2021 Table I gives only relative (normalised)
#       values; absolute baseline not stated in any cited paper.
#
#   TODO:
#     - Verify 5.4 pJ/bit against an LPDDR5-specific source (not HBM).
#     - Find absolute INT16 MAC energy at 20 nm (Horowitz 2014 ISSCC is the
#       standard reference; check Table 1 of that paper).
#     - If absolute INT16 value found, recompute: INT8 = 0.77 × INT16 pJ/op.
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
# Energy constants — see module docstring for full derivation
# ---------------------------------------------------------------------------

# Internal DRAM access energy (within PIM banks)
# Derived: 15% x 5.4 pJ/bit (LP-Spec Sec. II-A [23])
PIM_ENERGY_PJ_PER_BIT: float = 0.8     # pJ/bit

# Off-chip transfer energy (PIM -> NPU over external I/O)
# Source: SpecPIM Table 2 (5.47 pJ/bit for external HBM bandwidth)
OFFCHIP_ENERGY_PJ_PER_BIT: float = 5.4 # pJ/bit

# INT8 MAC energy on the NPU
# 63.6% of FP16 MAC energy (Samsung ISCA 2021 Table I, LP-Spec Sec. IV-B [32])
# Absolute baseline unverified — 0.5 pJ/op is an approximation (see docstring)
NPU_ENERGY_PJ_PER_INT8_OP: float = 0.5 # pJ/op

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
