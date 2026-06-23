"""
Hardware constants for CAPIM simulation.

Primary source: LP-Spec Table II (bandwidth, compute, capacity).

Energy model = 2 data-movement constants (pJ/bit, named by INTERFACE) + 2 compute
constants (pJ/MAC, named by DEVICE/process). The four feed the [off_mem, on_chip,
alu, comm] vector everywhere:
    off_mem <- MEM_INTERNAL (PIM)   or MEM_OFFCHIP (NPU)
    alu     <- PIM_MAC_PJ_PER_OP    or NPU_MAC_PJ_PER_OP
    comm    <- MEM_OFFCHIP (the PIM<->NPU crossing rides the same external bus)
    on_chip <- 0 (no mobile cache hierarchy modelled)
Sourcing for each, below:

  MEM_OFFCHIP -- off-chip transfer energy (5.47 pJ/bit):
    = 5.47 pJ/bit, HBM2 memory-access energy. ORIGIN: TPU-v4i (Jouppi et al.),
    quoted in SpecPIM (ASPLOS 2024) §7.1 ("energy of HBM2 and GDDR6 are 5.47pJ/b
    and 6.48pJ/b ... according to TPU-v4i [25]") -- NOT SpecPIM's own Table 2
    (that table is the HBM2-vs-GDDR6 design space, not the energy value).
    We use it because LP-Spec itself draws its energy "from prior works
    [24],[26],[29]" (LP-Spec §VI) and [29] IS SpecPIM -- so 5.4 keeps us on the
    SAME energy basis as the baseline we validate against. It is an HBM2 stand-in:
    LP-Spec publishes no absolute LPDDR5 off-chip figure. We use the exact
    cited 5.47 (not a rounded 5.4) for source fidelity.
    SENSITIVITY BOUND (not adopted): real mobile LPDDR5 off-chip I/O is much
    higher -- 10 pJ/bit (LPDDR5x/5T, Snapdragon 8 Gen 3 / Dimensity 9300) to
    20 pJ/bit (plain LPDDR5, A17 Pro), per PIM-AI (arXiv:2411.17309) Table 1
    (round-trip: SoC PHY + memory-side PHY). HBM2 is genuinely 2-4x more
    efficient per bit than LPDDR5, so 5.4 is OPTIMISTIC for off-chip/comm.
    Carry 10 and 20 pJ/bit as a sweep to show the AR/LP-Spec/CAPIM comparison
    is invariant (all three share this base); see AUDIT.

  MEM_INTERNAL -- internal near-bank DRAM access energy (0.8 pJ/bit):
    LP-Spec Section II-A states "data transfers within DRAM consume only
    15% of the energy required for off-DRAM transfers [23]" ([23] = Samsung
    Hot Chips 35, 2023). Derived: 0.15 × 5.47 = 0.82 → rounded to 0.8 pJ/bit.
    CORROBORATED: PIM-AI Table 1 independently lists 0.95 pJ/bit for internal
    near-bank DRAM access -- same ballpark, different (LPDDR5) source.

  INT8 MAC energy (0.23 pJ/op for BOTH NPU and PIM):
    Anchored to Horowitz (ISSCC 2014), the per-op energy table at 45 nm: 8-bit
    integer MULT 0.2 pJ + 8-bit integer ADD 0.03 pJ = 0.23 pJ/MAC. The one real,
    citable absolute we have for an INT8 MAC.

    Kept as TWO constants (NPU_MAC, PIM_MAC) -- same nominal value, different
    devices -- so the PIM side can be swept independently (below). We do NOT bake
    in a PIM/NPU difference, because:
      - Horowitz is a LOGIC-process number. Our NPU (4 nm logic) sits below 45 nm,
        so 0.23 is a conservative upper bound there. Same order as PAPI's alu =
        0.32 -- but note PAPI's value is UNCITED in its config (the only energy
        coefficient there with no source), so that is a sanity check, not support.
      - The PIM ALU is a 20 nm DRAM-process device. DRAM-process logic is
        DIRECTIONALLY costlier (LP-Spec: DRAM process ~10x less dense than logic;
        Samsung ISCA 2021 [32] gives INT8 = 63.6% of FP16 in DRAM vs Horowitz's
        ~15% in logic -- the energetics genuinely differ). But the MAGNITUDE is
        UNPUBLISHED (Samsung Table I is normalised to INT16=1.0, no absolute; and
        LP-Spec uses 63.6% only to justify its ALU design choice, not as an energy
        input -- its actual energy is borrowed "from [24],[26],[29]", §VI).
    So PIM_MAC = 0.23 is a logic-process FLOOR. The known-direction/unknown-
    magnitude DRAM-process upside is handled by a SWEEP on PIM_MAC, not an
    invented point penalty. Both MACs are second-order anyway (alu ~1% NPU /
    ~7% PIM of layer energy at batch=1 GEMV; movement dominates).

# TODO (energy verification):
#   VERIFIED (provenance traced to source, 2026-06-21):
#     - MEM_OFFCHIP 5.47 pJ/bit: HBM2, from TPU-v4i, quoted SpecPIM §7.1 (NOT its
#       Table 2). LP-Spec-aligned: LP-Spec §VI energy is "from [24],[26],[29]",
#       [29]=SpecPIM. So it matches the baseline's own basis.
#     - MEM_INTERNAL 15% internal/off-chip ratio: LP-Spec Sec. II-A, quoting
#       [23] = Samsung Hot Chips 35 (2023). Corroborated: PIM-AI Table 1 lists
#       0.95 pJ/bit internal (LPDDR5) -- same ballpark as our 0.8.
#     - NPU_MAC = PIM_MAC = 0.23 pJ/op: Horowitz ISSCC 2014 (8-bit INT mult 0.2 +
#       add 0.03) @ 45 nm logic -- the only citable absolute INT8 MAC.
#
#   KNOWN LIMITATION (documented, not a bug):
#     - MEM_OFFCHIP is an HBM2 stand-in; real LPDDR5 off-chip is 10-20 pJ/bit
#       (PIM-AI Table 1, round-trip). Kept at 5.47 for LP-Spec comparability;
#       carry 10/20 as a sensitivity sweep. The whole subfield (incl. LP-Spec)
#       uses HBM/GDDR energy for LPDDR5 designs -- state this, don't hide it.
#     - PIM_MAC = 0.23 is a logic-process FLOOR. DRAM-process logic is directionally
#       costlier (Samsung INT8/FP16 = 63.6% in DRAM vs ~15% in Horowitz logic) but
#       the absolute is UNPUBLISHED -> the upside is a SWEEP, not a baked-in penalty.
#     - NPU_MAC 0.23 is a 45 nm UPPER BOUND for a 4 nm part (true value lower).
#     - Both MACs are 2nd-order (alu ~1% NPU / ~7% PIM of layer energy at
#       batch=1 GEMV; movement dominates) -- precision is not load-bearing.
#
#   TODO:
#     - Mine the LPDDR5-PIM primaries for a citable LPDDR5 off-chip pJ/bit:
#       Aquabolt-XL (IEEE Micro 2022, [20]) and Samsung Hot Chips 35 (2023, [23]).
#       PIM-AI's 10/20 are uncited ("derived from the memory technologies used").
#     - If the PIM_MAC sweep shows sensitivity: find an absolute INT16/FP16 MAC
#       energy at 20 nm DRAM (then PIM_MAC = 0.636 x FP16_abs) to replace the floor.
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
# 2 movement (pJ/bit, by interface) + 2 compute (pJ/MAC, by device/process)
# ---------------------------------------------------------------------------

# --- data movement: energy per BIT, named by interface ---

# Internal near-bank movement, within the DRAM die (PIM path only).
# Derived: 15% x 5.47 (LP-Spec Sec. II-A [23]); corroborated by PIM-AI's 0.95 pJ/bit
MEM_INTERNAL_PJ_PER_BIT: float = 0.8    # pJ/bit

# Off-chip movement over the external LPDDR5 bus: NPU HOST-mode weight reads AND
# the PIM<->NPU crossing (comm) both ride this interface.
# 5.47 HBM2 (TPU-v4i via SpecPIM §7.1); LP-Spec-aligned ([29]=SpecPIM). HBM2
# stand-in -- real LPDDR5 is 10-20 pJ/bit (PIM-AI); sweep those. See docstring.
MEM_OFFCHIP_PJ_PER_BIT: float = 5.47    # pJ/bit

# --- compute: energy per INT8 MAC, named by device/process (SPLIT; see docstring) ---

# Horowitz ISSCC 2014: 8-bit INT mult 0.2 + add 0.03 = 0.23 pJ/MAC @ 45 nm logic.
# PIM near-bank ALU, 20 nm DRAM process. 0.23 is a logic-process FLOOR; DRAM
# process is directionally costlier but unpublished -> sweep PIM_MAC for the upside.
PIM_MAC_PJ_PER_OP: float = 0.23         # pJ/op

# NPU matrix/vector unit, 4 nm logic. 0.23 is a conservative upper bound (4 nm <
# 45 nm); same order as PAPI's alu=0.32 (which is uncited -- sanity check only).
NPU_MAC_PJ_PER_OP: float = 0.23         # pJ/op

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
