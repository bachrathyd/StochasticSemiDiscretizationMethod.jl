# Project Progress - GPU MF-SSDM

## Status Overview
- **Phase 1: Theoretical Formalization & CPU Prototyping** - ✅ COMPLETE
- **Phase 2: Full-GPU Infrastructure & Zero-Sync Kernels** - ✅ COMPLETE
- **Phase 3: Krylov Integration** - ✅ COMPLETE
- **Phase 4: Validation & Benchmarking** - 🟡 IN PROGRESS

---

## Detailed Task Tracking

| Task ID | Description | Status | CPU Consistency | Notes |
| :--- | :--- | :--- | :--- | :--- |
| T1.1 | Formalize $\Phi_L$ and $\Phi_R$ operators | ✅ | YES | Explicit congruence formulation |
| T1.2 | Implement CPU prototype | ✅ | YES | Using full matrix |
| T1.3 | Verify CPU Operators | ✅ | YES | Error = 0.00e+00 |
| T2.1 | Refactor to Device-Side $p$-loop | ✅ | YES | Fused into `kernel_M2_MF!` in `src/functions_gpu.jl` |
| T2.2 | Implement Shared Memory Buffering | ⚪ | - | Speed up the $d \times d$ matrix multiplications inside the kernel |
| T3.1 | Integrate Zero-Sync Kernel with KrylovKit | ✅ | YES | Using `KrylovKit.eigsolve` and `KrylovKit.linsolve` |
| T4.1 | Clean Benchmark Execution | 🟡 | - | Warm-up runs enforced. Benchmarking $p=10^4$ scaling. |

---

## Current Task
> **Executing Phase 4: Validation & Benchmarking.**

## Obstacles & Findings
- **Major Bottleneck Resolved:** Transitioned from $3 \times p$ kernel launches per iteration to exactly **1** kernel launch for the entire $p$-step mapping.
- **Resource Constraints:** The Zero-Sync kernel is resource-intensive. Reduced threads per block to 256 to avoid "too many resources requested" errors on some GPUs.
- **Scalar Indexing:** Identified and fixed scalar indexing issues by moving from `IterativeSolvers.gmres` to `KrylovKit.linsolve`, which is fully GPU-compatible.
- **Initialization/Extraction Bug:** Fixed a mapping bug where logical step 0 was incorrectly mapped to physical index `rp1` instead of `1` during initialization.

## Next Steps
1. Perform high-resolution benchmarks comparing CPU MF-SSDM vs GPU Zero-Sync.
2. Investigate Shared Memory Buffering (T2.2) if performance bottleneck shifts to memory latency.
