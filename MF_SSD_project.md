# MF-SSDM GPU Acceleration — Project Summary

## 1. Goal

Accelerate the **Multiplication-Free Stochastic Semi-Discretization Method (MF-SSDM)**
by moving the second-moment mapping to the GPU under a **"Zero-Sync"** policy:

1. **Upload once** — all per-step deterministic and stochastic coefficient tensors go to
   the GPU up front (`extract_gpu_coeffs`).
2. **Execute completely** — each application of the p-step mapping is a single
   cooperative kernel launch (grid-wide sync between steps), or a single CUDA-graph
   replay on devices without cooperative-launch support.
3. **Iterate internally** — `KrylovKit.eigsolve`/`linsolve` run with device-resident
   vectors; the Krylov basis never leaves the GPU.
4. **Download once** — only the Floquet multiplier ρ(H) (or the stationary moment
   vector) returns to the host.

## 2. Status: COMPLETE (2026-07-02)

All phases done: theory, zero-sync kernels, Krylov integration, validation,
benchmarking, tests, and API consolidation.

### Public API (`src/functions_gpu.jl`)

| function | purpose |
|---|---|
| `spectralRadiusOfMapping_GPU(dm)`  | ρ(H) fully on GPU (cooperative kernel; CUDA-graph fallback) |
| `fixPointOfMapping_GPU(dm)`        | stationary 1st+2nd moment via GPU `linsolve` |
| `spectralRadiusOfMapping_auto(dm)` | CPU below the measured crossover (D < 10⁴), GPU above |

Development variants v1/v2 (single-block) were removed after being superseded;
the internal kernels that remain are:
* `kernel_M2_MF!` / `kernel_M1_MF!` — additive-capable single-block kernels, used to
  evaluate the affine constant in the fixpoint solve;
* `kernel_M2_MF_v3!` — the cooperative multi-block kernel behind the public API;
* `kernel_M2_MF_v4_*!` + CUDA-graph wrapper — non-cooperative fallback path.

### Validation (Quadro P4000, CUDA 12.6)

* Hayes (d=1) and stochastic delayed Mathieu (d=2, periodic coefficients,
  multiplicative + delayed noise): GPU ρ(H) matches the CPU MF reference to
  ~1e-15 relative error on both the cooperative and the graph path.
* GPU stationary second moment matches `fixPointOfMapping_MF` to 1e-15.
* Guarded test set in `test/runtests.jl` (skips cleanly without a GPU).

### Transfer audit (CUDA profiler, p=40)

Per full eigsolve: coefficients upload once (~0.14 ms of HtoD), then only
KrylovKit's convergence scalars move (447 DtoH copies ≈ 0.5 ms total);
61% of device time is inside the single mapping kernel. The Zero-Sync goal is met:
bulk data never crosses the PCIe bus during iteration.

### Benchmark (stoch. Mathieu d=2, q=2 — `benchmark/benchmark_cpu_gpu.jl`)

* GPU overtakes the CPU MF path at **D ≈ 10⁴** (p ≈ 140).
* Speedup grows with size: 2.3× at p = 640 (D ≈ 2·10⁵), **3.5× at p = 1280
  (D ≈ 8·10⁵)** on a Quadro P4000 (a modest 2017 workstation GPU — newer cards
  will shift the crossover down and the asymptotic speedup up).

| p | D | CPU [s] | GPU [s] | speedup |
|---|---|---|---|---|
| 150 | 11 935 | 0.92 | 0.35 | 2.6× |
| 320 | 52 650 | 1.23 | 0.66 | 1.9× |
| 640 | 207 690 | 2.99 | 1.30 | 2.3× |
| 900 | 409 060 | 4.55 | 1.62 | 2.8× |
| 1280 | 824 970 | 8.06 | 2.33 | 3.5× |

* `spectralRadiusOfMapping_auto` uses the measured crossover (`cpu_threshold=10_000`).
* Results: `benchmark/cpu_vs_gpu.csv`, `benchmark/cpu_vs_gpu.png`.

### Perf notes (measured, do not regress)

* `krylovdim` defaults to KrylovKit's 30, capped only by free GPU memory.
  An earlier `D÷500` heuristic forced tiny subspaces → restart storms → ~2× slower.
* The cooperative (single-launch) path beats the CUDA-graph path by ~5–25%;
  the graph path exists for devices/drivers without cooperative launch.
* The fixpoint linsolve iterates on the fast homogeneous operator; only the two
  affine-constant evaluations use the slower additive-capable kernel.

## 3. Possible future work

* Shared-memory tiling of the small per-step coefficient matrices inside
  `kernel_M2_MF_v3!` (T2.2) — relevant only for larger d.
* Multi-GPU / batched parameter sweeps (stability maps launch many independent
  eigsolves — embarrassingly parallel across devices).

## 4. Key files

* GPU implementation: `src/functions_gpu.jl`
* CPU reference: `src/functions_multifree.jl`
* Benchmark: `benchmark/benchmark_cpu_gpu.jl` (+ CPU complexity: `benchmark/benchmark_mf_complexity.jl`)
* Tests: `test/runtests.jl`
