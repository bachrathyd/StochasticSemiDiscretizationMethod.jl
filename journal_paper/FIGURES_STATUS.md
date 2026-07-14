# Figure status (paper-sources)

Audit of the figures included by `main_MFSSD.tex`, after the two rounds of solver
speedups on the `package-cleanup` branch. **The speedups changed only wall-clock
timings — every ρ / Var / stability-boundary value is bit-identical**, so the
accuracy/convergence figures are unaffected.

| Figure | kind | status |
|---|---|---|
| `fig1_sykora_fig4_repro` | accuracy (Fig. 1 reproduction) | current — values unchanged |
| `pd_mathieu_orders` | convergence orders | current — values unchanged |
| `grand_orders_pub`, `grand_stored`, `grand_triple` | grand convergence diagram | current — values unchanged |
| `beam_mesh_convergence` | modal convergence | current — values unchanged |
| `ssv_chart` | 2-DOF SSV milling chart | current — values unchanged |
| `wp_ultra` | work-precision (timing) | **UPDATED** to the round-2 code (the multiplication-free curve is now ~10–22× faster from the StaticArrays small-`d` fast path + eager Krylov; the explicit-product and step-recursion curves are unchanged, so the MF advantage grows) |
| `gpu_chain_scaling` | CPU-vs-GPU scaling (timing) | **STALE — needs regenerating on a working GPU.** Its data CSV predates the speedups, so the CPU side is the old (slow) timing; the P4000 CUDA round-trip fails on the dev machine (toolkit too new for Pascal). Regenerate `benchmark_cpu_gpu.jl` / `benchmark_chain_dof.jl` on a working GPU, then `plot_chain_gpu_pub.jl`. |

**Action for the camera-ready:** re-check any `wp_ultra`-related sentence on
Overleaf against the refreshed figure (the qualitative claim — MF cost grows far
slower than the explicit product, and the gap widens with `p` — still holds and is
now stronger). Regenerate `gpu_chain_scaling` once a GPU is available.
