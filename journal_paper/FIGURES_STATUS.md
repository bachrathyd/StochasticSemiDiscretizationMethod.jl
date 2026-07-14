# Figure status (paper-sources)

Audit of the figures included by `main_MFSSD.tex`, after the two rounds of solver
speedups on the `package-cleanup` branch. **The speedups changed only wall-clock
timings — every ρ / Var / stability-boundary value is bit-identical**, so the
accuracy/convergence figures are unaffected.

| Figure | kind | status |
|---|---|---|
| `fig1_sykora_fig4_repro` | accuracy (Fig. 1 reproduction) | current — values unchanged |
| `pd_mathieu_orders` | convergence orders | current — values unchanged |
| `grand_orders_pub`, `grand_stored`, `grand_triple` | grand convergence diagram | **RE-TIMED** — regenerated with clean BenchmarkTools minima and a ~1 s CPU cap per method (`grand_triple`'s CPU-time axis now reflects the round-2 solver; SDM q=2/q=4 curves reach higher `p`). Every accuracy value is bit-identical, so the measured slopes (0.99 for SDM at any `q`; 2/4/6/… for GL-`S`) are unchanged. |
| `beam_mesh_convergence` | modal convergence | current — values unchanged |
| `ssv_chart` | 2-DOF SSV milling chart | current — values unchanged |
| `wp_ultra` | work-precision (timing) | **UPDATED** to the round-2 code (the multiplication-free curve is now ~10–22× faster from the StaticArrays small-`d` fast path + eager Krylov; the explicit-product and step-recursion curves are unchanged, so the MF advantage grows) |
| `gpu_chain_scaling` | CPU-vs-GPU scaling (timing) | **STALE — needs regenerating on a working GPU.** Its data CSV predates the speedups, so the CPU side is the old (slow) timing; the P4000 CUDA round-trip fails on the dev machine (toolkit too new for Pascal). Regenerate `benchmark_cpu_gpu.jl` / `benchmark_chain_dof.jl` on a working GPU, then `plot_chain_gpu_pub.jl`. |

**Text / tables updated to match (round-2 re-timing pass):**
- `wp_ultra` caption CPU numbers refreshed to the round-2 MF timings: at `p=192`
  MF is now `0.045 s / 39 MB` (was `0.30 s / 61 MB`), so the explicit-product
  advantage grows to `8.5×10³` in time and `1.0×10⁴` in allocation (was `1.3×10³`
  / `6.3×10³`); at `p=1024` the step-recursion is `507 s` vs `1.3 s` for MF.
- `tab:kron_scaling` (factored-vs-dense chain) re-timed from `factored_vs_dense.csv`:
  factored solves 1.42→0.19, 3.34→1.13, 2.14→1.78, 8.48→6.43, 37.1→27.3, 218.6→208.7 s.
- `tab:ssv_cost` MF layers (variance color map + the two MF boundary curves) re-timed
  on 56 threads with the round-2 solver; deterministic SD (LR) layers are unchanged
  (that solver was not touched by round-2; `cs_det` reproduces to 4193 pts / 36 s).
- Benchmark-environment note changed to Julia 1.12.6 (the machine of this re-timing pass).

**Action for the camera-ready:** the qualitative `wp_ultra` claim — MF cost grows far
slower than the explicit product, and the gap widens with `p` — still holds and is
now stronger. Regenerate `gpu_chain_scaling` once a GPU is available.
