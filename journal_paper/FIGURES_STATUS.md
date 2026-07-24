# Figure status (paper-sources)

## Round 3 re-timing (2026-07-23): real-Schur default + fused static extraction + per-step Mop-LU + d-adaptive Krylov dim

All ρ / Var / stability-boundary **values remain bit-identical** to rounds 1–2 — the
round-3 speedups changed only CPU time and memory — so every accuracy / convergence /
stability figure is pixel-identical and was **not** regenerated. Only CPU-bearing
tables, the two work-precision figures, and the ratio text changed.

**Protocol (essential):** all CPU numbers on the dual-socket Xeon Gold 6154
(2×18 cores, 192 GB), Julia 1.12, **single-threaded solve (BLAS 1)** unless a thread
count is stated. On this NUMA machine multi-thread BLAS *slows* the memory-bound
factored solve — an 8-thread BLAS re-time gave 2.7× **worse** d=200 (224 s → 573 s);
the paper's single-threaded protocol is required for the factored solves.

**Re-timed / regenerated:**
- `tab:kron_scaling` — factored-solve column, BLAS 1: `0.16 / 1.38 / 1.60 / 5.30 /
  24.3 / 224.5` s (d = 4…200). Fused static extraction speeds the d≤8 build; the
  real-Schur driver keeps the Krylov basis `Float64` (≈½ the basis memory at large
  d) at a small time cost, so d=8 / d=200 are marginally slower than round-2 while
  memory (not shown in this time-only table) is lower — the right default for the
  large-d thesis. Dense column unchanged (assembly untouched). d=200 = 3.74 min.
- `tab:ssv_cost` — variance color map **1673 → 566 s** (6048 pts, 56 threads); the two
  MF stochastic MDBM boundaries re-timed at niter=5 (matching the committed point
  counts): `ssv_sto` **652 → 186 s** / 846 pts, `ssv_var` **946 → 264 s** / 629 pts;
  the two deterministic (LR) boundaries (`cs_det` 39 s / 4193 pts, `ssv_det` 1025 s /
  8395 pts) **unchanged** (that solver was untouched). Fig 8 image value-identical.
  NB: `benchmark/ssv2dof_chart.jl` gained a `::Float64` assertion on the scalar MDBM
  curve functions — the factored spectral-radius solver's `return_vec` keyword makes
  its return type-inferred as a `Union` (boxed scalar), which broke MDBM's
  `zero(::Any)`; the runtime value is always `Float64`, so this is a value no-op
  (proper type-stability fix tracked as a follow-up task).
- `wp_ultra` (fig + abstract + caption + body + README L93) — MF curve re-timed: at
  p=192 MF is **0.030 s / 22 MB** (was 0.045 / 39), so the explicit-product advantage
  grows to **1.3×10⁴** time / **1.8×10⁴** allocation (was 8.5×10³ / 1.0×10⁴ in the
  body, and a *stale* 1.3×10³ / 6.3×10³ in the abstract **and** caption — now all
  consistent); the step-recursion advantage at p=1024 is **≈420×** (the caption's
  round-1 "≈32×" was stale). Classical + recursion curves reused verbatim (untouched).
- `HighOrderConvergence.png` (README, error vs CPU time) — **regenerated** round-3
  (values bit-identical; GL / classical curves shift left).
- beam text (d=32, D=2.16×10⁶): **134 → 142 s** (schur's small time cost at moderate d;
  value bit-identical, ρ(H)=1.2254511).

- `grand_triple` (Fig 5, PD-Mathieu work-precision, CPU-time axis) — **regenerated
  round-3**. The IBP dev-harness (`cov_colloc_v8_ibp.jl`) deleted at packaging was
  reconstructed from git history and adapted to the current package (imports the
  internal base symbols `StepV8`/`_lagr_coefs`/`_lint`/`_G8`/`gl_tab` so it loads
  without the also-deleted `cov_colloc_v8.jl`); the full 12-method sweep (SDM q=2/4,
  GL-1..5, IBP GL-1..5) was re-run at the 60 s/solve journal cap. The SDM cost curves
  now reach p=4096 (was p≈1536 at the same cap — a ~10× left-shift); accuracy values
  bit-identical. Generators committed to `benchmark/` (grand_orders_{pub,fix}.jl,
  plot_grand_triple.jl, cov_colloc_v8_ibp.jl) for reproducibility.
- σc=1.0 (5×) SSV milling chart — added as a pure appendix figure (`app:ssv5x`,
  `ssv_chart_5x.png` already committed); main-text Fig 9 stays at the physical
  σc=0.2, with its caption now pointing to the appendix.

**Not regenerated (with reason):**
- `grand_orders_pub`, `grand_stored`, `pd_mathieu_orders`, `fig1_sykora_fig4_repro`,
  `beam_mesh_convergence`, `ssv_chart` (Fig 8), `cmp_iklodi`, `ssv_vt_orders`,
  `helical_milling_order` — value / order / resolution axes ⇒ bit-identical, pixel-
  identical. Not regenerated.
- `gpu_chain_scaling` — HW-bound: CPU side stale, needs a working GPU; the P4000 CUDA
  round-trip fails (toolkit too new for Pascal). Unchanged from round-2.
- cmp_iklodi "about a minute" text — soft order-of-magnitude claim ("within roughly an
  order of magnitude of the algebraic method"); round-3 (~2×) does not change it. Kept.

---

## Rounds 1–2 audit (earlier)

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
