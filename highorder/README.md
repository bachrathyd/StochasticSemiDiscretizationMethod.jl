# highorder/ — superconvergent second-moment engine (v7) + validation instruments

Continuation of the higher-order research archived in
`demonstration/_highorder_research/` (v6 and its history are kept intact there).
Everything here was validated on 2026-07-02; raw sweep data in `out_*.csv`.

## The result in one paragraph

`cov_colloc_v7.jl` extends the v6 window-covariance collocation engine with a
**causal intra-block two-time noise fill** and this removes v6's O(h²) order
cap on the target problem class: on the mirror stochastic Mathieu benchmark
(the problem where v6 measured slope −2 for *every* GL order and lost to SDM
q=2 by two orders of magnitude), v7 with GL2 converges at **clean O(h⁴)** and
GL3 is **converged to 10 digits by p=12**. The converged values were confirmed
by an independent fine-grid arbiter to 4×10⁻¹⁰. As a byproduct, three archived
"trusted" reference values turned out to be wrong by ~1e-5 (unconverged SDM
q=2 extrapolations); see *Corrected reference values* below.

## Files

| file | role |
|---|---|
| `cov_colloc_v7.jl` | the engine: v6 + causal fill (`offdiag=:causal`; `:none` reproduces v6 exactly) |
| `arbiter_finegrid.jl` | independent reference: method-of-steps Heun/trapezoidal-Itô on the two-time window covariance kernel, O(h²) + Richardson; shares no code/philosophy with SDM or the collocation family |
| `v7_gates.jl` | structural gates: noise-off ρ(H)=ρ(U)² (exact, 1e-16, both modes); Krylov=dense; present-noise exact value `exp((2a+α²)T)` at O(h^2S) |
| `v7_mirror_order.jl` | the decisive A/B (v6 `:none` vs v7 `:causal`) on the mirror Mathieu |
| `v7_targets.jl` | Hayes + critical-Mathieu sweeps |
| `v7_isolate.jl` | term-isolation matrix (B×β×α) that localized the remaining O(h²) |
| `arbiter_run.jl`, `arbiter_mathieu_fine.jl`, `fg_debug.jl` | arbiter validation + arbitrations |
| `v7_wp.jl` | honest work-precision benchmark vs SDM q=0/1/2 (corrected reference) |

## The causal fill (what was wrong in v6)

v6 embedded the per-step noise increment ΔB with **diagonal blocks only**; the
intra-block two-time entries E[η(u_i)η(u_j)ᵀ] (O(h) quantities) were left zero.
r steps later the delayed-drift reads touch those entries through quadratic
forms with O(h²) weights → O(h²) error per unit time → global O(h²) for every
GL order, exactly and only when B≠0. The exact fill is causal transport with
the **present-drift propagator only**:

    E[η(u_i) η(u_j)ᵀ] = Δ(u_i,u_i) · Φ_A(u_j, u_i)ᵀ ,   u_i < u_j

(delayed reads of in-step noise vanish; dW after u_i is independent). The fill
is identically zero when the noise is off, so the exact noise-off gate
ρ(H)=ρ(U)² is preserved by construction. v6's own `FILL_OFFDIAG` experiment
failed because its impulse-congruence fill was non-causal (whole-step
quadrature, including noise injected after min(u_i,u_j)).

## What is now proven (measured)

* noise-off gate: exact (≤1e-15), any B, both modes.
* present-only noise (B=0): exact value, O(h²/h⁴/h⁶) for GL1/2/3 (unchanged).
* pure delayed noise (B=0, β≠0): GL2 O(h⁴) clean, GL3 to the reference floor.
* mirror stochastic Mathieu (periodic A, B≠0, α=β≠0): GL2 O(h⁴), GL3 converged
  to 1e-10 by p=12; value confirmed by the arbiter to 4e-10. **This is the
  problem where v6 capped at O(h²) and SDM q=2 was declared two orders better —
  that verdict is now reversed.**
* second-order mechanical structure (noise in the velocity row, delay coupling
  reading position — the delayed-Mathieu/oscillator class): fully
  superconvergent, because the delayed drift integrand reads the once-integrated
  (C^{3/2}-smooth) position component.

## The remaining, understood limitation

For problems where the delayed **drift** read touches a noise-carrying (rough,
C^{1/2}) component — e.g. scalar problems with B≠0 and any noise — all GL
orders cap at a clean **O(h²)** (measured: cases a/c/d of `v7_isolate.jl`;
case b, B=0, is clean). Mechanism: the stage equations approximate
∫B(s)x(s−τ)ds by Gauss sampling of the delayed path; for a Brownian-rough
integrand the sampled quadrature has an irreducible O(h²) second-moment error.
This is NOT the v6 defect (values are correct; the constant is small) and does
not affect the mechanical application class. A genuine fix (engine v8 sketch):
augment each window block with stage-weighted history integrals
∫ℓ_j(θ)x(θ)dθ as extra DOFs, give the delayed drift term its exact integral
representation, and extend ΔB with the integral–node and integral–integral
noise covariances (computable from the same causal kernel).

## Work-precision verdict (v7_wp.jl, corrected reference, BLAS 1 thread)

| method | slope (err vs p) | best error | at CPU time |
|---|---|---|---|
| SDM q=0 | −1.00 | 1.5e-4 | 3.9 s |
| SDM q=1 | −0.96 | 2.0e-5 | 3.9 s |
| SDM q=2 | −0.97 | 2.0e-5 | 3.9 s |
| v7 GL1 | −2.00 | 6.0e-6 | 5.4 s |
| v7 GL2 | −3.84 | **1.6e-10** | 4.0 s |
| v7 GL3 | (reference floor from p≈16) | 3.5e-11 | 4.6 s |

v7 GL2 is **five orders of magnitude** more accurate than SDM q=2 at equal
CPU time on the mirror Mathieu. The archived verdict ("SDM two orders better
than v6") was an artifact of measuring against SDM's own ~1.3e-5-biased
reference — and of v6's genuine O(h²) cap, both now fixed. Note SDM q=1 and
q=2 sit on the SAME curve at measured order ≈1 here: the nominal q-ladder does
not apply to this problem class.

## v8 — integrated-history engine (matrix, time-dependent B): 8th order+

`cov_colloc_v8.jl` stores per-block **B-weighted integrated-history DOFs**
`J_i = ∫ B_read(s)·x(s) ds` (with the *reading* step's B — exact for periodic
coefficients), so the delayed drift integral is exact even for a
Brownian-rough delayed path; the node–J / J–J noise covariances come from the
matrix causal kernel. Validation (`v8m_validate/targets/highorder.jl`):

* all v7 gates preserved (noise-off exact, present-noise O(h^2S), values);
* critical stoch. Mathieu: GL3 rates 5.82/5.91/5.97/6.06 → 0.15624206;
* rough-read d=2 (delayed **velocity** feedback, periodic A): v7 caps at
  exactly 2.0; v8 GL2 clean 4.0, GL3 to the arbiter floor (4e-10) by p=12;
* **hard mirror Mathieu (mixed noise): GL4 rates 6.93/7.99/7.93 = O(h⁸);
  GL5 opens above order 10** (reference-floor limited);
* present-noise exact test: GL4 7.93 → machine ε; GL5 at machine ε from p=4.

Measured order summary (ρ(H)):

| problem class | GL2 | GL3 | GL4 | GL5 |
|---|---|---|---|---|
| smooth-read (Mathieu class, any noise mix) | 4 | 6 | **8** | ≥10 (floor) |
| rough-read (delayed drift reads a noise-carrying component) | 4 | ~4–5 | ~4 | ~4 |

The rough-read ceiling is a **fixed O(h⁴) independent of S** (GL4/5/6 all
rate ≈4.06 — the S+2 conjecture is rejected). Its source is the within-step
response covariance to rough delayed forcing (the J construction's interior
representation); lifting it needs second-level integrated DOFs — open.
Mixed αβ×B (case d) shows O(h³); the pure cross (B=0) is clean O(h^2S).

## Corrected reference values (arbiter-confirmed)

| problem | old archived "reference" | correct value |
|---|---|---|
| mirror Mathieu (A=1, ε=.5, B=.2, ζ=.1, α=β=.2, τ=P=1) | 0.7389535725 (raw SDM q2 p800) | **0.7389661254 ± 4e-10** |
| Hayes (A=−1, B=−0.4, β=0.3) | "≈0.148" | **0.1473709451 ± 2e-11** |
| critical stoch. Mathieu (A=3, ε=2, B=.5, ζ=.1, α=.1, P=4π, τ=2π) | 0.15622747 (v6 GL4 p120) / 0.15622870 (SDM-q2 "Richardson") | **0.15624206** (v7 GL3/GL4 agree to 6e-9 at p=96; arbiter h²-sequence marches to it — see out_mathieu_fine.png) |

Root cause of all three: SDM q=2 converges at *measured* order ≈1 (sometimes
0.25) on problems with delayed/multiplicative noise — far from its nominal
order — so raw fine-p values and wrong-exponent Richardson extrapolations were
systematically ~1e-5 off. Always extrapolate SDM references with the measured
order, or use the arbiter.

## How to reproduce

```
julia --project=. highorder/v7_gates.jl          # structural gates (fast)
julia --project=. highorder/v7_mirror_order.jl   # the decisive A/B
julia --project=. highorder/v7_isolate.jl        # residual isolation
julia --project=. highorder/arbiter_run.jl       # arbiter gates + arbitration
julia --project=. highorder/v7_wp.jl             # work-precision vs SDM
```
