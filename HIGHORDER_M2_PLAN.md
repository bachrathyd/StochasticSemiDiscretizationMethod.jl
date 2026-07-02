# High-order stochastic second-moment IRK inside StochasticSemiDiscretizationMethod.jl

> **Note (2026-07-02, morning):** the scratch scripts referenced below (`scratch_*.jl`,
> `moment_colloc.jl`, `benchmark_mf_v6.jl`, `cov_colloc_v6.jl`, …) were archived to
> `demonstration/_highorder_research/`. Status: the moment-equation collocation shows
> the predicted superconvergent *order*, but the reduced (M,P) prototype converges to a
> **wrong value** (fails the noise-off ρ(Φ)² gate) — see `demonstration/README.md`.
>
> **RESOLVED (2026-07-02, afternoon) — see `highorder/README.md`.** The v6 O(h²) cap
> was the missing intra-block two-time noise covariance; the causal fill in
> `highorder/cov_colloc_v7.jl` restores full superconvergence on the target
> (mechanical/Mathieu) class — GL2 O(h⁴), GL3 converged to 1e-10 by p=12 on the very
> benchmark where v6 lost to SDM. Values independently confirmed by a fine-grid
> arbiter; three archived reference values were themselves wrong by ~1e-5 (unconverged
> SDM q=2 — its measured order is ≈1, not its nominal). Remaining known limit: O(h²)
> when the delayed drift reads a Brownian-rough component (scalar B≠0+noise); v8
> sketch (stage-weighted history integrals) documented in highorder/README.md.

## Context / problem

The IRK collocation (GL(S), deterministic order 2S) DDE solver in the MFCM package
(`Integration_based_stab_general_order`) is fast and high-order **deterministically**,
but its hand-rolled stochastic second-moment map collapses to **O(h¹)** for ρ(H) at
every order GL1–GL5 — useless, since the whole point of IRK is superconvergence.

Root cause (confirmed by exploration): the MFCM `_build_block_transitions`
(`src/stoch_irk_map.jl`) flattens the collocation step into an **explicit** per-block
transition list, losing the **implicit intra-step stage coupling** that
`build_explicit_matrices` (the trusted `L·x = R·x_prev` form behind `MonodromyMap`)
keeps. The deterministic core (`build_system_matrices` → `M_prop`, `M_del`) is verified
correct; only the explicit-flattening for the stochastic map is wrong.

Meanwhile SSDM's second-moment machinery (`M2_Mapping_from_Sparse`,
`DiscreteMapping_M2`, `spectralRadiusOfMapping_MF`) is **order-preserving** — verified
O(h³) when fed SDM order-2 matrices. So: feed it high-order step matrices and we get
high-order ρ(H).

## The bridge

SSDM's M2 path needs **per-step `n×n` deterministic `F̂ₙ`** so that the monodromy is the
product `∏ F̂ₙ` and `H = ∏ (F̂ₙ⊗F̂ₙ + Λₙ)`. The trusted MFCM stepping is implicit
(`L⁻¹R`), and its augmented block (BSIZE=(S+1)·D, endpoint+stages) carries internal
stage DOFs that SSDM's plain (r+1)·d state does not.

### Decision: build augmented per-step F̂ in the SSDM shift convention

Each MFCM step n maps an augmented history window to the next augmented endpoint+stages
block. We build, per step, a single sparse `N×N` matrix `F̂ₙ` (N = (R+1)·BSIZE, the
augmented window) in SSDM's shift form:

```
F̂ₙ = spdiagm(-BSIZE => ones(N-BSIZE))            # history shift by one augmented block
      + (rows 1..BSIZE) ← M_prop[n], M_del[k][n][j] contributions  # the new block
```

The new-block rows are filled exactly as in the **collocation branch of
`build_explicit_matrices`** (the trusted routing): M_prop into the previous endpoint
x-block; each delay into block `m_idx` (x-part, w[1]) and block `m_idx+1`
(stages w[2..S+1] + endpoint w[S+2]). Crucially the stage DOFs of the *current* block
are solved within the same block (the augmented state carries them), so the implicit
coupling is preserved — no O(h) collapse.

ρ(monodromy) of `∏ reverse(F̂ₙ)` must converge at order 2S deterministically. This is
the concept gate.

## Phase 0 — concept test (BEFORE touching the package)

`concept_highorder_m2.jl` (scratch, repo root):
- Scalar test DDE `x'(t) = a x(t) + b x(t-τ)` (D=1), known exact char. roots.
- Build augmented `F̂ₙ` per step from a GL(S) tableau (reuse the math from MFCM
  `build_system_matrices`: `M = I - h·aᵢⱼ·A`, `Minv`, `M_prop`, `M_del`, collocation
  weights `ce(θ)`), assembled in SSDM shift form above.
- monodromy `P = prod(reverse(F̂ₙ))`, `ρ = max|eig(P)|`.
- Print convergence rate of `|ρ − ρ_exact|` vs h for GL1, GL2, GL3.
- **Gate: GL2 ≥ O(h⁴), GL3 ≥ O(h⁶) (deterministic).** If it collapses to O(h), the
  shift-form embedding of the augmented block is wrong — fix before proceeding.

## Phase 1 — deterministic high-order F̂ builder in the package

New `src/functions_irk_discretization.jl`:
- `GLTableau(S)` — Gauss–Legendre nodes/weights/`a`,`b`,`c` + collocation interp weights
  `ce(θ)` (Lagrange on {0,c₁..c_S,1}). (Mirror MFCM `tableau_library.jl` / `ce`.)
- `irk_step_matrices(lddep, ts, S, r)` → per-step augmented `M_prop`, `M_del`,
  `delay_indices`, `delay_weights` (mirror MFCM `build_system_matrices`, utils.jl:43–120).
- `irk_detMX(...)` → per-step sparse augmented `F̂ₙ` (N×N) in SSDM shift form.

## Phase 2 — exact Itô-isometry stochastic block (the part MFCM got wrong)

New `src/functions_irk_m2.jl`:
- For each step build augmented stochastic `Ĝₙ` (entries `SVector{K,Float64}` Itô-kernel
  samples) so SSDM's `M2_Mapping_from_Sparse` applies the exact isometry — REUSE
  SSDM's verified `itoisometrymethod`/`Trapezoidal(K)` (`functions_stoch_utilities.jl`)
  rather than MFCM's `GaussKernel`.
- Multiplicative α (present) and β (delay) noise routed through the same augmented
  block structure as the deterministic delays.
- Assemble `stDiscreteMapping(ts, detMXs, detVs, stMXs, stVs)` and feed to the EXISTING
  `DiscreteMapping_M2(stdm, rst)` (functions_discretization.jl:85). Need a `Result`
  carrying the right `n`(=N), `n_steps`, `itoisometrymethod`.

## Phase 3 — user API + verification

- `calculateResults_IRK(lddep, GL(S), DiscretizationLength; n_steps)` → `Result`-like,
  reusing `spectralRadiusOfMapping_MF` / `DiscreteMapping_M2`.
- Verify: scalar SDDE with known second-moment exponent → GL2 gives O(h⁴) for ρ(H).
- Cross-check vs in-package SDM order-2 on the stochastic Mathieu point
  (A=3,B=0.5,α=0.1; ρ_ref≈0.1562208339).
- Redo the work-precision diagram showing IRK superconvergence vs SDM order-2.

## Critical files
- SSDM M2 contract: `src/structures_result.jl` (stDiscreteMapping 54–60),
  `src/functions_discretization.jl` (DiscreteMapping_M2 85–97; ρ 144),
  `src/functions_stoch_utilities.jl` (M2_Mapping_from_Sparse 120–141, CovVecIdx 86–101),
  `src/functions_multifree.jl` (DiscreteMapping_M2_MF 165, spectralRadiusOfMapping_MF 349).
- MFCM trusted math (REFERENCE ONLY, do not import):
  `…/Integration_based_stab_general_order/src/utils.jl:43–120` (build_system_matrices),
  `…/src/sparse_builder.jl:1–85` (build_explicit_matrices — the CORRECT collocation
  routing), `…/src/tableau_library.jl` (GL, ce).

## Phase 0 RESULTS (verified empirically)

1. **MFCM deterministic stepping is exactly correct & superconvergent.** On scalar
   `x'=ax+bx(t-τ)` (a=-1,b=-0.5,τ=1), the exact dominant multiplier is 0.3319869969.
   MFCM `MonodromyMap` / `Φ_window` reproduces it: GL1→O(h²), GL2→O(h⁴), GL3→O(h⁶).
   (GL3 p=4 already correct to 8 digits.)

2. **My per-step block math matches MFCM exactly** (M_prop, M_del, weights identical).

3. **BUT: the per-step explicit `∏ F̂ₙ` product does NOT reproduce `Φ_window`.** Every
   attempt converges to a wrong value (~0.19 instead of 0.332). Root cause is fundamental,
   not a coding bug:
   - MFCM's monodromy is **implicit** (`L·X = R·X_prev`, then `L⁻¹R`). With a collocation
     method the stage DOFs couple **across the whole period** (L is block-bidiagonal, its
     inverse is dense-lower-triangular). A single-step shift+inject `F̂ₙ` cannot encode
     that global coupling — each step drops the block that falls off the window, breaking
     the implicit stage interpolation that later steps read back.
   - Confirmed: MFCM itself only ever uses `r = p` (delay = one period); its
     deterministic `base_sweep` bounds-errors for `r < p`.

**Conclusion: the per-step `F̂ₙ⊗F̂ₙ` factorization (SSDM's native M2 form) is incompatible
with implicit high-order collocation.** SDM works in that form only because SDM's step is
explicit (matrix-exponential, no cross-window implicit coupling).

## Revised architecture — per-PERIOD second moment (Option B)

Build the M2 map on the **whole-period** trusted operator instead of per-step:
- Deterministic per-period operator `Φ = Φ_window` (the verified `L⁻¹R` window map),
  size `W=(r+1)·BSIZE`. ρ(Φ) is already superconvergent.
- Second moment over one period: `H = Φ⊗Φ + Λ`, where `Λ = Σₙ Ψₙ·Covₙ·Ψₙᵀ` is the
  accumulated noise covariance, `Ψₙ` = propagator from step n's noise injection to the
  period end, and `Covₙ` = per-step noise covariance built with SSDM's **exact Itô
  isometry** (`Trapezoidal(K)` / `itoisometrymethod`).
- ρ(H) via Krylov on the `W²`-or-`W(W+1)/2`-sized symmetric vectorized map.

`Ψₙ` is obtained from the same `L`/`R` machinery (the columns of `L⁻¹` give the
step-n→end propagation). With noise off, `H = Φ⊗Φ` ⇒ `ρ(H)=ρ(Φ)²` automatically
high-order — the gate becomes trivially satisfied.

### Self-contained port
Port into SSDM (no MFCM import): GL tableau + `colloc_weights` + `build_system_matrices`
equivalent producing `M_prop,M_del,delay_*`, then `build_explicit_matrices` equivalent
producing `L,R` ⇒ `Φ` and the per-step propagators `Ψₙ`. All three are short and already
transcribed in the scratch scripts.

### Verification
- noise-off: `ρ(H)=ρ(Φ)²`, GL2 O(h⁴), GL3 O(h⁶).
- scalar SDDE with known 2nd-moment exponent: GL2 → O(h⁴) for ρ(H).
- cross-check vs in-package SDM order-2 on stochastic Mathieu (ρ_ref≈0.1562208339).
- work-precision diagram: IRK superconvergence vs SDM order-2.

## ===== MAJOR FINDINGS (empirical, this session) =====

### Deterministic side: superconvergence CONFIRMED
- Self-contained GL+L/R `Φ` builder reproduces MFCM exactly; ρ(Φ): GL1 O(h²),
  GL2 O(h⁴), GL3 O(h⁶). Noise-off `H=Φ⊗Φ` ⇒ ρ(H)=ρ(Φ)² superconvergent.

### Stochastic side: the order is capped, and it comes from the DELAY interpolation
- The Sykora-2020 paper (Fig. 2, in repo) states & we REPRODUCED on the Mathieu:
  **q=0 → O(h¹), q=1–3 → O(h²), q=4–5 → O(h³)** for ρ(H). `q` = `SemiDiscretization(q,Δt)`
  = the Lagrange order of the **delay-term** interpolation.
- The higher order is delivered ENTIRELY by the delay interpolation order. A no-delay
  problem makes q inert ⇒ always O(h¹) (this misled an earlier analysis — corrected).
- **SDM ceiling ≈ O(h³)**: even q=7 the second-moment ρ(H) does not exceed ≈order 3.
  The cap is the stochastic Itô term, not the deterministic order.

### The real research question
Can IRK collocation exceed the SDM **O(h³)** ceiling for ρ(H)? The per-step 2nd-moment
contribution is the (exact) Itô isometry `∫ Ψ(s)G(s)G(s)ᵀΨ(s)ᵀ ds`. Its discretization
error has 3 sources: (1) propagator Ψ(s) within-step, (2) delayed-state interpolation in
G(s) [the q part], (3) the `∫ds` quadrature [trapezoidal K-sample]. IRK makes (1),(2)
high-order via the collocation polynomial and (3) can use Gauss quadrature. IF the O(h³)
cap is from these approximations, IRK beats it; if it's a deeper Itô structural limit,
IRK ties at O(h³).

### Next decisive experiment
Build the IRK 2nd-moment for the scalar delay SDDE, measure ρ(H) order vs a
high-precision reference; compare to SDM q=4 (O(h³)). Outcome decides whether to (a)
implement high-order IRK stochastic moment in SSDM, or (b) document the O(h³) ceiling.

## ===== ANALYTICAL DERIVATION OF THE ORDER CEILING =====

From the paper's Eq. (2) (the SDM discretization), each step on [tₙ,tₙ₊₁] approximates:
- drift present state `A xₜ`: kept continuous, propagated by AVERAGED `Ā(n)` (semi-disc).
- delayed state `x_{t−τ}`: frozen + **q-Lagrange interpolated** → error O(h^{q+1}).
- **noise present state `α xₜ` → `α(t)·e^{Ā(n)(t−tₙ)}·x_{tₙ}`**: x FROZEN at tₙ, propagated
  by the AVERAGED matrix exponential.
- noise delayed state `β x_{t−τ}`: frozen at grid (q-Lagrange).

Second moment (Itô isometry, exact GIVEN the integrand):
  `C_{n+1} = F̂ Cₙ F̂ᵀ + ∫₀ʰ G(s) Cₙ(s) G(s)ᵀ ds`,  G(s)=e^{Ā(h−s)}[α(s)e^{Ā s} …].

Order accounting of the noise term (the cap):
- The integral magnitude is **O(h)** (one factor of dt from dW·dW).
- Integrand errors: (i) averaged-`Ā` propagator vs true time-ordered flow → O(h²) local,
  but centered ⇒ integrates to O(h³)/step; (ii) **within-step covariance frozen
  `Cₙ(s)≈Cₙ`** → O(h) integrand error ⇒ O(h²)/step ⇒ O(h) global UNLESS corrected;
  (iii) `∫ds` quadrature (trapezoidal K).
- The q-Lagrange + SD-order corrections successively cancel the (i)/(ii) leading terms,
  lifting the global order to **O(h³) at q≥4** — the empirically observed ceiling.

### Verdict
The O(h³) cap is **NOT fundamental to the Itô isometry.** It comes from two fixable
approximations: (a) the averaged-`Ā` propagator, (b) freezing the within-step covariance
`Cₙ(s)≈Cₙ`. A collocation (IRK) method addresses BOTH: its stage values represent the
within-step trajectory to order 2S (so `Cₙ(s)` evolves at high order), and the collocation
polynomial replaces the averaged exponential. **Prediction: IRK collocation + Gauss
quadrature of the `∫ds` CAN exceed O(h³)** and approach the deterministic order, limited
only by the weak/Itô order of the moment scheme itself (typically the strong/weak order of
the underlying SRK — for 2nd moments of linear SDEs, can reach the deterministic order
when the noise covariance integral is integrated exactly).

### Caveat that bounds the upside
For a SCALAR geometric SDE `dx=ax dt+σx dW` the 2nd moment is `exp((2a+σ²)t)` — the `σ²`
Itô term is exact and an EXACT within-step treatment gives spectral order = deterministic
order. The earlier "O(h¹) for no-delay" was the trapezoidal-`∫ds` + frozen-`Cₙ` artifact,
NOT a fundamental wall. So IRK with exact within-step covariance should recover high order
even there. This is the falsifiable claim to test next.

## ===== PROVEN: superconvergence via MOMENT-EQUATION collocation =====

The claim is CONFIRMED numerically, both no-delay and delay:

**Method.** For a linear SDDE the second moment `M(t)=E[xxᵀ]` and its delay
cross-correlations `P(t)=E[x_t x_{t−τ}ᵀ]` satisfy a DETERMINISTIC delay system. The Itô
corrections enter as ordinary coefficients/sources:
  scalar Hayes `dx=(A x+B x_{t-1})dt+(β x_{t-1}+σ)dW`:
    dM/dt = 2A M + 2B P + β² M(t-1) + σ²        ← β²M(t-1), σ² are the Itô corrections
    dP/dt = A P + B M(t-1)
Apply GL(S) collocation to THIS deterministic moment-DDE ⇒ order = 2S.

**Measured (scratch_theory.jl, scratch_delay_moment.jl):**
| GL(S) on moment eqn | no-delay scalar | delay (Hayes) |
| GL1 | O(h²) | O(h²) |
| GL2 | O(h⁴) | O(h⁴) |
| GL3 | O(h⁶) | O(h⁶) |
vs SDM trajectory-level: O(h¹) (no-delay) / O(h³) ceiling (delay).

**This BEATS the SDM O(h³) ceiling and delivers the requested superconvergence.**

### Implementation route in SSDM (revised, final)
Build the deterministic SECOND-MOMENT delay system from the SDDE coefficients, then
discretize it with the verified high-order GL collocation `L`/`R` window monodromy:
1. From `(A,B,α,β,σ)` assemble the moment-system coefficient functions for the vectorized
   covariance state `vech(E[y yᵀ])` over the delay window (y = augmented history). The
   present-state multiplicative term α contributes an Itô source `αα ᵀ`-type coupling into
   the diagonal; β contributes the delayed `β·β`/`β·(…)` couplings. (This is the
   deterministic "squared-process generator" — cf. SSDM `calculate_noise_mxelems`.)
2. Feed that deterministic moment-DDE to the GL collocation monodromy builder
   (scratch_port.jl / scratch_delay_moment.jl, generalized to the covariance dimension).
3. ρ(window monodromy) = ρ(H), now superconvergent.

KEY DIFFERENCE from current SSDM: do NOT build `H=F̂⊗F̂+Λ` at the trajectory level
(that freezes within-step covariance ⇒ O(h³)). Instead collocate the moment equation.

### Cost note
The moment state is the symmetric covariance `n(n+1)/2` per block — same size class as
SSDM's existing M2 vector. The collocation augments by stage DOFs (×(S+1)) but reaches
order 2S, so far fewer steps p are needed ⇒ Pareto win in the work-precision diagram.
