# v8 design note — integrated-history DOFs for rough delayed drift reads

> **STATUS (2026-07-02, implemented & validated):** scalar prototype
> (`cov_colloc_v8_scalar.jl`) and matrix engine (`cov_colloc_v8.jl`, d≥1,
> time-dependent B via reading-step-weighted J DOFs) both built; full ladder
> in `v8m_validate/targets/highorder.jl`. Outcome: smooth-read class reaches
> **GL4 = O(h⁸), GL5 ≥ O(h¹⁰)**; rough-read class lifts from v7's O(h²) to a
> **fixed O(h⁴) independent of S** (S+2 conjecture rejected by GL4/5/6 all
> measuring ≈4.06). Validation ladder items 1–4 below: all PASS (case d keeps
> an O(h³) term; pure αβ cross clean). Next frontier: second-level integrated
> DOFs for the within-step response covariance to rough delayed forcing.

## Motivation (measured, v7_isolate.jl)

v7 caps at O(h²) for every GL order exactly when the delayed **drift** read
touches a Brownian-rough (C^{1/2}) component: scalar B≠0 with any noise
(cases a/c/d); B=0 is clean (case b); mechanical systems reading the smooth
position component are clean. Mechanism: the stage equations replace
I = ∫ B(s) x(s−τ) ds by Gauss sampling of the delayed path; for a rough
integrand the sampled quadrature carries an irreducible O(h²) second-moment
error. This is the moment-level counterpart of why pathwise Euler/sampling
schemes cap at low strong order for SDDEs, and the fix is the counterpart of
the Itô–Taylor iterated-integral terms (user's suggestion 2026-07-02).

## State augmentation

Per window block (one step of history), in addition to the S+1 node values,
store S integrated-history functionals

    W_j = ∫_step ℓ_j(θ) x(t_block + θ h) dθ ,   j = 1..S

(ℓ_j = the Lagrange basis used by the collocation quadrature). Block size
becomes (2S+1)d.

## Deterministic step changes

The stage equation currently uses B_j·(delayed point read). Replace the
delayed drift accumulation over the step by its exact integral representation:
the term h Σ_j a_ij B_j x(u_j−τ) is the Gauss approximation of
∫_0^{c_i h} B(s) x(s−τ) ds; with W_j of block n−r available, the integral of
the delayed path against any polynomial weight is EXACT in the stored DOFs
(for B(s) polynomial-approximated at collocation order — B itself is smooth,
so expanding B(s) in the same basis keeps 2S order):

    ∫_0^{c_i h} B(s) x(s−τ) ds ≈ Σ_j [B-weights_{ij}] · W_j^{(n−r)}   (exact in x)

The new block's own W_j must also be produced by the step: for the
deterministic part, W_j is a linear functional of the collocation polynomial →
extra rows in the block map (exact). Note the collocation polynomial is the
smooth drift interpolant; the noise contribution to W_j is handled in ΔB.

## Noise increment changes

ΔB gains integral–node and integral–integral entries:

    E[∫ℓ_i η ds · η(u_k)ᵀ] and E[∫ℓ_i η ds · (∫ℓ_j η ds)ᵀ]

computable to the needed order from the same causal kernel
Δ(u,v) = Δ(min)·Φ_A(max,min)ᵀ used by the v7 fill:

    E[Wη_i η(u_k)ᵀ] = ∫ ℓ_i(s) Δ(s, u_k) ds       (piecewise: s<u_k and s>u_k)
    E[Wη_i Wη_jᵀ]   = ∬ ℓ_i(s) ℓ_j(v) Δ(s,v) ds dv (triangle split at s=v)

with Δ(s,v) = Δd(min(s,v))·Φ_A(max,min)ᵀ and Δd(·) the collocation polynomial
of the Σ_noise stage solve. The integrals are smooth on each triangle → Gauss
quadrature at collocation order.

## Validation ladder for v8 (same discipline)

1. noise-off gate exact (the W-DOF deterministic map must reproduce ρ(U)² —
   note ρ(U) itself changes slightly: the delayed integral is now exact, which
   should only IMPROVE the deterministic order).
2. deterministic order on scalar Hayes (exact multiplier 0.3319869969).
3. present-noise exact value (unchanged paths).
4. THE TARGET: scalar cases a/c/d of v7_isolate.jl must lift from O(h²) to
   ≥O(h⁴) (GL2). Reference values already arbitrated:
   a: 0.1473709451, c: 0.0868082230, d: 0.1701952437.
5. mirror + critical Mathieu regression (must stay superconvergent/correct).
