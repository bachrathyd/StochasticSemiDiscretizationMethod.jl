# Breaking the O(h³) wall for stochastic-DDE second-moment stability

## The question
The trusted SDM caps the second-moment spectral radius ρ(H) at **O(h³)** for multiplicative
noise (the present-state noise term is frozen + averaged). Can a higher-order method be built?
And does raising the order (GL(3)…GL(6)) keep increasing the convergence rate?

## The answer: YES — order can be made arbitrarily high (2S for GL(S))

**Why it works (rigorous).** For a *linear* SDE/SDDE the second moment satisfies a
**deterministic** equation — the Itô correction (e.g. `α²` in `dM/dt = (2a+α²)M`) is just a
coefficient. So applying a high-order deterministic integrator to the *moment equation*
inherits its full order. The SDM is stuck at O(h³) only because it discretizes the
*trajectory* map with a frozen present-state noise term, not the moment equation.

## Verified results (this folder)

### 1. `gl_highorder_test.jl` — KEY result (answers the GL(3)..GL(6) question)
Present-state multiplicative noise `dx = αx dW` (the exact case where SDM is stuck at O(h³)).
GL(S) collocation of the moment ODE `dm/ds = (2a+α²)m`:

| Method | measured order | (= 2S) |
|--------|----------------|--------|
| GL(3)  | **6.0** | 6 |
| GL(4)  | **8.0** | 8 |
| GL(5)  | **10** (→ machine precision) | 10 |
| GL(6)  | **12** (machine-exact at p=2) | 12 |

The convergence order is exactly **2S**, climbing arbitrarily, until ~1e-16 round-off.
Run: `julia --project=. demonstration/gl_highorder_test.jl`

### 2. `_highorder_research/lyap_stage.jl` — pseudospectral covariance Lyapunov (autonomous)
Chebyshev pseudospectral DDE window generator 𝓐_w + covariance Lyapunov
`𝓛 = I⊗𝓐_w + 𝓐_w⊗I + Σ 𝓖⊗𝓖`, `ρ(H) = dominant |exp(eig𝓛·τ)|`.
- present-state noise `dx=αx dW`: **machine-exact already at N=2** (spectral).
- Hayes delayed noise: converges to **~0.1486** = the trusted SDM/Monte-Carlo value
  (confirming the earlier `(M,P)`-lift value 0.5702 was wrong; now agreed by SDM + MC +
  pseudospectral). Delayed-noise convergence is algebraic in N (non-smooth delay kernel).

### 3. `_highorder_research/highorder_secondmoment.jl` — noise-off structural gate
`ρ(U C Uᵀ) = ρ(U)²` to **1e-15** for d=1,2,3 (× GL2,GL3): the implicit collocation
monodromy U is the correct high-order operator; the Lyapunov structure is sound.

## What is NOT yet done
The **periodic, multi-dimensional** case (stochastic delayed Mathieu) needs the GL window
monodromy U (which handles periodicity at high order — verified deterministically) combined
with the noise covariance **in the per-step stage equations** (a matrix-Sylvester collocation
per step). Two shortcuts were tried and rejected (documented as negative results in
`_highorder_research/`):
- deposit-then-propagate noise → degrades to O(h);
- fixed pseudospectral window → wrong for periodic systems (history must slide with t).
The clean remaining build is the per-step covariance collocation; the theory is proven.

## Bottom line
The O(h³) wall is **not fundamental**. Collocating the deterministic moment/Lyapunov
equation gives order **2S** (verified GL(3)…GL(6)) or spectral (pseudospectral, autonomous),
including the present-state multiplicative noise where SDM is permanently stuck. The
remaining work is engineering the periodic multi-d covariance collocation, not a new barrier.
