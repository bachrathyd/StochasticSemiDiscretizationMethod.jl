# Response to Reviewer 3 — Round 4 Addendum (post-acceptance QA report)

Manuscript: *Multiplication-Free Stochastic Semi-Discretization for the
Efficient Moment Stability Analysis of Vibration Systems with Delay*

Following the Round-4 acceptance we completed the production run of the
two-DOF SSV process map announced in our Round-3/Round-4 responses. In the
course of that final quality-assurance pass we found and fixed a genuine
defect. In the interest of full transparency we report it here, numbered as
before, together with the finalized figure and its cost accounting.

**1. (QA finding and correction.)** The independent Monte-Carlo cross-check of
Section 7.4 — added at this review's insistence on validation — caught a
factor-of-two error in the *additive-noise accounting* of the accompanying
package: an `stAdditive` source specified on its own Wiener channel was
registered once per *channel* rather than once per *source*, so its
variance contribution was double-counted in the factored fixed point (and
quadruple-counted in the classical assembled mapping). All previously
published single-axis studies are unaffected (there the additive source
shares the multiplicative channel, and the accounting is correct — which is
precisely why the error stayed hidden until the two-DOF configuration).
The spectral radius ρ(H) is blind to additive terms and was never affected;
only the stationary-variance layer of the new chart moved. The fix is one
line (container sized per source), and is now guarded by a regression test
against the exact analytic limit Var(x) = σ_a²/(4ζ) at w = 0 (factored and
classical routes, d = 2 and d = 4, including the two-independent-source case
that previously threw an error). A freshly recomputed variance map agrees
with the analytically corrected data to 10⁻⁸ relative.

**2. (Corrected validation numbers.)** At the representative stable point
(Ω₀, w) = (1.0, 0.30): Var(x) = 0.2267 (MF-SSDM, converged) against
0.2296 ± 0.0010 (95%) from the 2·10⁴-path semi-implicit Euler–Maruyama
ensemble — a 1.3% agreement consistent with the O(Δt) weak error of the
simulation. We further note that the *explicit* Euler–Maruyama ensemble is
numerically unstable outright on the two-DOF model at the same time step
(on the single-axis model it "only" overestimated the variance by ≈80%);
the manuscript sentence has been updated accordingly. We believe this
episode strengthens rather than weakens the paper's methodological message:
the moment solver, the trajectory ensemble, and the exact limit caught the
defect precisely because they are independent.

**3. (Finalized figure and cost table.)** The four boundary curves of the
process map are now located by the Multi-Dimensional Bisection Method in a
two-phase pattern (zeroth-order bracketing search + one linear-interpolation
pass). The two *deterministic* boundaries are computed with the deterministic
semi-discretization package (sparse left/right monodromy pair + Krylov
eigensolve of the first moment) — three orders of magnitude cheaper per
point, resolving the fine lobe sub-structure of the modulated system at a
577×705-equivalent resolution in 39 s / 1025 s. The two *stochastic*
boundaries (ρ(H) = 1 and Var(x) = 0.25) require the MF-SSDM solver of this
paper (1751 s / 1616 s at 289×161-equivalent resolution), and the 96×64
variance color map costs 2400 s — all measured job-alone on 56 threads. A
new Table (tab:ssv_cost) in Section 7.4 reports this per-layer accounting,
and the hardware is now specified precisely (dual-socket Intel Xeon Gold
6154, 2×18 cores, 192 GB RAM). The agreement of the deterministic package's
ρ(Φ) = 1 curve with the stable-region edge of the MF-SSDM color map, and of
the MDBM-located ρ(H) = 1 curve with the same edge, constitutes an
additional independent cross-validation of the entire chain.

**4. (Scope of changes.)** No mathematical content of Sections 1–6 changed.
The changes are: the one-line package fix and its regression test; the
corrected variance layer and finalized boundary curves of Fig. 12 (fig:ssv);
the updated Monte-Carlo paragraph and new cost table in Section 7.4; and the
precise hardware statement in Section 7. The engineering conclusions are
qualitatively unchanged — the quality boundary lies strictly and
non-uniformly below all stability boundaries; quantitatively the usable
window is somewhat larger than in the pre-correction draft (the variance was
overestimated by the factor of two).

We would be grateful for the reviewer's assessment of these corrections.
