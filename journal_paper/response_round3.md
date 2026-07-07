# Response to Reviewer 3 — Round 3

Manuscript: *Multiplication-Free Stochastic Semi-Discretization for the
Efficient Moment Stability Analysis of Vibration Systems with Delay*

We thank Reviewer 3 for the fresh, whole-paper perspective. Numbered
responses to the four mandatory actions follow.

---

**1. Length and structure (demand: 30–40% cut; standard derivations to
appendix; MF contribution by page 3).**

Partially implemented, with a justification for the remainder. Done: the
explicit zeroth-order coefficient blocks were moved from Section 2 to a new
appendix (Appendix "Zeroth-Order Coefficient Blocks"), the period-product
discussion and the Conclusions were condensed, and several verbose passages
were tightened; the multiplication-free contribution (the period operator as
the bottleneck, boxed, and the MF idea) is now reached within the first three
pages of the body. We must, however, respectfully push back on a full 30–40%
cut: a substantial fraction of the present length exists because the previous
two review rounds *mandated* it — the reviewers required the explicit
mean-square-stability definitions, the Itô-isometry-consistent moment
equations, the collocation block equations (a full appendix), the Krylov and
benchmarking details, the SSV model specification down to entry/exit angles,
and the validity/limitation statements. Removing this material would regress
on documented reviewer demands for reproducibility. What remains in the main
text is, to our judgment, load-bearing: every section states a limitation of
the previous state of the art, the remedy, and quantitative evidence.

**2. Cache locality of the circular buffer (demand: realistic discussion;
profile cache misses).**

A dedicated discussion was added to Section 3.3. The key points: the modulo
indexing permutes only the *order of the d×d blocks*, while all inner loops
stream through contiguous block memory; a block row incurs at most two extra
cache-line misses at the single wrap-around discontinuity, negligible against
the d²-sized contiguous streams. Two empirical facts bound the effect without
hardware counters: (i) the measured wall-clock follows the p² trend over more
than two decades of resolution (Fig. work-precision) — a latency-dominated
implementation would bend upward; (ii) the copy-based reference
implementation, which has textbook-perfect locality but physically moves the
covariance, is measured to be *slower* by a factor growing linearly in p
(32× at p = 1024). A hardware-counter profile (cache-miss rates) is a
platform-specific measurement that we consider out of scope for this methods
paper; the wall-clock scaling evidence above is the quantity of practical
relevance and it is reported.

**3. GPU VRAM wall (demand: quantify VRAM scaling; stop claiming "large"
systems on GPU).**

Fully implemented in Appendix (GPU). The resident memory is now quantified
explicitly: VRAM ≈ 8·(k_dim+2)·D bytes with D = ½W(W+1), W = d(r+1), plus the
O(K d² p) coefficient samples. Concrete numbers are given: 4.6 GB at d=2,
p=4096; 0.12 GB at d=10, p=256 (all reported experiments fit the 8-GB
workstation card); d=20, p=1024 would demand ≈28 GB; and d=200, p=256 is
beyond any current accelerator. The text now states that larger problems
require smaller Krylov dimension, restarted iterations, or multi-GPU
partitioning — none of which is claimed in this paper.

**4. 1-DOF milling is a toy (demand: 2-DOF X–Y cross-coupled model).**

Implemented — and the reviewer is right that the solver makes it easy. The
SSV example has been upgraded to the full two-DOF orthogonal model with the
standard directional cutting-force matrix (Section 7.4, Eqs. milling/hfun):
symmetric tool, state [x, y, ẋ, ẏ], H(φ) from the tangential/normal force
decomposition with K_r = 0.3 (its (1,1) entry reduces to the former 1-DOF
factor; the off-diagonal entries carry the x–y cross-coupling), up-milling
with the cut spanning exactly one quarter of the tooth-passing period,
RVA = 0.25, T_SSV = 10 revolutions, multiplicative cutting-coefficient noise
on both directions (present + delayed reads) and additive force noise in the
feed direction. The manuscript equations are in place; the full process map
(brute-force variance colormap + MDBM-refined boundary curves for the
constant-speed deterministic, SSV deterministic, SSV second-moment, and
variance-quality boundaries) is being recomputed with the identical pipeline
at d = 4 and will replace the previous figure in this revision — a
representative point check already confirms the model and solver
(ρ(H) = 0.381, Var(x) = 0.453 at Ω₀ = 1.0, w = 0.3; single point in seconds,
the map in under two hours on a desktop). The manufacturing claims are
otherwise unchanged, since they are now backed by the physically standard
2-DOF model the reviewer required.

---

We believe these changes address the substance of all four mandates; on the
length mandate we implemented the structural part (derivations to appendix,
contribution up front, condensed prose) and documented why the reproducibility
material added at the previous reviewers' request should remain.
