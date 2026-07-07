# Response to the Reviewers

Manuscript: *Multiplication-Free Stochastic Semi-Discretization for the Efficient
Moment Stability Analysis of Vibration Systems with Delay*

We thank both reviewers for their thorough and demanding reports. All ten major
concerns have been addressed; several prompted new computations that are now
part of the manuscript. Point-by-point responses follow (reviewer text in
italics, our response in plain text; section/equation numbers refer to the
revised manuscript).

---

## Reviewer 1 — Theoretical and Numerical Methods

**1.1 — *Inconsistent quadrature schemes: trapezoidal Itô isometry (Sec. 4) vs
high-order collocation claims (Sec. 5).***

The two quadratures belong to two different schemes, and the manuscript now
says so explicitly. The trapezoidal $K$-point rule of Section 4 is the
isometry quadrature of the **classical-order** MF-SSDM operator (and of its
Kronecker-factored form); its error is part of that scheme's overall
first-order moment accuracy, so a low-order rule with $K=20$ is not the
binding error term there. The **collocation blocks** of Section 5 do not use
this rule: all within-step covariance entries are filled by Gauss quadrature
of the causal two-time kernel, consistent with the collocation order (the
full construction is now written out in Appendix A, added in revision). The
measured order-$2S$ convergence on problems with time-periodic drift *and*
time-periodic multiplicative noise coefficients (Figs. 5–7) confirms that no
low-order stochastic quadrature limits the composed scheme. We have also
re-scoped the "exactness" claim of Section 4: the factored operator is
algebraically identical to the dense tensor *assembled with the same rule*;
both share the quadrature error of the isometry integral.

**1.2 — *Hidden constants; where does classical SSDM outperform MF-SSDM at
small $p$?***

Measured directly and now reported: below $p \approx 12$ all three evaluation
routes (explicit product, single-step recursion, MF) complete within
milliseconds and their ordering is dominated by constant overheads; the
explicit product is marginally faster only in this regime. The work-precision
figure now also contains the *intermediate* classical formulation the referee
implicitly asks about — the sparse single-step recursion with copying, which
avoids the explicit product without any of the new machinery. Its measured
cost follows $\mathcal{O}(p^3)$ per operator application; MF-SSDM overtakes it
already at $p \approx 16$ and is $32\times$ faster at $p = 1024$, a gap that
grows linearly in $p$. Since useful stochastic resolutions start around
$p \sim 10^2$, the small-$p$ regime has no practical relevance; a zoom plot
would show three near-flat millisecond curves, so we report the crossover in
the text instead.

**1.3 — *Krylov solver black box: spectrum, stalling, preconditioning.***

Added to Sections 3.3 and 7.2: the eigensolve uses KrylovKit's Arnoldi with
Krylov dimension 15 and relative tolerance $10^{-12}$; in every experiment of
the paper it converges within a single Krylov factorization (15 operator
applications), independently of $p$ — reported explicitly now — because the
dominant eigenvalue of $\mathcal{H}$ is real, positive (PSD-cone preserving
map) and well separated in all tested problems. The stationary solve uses
unpreconditioned GMRES; conditioning of $\mathcal{I}-\mathcal{H}$ degrades as
$\rho(\mathcal{H})\to 1$, which is intrinsic to the problem (the variance
itself diverges there), and the variance boundary of the SSV study
($\operatorname{Var}=0.25$, i.e. $\rho$ well below 1) is computed away from
this regime. The referee is right that matrix-free evaluation precludes
conventional preconditioning; none was needed in any experiment reported, and
we state this openly.

**1.4 — *Combined $d \times S$ memory explosion; show $d=100$, $S=4$.***

We agree with the arithmetic and now state the combined scaling explicitly in
Section 5: the collocation covariance costs $(2S+2)^2$ relative to the
classical block, i.e. $\mathcal{O}\big(d^2 r^2 (2S{+}2)^2\big)$ storage, and a
$d=200$, $S=4$ configuration is beyond desktop memory. This is a genuine,
stated limitation: the high-order blocks are presently intended for low- and
moderate-dimensional systems where tight tolerances matter, while the
classical-order factored operator covers the high-$d$ regime (Table 1, beam
appendix). The automatic pruning of Section 5.3 reduces the factor to
$(S+2)^2$ (measured $2.6$–$2.8\times$) in the common no-delayed-noise case.
A memory-lean high-$d$/high-$S$ variant is future work and is listed as such.

**1.5 — *IBP remedy restrictive; block-local fallback unanalyzed.***

Correct on scope: the structural (companion-form) IBP requires the
antiderivative in the state, which is the case for the mechanical
(delayed-PD-control) class that motivates the paper. The block-local form
introduces no dynamic state augmentation — it reuses exactly the type of
integrated-history quantities the collocation block already carries, so its
per-step overhead is that of $S{+}1$ additional quadratures and it does not
change the Krylov operator structure. We have kept its full analysis outside
this paper (stated openly in Section 6 and in the outlook), because the
measured evidence in this manuscript covers the mechanical class only.

---

## Reviewer 2 — Applied Dynamics and Manufacturing

**2.1 — *$\sigma_c = 0.3$ multiplicative noise is absurdly high; justify.***

Accepted. The SSV study was recomputed with $\sigma_c = 0.2$, and the text now
positions this value explicitly: stochastic cutting-force models identified
from turning measurements (Fodor et al. 2020, IJAMT; Fodor & Bachrathy 2024,
IJAMT) support relative force-variation intensities of this order at the
upper end, and the value is deliberately chosen near that upper end so that
the noise effect is clearly resolvable on the process map. We also note in
the text that the qualitative conclusion — the variance-based quality
boundary lies strictly below the stability boundary — persists at lower
intensities, with the two boundaries approaching each other as
$\sigma_c, \sigma_a \to 0$.

**2.2 — *$R_a$ mapping is a gross oversimplification.***

Accepted and reworded throughout. Equation (26) is now introduced as the
**vibration-induced contribution** to the surface irregularity — a
lower-bound proxy, not the measured $R_a$: the revised text states explicitly
that kinematic feed marks, tool nose radius, runout, and plastic side flow
contribute additively and are outside the linear model, cites the
surface-location-error literature (Honeycutt & Schmitz 2017), and adds the
assumptions under which the Gaussian mean-absolute-value identity holds
(zero-mean response; approximately Gaussian in the additive-noise-dominated
regime, with excess kurtosis under strong parametric noise). The
corresponding highlight was deleted.

**2.3 — *Linear theory in a regime where fly-over nonlinearity dominates.***

The revised text now makes the intended logic explicit: the variance limit is
precisely the instrument that **keeps the process inside the linear regime**.
The quality threshold is set at a vibration level far below the amplitude at
which tool–workpiece contact loss occurs, so the linear SDDE analysis is used
only where it is valid; in the stable-but-noisy band beyond the threshold the
linear model underestimates none of the practical unusability of the region
(nonlinear effects make it worse, not better), so the boundary is
conservative in the safe direction.

**2.4 — *White noise on the delayed velocity is unphysical; encoder noise is
colored.***

We agree on the physics and now address it in Section 6: white noise is the
**worst case** for the regularity classification. If the velocity read is
band-limited (filtered encoder signal), the sample paths of the read quantity
are smooth, the rough-read collapse does not occur, and the discretization
attains its nominal order without any special treatment — the IBP
reformulation is then simply inactive and harmless. Modeling-wise, a colored
noise is accommodated by augmenting the state with the filter dynamics, after
which the automatic regularity classification of the coefficient matrices
detects the smoothness by itself. The white-noise analysis presented covers
the conservative end of this spectrum, and a Wong–Zakai remark (added in
Section 2) covers the interpretation question for physical broadband noise.

**2.5 — *No experimental validation; at minimum validate against Monte
Carlo.***

A Monte-Carlo validation was added (Section 7.4): the stationary variance of
the SSV milling equation at the representative stable point
$(\Omega_0, w) = (1.0, 0.30)$ was estimated by a semi-implicit (symplectic)
Euler–Maruyama ensemble ($2\times10^4$ paths, time step $\Delta t/8$, 40
discarded + 20 averaged modulation periods, path-wise time-varying delay with
linear history interpolation). Result: $\operatorname{Var}(x) = 0.2681 \pm
0.0006$ (95% CI) against the MF-SSDM fixed point $0.2699$ (self-converged to
$0.2707$ at fourfold resolution) — agreement within 1%, the residual being
consistent with the $\mathcal{O}(\Delta t)$ weak error of the simulation.
Two remarks the referee may find interesting: (i) the MF-SSDM value converges
monotonically under mesh refinement ($0.26988 \to 0.27057 \to 0.27071$ at
$r = 24/48/96$), so the semi-discretization introduces no systematic bias;
(ii) a \emph{naive explicit} Euler–Maruyama ensemble at the same time step
overestimates the variance by $\approx 80\%$ (numerical energy drift on
oscillatory systems) — trajectory averaging is itself delicate, which
reinforces the case for a consistent moment solver. Full experimental
validation on a machine tool is beyond the scope of this methods paper and
is stated as such in the outlook.

---

We believe the revision addresses all concerns; we are grateful for the
review, which in particular prompted the added Monte-Carlo validation, the
intermediate-baseline measurement, and the explicit collocation appendix.

---
---

# Response — Round 2

## Reviewer 1

**R2-1.1 — *Preconditioning limitation must be documented, not brushed off.***
Done. Section 3.3 now states explicitly that the matrix-free formulation
prohibits standard preconditioning (no operator entries exist to factorize),
that the GMRES iteration count of the stationary-moment solve therefore
degrades as $\rho(\mathcal{H})\to1$, and that this is part of the price of
the $\mathcal{O}(p^2)$ complexity. We also state the two mitigating facts
precisely rather than as a dismissal: the diverging quantity in that limit is
the stationary variance itself, and stability-boundary searches do not use
the GMRES solve at all — the boundary is located through the Krylov
eigensolve of $\rho(\mathcal{H})=1$, which remains well conditioned at the
boundary (this is also how every boundary curve in the revised process map is
computed).

**R2-1.2 — *Do not conflate high order and high dimension.***
Done, in both places the reviewer names: the abstract now states that the
$(2S+2)^2$ covariance scaling restricts the high-order collocation route to
low- and moderate-dimensional systems, complementary to the factored operator
that covers the high-dimensional regime at classical order; the Conclusions
carry the same statement and add that a formulation combining both regimes is
an open problem.

**R2-1.3 — *Unanalyzed block-local IBP presented as a remedy.***
Accepted. The block-local form is demoted to explicitly labeled outlook: the
enumeration item in Section 6 now reads "outlook only", states that the
variant is neither analyzed nor tested in this paper, and that all
quantitative IBP results concern the structural (mechanical) form; the
corresponding sentence in Section 7.3 was reworded the same way.

## Reviewer 2

**R2-2.1 — *The white-noise rough-read collapse is a modeling artifact.***
We agree with the physics and now say so in a dedicated paragraph at the end
of Section 6, as requested: in physically realizable control systems with
band-limited (filtered) signals the delayed read is smooth and no collapse
occurs — the phenomenon is a property of the white-noise idealization. The
paragraph then states why the analysis is still necessary: the white-noise
SDDE is the standard idealization for broadband disturbances, and whoever
adopts it inherits the collapse numerically; the treatment removes that
numerical consequence at negligible cost, and the automatic regularity
classification keeps the machinery inactive whenever the filter dynamics are
modeled explicitly.

**R2-2.2 — *The stable-but-beyond-quality region is physically meaningless;
truncate or shade it.***
Done graphically and textually. In the revised process map the band between
the quality limit and the stability limit is heavily shaded (the variance
colormap remains faintly visible under the shading purely as a growth
indicator), the caption states that the band is not a usable operating
region, and the text adds that fly-over nonlinearity would dominate there —
making the linear quality boundary conservative in the safe direction. We
have kept the (shaded) stability curve in the figure because its distance
from the quality limit is precisely the quantitative message about how
misleading a stability-only analysis would be.

**R2-2.3 — *Monte Carlo is not experimental validation; validate the model
against published data.***
We agree on the epistemology and have made the two claims explicit and
separate in Section 7.4: (i) the Monte-Carlo check validates the numerics,
not the physics — stated verbatim; (ii) the physical model class is anchored
to published experimental work: the interrupted-cut directional-factor model
and the SSV lobe modification belong to the model family validated against
milling experiments by Zatarain et al. (CIRP Annals 2008, with experimental
correlation) and Seguy et al. (IJAMT 2010, experimental SSV milling), and the
force-noise intensity is taken from published cutting-force measurements
(Fodor et al. 2020; Fodor & Bachrathy 2024). An experimental validation of
the stochastic process map itself — the new prediction of this paper — is
stated as future work in both Section 7.4 and the Conclusions; we respectfully
maintain that performing new machining experiments is outside the scope of
this methods manuscript.
