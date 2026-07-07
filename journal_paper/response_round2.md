# Response to the Reviewers — Round 2

Manuscript: *Multiplication-Free Stochastic Semi-Discretization for the
Efficient Moment Stability Analysis of Vibration Systems with Delay*

All six remaining concerns of the Round-2 reports have been addressed in the
revised manuscript. Numbered point-by-point responses follow.

---

## Reviewer 1 (Theoretical and Numerical Methods)

**1. Preconditioning limitation must be documented, not evaded.**
Done. Section 3.3 now contains an explicit limitation statement: because
$\mathcal{H}$ is never assembled, standard preconditioning is not available
(no operator entries exist to factorize), and the GMRES iteration count of
the stationary-moment solve degrades as $\rho(\mathcal{H}) \to 1$ — stated
verbatim as "part of the price of the $\mathcal{O}(p^2)$ complexity". Two
precise (not dismissive) mitigations are given: (i) the diverging quantity in
that limit is the stationary variance itself, so no finite answer is missed;
(ii) stability-*boundary* searches never use the GMRES solve — the boundary
is located via the Krylov eigensolve of $\rho(\mathcal{H}) = 1$, which stays
well-conditioned at the boundary (the dominant eigenvalue is simple and well
separated in every experiment of the paper). All boundary curves in the
revised process map are computed exactly this way.

**2. Do not conflate high order and high dimension.**
Done in both requested places. The abstract now states that the $(2S+2)^2$
covariance scaling restricts the high-order collocation route to low- and
moderate-dimensional systems, complementary to the factored operator that
covers the high-dimensional regime at classical order. The Conclusions carry
the same non-conflation statement and add that a formulation combining both
regimes is an open problem.

**3. Unanalyzed block-local IBP presented as a remedy.**
Accepted. The block-local variant is demoted to explicitly labeled outlook:
the Section-6 enumeration item is retitled "General (block-local) form ---
outlook only", states that the variant is neither analyzed nor tested in this
paper, and that all quantitative IBP results concern the structural
(mechanical) form; the corresponding forward reference in Section 7.3 was
reworded identically.

## Reviewer 2 (Applied Dynamics and Manufacturing)

**4. The white-noise rough-read collapse is an artifact of idealized
modeling.**
We agree with the physics, and the requested dedicated paragraph now closes
Section 6 ("Scope of the rough-read phenomenon: a property of the white-noise
idealization"). It states plainly that in physically realizable control
systems with band-limited (filtered) signals the delayed read is smooth and
no collapse occurs; that the white-noise SDDE is nevertheless the standard
idealization for broadband disturbances, whose adopters inherit the collapse
*numerically* regardless of the physics; that the treatment removes this
numerical consequence at negligible cost; and that with explicitly modeled
filter dynamics the automatic regularity classification keeps the IBP
machinery inactive. The white-noise case is presented as the conservative
envelope of the band-limited family, not as a claim about physical encoder
noise.

**5. The stable-but-beyond-quality region is physically meaningless —
truncate or shade it.**
Done graphically and textually. In the revised process map the band between
the quality limit and the stability limit is heavily shaded (the variance
colormap remains faintly visible beneath the shading purely as a growth
indicator); the caption states the band "is displayed only to show the
variance growth, not as a usable operating region"; and the text adds that
fly-over nonlinearity would dominate there, making the linear quality
boundary conservative in the safe direction. The (shaded) stability curve is
retained in the figure deliberately: its distance from the quality limit *is*
the quantitative message about how misleading a stability-only analysis would
be.

**6. Monte Carlo is not experimental validation — validate against published
data.**
We agree on the epistemology, and Section 7.4 now separates the two claims
explicitly: (i) "the Monte-Carlo check verifies the numerics, not the
physics" — stated verbatim; (ii) the physical model class is anchored to
published experiments: the interrupted-cut directional-factor model and the
SSV lobe modification belong to the family validated experimentally by
Zatarain et al. (CIRP Annals 2008, experimental correlation) and Seguy et al.
(IJAMT 2010, experimental SSV milling), and the force-noise intensity is of
the order identified from published cutting-force measurements (Fodor et al.
2020; Fodor & Bachrathy 2024). Experimental validation of the stochastic
process map itself — the genuinely new prediction — is declared future work
in Section 7.4 and the Conclusions; new machining experiments are outside the
scope of this methods manuscript.

---

We thank both reviewers; the requested changes have made the limitations of
the method and of the model explicit, which we believe strengthens the paper.
