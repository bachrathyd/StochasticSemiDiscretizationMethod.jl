# Peer Review Report - Round 2

**Journal:** Journal of Sound and Vibration (JSV)
**Manuscript Title:** Multiplication-Free Stochastic Semi-Discretization for the Efficient Moment Stability Analysis of Vibration Systems with Delay
**Authors:** Dániel Bachrathy, Henrik T. Sykora

---

## Reviewer 1: Theoretical and Numerical Methods (Strictly Critical)

**Recommendation:** Minor Revision (with strict conditions)

**General Comments:**
The authors have made a commendable effort to clarify the manuscript and address the initial concerns. The addition of the intermediate baseline ($\mathcal{O}(p^3)$) and the admission of the memory explosion limits are appreciated steps toward transparency. However, admitting to severe limitations is not the same as solving them. Several claims remain overly broad given the constraints that have now been forced into the open. The manuscript is mathematically sound but its framing must be tightened before publication.

**Remaining Concerns:**
1. **The Preconditioning Evasion:** The authors admit that unpreconditioned GMRES degrades as $\rho(\mathcal{H}) \to 1$ but brush this off by claiming they only compute "away from this regime." This is a severe algorithmic limitation. If a user utilizes your package to search for the exact stability boundary (which is the primary use case for such software), the solver will stall. The manuscript must explicitly state in Section 3.3 that the matrix-free formulation fundamentally prohibits standard preconditioning, leading to critical slowdowns near the stability boundary. This is the price paid for $\mathcal{O}(p^2)$ complexity, and it must be documented as such.
2. **High-Order Memory Limitations Must Be Front-and-Center:** The authors now concede that the $d=200, S=4$ configuration "is beyond desktop memory." Yet, the abstract and introduction still heavily sell both "high-dimensional systems" and "high-order convergence" in the same breath. The authors must explicitly state in the Abstract and Conclusion that the high-order Gauss-Legendre collocation blocks are strictly limited to low-to-moderate dimensional systems due to the $(2S+2)^2$ memory scaling, while high-dimensional systems are restricted to the classical first-order formulation. Do not conflate the two achievements.
3. **Unanalyzed "Fallback" for IBP:** The authors admit that the block-local form of the Integration-by-Parts (IBP) remedy is "unanalyzed" in this paper. Presenting an unanalyzed, untested mathematical fallback as a "remedy" in a D1 journal paper is unacceptable. The block-local form must either be thoroughly analyzed with convergence plots (similar to the structural form) or entirely moved to the "Outlook/Future Work" section. It cannot be presented as a finalized solution.

---

## Reviewer 2: Applied Dynamics and Manufacturing Engineering (Strictly Critical)

**Recommendation:** Major Revision

**General Comments:**
While the authors have walked back some of their most egregious physical claims (e.g., modifying the $R_a$ mapping and reducing the noise intensity), the application section still reads like a purely mathematical exercise divorced from physical reality. The addition of a Monte Carlo simulation is a welcome numerical check, but it completely misses the point of the original critique: numerical self-consistency does not prove physical validity. For a paper in the *Journal of Sound and Vibration*, the mechanical modeling relies on highly idealized assumptions that render the "process window" plots physically questionable. 

**Remaining Concerns:**
1. **The "White Noise" Artifact:** The authors argue that modeling velocity noise as white noise is the "worst case" for regularity, thus necessitating their IBP fix. But as they admit in the response: if real encoder noise is band-limited (which it always is), the path is smooth, and the "rough-read collapse" does not occur. This means the entire mathematical apparatus built in Section 6 is solving a problem that does not exist in real physical control systems. It is an artifact of over-idealized white-noise modeling. The authors must add a dedicated paragraph in Section 6 explicitly stating that in practical, physically realizable control systems with filtered signals, this first-order collapse is a mathematical fiction.
2. **The "Linearity" Defense is Flawed:** The authors defend their linear model by stating the variance limit "keeps the process inside the linear regime." If this is true, then the entire region in Fig. 8 between the red dashed line (quality limit) and the blue line (stability limit) is physically meaningless, because fly-over nonlinearity will have taken over long before the stability limit is reached. Therefore, plotting the theoretical linear stability limit (blue line) is highly misleading. The authors should graphically truncate or heavily shade the region where the linear assumption breaks down, rather than presenting it as a valid "stable but unusable" mathematical space. 
3. **Monte Carlo is NOT Experimental Validation:** The authors state that "full experimental validation... is beyond the scope of this methods paper." This is a significant weakness for a paper making claims about "surface-quality process windows" in SSV milling. While I accept that conducting new machining experiments may be too burdensome for a revision, the authors must validate their model against *existing, published experimental data* on SSV chatter and surface roughness. A Monte Carlo simulation only proves that your semi-discretization solves your idealized equation correctly; it does absolutely nothing to prove that Eq. (25) accurately describes a real milling machine. Validate the physical model, not just the math.