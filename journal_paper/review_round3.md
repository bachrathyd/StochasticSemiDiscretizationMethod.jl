# Peer Review Report - Final Round

**Journal:** Journal of Sound and Vibration (JSV)
**Manuscript Title:** Multiplication-Free Stochastic Semi-Discretization for the Efficient Moment Stability Analysis of Vibration Systems with Delay
**Authors:** Dániel Bachrathy, Henrik T. Sykora

---

## Reviewer 1: Theoretical and Numerical Methods

**Recommendation:** Accept

**Final Comments:**
The authors have fully complied with the strict conditions outlined in my previous report. 

1. **Preconditioning:** The explicit documentation of the preconditioning limitation in Section 3.3 is appreciated. Stating plainly that GMRES degrades near $\rho(\mathcal{H}) \to 1$ as a structural price of the $\mathcal{O}(p^2)$ complexity is the exact level of transparency required for a high-quality methods paper.
2. **Memory Scaling Constraints:** The modifications to the Abstract and Conclusions clearly separate the high-order (Gauss-Legendre) achievements from the high-dimensional (Kronecker-factored) achievements. Acknowledging that the $(2S+2)^2$ block size restricts the high-order scheme to low/moderate dimensions prevents any misinterpretation by future users.
3. **IBP Fallback:** Relegating the untested block-local IBP variant to an "outlook only" status resolves my concern regarding unanalyzed claims.

The mathematical rigor and algorithmic transparency of the manuscript are now up to the standards of the *Journal of Sound and Vibration*. I recommend publication.

---

## Reviewer 2: Applied Dynamics and Manufacturing Engineering

**Recommendation:** Accept

**Final Comments:**
The authors have satisfactorily addressed the physical modeling critiques.

1. **White Noise Idealization:** The newly added paragraph at the end of Section 6 perfectly captures the physical reality. By acknowledging that the "rough-read collapse" is a numerical consequence of the standard broadband white-noise idealization—and that physically band-limited (filtered) signals would not suffer from this—the authors have successfully bridged the gap between mathematical theory and physical control systems. 
2. **Process Window Shading:** Graphically shading the region between the variance-quality limit and the stability limit is a highly effective way to communicate the dangers of a purely linear stability analysis. Acknowledging that "fly-over" nonlinearity would dominate this region correctly grounds the linear mathematical results in physical reality. 
3. **Experimental Anchoring:** While new physical experiments would have been ideal, I accept the authors' argument that it is outside the scope of this specific methodological paper. Anchoring the physical model class to the widely accepted experimental works of Zatarain et al. (2008) and Seguy et al. (2010) provides sufficient justification for the SSV model structure. 

The integration of the 2-DOF orthogonal model (as demanded by Reviewer 3) further solidifies the engineering relevance of this work. The manuscript now represents a valuable and physically grounded contribution to manufacturing dynamics. I recommend acceptance.