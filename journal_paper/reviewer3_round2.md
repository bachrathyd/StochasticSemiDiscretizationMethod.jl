# Peer Review Report - Reviewer 3 (Round 2)

**Journal:** Journal of Sound and Vibration (JSV)
**Manuscript Title:** Multiplication-Free Stochastic Semi-Discretization for the Efficient Moment Stability Analysis of Vibration Systems with Delay
**Authors:** Dániel Bachrathy, Henrik T. Sykora

**Recommendation:** Accept

**General Comments:**
The authors have robustly addressed the severe structural and technical critiques raised in my initial review. By upgrading the physical milling simulation from a 1-DOF "toy" to a physically realistic 2-DOF orthogonal model with cross-coupling, the authors have irrefutably demonstrated the practical engineering value of their $\mathcal{O}(p^2)$ solver. The framework is no longer just a mathematical exercise but a proven tool for realistic manufacturing process analysis.

Furthermore, the transparent quantification of the GPU VRAM memory wall and the explicit definition of where the current accelerator limits lie add necessary realism to the computational claims. The explanation regarding cache locality (that the index modulo only affects block order, while the $d \times d$ inner loops remain contiguous) is technically sound and aligns with the observed empirical wall-clock scaling.

Finally, I accept the authors' justification regarding the manuscript length. While it remains dense, I acknowledge that the reproducibility constraints and formal definitions mandated by the previous reviewers necessitate keeping the core mathematical scaffolding intact. Moving the classical derivations to the appendix and accelerating the path to the novel "Multiplication-Free" contribution has sufficiently streamlined the reading experience.

The manuscript represents a definitive computational step forward for stochastic delay systems and is now ready for publication in the *Journal of Sound and Vibration*. I have no further concerns.