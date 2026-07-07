# Peer Review Report - New Reviewer 

**Journal:** Journal of Sound and Vibration (JSV)
**Manuscript Title:** Multiplication-Free Stochastic Semi-Discretization for the Efficient Moment Stability Analysis of Vibration Systems with Delay
**Authors:** Dániel Bachrathy, Henrik T. Sykora
**Reviewer:** Reviewer 3 (Newly Assigned)

**Recommendation:** Major Revision / Reject and Resubmit

**General Comments:**
I have been invited to review this manuscript as an additional reviewer. While the authors have apparently survived two rounds of back-and-forth regarding specific mathematical limitations with previous reviewers, I am stepping back to evaluate the paper as a whole. 

Frankly, the manuscript in its current form is unacceptable for publication in the *Journal of Sound and Vibration*. The paper is excessively verbose, dense, and bloated. It reads more like a Ph.D. thesis or an unedited internal laboratory report than a concise, impactful journal article. The authors bury their actual contributions under pages of pedagogical derivations of standard semi-discretization that have been published dozens of times over the last two decades. Furthermore, while the authors make grand theoretical claims about computational complexity, they completely ignore fundamental modern hardware realities (cache hierarchies and VRAM) and rely on a physical "toy" model that undermines their manufacturing claims.

Before this manuscript can be considered, it must undergo a drastic structural editing process and address severe hardware-level and physical modeling oversights.

**Major Concerns (Must be addressed before acceptance):**

**1. Unacceptable Length and Bloat (Demand for 30-40% Reduction):**
The manuscript is exhausting to read. The authors take 5 paragraphs to explain concepts that require 5 sentences. Section 2 (Stochastic Semi-Discretization Framework) is essentially a textbook rehash of the classical SSDM. Why is this taking up so much space? The journal is not a repository for lecture notes. 
*   **Mandatory Action:** Cut the manuscript length by at least 30-40%. Move all standard derivations (like the explicit matrix components in Section 2) to an Appendix or Supplementary Material. Get straight to the "Multiplication-Free" contribution by page 3. Be concise.

**2. The $\mathcal{O}(p^2)$ Illusion and CPU Cache Thrashing:**
The authors proudly claim strict $\mathcal{O}(p^2)$ complexity achieved via "virtual circular buffers" and index mapping ($idx(k) = (n-k) \pmod{p+1}$). Theoretically, this is true. However, in software engineering, theoretical Big-O notation means nothing if it destroys memory locality. Iterating through large covariance blocks using modulo-based pointer jumping will cause massive CPU cache thrashing (L2/L3 cache misses). For moderate to large systems, the memory fetch latency will completely dominate the FLOPs, turning the $\mathcal{O}(p^2)$ algorithm into a wall-clock nightmare. 
*   **Mandatory Action:** The authors must provide a realistic discussion on memory locality and cache-miss penalties inherent to their circular buffer design. Showing a single idealized time-scaling plot is insufficient; profile the cache misses.

**3. Ignored GPU VRAM Limitations in Krylov Iterations:**
In Appendix A, the authors brag about keeping the "entire computation resident on the device (GPU)" to avoid catastrophic PCIe bus sync overheads. However, they are using Arnoldi iterations (KrylovKit.jl) to find the spectral radius. For high-dimensional systems (which the authors claim to solve), storing the Arnoldi subspace vectors (the full history covariance matrices) will instantaneously blow past the VRAM limits of any standard or even data-center GPU (e.g., 16GB - 80GB). If the Krylov vectors trigger host-to-device memory paging, the "Zero-Sync" policy is destroyed and performance will tank.
*   **Mandatory Action:** The authors must explicitly quantify the GPU VRAM consumption scaling as a function of the Krylov dimension, $p$, and $d$. Stop claiming the method works on the GPU for "large" systems without defining the hard VRAM wall.

**4. The 1-DOF Milling Model is a Toy (Physical Irrelevance):**
In Section 7.4, the authors model Spindle Speed Variation (SSV) milling as a single Degree-Of-Freedom (1-DOF) system. This is completely inadequate for a D1 journal paper attempting to make serious claims about manufacturing process windows. Real milling chatter is inherently defined by the dynamic cross-coupling between the X and Y orthogonal directions through the directional cutting-force matrix. A 1-DOF model completely ignores mode-coupling chatter and the directional orientation of the tool path, which drastically shifts stability lobes in reality. 
*   **Mandatory Action:** To demonstrate that this $\mathcal{O}(p^2)$ solver is actually useful for real-world manufacturing (and not just an academic toy), the authors must upgrade the SSV milling example to at least a 2-DOF (orthogonal X-Y) model with cross-coupling. If the solver is as fast as claimed, adding one more DOF should be trivial. If they cannot do this, the manufacturing claims must be heavily downgraded.

**Summary:**
Shorten the paper drastically. Address the cache latency. Address the GPU memory wall. Use a physically relevant 2-DOF milling model. Until then, I recommend rejection.