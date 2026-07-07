# Review process — main_MFSSD.tex

Multi-role internal review prior to submission. Each round: five independent
reviewer roles (applied mathematician / SDE numerics, manufacturing engineer,
numerical linear algebra & HPC, stochastic-processes expert, journal editor)
review the full manuscript; findings are triaged and applied; the next round
verifies the fixes. Bibliography verified entry-by-entry against the web in a
separate audit (see "Bibliography audit" at the end).

---

## Round 1

### Reviewer A — Manufacturing engineer (machining dynamics). Verdict: major revision.
MAJOR:
A1. Directional factor h(t) never defined (Eq. milling): give explicit sum over
    teeth, screen function, entry/exit angles for down-milling a_D=0.5, and the
    tooth angular position as the INTEGRAL of the modulated speed φ(t)=∫Ω ds.
A2. τ(t)=(2π/N)/Ω(t) is the instantaneous approximation of the regenerative
    delay; exact delay solves ∫_{t-τ}^t Ω ds = 2π/N (differs at O(RVA)). State
    the choice + justify; also specify how a time-varying delay is discretized
    (buffer τ_max, interpolation between slots) — currently unspecified.
A3. SSV study not reproducible: ζ missing, T_SSV definition ambiguous, Floquet
    period unstated, p/S/engine unstated, grid ranges only in figure, contour
    extraction method unstated, dimensionless↔physical mapping absent.
    → Parameter table needed.
A4. Ra–variance link: assumptions (Gaussian, zero-mean, surface inheritance)
    unstated; no citation at the equation; deterministic forced-vibration
    contribution to Ra excluded by construction (c=0) — say the bound covers
    only the stochastic contribution; Var=0.25 threshold arbitrary — discuss
    scaling to physical Ra spec.
A5. No SSV literature cited in the SSV section (Sexton & Stone; Insperger &
    Stépán variable-speed; Zatarain et al. CIRP 2008; Seguy et al.; Otto &
    Radons; Long & Balachandran).
A6. Bibliography integrity — ~35 placeholder entries, bundle-citations of
    8–12, Guo2024 duplicated within one \cite. Submission blocker.
A7. Beam appendix unspecified (EI/ρA/ω_i, damping model, P0,P1,kP,kD,x_s,τ,
    noise term written literally as "+ noise"). → Parameter table.
A8. No RVA=0 baseline in the SSV chart; justify σ_c=0.3 magnitude.
Physical plausibility of the charts: confirmed credible (lobe morphology,
quality boundary strictly below stability, widest separation inside pockets).
MINOR (selection): A_avg undefined; q=2/4 blocks used but only q=0 derived
(cite Sykora2019/Insperger2011); N and K symbol collisions; light-blue zone
not identified in fig:ssv caption/legend; abstract/highlights over limits;
"n_m ≳ 8 for three digits" inconsistent with figure; "2.16-million-unknown
FE beam problem" misreadable → "covariance problem"; h(t)≡0 intervals remark;
"symmetrically contract" wrong for σ=0.3 island topology.
TYPOS: fig1 axis labels A/B/α vs text δ/ε/σ; "[Grant Numbers]"; bib key
s00170-024-14059-9 raw DOI; "D{\}vid" corrupted author; Var notation drift.

### Reviewer B — Journal editor (JSV). Verdict: major revision.
MAJOR:
B1. Scope overload: four contributions + GPU + beam + 3 case studies; title
    names only MF. Reframe under an umbrella claim (or split — authors chose
    reframe).
B2. Section 5 (collocation) not reproducible: only the J_i equation is given;
    stage equations + covariance-kernel fill + block update needed (appendix).
B3. "Order eight demonstrated" thin (S=4/5 hit solver floor by p≈12, ~3 usable
    points). Soften to "consistent with order eight" or add resolvable range.
B4. Claim inconsistencies: abstract "four steps" vs body "p=8" (variant mix);
    conclusions "faster by 3–5 orders at equal time" self-contradictory →
    "more accurate"; S=2 slope "4.0–4.6" vs "4.1–4.6"; d≲10 vs d≈10–30.
B5. Citation padding: 6–12-cite chains, identical list reused for different
    claims, Guo2024 twice in one bracket, Roberts1986/Namachchivaya1990
    misattached to Krylov methods. Trim to 2–4 load-bearing citations each.
B6. Reproducibility: Fig 2 benchmark parameters absent; WP hardware/software
    unstated; "desktop" vs Xeon Gold inconsistency; Julia package unnamed/
    unversioned/no URL; SSV grid ranges missing.
MINOR (selection): ρ(S)=1 wrong — truncated shift is nilpotent (ρ=0), restate
via norm; intro roadmap says beam is in Sec 7 but it is Appendix B; N defined
twice in nomenclature, d/m/q/w/σ_c/σ_a/RVA/T_SSV/a_D missing, a_p unused;
M*_{1,1} vs M*_{x,x} drift; \ref{app:gpu} → Appendix~\ref; R_a formula needs
citation + Highlight 7 overclaims ("established"); "never be practically
approached" soften; "tool's period of rotation" → tooth-passing period (N>1);
covariance vs second-moment wording; repeated phrase "exercising every
structural feature" (Sec 7.4 + Conclusions); author-year in subsection
heading; second author email/ORCID.
LANGUAGE: marketing register in Conclusions ("successfully addressed",
"debilitating", "fundamentally shifts this paradigm", "plagued",
"dramatically", "paving the way"), "exhaustive"→"extensive", "giant"→"large",
"honest"→drop, "ultra-high fidelity"→"high-resolution", "decisive"×3 → once,
"for free"→"negligible additional cost", straight quotes → LaTeX quotes,
"curse of dimensionality" misnomer, grant sentence grammar, "0th-order"→
"zeroth-order", 84-word sentences split, abstract ~420 words → ≤250.
MISSING FORMALITIES: Highlights — 7 items (max 5) and all but one over 85
chars (compliant set proposed); Declaration of competing interest; CRediT;
Data availability (package must be named + URL); Funding statement; abstract
length; Fig 2 caption parameters.

### Reviewer C — Applied mathematician (SDE numerics). Verdict: major revision.
MAJOR:
C1. Δt-power inconsistency across Eqs. (2)/(6)/(7)/(10) vs Sec. 4 (= D1/E3);
    adopt the isometry-integral convention once.
C2. Additive-injection Q00 valid only for E[y]≡0; missing F E[y] ĉᵀ and
    G E[y] γ̂ᵀ cross terms; γ̂γ̂ᵀΔt has the O(Δt³) pathology (= E4/D10).
C3. vec/vech confusion in eq:Hn_kron; Table 1 D-values ~3.5–4× below
    dim vech for stated (d,p) — undocumented reduced/delay-vs-period
    bookkeeping; beam value matches exactly. Define D once, annotate table.
C4. All order claims empirical; "no choice of q can lift the order" exceeds
    evidence (q=2,4, one benchmark); S=4/5 measurable window <1 decade (~3
    points) — "eighth order demonstrated" overclaims; O(Δt²) rough ceiling
    asserted without derivation. Add heuristic local-error analysis + caveats;
    keep "measured" qualifier in abstract/highlights.
C5. Baseline fairness: explicit period product is the weakest classical
    formulation; the natural intermediate baseline (single-step recursion
    with copy, O(p³) time / O(p²) memory, no new machinery) is unmeasured;
    "strictly quadratic" presumes p-independent Krylov iteration counts —
    report them.
C6. T=τ assumed silently; SSV violates it (τ(t), T_SSV≠τ); time-varying-delay
    discretization + SSV solver settings must be documented (= A2/D8).
C7. Collocation section not reproducible; Δ(u,u), c_i, endpoint index e, p′
    undefined (= B2/E5/D7). Appendix with explicit block equations needed.
MINOR (selection): arbiter certificate (9.4e-13) tighter than reference
self-convergence (4.6e-11) — report arbiter's own error estimate; Var_ref has
no independent arbiter — acknowledge; "allocated memory" ≠ footprint (= D3);
R_a: Gaussian/zero-mean/cyclostationary-phase assumptions + feed-mark
kinematic roughness ignored — state; C^{1/2−} notation define; IBP holds
pathwise (Y_c ∈ C¹, no Itô correction) — say it; Sykora-benchmark parameters
missing; h(t) discontinuities vs exact integration — discuss mesh alignment;
Sec-4 exactness scoped to same-quadrature (= D5); Q̂ undefined; GMRES near
ρ→1; Table 1 mixes p across rows — note.
NOTATION: N duplicated; eq:beam M/C/G/K collide with global symbols (add
disclaimer or rename); q triple-duty; w triple-duty; "v8" code name leaks in
figure legends; covariance vs second moment; M*_{1,1} vs M*_{x,x};
idx(−1); [Grant Numbers]; ssv filename "iter29" artifact.

### Reviewer D — Numerical linear algebra / HPC. Verdict: major revision.
Cross-checked headline numbers vs figures: 384/0.30 s and 386 GB/61 MB ratios
consistent (1.28e3 / 6.33e3). MAJOR:
D1. Same Δt-normalization inconsistency as E3 (Eqs. 2/7/12 vs Sec-4 integral
    form; O(Δt³) as printed). Adopt one convention, audit all equations.
D2. O(p⁴) classical claim asserted not derived; measured allocation slope in
    wp_ultra(b) is ~p⁵ (cumulative GC allocation of p products of O(p⁴)
    objects). State both exponents: peak footprint O(p⁴), cumulative
    allocation O(p⁵); give an operation count for the actual comparator.
D3. "Allocated memory" metric must be defined (cumulative GC allocation, not
    peak RSS — 386 GB can't be resident); fix Conclusions "peak memory
    allocation" mislabel; note metric-dependence of the 6.4e3 ratio.
D4. vec/vech conflation in eq:Hn_kron (Kronecker acts on vec, D declared as
    vech); Table 1 D-values don't match dim vech for d=4/200 (beam matches);
    reconcile bookkeeping (likely DOF vs d or delay≠period) and state it.
D5. Kronecker "exactness" mis-scoped: identity holds between factored and
    dense built from the SAME K-point quadrature; both approximate the
    continuous isometry at the rate of the rule. State K choice + error vs K;
    define Δs; fix free superscript w in Eq. 15.
D6. Krylov details absent: package/version, tolerances, krylovdim, iteration
    counts vs p (O(p²) end-to-end needs #it ≈ const), why dominant eigenvalue
    is real/positive (PSD cone), GMRES settings, conditioning of I−H near
    ρ→1 (SSV boundary!), symmetry maintenance in iteration, and what the
    ~1e-11 'solver floor' actually is (reference uncertainty 4.6e-11!).
D7. Collocation section not reproducible (= B2/E5). Δ(u,u), p′ undefined.
D8. Time-varying delay never formulated (= A2); p 'per delay' vs 'per period'
    bookkeeping global fix.
D9. Benchmarking hygiene: machine spec for wp_ultra unstated; BLAS threads;
    timing protocol; JIT warm-up; code/data availability statement with
    repo/version required.
D10. Same missing mean–moment cross terms as E4.
MINOR (selection): GPU appendix wording contradicts figure (GPU 10–500×
slower at small p for d≤4; break-even d=2 at p≈200; plateau reads 5–6× not
7×; FP64 1/32-rate P4000 strengthens the case — state it; data-center claim
is speculation → label as expectation); A_avg undefined; R_a assumptions;
arbiter details for the 9.4e-13 certification; abstract/highlights limits;
symbol collisions q/Y/N; pruning "1e-13" — on which norm; GMRES near lobe
peaks; IBP requires b_c ∈ C¹ and constant τ — note restriction.
TYPOS: [Grant Numbers]; duplicate N; "in \ref{app:gpu}" → "Appendix~\ref";
Eq.15 free w + ordering; Φ bolding inconsistent; idx(−1) domain; Δs undefined.

### Reviewer E — Stochastic-processes expert. Verdict: major revision.
MAJOR:
E1. Mean-square stability never defined; ρ(H)<1 criterion never stated; H must
    be framed as a finite-dimensional approximation of the infinite-dimensional
    two-time-covariance (Volterra) moment operator, with convergence either
    cited (Sykora 2019/2020a) or stated as numerically verified.
E2. Itô interpretation of the SDDE never stated; add "interpreted in the Itô
    sense" + a Wong–Zakai remark (Stratonovich correction relevance for
    physical broadband noise).
E3. Per-step noise normalization internally inconsistent across Eqs: (6) has
    no Δt, (12) has Δt with G-blocks that are already O(Δt) integrals (→O(Δt³)
    scaling, wrong); additive γ̂ (time-integrated) with ΔW dimensionally wrong.
    The Sec-4 isometry-integral form is the correct one — rewrite Eqs.
    (2),(5),(6),(7),(9),(12) in one consistent convention.
E4. Second-moment update silently assumes zero mean: for c≠0 cross terms
    F E[y] ĉᵀ + Gₖ E[y] γ̂ₖᵀ missing. Either restrict the model or state joint
    mean+moment propagation. "Covariance" vs uncentered second moment wording.
E5. Collocation section: no derivation; kernel symbols Δ(u,u), Φ_A undefined;
    "exact causal two-time covariance kernel of locally linearized SDE" is an
    overclaim as stated — specify what is exact vs collocation-order
    approximate; state order 2S is measured; add block equations (appendix).
E6. Stationary logic: Q̂ undefined (period-accumulated injection); existence
    condition ρ(H)<1 unstated; PSD-ness of GMRES solution not discussed;
    cyclostationarity — which phase of Var(t) enters the quality boundary.
MINOR (selection): ρ(S)=1 wrong (nilpotent, ρ=0) — same finding as Reviewer B;
"iff" too strong in rough/smooth classification → "structurally/generically";
prove ρ(H)=ρ(Φ)² in one line (H=Φ⊗Φ restricted to symmetric subspace,
eigenvalues μᵢμⱼ — exact incl. complex multipliers); period–delay
commensurability T=τ assumed silently, SSV handling of τ(t) unspecified;
Table 1 D-values ≈3.8× smaller than dim vech (pruned dimension? label);
R_a formula assumptions (zero-mean Gaussian); "coherence resonance"
terminology incorrect → "stochastic amplification"; Eq. (14) E[·] is a time
integral not expectation, unsummed w; vec/vech restriction to symmetric
subspace state once; "exact algebraic consistency with continuous-time
Itô–Volterra operators" overclaim → "with the explicitly assembled discrete
operator"; well-posedness sentence + citation; A_avg undefined.
TYPOS: N defined twice; Δ overloaded (Δt/ΔW/Δ-kernel); endpoint index e
undefined; noise-index letters drift k/w; "fixpoint" vs "fixed point";
idx(−1) domain note; [Grant Numbers]; highlights >85 chars; row-index p
collides with resolution p.

### Fixes applied after Round 1

**Mathematics / equations (C1–C3, D1–D5, E1–E6, A-minor, B-minor):**
1. Itô interpretation stated after Eq. (1) + Wong–Zakai/Stratonovich remark +
   well-posedness sentence (cite Mao). [E2, E-minor 11]
2. New paragraph "Mean-square stability and the quantity computed": definition,
   ρ(H)<1 criterion, infinite-dimensional two-time-kernel framing, convergence
   cited to Sykora 2019/2020a + declared numerically verified, zero-mean
   convention stated (mean propagated jointly in general case). [E1, E4, C2]
3. Δt-normalization made consistent everywhere: stochastic blocks defined as
   sampled functions G(s)=Φ(s)α(t+s); Eqs. (mom_update), (Q-block),
   (ito_injection), (Hn_kron), (emat), (kron_quad) all written as isometry
   integrals ∫G(s)·C·G(s)ᵀds; O(Δt) scaling explicit. [C1, D1, E3]
4. vec/vech fixed: H_n defined on vec space, restriction to symmetric subspace
   stated with D = d(r+1)(d(r+1)+1)/2; r (delay steps) vs p (period steps)
   bookkeeping introduced globally; Table 1 caption states τ=T/2 ⇒ r=p/2 and
   the D formula (values now reproducible). [C3, D4, E-minor 5]
5. ρ(S)=1 corrected: shift is nilpotent/relabeling; growth from injections. [B,E]
6. Q̂ defined (period-accumulated injection, evaluated matrix-free); existence
   (1∉spec) vs meaningful (ρ<1) fixed point; PSD symmetrization + monitoring
   stated; cyclostationary phase of Var stated at Eq. (ra). [E6, C-minor 9]
7. ρ(H)=ρ(Φ)² proved in one line (H=Φ⊗Φ on symmetric subspace, eigenvalues
   μiμj, exact incl. complex pairs); gate must survive pruning noted. [E-minor 3]
8. Kronecker exactness rescoped: identity holds between factored and dense
   built from the SAME K-point rule (K=20 stated); Σ_w restored in Eq.;
   √(w_m) scaling fixed (Δs removed). [D5, C-minor 7]
9. New Appendix "The Collocation Block Equations": block layout, stage system,
   J-rows, within-step Lyapunov equation, causal kernel Σ_η(u)Ψ(v,u)ᵀ with
   exactness scope, quadrature fill; kernel symbols renamed (Δ→Σ_η, Φ_A→Ψ);
   endpoint index c_e=1 defined; p′→r. [B2, C7, D7, E5]
10. "no choice of q" and "order eight demonstrated" softened to measured/
    "consistent with order eight, within the resolvable range"; abstract "four
    steps" claim removed; "S=4 at p=8 vs classical at p=10⁴" now says
    "extrapolating its measured first-order trend". [C4, B3, B4]
11. Missing intermediate baseline measured and added (reviewer C5): single-step
    recursion with copy, O(p³) confirmed (470 s vs 15 s at p=1024, ≈32×, one
    power of p); Krylov iteration count = 15 for every p and method → reported
    in text; figure updated with third curve. [C5, D6-part]
12. Memory metric defined honestly (cumulative solver allocation; peak footprint
    O(p⁴) vs O(p²) stated; allocation slope O(p⁵) explained); Conclusions
    "peak memory allocation" wording fixed. [D2, D3, C-minor 2]

**SSV milling (A1–A5, A8, C6, D8):**
13. h(t) fully specified (Eq. hfun): tooth sum, screen function, φ_en=π/2,
    φ_ex=π, and φ_j(t) = ∫Ω ds (integral of modulated speed).
14. Delay definition stated as instantaneous approximation vs exact implicit
    delay, O(RVA) deviation, citations (Long2007, Seguy2010); discretization of
    the time-varying delay (rounded read, τ_max buffer) described.
15. All parameters now in text: ζ=0.02, T_SSV = 2 nominal revolutions =
    principal period, q=2, r=24, grid ranges, contour extraction; σ_c=0.3
    flagged as deliberately strong; noise-off gate number (2e-5 @ r=32).
16. SSV literature added: Zatarain 2008, Seguy 2010, Long 2007 (verified
    replacements); RVA=0 constant-speed stability baseline computed and
    overlaid in the chart; lobe-raising effect noted.
17. R_a link: zero-mean Gaussian assumptions stated, stochastic-contribution-
    only caveat (feed marks/runout excluded), threshold-scaling discussion,
    citation (Honeycutt & Schmitz 2017), cyclostationary phase stated. [A4]
18. fig:ssv legend: light-blue "stable but quality-violating" zone labeled.

**Beam appendix (A7):** full parameter set added (EI=ρA=L=1, ω_i=(iπ)²,
P0=3, P1=4, Ω=2ω₃, τ=2π/Ω, x_s=0.37, k_P=k_D=3, modal damping 2ζ₁ω_i with
ζ₁=0.002, noise term −σ_P G q dW with σ_P=1.5, q=0, K=20); M/C/K/G symbol
clash disclaimed; "three digits" claim replaced by quantitative statement.

**Benchmark study (B6):** Sykora-benchmark subsection rewritten with the full
model, all parameters (ε=2, ζ=0.1, τ=2π, period 4π, p=40, q=2), MDBM cited,
axes/labels aligned with the figure ((A,B), α), caption completed;
"symmetrically contract" → "shrinks monotonically; closed island at α=0.3".

**Editorial / formalities (B):** abstract rewritten ≤250 words; highlights
reduced to 5 items ≤85 chars; Declaration of competing interest, CRediT,
Data availability (package named + GitHub URL) added; grant sentence fixed;
marketing register removed from Conclusions ("successfully addressed",
"debilitating", "paradigm", "plagued", "dramatically", "paving the way",
"exhaustive", "giant", "ultra-high fidelity", "honest", "for free");
"curse of dimensionality" → resolution bottleneck; straight quotes fixed;
intro roadmap updated (appendices); citation chains trimmed to 2–4
load-bearing refs (all fabricated keys removed — see audit); "tooth-passing
period" fix; nomenclature overhauled (N dedup → z teeth, added d/m/q/w/r/
σ_c/σ_a/RVA/T_SSV/a_D, removed unused a_p/K_t/K_n/σ/RPM); "0th"→"zeroth";
Appendix~\ref formatting; internal code name "v8" removed from all figure
legends (GL-S / IBP GL-S); M*_{1,1}→M*_{x,x}; GPU appendix wording aligned
with the measured data (break-even points, 5–6×, FP64 1/32-rate note,
data-center claim labeled expectation). Duplicate "exercising every
structural feature" removed.

---

## Round 2 (five fresh reviewers on the revised 40-page manuscript)

All five verdicts improved from "major revision" to "minor revision".
Round-1 fixes were verified item by item; confirmed OK: Itô/Wong–Zakai
statement (correction formula re-derived and confirmed), mean-square-stability
paragraph, isometry-integral rewrite (dimensional consistency re-checked),
mean-coupling cross terms (independently re-derived — complete and correct),
nilpotent shift, Q̂/existence/PSD, ρ(H)=ρ(Φ)² argument (airtight incl. complex
and defective cases), collocation appendix (stage system and causal kernel
verified; scope honest), Kronecker same-quadrature scoping, baseline +
iteration counts (470/15 s ≈ 32× checked against figure), memory metric,
GPU wording vs data, h(t) convention (verified against standard directional
factor), R_a caveats, SSV citations + RVA=0 overlay, benchmark parameters,
declarations/highlights/abstract (245 words; 5×≤85 chars), bibliography
integrity (all 37 cited keys exist; no fabricated keys).

Remaining blockers found (all mechanical): (1) tab-corrupted `\times` in the
beam appendix (script artifact); (2) Table 1 D-values match a buffer of r+2
blocks (r+1 history + write slot), not the printed r+1 formula; (3) the SSV
step-count arithmetic was internally inconsistent ("minimal delay" vs
τ_max/24; step count claimed Ω₀-dependent while the stated rule made it
constant); (4) incomplete p→r migration in Secs. 2–3/Algorithm 1; (5) intro
still had "(order eight demonstrated)" + unqualified p=10⁴ claim; (6) S=2
slope 4.0–4.6 vs 4.1–4.6; (7) new symbol collisions from the Kronecker
rewrite (quadrature index vs m, q); (8) Gaussianity premise of Eq. (ra)
overstated under strong multiplicative noise; (9) minor: vec(Q̂) double-vec,
dW argument, iff→generically, Var_ref-no-arbiter acknowledgment, hardware
naming drift, 6.4 vs 6.3×10³ rounding, fig:ssv caption zones/baseline,
"for free"/"honest" leftovers, 5-citation chain, uncited bib entries.

### Fixes applied after Round 2
All items above fixed: `\times` repaired; Table 1 caption now states the
r+2-block buffer (r+1 history + write slot) with the matching formula; the
SSV subsection rewritten around the NEW study (T_SSV = 10 revolutions,
Δt = min(τ_max/24, 2π/30) resolving the natural frequency, w up to 2.4,
σ_c = 0.2) with measured step counts and timings — this longer-period
configuration is feasible with the MF method only; p→r sweep completed in
Secs. 2.1–2.2/Algorithm 1 with the T=τ display convention stated; intro
claims softened and qualified; slopes unified to 4.1–4.6; quadrature index
renamed κ everywhere, stage derivatives renamed Ẏ_j; Gaussianity clause
added at Eq. (ra); all minor/typo items applied; bibliography trimmed by 7
further entries (uncited or single-use tangential: Lamba, Baker, Ghanem–
Spanos, Wedig, Bhattacharya, Øksendal, Grigoriu) → 32 verified entries;
figure sizes reduced and two verbose passages condensed for length.
Additionally, per external review: Monte-Carlo validation of the stationary
variance added; Krylov tolerances stated; Itô-choice note in the SSV section
(see response_to_reviewers.md for the external points).

---

## Round 3 (spot-check of Round-2 blocker fixes + new SSV/MC passages)

All nine mechanical blockers verified FIXED, with independent recomputation:
Table 1 D-values reproduce exactly from the r+2-buffer formula for all six
rows; SSV step counts (432 at Ω₀>0.694 delay-bound branch, 1875 at Ω₀=0.16
natural-frequency branch, T_SSV=392.7, τmax=21.8) reproduce from the printed
min-rule; MC percentages (0.67%/0.96%, "within 1%") and cost arithmetic
(13,440 problems / 1380 s ≈ 0.103 s each) check out; p→r migration complete
incl. Algorithm 1 under the stated T=τ convention; no tab corruption; slopes
unified; κ index and Ẏ_j consistent; all 32 cite keys exist with no orphans
and no chain >4; abstract 247 words; 5 highlights ≤85 chars.
Three wording suggestions in the new passages — applied: "stochastic
resonance" → "noise-induced resonance" (the authors' published term, MSSP
2021, now cited at first use); ambiguous "displacement" → "downward shift of
the boundary"; scare-quoted self-quotation removed; "infeasible by many
orders of magnitude" and "unconditionally consistent" reworded.
**Verdict: submission-ready.** Remaining for the authors: real grant numbers
in the funding statement.

---

## Round 3 — external Reviewer 3 (Gemini), reviewer3_report.md

Verdict: major revision / reject-and-resubmit. Four mandates: (1) 30–40%
length cut, standard derivations to appendix, contribution by page 3;
(2) cache-locality discussion of the circular buffer (+ profiling demand);
(3) GPU VRAM scaling quantification for the Krylov subspace; (4) upgrade the
1-DOF SSV milling model to 2-DOF X–Y with directional cross-coupling.

### Fixes applied after Round 3
(1) Zeroth-order coefficient blocks moved to new appendix (app:blocks);
period-product paragraph and Conclusions condensed; MF contribution reached
by page 3; full 30–40% cut respectfully declined with justification (material
mandated by Rounds 1–2 for reproducibility). (2) Cache-locality paragraph in
Sec. 3.3: modulo maps permute block order only, inner loops stream contiguous
blocks, ≤2 extra cache-line misses per block row; empirical bounds: measured
p² wall-clock over two decades + copy-based (perfect-locality) reference is
32× slower at p=1024; hardware-counter profile declared out of scope.
(3) VRAM formula + concrete numbers in the GPU appendix (4.6 GB @ d=2,p=4096;
0.12 GB @ d=10,p=256; 28 GB @ d=20,p=1024 infeasible on the 8-GB card;
d=200 beyond any accelerator); mitigation options listed, none claimed.
(4) SSV example upgraded to the 2-DOF cross-coupled model (directional matrix
H(φ) from the tangential/normal decomposition, K_r=0.3, up-milling, cut = 25%
of tooth pitch, RVA=0.25, T_SSV=10 rev, log-Ω axis); equations in manuscript;
d=4 process map recomputing with the same BF+MDBM pipeline (point check:
ρ=0.381, Var=0.453 @ Ω₀=1, w=0.3). Numbered response: response_round3.md.

---

## Bibliography audit (web verification, entry by entry)

Verified by four parallel agents against Crossref/publisher pages (2026-07-05).
Result: **25 entries VERIFIED** (kept, some with metadata fixes), **13
CORRECTED** (real works, wrong metadata — including the authors' own papers:
Fodor2020 vol. 111 not 108; Fodor2023 vol. 74 art. 103515; Bachrathy2021_CIRP
pages 329–332 + full author list; Sykora2021_MSSP author list Sykora–Hajdu–
Dombovari–Bachrathy; Sykora2020b full five-author list; Stepan1989 is a book;
Bhattacharya 1990; Arnold English ed. 1974; Ritto→Engineering Optimization
2011; Spanos→Ghanem & Spanos PEM 1993; Wedig→Springer LNP 451 1995;
Lamba→JCAM 161; Roberts & Spanos→IJNLM 21; Bachrathy multi-frequency→CIRP
Annals 2013; deterministic MF-SD→Bachrathy JVC 2026), and **41 entries
NOT FOUND / FABRICATED** — all deleted and their citations removed:
Long2007*(replaced by real Long–Balachandran–Mann NonlinDyn 2007),
Dombovari2010 (→Zatarain 2008), Seguy2008 (→Seguy 2010), Bouche1999,
Bobrik1998, Iglesias2022, Sykora2021 (duplicate of _MSSP with invented title),
Zeldin1998, Namachchivaya1990, Sun2006, Xu2011, Liu2023, Wang2022, Zhang2022,
Li2020, Wang2021, Chen2023, Wu2024, Kim2019, Liu2020, Yang2022, Gao2023,
Zhou2021, Zhao2024, Ma2019, Sun2020, Ding2022, Peng2023, Ren2021, Guo2024,
Zhang2023, Liu2021, Ritto2020, Wang2024, Chen2020, Ma2022, Li2023, Guo2021,
Wu2022, Peng2020, Zhao2021, Sun2023, Ding2024, Yang2020, Kim2022, Zhou2024,
Gao2021, Ren2024, Bakar2023, Lee2020. New verified entries added where needed:
Zatarain2008, Seguy2010, HoneycuttSchmitz2017, GhanemSpanos1993, BachrathyMFSD
(JVC 2026), Fodor2024_PSD (renamed from raw-DOI key). \nocite{*} removed.

---

## Round 4 — external Reviewer 3, second look (reviewer3_round2.md)

**Verdict: ACCEPT.** The reviewer confirms: the 2-DOF cross-coupled milling
model "irrefutably" demonstrates practical value; the VRAM quantification and
cache-locality explanation are technically sound and consistent with the
measured scaling; the length justification is accepted given the earlier
reproducibility mandates. "No further concerns."
Closing response: response_round4.md (notes the 96×64 hi-res colormap
completed at 7134 s / 6 cores; MDBM curves finalizing; figure + Sec 7.4
numbers to reflect final values in production files).

---

## Round 4 addendum — post-acceptance QA correction (2026-07-06)

During the production run of the 2-DOF chart, the Monte-Carlo cross-check
(added at this review's insistence) caught a factor-2 additive-variance
double count in the package: an stAdditive source on a distinct Wiener
channel was registered once per channel instead of once per source (factored
fixpoint ×2, classical mapping ×4; all 1-DOF studies unaffected — shared
channel there). Fixed in src/structures_result.jl; regression testset vs the
exact w=0 limit Var=σa²/(4ζ) added (Pkg.test all pass); fresh 56-thread
recompute of the variance map matches the exact ÷2 correction to 1e-8.
Corrected point (Ω₀=1, w=0.3): Var=0.2267 vs MC 0.2296±0.0010 (1.3%; the
0.453 quoted in response_round3.md is superseded). Explicit Euler diverges
outright on the 2-DOF model (was +80% bias at 1-DOF) — manuscript sentence
updated. Figure finalized: 4 MDBM boundary curves, two-phase pattern
(zeroth-order bracketing + linear-interpolation pass); deterministic curves
via the deterministic SemiDiscretizationMethod.jl package (LR monodromy +
KrylovKit; 577×705-equivalent; 39 s / 1025 s), stochastic curves via MF-SSDM
(289×161-equivalent; 1751 s / 1616 s), BF map 2400 s — all job-alone at 56
threads; new cost table tab:ssv_cost; precise hardware statement (dual Xeon
Gold 6154, 2×18 cores, 192 GB). 43 pages, compiles clean. Numbered response:
response_round4_addendum.md (awaiting round 5).

---

## Round 5 — external "Final Round" report (review_round3.md, 2026-07-06 09:08)

**Both reviewer roles: ACCEPT.** Reviewer 1 (theory/numerics): preconditioning
limitation documented at the right level of transparency; high-order vs
high-dimensional achievements cleanly separated; block-local IBP correctly
demoted to outlook. Reviewer 2 (applied dynamics/manufacturing): white-noise
idealization paragraph "perfectly captures the physical reality"; experimental
anchoring via Zatarain/Seguy accepted as sufficient for a methods paper; 2-DOF
model "solidifies the engineering relevance". Note: Reviewer 2 praised the
validity-band shading; the shading was subsequently replaced by a uniform
colormap at the corresponding author's request — the qualitative-validity
caveat is retained in the body text and the caption, which preserves the
reviewer's substantive point. This report predates response_round4_addendum.md
(11:30), which now stands as the latest numbered response documenting the
post-acceptance QA correction. **Five review rounds completed; final verdict
across all roles: ACCEPT.**

---

## Bibliography audit (web verification, entry by entry)
[superseded stub — see the detailed audit section above (25 verified, 13
corrected, fabricated entries removed); kept for round-count integrity]
