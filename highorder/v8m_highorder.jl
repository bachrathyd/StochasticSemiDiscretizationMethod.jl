# =============================================================================
# v8 MATRIX engine — stage 3: push to 8th order and beyond
#
#   H-1 smooth-read (Mathieu class): GL4 → expect O(h⁸), GL5 → O(h¹⁰).
#       "Hard" mirror variant (stronger noise/excitation) so the convergence
#       window is visible above the Krylov floor; reference = GL5 fine value;
#       Krylov tol tightened to 1e-13.
#   H-2 rough-read scalar case a: GL3/GL4/GL5/GL6 → test the order = S+2 law
#       (would mean GL6 = 8th order in the worst case).
#   H-3 present-noise exact value at GL4/GL5 (analytic reference, the cleanest
#       possible high-order measurement).
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v8.jl"))
using Printf

rr(pb,S,p) = rho_H_krylov_v8m(build_v8m(pb,S,p); tol=1e-13)

println("── H-3 present-noise exact: exp((2a+α²)T), GL4/GL5 ──")
ρex = exp(2*(-0.8)+0.16)
pbc = Prob(1, 1.0, 0.25, t->fill(-0.8,1,1), t->zeros(1,1),
           t->fill(0.4,1,1), t->zeros(1,1))
for S in (4,5)
    errs=Float64[]; ps=[4,8,16]
    for p in ps; push!(errs, abs(rr(pbc,S,p)-ρex)); end
    rates=[log2(errs[i]/errs[i+1]) for i in 1:length(errs)-1]
    @printf("  GL%d errs=%s rates=%s (nominal %d)\n", S,
            join([@sprintf("%.2e",e) for e in errs],","),
            join([@sprintf("%.2f",r) for r in rates],","), 2S)
    flush(stdout)
end

println("── H-1 hard mirror Mathieu (α=β=0.35, ε=0.8), GL4/GL5 ──")
pbh = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.8*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.3 0.0],
    t->[0.0 0.0; 0.35 0.0],
    t->[0.0 0.0; 0.35 0.0])
println("  reference: GL5 p=16 (and GL5 p=12 consistency)")
ρ16 = rr(pbh, 5, 16); ρ12 = rr(pbh, 5, 12)
@printf("  GL5 p=12 %.13f  p=16 %.13f  Δ=%.1e\n", ρ12, ρ16, abs(ρ16-ρ12))
ρrefh = ρ16
for S in (3,4,5)
    errs=Float64[]; ps=[3,4,6,8]
    for p in ps; push!(errs, abs(rr(pbh,S,p)-ρrefh)); end
    rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
    @printf("  GL%d errs=%s rates=%s (nominal %d)\n", S,
            join([@sprintf("%.2e",e) for e in errs],","),
            join([@sprintf("%.2f",r) for r in rates],","), 2S)
    flush(stdout)
end

println("── H-2 rough case a (B=−.4, β=.3): the order=S+2 law? ──")
pba = Prob(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(-0.4,1,1),
           t->zeros(1,1), t->fill(0.3,1,1))
# reference: GL6 fine (its own family, well below any measured error here)
ρrefa = rr(pba, 6, 40)
@printf("  ref (GL6 p=40) = %.13f  [arbited value 0.1473709451]\n", ρrefa)
for S in (3,4,5,6)
    errs=Float64[]; ps=[4,6,8,12,16,24]
    for p in ps; push!(errs, abs(rr(pba,S,p)-ρrefa)); end
    rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
    @printf("  GL%d errs=%s\n", S, join([@sprintf("%.2e",e) for e in errs],","))
    @printf("  GL%d rates=%s (S+2=%d, 2S=%d)\n", S,
            join([@sprintf("%.2f",r) for r in rates],","), S+2, 2S)
    flush(stdout)
end
println("done")
