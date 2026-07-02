# =============================================================================
# Isolate the remaining O(h²) mechanism seen on Hayes (v7 causal, all GL).
# Scalar d=1, T=τ=1, A=−1. Four coefficient combinations:
#   (a) B=−0.4, β=0.3, α=0    — baseline Hayes: measured O(h²)
#   (b) B= 0.0, β=0.3, α=0    — delayed noise WITHOUT delay drift
#   (c) B=−0.4, β=0.0, α=0.3  — present noise + delay drift
#   (d) B=−0.4, β=0.3, α=0.3  — full mix
# Reference per case: fine-grid arbiter (N=512/1024/2048, drift ~1e-11 at d=1).
# Whichever combination keeps the O(h²) tail identifies the responsible term.
# =============================================================================
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))
using Printf

cases = [
    ("a: B≠0, β≠0, α=0 ", -0.4, 0.3, 0.0),
    ("b: B=0, β≠0, α=0 ",  0.0, 0.3, 0.0),
    ("c: B≠0, β=0, α≠0 ", -0.4, 0.0, 0.3),
    ("d: B≠0, β≠0, α≠0 ", -0.4, 0.3, 0.3),
]
ps = [4, 6, 8, 12, 16, 24, 32]

for (name, Bv, βv, αv) in cases
    println("── case $name ──")
    fgpb = FGProb(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(Bv,1,1),
                  t->fill(αv,1,1), t->fill(βv,1,1))
    ρ_ref = fg_arbiter(fgpb, [512, 1024, 2048])
    pb = Prob(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(Bv,1,1),
              t->fill(αv,1,1), t->fill(βv,1,1))
    for S in (2, 3)
        errs = Float64[]
        for p in ps
            ρ = rho_H_krylov(build_v7(pb, S, p); offdiag=:causal)
            push!(errs, abs(ρ - ρ_ref))
        end
        rates = [log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
        @printf("  GL%d errs: %s\n", S, join([@sprintf("%.2e",e) for e in errs], " "))
        @printf("  GL%d rates: %s\n", S, join([@sprintf("%5.2f",r) for r in rates], " "))
    end
end
println("done")
