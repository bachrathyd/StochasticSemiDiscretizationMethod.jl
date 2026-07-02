# =============================================================================
# DECISIVE IBP experiment on a HARD PD-Mathieu (strong gains + strong noise so
# convergence is measurable over ~3 decades before any reference floor).
#   q̈ + 2ζq̇ + (1+0.8cos2πt)q = k_P(t)q(t−τ) + k_D(t)q̇(t−τ) + [αq+βq_τ]dW
#   k_P = 0.40(1+0.3cos2πt), k_D = 0.45(1+0.4cos2πt), ζ=0.05, α=0.5, β=0.35
# Reference: v8-IBP GL4 p=40 (self-converged; consistency GL4 p=32 vs p=40
# reported), cross-checked against the fine-grid arbiter within its accuracy.
# Verdict: v8-direct GL3 rates (~4 expected) vs v8-IBP GL3 rates (≥5.5 = win).
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
using Printf

const ROUGH=[2]; const POSMAP=Dict(2=>1)
Afun(t)=[0.0 1.0; -(1.0+0.8cos(2π*t)) -0.1]
Bfun(t)=[0.0 0.0; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
αfun(t)=[0.0 0.0; 0.5 0.0]
βfun(t)=[0.0 0.0; 0.35 0.0]
pb  = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun)
pbfg= FGProb(2,1.0,1.0, Afun, Bfun, αfun, βfun)

println("── reference: v8-IBP GL4 self-convergence + arbiter cross-check ──")
ρ32 = rho_H_krylov_v8m(build_v8ibp(pb,4,32,ROUGH,POSMAP))
ρ40 = rho_H_krylov_v8m(build_v8ibp(pb,4,40,ROUGH,POSMAP))
@printf("  v8-IBP GL4: p=32 %.12f  p=40 %.12f  Δ=%.1e\n", ρ32, ρ40, abs(ρ40-ρ32))
ρarb = fg_arbiter(pbfg, [512, 1024, 2048])
@printf("  arbiter %.12f  |ref−arb| = %.2e (arbiter accuracy ~2e-10)\n", ρ40, abs(ρ40-ρarb))
ρref = ρ40

ps=[4,6,8,12,16,24]
for (nm, eng) in (("v8-direct", (S,p)->rho_H_krylov_v8m(build_v8m(pb,S,p))),
                  ("v8-IBP   ", (S,p)->rho_H_krylov_v8m(build_v8ibp(pb,S,p,ROUGH,POSMAP))))
    for S in (3,4)
        errs=[abs(eng(S,p)-ρref) for p in ps]
        rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
        @printf("  %s GL%d errs: %s\n", nm, S, join([@sprintf("%.2e",e) for e in errs]," "))
        @printf("  %s GL%d rates: %s\n", nm, S, join([@sprintf("%5.2f",r) for r in rates]," "))
        flush(stdout)
    end
end
println("done")
