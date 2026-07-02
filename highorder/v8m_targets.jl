# =============================================================================
# v8 MATRIX engine — validation ladder, stage 2 (d=2 targets)
#
#   T-A mirror Mathieu (arbiter-confirmed 0.7389661254): v8 must keep v7's
#       superconvergence and value (delay reads the SMOOTH position — v7 was
#       already clean here).
#   T-B critical stochastic Mathieu (settled 0.15624206): value regression.
#   T-C ROUGH-READ oscillator — delayed VELOCITY feedback:
#         x'' + 2ζ x' + k(t) x = b·x'(t−τ) + [α x + β x(t−τ)]dW
#       the delayed drift reads x₂ (rough, noise-carrying) ⇒ v7 caps at O(h²);
#       v8's exact delayed integrals must lift it. Reference: fine-grid
#       arbiter (independent). THE decisive d=2 A/B.
# =============================================================================
include(joinpath(@__DIR__, "cov_colloc_v8.jl"))
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
using Printf

function sweep8(pb, ref, ps, S; engine=:v8)
    errs=Float64[]
    for p in ps
        ρ = engine==:v8 ? rho_H_krylov_v8m(build_v8m(pb,S,p)) :
                          rho_H_krylov(build_v7(pb,S,p); offdiag=:causal)
        push!(errs, abs(ρ-ref))
    end
    rates=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
    @printf("    %-3s GL%d errs: %s\n", engine, S, join([@sprintf("%.2e",e) for e in errs]," "))
    @printf("    %-3s GL%d rates: %s\n", engine, S, join([@sprintf("%5.2f",r) for r in rates]," "))
    flush(stdout)
    errs
end

println("══ T-A mirror Mathieu (ref 0.7389661254) ══")
pbA = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0],
    t->[0.0 0.0; 0.2 0.0],
    t->[0.0 0.0; 0.2 0.0])
for S in (2,3); sweep8(pbA, 0.7389661254, [6,8,12,16,24,32], S); end

println("══ T-B critical stoch. Mathieu (ref 0.15624206) ══")
pbB = Prob(2, 4π, 2π,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])
for S in (3,); sweep8(pbB, 0.15624206, [16,24,32,48,64], S); end

println("══ T-C rough-read: delayed VELOCITY feedback ══")
Afun(t)=[0.0 1.0; -(1.0+0.3*cos(2π*t)) -0.2]
Bfun(t)=[0.0 0.0; 0.0 0.15]          # reads x₂(t−τ): rough
αfun(t)=[0.0 0.0; 0.2 0.0]
βfun(t)=[0.0 0.0; 0.1 0.0]
pbC_v = Prob(2, 1.0, 1.0, Afun, Bfun, αfun, βfun)
pbC_fg = FGProb(2, 1.0, 1.0, Afun, Bfun, αfun, βfun)
println("  arbiter reference:")
ρrefC = fg_arbiter(pbC_fg, [512, 1024, 2048])
ps=[6,8,12,16,24,32]
for S in (2,3)
    sweep8(pbC_v, ρrefC, ps, S; engine=:v7)   # expect O(h²) cap
    sweep8(pbC_v, ρrefC, ps, S; engine=:v8)   # expect ≥O(h⁴) at GL2
end
println("done")
