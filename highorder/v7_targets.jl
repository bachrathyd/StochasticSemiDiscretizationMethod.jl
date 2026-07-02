# =============================================================================
# v7 validation ladder — stage 3: target problems with trusted references
#
# T-1 Hayes delayed-noise:  dx = (A x + B x_{t-1})dt + β x_{t-1} dW
#     A=-1, B=-0.4, β=0.3, T=τ=1. Trusted value ≈ 0.148 (SDM; the wrong
#     engines gave 0.2084 / 0.5702 — this is the value trap test).
# T-2 Critical stochastic delayed Mathieu (the original research target):
#     x'' + 2ζx' + (A+ε cos(t/2))x = B x(t−2π) + mult. noise α
#     A=3, ε=2, B=0.5, ζ=0.1, α=0.1, P=4π, τ=2π. Converged refs from the
#     archive: v6 GL4 p120 → 0.15622747; SDM-q2-Richardson → 0.15622870.
# Both: value against SDM q=2 Richardson computed fresh, order table for
# :none vs :causal (B≠0 in both, so the v6 cap should show in :none).
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))

function sweep(io, tag, pb, ρ_ref, ps; modes=(:none,:causal), Ss=(2,3))
    for mode in modes, S in Ss
        errs=Float64[]
        for p in ps
            eng = build_v7(pb, S, p)
            ρ = rho_H_krylov(eng; offdiag=mode)
            err = abs(ρ - ρ_ref)
            push!(errs, err)
            @printf("  %-7s GL%d p=%3d ρ=%.10f err=%.3e\n", mode, S, p, ρ, err)
            println(io, "$tag,$mode,$S,$p,$ρ,$err"); flush(io); flush(stdout)
        end
        slopes=[log(errs[i]/errs[i+1])/log(ps[i+1]/ps[i]) for i in 1:length(ps)-1]
        @printf("  %-7s GL%d slopes: %s\n", mode, S,
                join([@sprintf("%5.2f",s) for s in slopes]," "))
    end
end

io = open(joinpath(@__DIR__, "out_targets.csv"), "w")
println(io, "problem,mode,S,p,rho,err")

# ───────────────────────── T-1: Hayes ─────────────────────────
println("══ T-1 Hayes delayed noise (A=-1, B=-0.4, β=0.3) ══")
function hayesProblem()
    AMx  = ProportionalMX(fill(-1.0,1,1))
    BMx1 = DelayMX(1.0, fill(-0.4,1,1))
    cVec = Additive(1)
    αMx1  = stCoeffMX(1, ProportionalMX(zeros(1,1)))
    βMx11 = stCoeffMX(1, DelayMX(1.0, fill(0.3,1,1)))
    σVec  = stAdditive(1, Additive(ones(1)))
    LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end
ρh(p) = spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(
    StochasticSemiDiscretizationMethod.calculateResults(
        hayesProblem(), SemiDiscretization(2, 1.0/p), 1.0)))
ρh512=ρh(512); ρh1024=ρh(1024); ρ_ref_h = ρh1024 + (ρh1024-ρh512)/7
@printf("SDM q2: ρ(512)=%.10f ρ(1024)=%.10f → Richardson %.10f\n", ρh512, ρh1024, ρ_ref_h)
pb_h = Prob(1, 1.0, 1.0,
    t->fill(-1.0,1,1), t->fill(-0.4,1,1), t->zeros(1,1), t->fill(0.3,1,1))
sweep(io, "hayes", pb_h, ρ_ref_h, [4,6,8,12,16,24,32])

# ───────────────────────── T-2: critical Mathieu ─────────────────────────
println("══ T-2 critical stochastic Mathieu (A=3, ε=2, B=0.5, ζ=0.1, α=0.1) ══")
const P2=4π; const τ2=2π
function mathieuProblem()
    AMxfun(t) = @SMatrix [0. 1.; -(3.0 + 2.0*cos(0.5*t)) -0.2]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ2, @SMatrix [0. 0.; 0.5 0.])
    cVec = Additive(2)
    αMxfun(t) = @SMatrix [0. 0.; -0.1*(3.0 + 2.0*cos(0.5*t)) -0.1*0.2]
    αMx1  = stCoeffMX(1, ProportionalMX(αMxfun))
    βMx11 = stCoeffMX(1, DelayMX(τ2, @SMatrix [0. 0.; 0.1*0.5 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., 0.]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end
ρm(p) = spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(
    StochasticSemiDiscretizationMethod.calculateResults(
        mathieuProblem(), SemiDiscretization(2, P2/p), τ2, n_steps=p)))
ρm512=ρm(512); ρm1024=ρm(1024); ρ_ref_m = ρm1024 + (ρm1024-ρm512)/7
@printf("SDM q2: ρ(512)=%.10f ρ(1024)=%.10f → Richardson %.10f\n", ρm512, ρm1024, ρ_ref_m)
@printf("archived refs: v6 GL4 p120 = 0.15622747, old SDM-q2-Richardson = 0.15622870\n")
pb_m = Prob(2, P2, τ2,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])
sweep(io, "mathieu_crit", pb_m, ρ_ref_m, [8,12,16,24,32,48])

close(io)
println("done — CSV → highorder/out_targets.csv")
