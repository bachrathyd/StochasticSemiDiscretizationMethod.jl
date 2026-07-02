# =============================================================================
# v7 validation ladder — stage 2: THE decisive A/B on the mirror Mathieu
#
# Problem (identical to benchmark_mf_v6.jl / benchmark_mf_complexity.jl):
#   x'' + 2ζx' + (A+ε cos 2πt/P)x = B x(t−τ) + noise,  τ = P = 1
#   multiplicative present α and delayed β noise, both 0.2 (σ additive is
#   irrelevant for ρ of the homogeneous second-moment map).
# v6 measured: slope −2 for ALL GL orders (the cap). v7 hypothesis: the causal
# intra-block fill restores high order.
#
# Reference: SDM q=2 Richardson extrapolation (p=512,1024, order-3 elimination)
# — the archived analysis showed raw SDM q2 values bounce at ~1e-5; Richardson
# is the trusted reference recipe.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))

const Ac=1.0; const εc=0.5; const Bc=0.2; const ζc=0.1
const τc=1.0; const σc=0.1; const αc=0.2; const Pc=1.0

function createStochMathieuProblem()
    AMxfun(t) = @SMatrix [0. 1.; -(Ac + εc*cos(2π*t/Pc)) -2ζc]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τc, @SMatrix [0. 0.; Bc 0.])
    cVec = Additive(2)
    αMx1  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; αc 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τc, @SMatrix [0. 0.; αc 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σc]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

ρ_sdm(p) = spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(
    StochasticSemiDiscretizationMethod.calculateResults(
        createStochMathieuProblem(), SemiDiscretization(2, Pc/p), τc)))

println("Computing SDM q=2 reference (p=512, 1024) ...")
t0=time(); ρ512 = ρ_sdm(512); ρ1024 = ρ_sdm(1024)
ρ_ref = ρ1024 + (ρ1024 - ρ512)/(2^3 - 1)     # order-3 Richardson
@printf("  ρ(512)=%.10f ρ(1024)=%.10f → Richardson ρ_ref=%.10f  (Δ=%.2e, %.0fs)\n",
        ρ512, ρ1024, ρ_ref, abs(ρ1024-ρ512), time()-t0)

pb = Prob(2, Pc, τc,
    t->[0.0 1.0; -(Ac + εc*cos(2π*t/Pc)) -2ζc],
    t->[0.0 0.0; Bc 0.0],
    t->[0.0 0.0; αc 0.0],
    t->[0.0 0.0; αc 0.0])

ps = [6, 8, 12, 16, 24, 32, 48]
results = Dict{Tuple{Symbol,Int},Vector{Float64}}()

open(joinpath(@__DIR__, "out_mirror_order.csv"), "w") do io
    println(io, "mode,S,p,rho,err")
    for mode in (:none, :causal), S in (1, 2, 3)
        errs = Float64[]
        for p in ps
            t1 = time()
            eng = build_v7(pb, S, p)
            ρ = rho_H_krylov(eng; offdiag=mode)
            err = abs(ρ - ρ_ref)
            push!(errs, err)
            @printf("  %-7s GL%d p=%2d ρ=%.10f err=%.3e  (%.1fs)\n", mode, S, p, ρ, err, time()-t1)
            println(io, "$mode,$S,$p,$ρ,$err")
            flush(io); flush(stdout)
        end
        results[(mode,S)] = errs
    end
end

println("\n══ ORDER TABLE (local log-log slopes between consecutive p) ══")
for mode in (:none, :causal), S in (1, 2, 3)
    errs = results[(mode,S)]
    slopes = [ log(errs[i]/errs[i+1]) / log(ps[i+1]/ps[i]) for i in 1:length(ps)-1 ]
    @printf("%-7s GL%d  slopes: %s\n", mode, S,
            join([@sprintf("%5.2f",s) for s in slopes], " "))
end

println("\nVERDICT:")
for S in (2,3)
    sN = results[(:none,S)]; sC = results[(:causal,S)]
    slopeN = log(sN[3]/sN[end]) / log(ps[end]/ps[3])
    slopeC = log(sC[3]/sC[end]) / log(ps[end]/ps[3])
    @printf("  GL%d overall slope  :none %.2f   :causal %.2f   → %s\n",
            S, slopeN, slopeC,
            slopeC > slopeN + 0.7 ? "CAUSAL FILL LIFTS THE ORDER" : "no significant lift")
end
