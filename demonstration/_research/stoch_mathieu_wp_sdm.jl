# Trusted SDM (order 2) work-precision emitter for the stochastic delayed Mathieu
# equation — same parameter point and same ρ_ref as the IRK script in MFCM.
# Produces stoch_mathieu_wp_sdm.csv (method,order,p,rho,cputime,abserr).
#
# Usage: julia stoch_mathieu_wp_sdm.jl <rho_ref>
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1
const RHO_REF = 0.1562208339   # trusted SDM order-2 Richardson (shared with IRK script)

function mathieu_lddep()
    AMxfun(t) = @SMatrix [0.0 1.0; -(Aval + EPS*cos(0.5*t)) -2ZETA]
    AMx  = ProportionalMX(AMxfun)
    BMx  = DelayMX(TAU, @SMatrix [0.0 0.0; Bval 0.0])
    af(t) = @SMatrix [0.0 0.0; -ALPHA*(Aval + EPS*cos(0.5*t)) -ALPHA*2ZETA]
    bf(t) = @SMatrix [0.0 0.0; ALPHA*Bval 0.0]
    αMx = stCoeffMX(1, ProportionalMX(af))
    βMx = stCoeffMX(1, DelayMX(TAU, bf))
    cV  = Additive(@SVector [0.0, 0.0])
    σV  = stAdditive(1, Additive(@SVector [0.0, 0.0]))
    LDDEProblem(AMx, [BMx], [αMx], [βMx], cV, [σV])
end

function sdm_rho(p::Int)
    lddep = mathieu_lddep()
    method = SemiDiscretization(2, PER / p)        # order-2 SDM, p steps per period
    rst = SSDM.calculateResults(lddep, method, TAU; n_steps = p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst))
end

function timed_rho(p; nrep = 3)
    sdm_rho(p)                                       # warm up
    best = Inf; rho = 0.0
    for _ in 1:nrep
        t0 = time_ns(); rho = sdm_rho(p); best = min(best, (time_ns() - t0) / 1e9)
    end
    return (rho, best)
end

rho_ref = RHO_REF
p_list = [10, 20, 40, 80, 160, 320, 640]
open(joinpath(@__DIR__, "stoch_mathieu_wp_sdm.csv"), "w") do io
    println(io, "method,order,p,rho,cputime,abserr")
    @printf("  %-6s %-5s %-6s %-14s %-12s %-12s\n", "method", "ord", "p", "rho", "time[s]", "abs.err")
    for p in p_list
        rho, t = timed_rho(p)
        err = abs(rho - rho_ref)
        @printf(io, "SDM,2,%d,%.12g,%.12g,%.12g\n", p, rho, t, err)
        @printf("  %-6s %-5d %-6d %-14.9f %-12.4f %-12.3e\n", "SDM", 2, p, rho, t, err)
    end
end
println("Wrote stoch_mathieu_wp_sdm.csv")
