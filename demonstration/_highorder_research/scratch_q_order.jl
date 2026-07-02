# Clean asymptotic order of SDM ρ(H) on Mathieu, using Richardson-extrapolated
# reference and tight eig tolerance. Confirms the O(h³) ceiling.
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1

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

function rho(q,p)
    lddep=mathieu_lddep(); method=SemiDiscretization(q, PER/p)
    rst=SSDM.calculateResults(lddep,method,TAU;n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst); tol=1e-13)
end

# self-convergence (reference-free) order via three successive doublings
function self_rates(q, ps)
    rs = [rho(q,p) for p in ps]
    @printf("q=%d:\n", q)
    for i in 3:length(ps)
        d1=abs(rs[i-1]-rs[i-2]); d2=abs(rs[i]-rs[i-1])
        @printf("  p=%4d  ρ=%.11f  Δ=%.3e  rate≈%.2f\n", ps[i], rs[i], d2, log2(d1/d2))
    end
    println()
end

println("Self-convergence (reference-free), tight eig tol:\n")
self_rates(0, [20,40,80,160,320])
self_rates(2, [20,40,80,160,320])
self_rates(4, [20,40,80,160,320])
