# Resolve the asymptotic order of ρ(H) on the delayed stochastic Mathieu.
# Use successive-difference (self-convergence) rates that need NO external reference:
#   rate_p = log2( |ρ(p)-ρ(p/2)| / |ρ(2p)-ρ(p)| )  → the true convergence order.
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

function sdm_rho(order,p)
    lddep=mathieu_lddep(); method=SemiDiscretization(order, PER/p)
    rst=SSDM.calculateResults(lddep,method,TAU;n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst))
end

# self-convergence: rate from three successive halvings (no external ref).
for order in [1,2]
    @printf("SDM order %d — self-convergence rate of ρ(H):\n", order)
    ps = [20,40,80,160,320,640,1280]
    rhos = [sdm_rho(order,p) for p in ps]
    for i in 3:length(ps)
        d1 = abs(rhos[i-1]-rhos[i-2])
        d2 = abs(rhos[i]-rhos[i-1])
        rate = log2(d1/d2)
        @printf("  p=%4d  ρ=%.10f  Δ=%.2e  rate≈%.2f\n", ps[i], rhos[i], d2, rate)
    end
    println()
end
