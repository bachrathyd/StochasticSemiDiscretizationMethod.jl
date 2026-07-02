# Isolate the Mathieu bug: compare
#  (a) my first-moment ρ (deterministic backbone)  vs trusted SDM deterministic ρ
#  (b) my 2nd-moment ρ with noise OFF (should be ρ_det²)
#  (c) trusted SDM 2nd-moment ρ (the target 0.1562)
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__,"moment_engine.jl"))
include(joinpath(@__DIR__,"moment_engine2.jl"))

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1

# my SDDEProblem (Mathieu)
prob = SDDEProblem(2, PER,
    t -> [0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA],
    [(t->TAU, t->[0.0 0.0; Bval 0.0])],
    [ ( t->[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA],
        [t->[0.0 0.0; ALPHA*Bval 0.0]], t->[0.0,0.0] ) ])

# noise-OFF clone
prob0 = SDDEProblem(2, PER, prob.A, prob.delays,
    [ (t->zeros(2,2), [t->zeros(2,2)], t->[0.0,0.0]) ])

# my first-moment ρ
function my_first_rho(S,p)
    Phi,_,_,_ = first_moment_phi(prob,S,p)
    maximum(abs.(eigen(Phi).values))
end

# trusted SDM deterministic ρ (first moment) via package
function sdm_det_rho(q,p)
    AMx=ProportionalMX(t->SMatrix{2,2}(prob.A(t)))
    BMx=DelayMX(TAU, t->SMatrix{2,2}([0.0 0.0; Bval 0.0]))
    cV=Additive(SVector{2}(0.0,0.0))
    # deterministic LDDE (SemiDiscretizationMethod path inside SSDM): use M1 of zero-noise
    αMx=stCoeffMX(1,ProportionalMX(t->SMatrix{2,2}(zeros(2,2))))
    βMx=stCoeffMX(1,DelayMX(TAU,t->SMatrix{2,2}(zeros(2,2))))
    σV=stAdditive(1,Additive(SVector{2}(0.0,0.0)))
    lddep=LDDEProblem(AMx,[BMx],[αMx],[βMx],cV,[σV])
    rst=SSDM.calculateResults(lddep,SemiDiscretization(q,PER/p),TAU;n_steps=p)
    # first-moment spectral radius
    spectralRadiusOfMapping(DiscreteMapping_M1(rst))
end

println("(a) FIRST-moment ρ — my engine vs trusted SDM:")
for p in [40,80,160]
    @printf("  p=%3d  mine GL2=%.8f  SDM q2=%.8f\n", p, my_first_rho(2,p), sdm_det_rho(2,p))
end

println("\n(b) my 2nd-moment ρ NOISE-OFF (should equal ρ_det²):")
for p in [40,80]
    ρ2=second_moment_rho(prob0,2,p); ρ1=my_first_rho(2,p)
    @printf("  p=%3d  ρ2=%.8f  ρ1²=%.8f  diff=%.2e\n", p, ρ2, ρ1^2, abs(ρ2-ρ1^2))
end

println("\n(c) my 2nd-moment WITH noise vs trusted SDM target 0.1562:")
for p in [40,80,160]
    @printf("  p=%3d  mine GL2=%.8f\n", p, second_moment_rho(prob,2,p))
end
