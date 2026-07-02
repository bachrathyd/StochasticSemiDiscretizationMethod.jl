# Reconcile moment-DDE (A0/A1) vs trusted SDM on scalar Hayes — find which is right.
# Test 1: NOISE OFF (β=0) → both must give ρ(Φ_det)². Isolates the β² Itô term.
# Test 2: with β, compare to a DIRECT covariance-recurrence ground truth (brute force
#         step-by-step E[x²] via fine Euler-Maruyama-free exact moment propagation).
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__,"..","moment_colloc.jl"))

const A=-1.0; const B=-0.4; const TAU=1.0

function sdm_rho(β,q,p)
    AMx=ProportionalMX(t->SMatrix{1,1}(A)); BMx=DelayMX(TAU,SMatrix{1,1}(B))
    αMx=stCoeffMX(1,ProportionalMX(t->SMatrix{1,1}(0.0)))
    βMx=stCoeffMX(1,DelayMX(TAU,SMatrix{1,1}(β)))
    cV=Additive(SVector{1}(0.0)); σV=stAdditive(1,Additive(SVector{1}(0.0)))
    lddep=LDDEProblem(AMx,[BMx],[αMx],[βMx],cV,[σV])
    rst=SSDM.calculateResults(lddep,SemiDiscretization(q,TAU/p),TAU;n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst); tol=1e-13)
end

hi_rho(β,S,p) = rho_moment([2A 2B; 0.0 A], [β^2 0.0; B 0.0], S, p, TAU)

# deterministic first-moment ρ via package M1 (for ρ²)
function sdm_det1(q,p)
    AMx=ProportionalMX(t->SMatrix{1,1}(A)); BMx=DelayMX(TAU,SMatrix{1,1}(B))
    αMx=stCoeffMX(1,ProportionalMX(t->SMatrix{1,1}(0.0)))
    βMx=stCoeffMX(1,DelayMX(TAU,SMatrix{1,1}(0.0)))
    cV=Additive(SVector{1}(0.0)); σV=stAdditive(1,Additive(SVector{1}(0.0)))
    lddep=LDDEProblem(AMx,[BMx],[αMx],[βMx],cV,[σV])
    rst=SSDM.calculateResults(lddep,SemiDiscretization(q,TAU/p),TAU;n_steps=p)
    spectralRadiusOfMapping(DiscreteMapping_M1(rst))
end

println("TEST 1: NOISE OFF (β=0) — both must give ρ(Φ_det)²")
for p in [16,32,64]
    ρ1=sdm_det1(2,p)
    @printf("  p=%3d  ρ(Φ_det)²=%.10f   SDM ρ2=%.10f   moment-DDE ρ2=%.10f\n",
        p, ρ1^2, sdm_rho(0.0,2,p), hi_rho(0.0,3,p))
end

println("\nTEST 2: WITH β=0.3 — SDM (fine) vs moment-DDE")
@printf("  SDM q=2 p=256 = %.10f\n", sdm_rho(0.3,2,256))
@printf("  SDM q=2 p=512 = %.10f\n", sdm_rho(0.3,2,512))
@printf("  moment-DDE GL3 p=64 = %.10f\n", hi_rho(0.3,3,64))
@printf("  moment-DDE GL3 p=128= %.10f\n", hi_rho(0.3,3,128))
