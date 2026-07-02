using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using LinearAlgebra, Printf
include(joinpath(@__DIR__,"moment_engine.jl"))
include(joinpath(@__DIR__,"moment_engine3.jl"))

# ---- Mathieu ----
const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1
prob = SDDEProblem(2, PER,
    t -> [0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA],
    [(t->TAU, t->[0.0 0.0; Bval 0.0])],
    [ ( t->[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA],
        [t->[0.0 0.0; ALPHA*Bval 0.0]], t->[0.0,0.0] ) ])
prob0 = SDDEProblem(2, PER, prob.A, prob.delays,
    [ (t->zeros(2,2), [t->zeros(2,2)], t->[0.0,0.0]) ])

function myfirst(S,p)
    Phi,_,_,_ = first_moment_phi(prob,S,p)
    maximum(abs.(eigen(Phi).values))
end

println("NOISE-OFF: cov-engine ρ2 should = ρ(Φ)²")
for p in [40,80]
    ρ1=myfirst(2,p); ρ2=second_moment_rho_cov(prob0,2,p)
    @printf("  p=%3d  ρ1²=%.8f  ρ2=%.8f  diff=%.2e\n", p, ρ1^2, ρ2, abs(ρ2-ρ1^2))
end

println("\nWITH noise: cov-engine ρ2 vs trusted SDM 0.156228:")
for S in 1:2, p in [40,80,160]
    @printf("  GL%d p=%3d  ρ2=%.8f\n", S, p, second_moment_rho_cov(prob,S,p))
end

# ---- Hayes scalar cross-check ----
println("\nHayes scalar (ref 0.57022372583):")
hayes=SDDEProblem(1,1.0,t->reshape([-1.0],1,1),[(t->1.0,t->reshape([-0.4],1,1))],
    [(t->reshape([0.0],1,1),[t->reshape([0.3],1,1)],t->[0.0])])
for S in 1:2, p in [8,16,32]
    @printf("  GL%d p=%2d  ρ2=%.10f\n", S,p,second_moment_rho_cov(hayes,S,p))
end
