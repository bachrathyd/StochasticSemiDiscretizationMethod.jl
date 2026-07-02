# Dimension-INDEPENDENT validation of the covariance-propagation engine.
# Critical structural check: with noise OFF, ρ(H) must equal ρ(Φ)² for EVERY d.
# Then a zaj-on check on a d=3 system (no special analytic degeneracy like d=1 Hayes).
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using LinearAlgebra, Printf, Random
include(joinpath(@__DIR__,"moment_engine.jl"))
include(joinpath(@__DIR__,"moment_engine3.jl"))

# random stable-ish linear DDE of dimension d, single delay τ, with multiplicative noise.
function rand_prob(d; τ=1.0, T=1.0, withnoise=true, seed=1)
    rng=MersenneTwister(seed)
    A = -2.0*Matrix(I,d,d) .+ 0.3*randn(rng,d,d)
    B = 0.2*randn(rng,d,d)
    α = withnoise ? 0.15*randn(rng,d,d) : zeros(d,d)
    β = withnoise ? 0.1*randn(rng,d,d)  : zeros(d,d)
    SDDEProblem(d, T, t->A, [(t->τ, t->B)], [(t->α, [t->β], t->zeros(d))])
end

function firstrho(prob,S,p)
    Phi,_,_,_=first_moment_phi(prob,S,p); maximum(abs.(eigen(Phi).values))
end

println("STRUCTURAL CHECK: noise-off ρ2 must = ρ(Φ)² for all d")
for d in 1:4
    prob0=rand_prob(d; withnoise=false)
    p=24; S=2
    ρ1=firstrho(prob0,S,p); ρ2=second_moment_rho_cov(prob0,S,p)
    @printf("  d=%d:  ρ1²=%.10f  ρ2=%.10f  diff=%.2e  %s\n",
            d, ρ1^2, ρ2, abs(ρ2-ρ1^2), abs(ρ2-ρ1^2)<1e-8 ? "OK" : "*** MISMATCH ***")
end

println("\nORDER CHECK (d=3, with multiplicative noise) — self-convergence rate:")
prob3=rand_prob(3; withnoise=true, seed=7)
function order_table(prob, ps)
    for S in 1:3
        @printf("GL(%d):\n",S)
        rs=[second_moment_rho_cov(prob,S,p) for p in ps]
        for i in 3:length(ps)
            d1=abs(rs[i-1]-rs[i-2]); d2=abs(rs[i]-rs[i-1])
            @printf("  p=%2d  ρ2=%.10f  Δ=%.2e  rate≈%.2f\n", ps[i], rs[i], d2, log2(d1/d2))
        end
    end
end
order_table(prob3, [8,16,32,64])
