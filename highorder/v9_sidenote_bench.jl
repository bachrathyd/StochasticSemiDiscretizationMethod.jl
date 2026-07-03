# Paired v9-vs-v8 benchmark for the paper side-note: the same solver call on
# (a) a delayed-P/PD DRIFT control problem with present-state + additive noise
#     (β ≡ 0 → the automatic pruning activates), and
# (b) the hard PD-Mathieu with delayed MULTIPLICATIVE noise (β ≠ 0 → the
#     engine transparently falls back to the full block).
# Reported: persistent covariance DOF (memory) and wall-clock of one ρ(H) solve.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v9.jl"))

vech(W)=W*(W+1)÷2
tmin(f)=minimum((@elapsed(f()) for _ in 1:3))

# (a) β≡0: PD-drift control, present + additive noise
Aa(t)=[0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
Ba(t)=[0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.12*(1+0.4cos(2π*t))]
αa(t)=[0.0 0.0; 0.30 0.0]
βa(t)=[0.0 0.0; 0.0 0.0]
σa(t)=reshape([0.0,0.3],2,1)
pbA = Prob(2,1.0,1.0,Aa,Ba,αa,βa,σa)

# (b) β≠0: hard PD-Mathieu (delayed multiplicative noise)
Ab(t)=[0.0 1.0; -(1.0+0.8cos(2π*t)) -0.1]
Bb(t)=[0.0 0.0; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
αb(t)=[0.0 0.0; 0.5 0.0]
βb(t)=[0.0 0.0; 0.35 0.0]
pbB = Prob(2,1.0,1.0,Ab,Bb,αb,βb)

for (tag,pb) in (("(a) β≡0, delayed-PD drift", pbA), ("(b) β≠0, hard PD-Mathieu", pbB))
    println("── ", tag, " ──")
    for (S,p) in ((3,16),(4,16))
        e8 = build_v8m(pb,S,p)
        e9 = build_v9m(pb,S,p)          # auto: pruned for (a), fallback for (b)
        t8 = tmin(()->rho_Hlin_krylov_v8m(e8))
        t9 = tmin(()->rho_H_krylov_v9m(e9))
        pruned = haskey(e9,:engine) && e9.engine==:v9
        @printf("S=%d p=%2d | full block: D=%6d %6.2fs | auto (%s): D=%6d %6.2fs | mem %.2fx  time %.2fx\n",
                S, p, vech(e8.W), t8, pruned ? "pruned" : "fallback=v8", vech(e9.W), t9,
                vech(e8.W)/vech(e9.W), t8/t9)
    end
end
