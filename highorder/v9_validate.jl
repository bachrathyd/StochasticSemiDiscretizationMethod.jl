# Validation + benchmark for the DOF-pruned v9 engine (cov_colloc_v9.jl).
# Gates (β≡0 problem — delayed PD-DRIFT + present-state mult. noise + additive):
#   (1) homogeneous ρ(H): v9 reproduces the corrected v8 (rho_Hlin) to ~1e-13
#   (2) stationary fixpoint Var(q): v9 reproduces v8 to ~1e-13
#   (3) memory (vech-DOF) reduction ≈ (2S+2)^2/(S+2)^2 and wall-clock speedup
#   (4) β≠0 problem transparently falls back to the full v8 engine
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using LinearAlgebra, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v9.jl"))

Afun(t)=[0.0 1.0; -(1.0+0.5cos(2π*t)) -0.4]
Bfun(t)=[0.0 0.0; 0.20*(1+0.3cos(2π*t)) 0.12*(1+0.4cos(2π*t))]   # delayed PD drift
αfun(t)=[0.0 0.0; 0.30 0.0]                                       # present-state noise
βfun(t)=[0.0 0.0; 0.0  0.0]                                       # NO delayed noise ⇒ prune
σfun(t)=reshape([0.0, 0.3], 2, 1)
pb = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun, σfun)
@assert _no_delay_noise(pb)

println("── GATE 1&2: v9 vs v8 (homogeneous ρ, stationary Var) ──")
@printf("%4s | %16s %16s %9s | %14s %14s %9s\n","p","ρ_Hlin v8","ρ v9","relΔ","Var v8","Var v9","relΔ")
for p in (6,10,16,24)
    r8=rho_Hlin_krylov_v8m(build_v8m(pb,3,p)); r9=rho_H_krylov_v9m(build_v9m(pb,3,p))
    v8=fixPoint_v8m(build_v8m(pb,3,p))[1,1];   v9=fixPoint_v9m(build_v9m(pb,3,p))[1,1]
    @printf("%4d | %16.12f %16.12f %9.1e | %14.11f %14.11f %9.1e\n",
            p, r8, r9, abs(r8-r9)/abs(r8), v8, v9, abs(v8-v9)/abs(v8))
end

println("\n── GATE 3: memory + speed across S (p=16) ──")
vech(W)=W*(W+1)÷2; tmin(f)=minimum((@elapsed(f()) for _ in 1:2))
@printf("%3s | %7s %7s %7s | %8s %8s %7s\n","S","W v8","W v9","mem×","t v8","t v9","spd×")
for S in (2,3,4,5)
    e8=build_v8m(pb,S,16); e9=build_v9m(pb,S,16)
    t8=tmin(()->rho_Hlin_krylov_v8m(e8)); t9=tmin(()->rho_H_krylov_v9m(e9))
    @printf("%3d | %7d %7d %6.2fx | %7.2fs %7.2fs %5.2fx\n",
            S, e8.W, e9.W, vech(e8.W)/vech(e9.W), t8, t9, t8/t9)
end

println("\n── GATE 4: β≠0 falls back to v8 ──")
pb2=Prob(2,1.0,1.0,Afun,Bfun,αfun,(t->[0.0 0.0; 0.25 0.0]),σfun)
e=build_v9m(pb2,3,8)
@printf("β≠0: fell back to v8 = %s (BSIZE=%d, expect 2S+2 → %d)\n",
        !haskey(e,:engine), e.BSIZE, (2*3+2)*2)
