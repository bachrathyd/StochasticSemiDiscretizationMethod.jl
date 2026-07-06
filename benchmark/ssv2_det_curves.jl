# Deterministic stability boundaries for the 2-DOF SSV milling chart, computed
# with the FAST deterministic SemiDiscretizationMethod.jl in its Left/Right
# (LR) form: instead of multiplying p step matrices into a dense monodromy, it
# assembles two SPARSE matrices ΦL, ΦR and takes the largest |λ| of the
# generalized problem λ ΦL x = ΦR x. We drive that through KrylovKit
# (thread-safe, unlike Arpack `eigs`) as  x → ΦL⁻¹(ΦR x)  with a cached LU, so
# MDBM can evaluate with doThreadprecomp=true. This is the first-moment ρ(Φ),
# NOT the second-moment ρ(H) — hence seconds, not hours.
#
#   julia ssv2_det_curves.jl cs_det        # constant speed (no SSV), N_ITER=2
#   julia ssv2_det_curves.jl cs_det 5      # ... with 5 MDBM refinements
#   julia ssv2_det_curves.jl ssv_det 5     # SSV deterministic boundary  ρ(Φ)=1
# 2nd arg (optional) overrides N_ITER (MDBM refinement iterations).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using SemiDiscretizationMethod                     # deterministic package ONLY
using MDBM, StaticArrays, LinearAlgebra, SparseArrays, KrylovKit, Printf
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const R_RES=24; const NAT_RES=30
const ΩLO=0.125; const ΩHI=1.5; const WHI=4.0
const N_ITER = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2   # MDBM refinements (2nd arg)

φ0fun(t, Ω0, Tssv, rva) = rva==0 ? Ω0*t :
    Ω0*t - (Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)

function Hdir(t, Ω0, Tssv, rva)
    φ0 = φ0fun(t, Ω0, Tssv, rva)
    h11=0.0;h12=0.0;h21=0.0;h22=0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        φ ≤ PHI_EX || continue
        s,c = sincos(φ)
        a1=(c+Kr*s); a2=(s-Kr*c)
        h11+=a1*s;h12+=a1*c;h21+=-a2*s;h22+=-a2*c
    end
    @SMatrix [h11 h12; h21 h22]
end

const Z2 = @SMatrix zeros(2,2); const I2 = SMatrix{2,2}(1.0I)

# largest |λ| of the deterministic 2-DOF milling monodromy (boundary at ρ=1),
# via the sparse LR generalized eigenproblem + KrylovKit.
function rho_det(Ω0::Float64, w::Float64; rva::Float64)::Float64
    Tssv = NT*2π/Ω0                              # SSV modulation reference for H, τ
    T    = rva == 0.0 ? (2π/N_TEETH)/Ω0 : Tssv   # cs: one tooth pass; SSV: modulation period
    τmax = (2π/N_TEETH)/(Ω0*(1-rva))
    # ONE delay closure — constant when rva=0 (sin term ×0), so no
    # Union{Float64,Function} to poison the DelayMX / LDDEProblem types.
    delay(t) = (2π/N_TEETH)/(Ω0*(1+rva*sin(2π*t/Tssv)))
    Hf(t) = Hdir(t, Ω0, Tssv, rva)
    # 4×4 companion matrices built column-major from scalars — NO block hvcat,
    # so no runtime typed_hvcat/hvcat_internal and no per-call heap Array.
    Af(t) = (H = Hf(t); SMatrix{4,4,Float64}(
        0.0, 0.0, -1-w*H[1,1], -w*H[2,1],        # column 1
        0.0, 0.0, -w*H[1,2], -1-w*H[2,2],        # column 2
        1.0, 0.0, -2ζ, 0.0,                      # column 3
        0.0, 1.0, 0.0, -2ζ))                     # column 4
    Bf(t) = (H = Hf(t); SMatrix{4,4,Float64}(
        0.0, 0.0, w*H[1,1], w*H[2,1],
        0.0, 0.0, w*H[1,2], w*H[2,2],
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0))
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(delay, Bf)], Additive(zeros(4)))
    Δt = min(τmax/R_RES, 2π/NAT_RES); nst = max(1, Int(round(T/Δt))); Δt = T/nst
    m = DiscreteMapping_LR(lddep, SemiDiscretization(2, Δt), τmax;
                           n_steps=nst, calculate_additive=false)   # n_steps needed for SSV (T≠τmax)
    F = lu(m.LmappingMX); R = m.RmappingMX
    vals, = KrylovKit.eigsolve(x -> F \ (R * x), ones(size(R,1)), 1, :LM;
                               tol=1e-10, maxiter=400, krylovdim=10)
    abs(vals[1])
end

name = isempty(ARGS) ? "ssv_det" : ARGS[1]
rva  = name == "cs_det" ? 0.0 : RVA
f(ξ, w) = (rho_det(10.0^ξ, w; rva=rva) - 1.0)::Float64

ax = [Axis(range(log10(ΩLO), log10(ΩHI), length=10), :ξ),
      Axis(range(0.0, WHI, length=12), :w)]        # doubled along depth of cut
prob = MDBM_Problem(f, ax)
t = @elapsed begin                                  # total time: both phases
    solve!(prob, 6, verbosity=4, doThreadprecomp=true,interpolationorder=0,checkneighbourNum=5)
    solve!(prob, 0, verbosity=3, doThreadprecomp=true,interpolationorder=1,checkneighbourNum=0)
end
pts = getinterpolatedsolution(prob)

# #points where the function foo was evaluated
# x_eval,y_eval=getevaluatedpoints(prob)
# #interpolated points of the solution (approximately where foo(x,y) == 0 and c(x,y)>0)
# x_sol,y_sol=getinterpolatedsolution(prob)
# using Plots; gr()
# scatter(x_eval,y_eval,s=2)
# scatter!(x_sol,y_sol,s=2)


open(joinpath(@__DIR__, "ssv2_mdbm_$(name).csv"), "w") do io
    println(io, "curve,xi,w")
    for k in eachindex(pts[1])
        @printf(io, "%s,%.6f,%.6f\n", name, pts[1][k], pts[2][k])
    end
end
@printf("MDBM %-8s (deterministic LR+KrylovKit): %d pts (%.0fs)\n", name, length(pts[1]), t)
println("done")
