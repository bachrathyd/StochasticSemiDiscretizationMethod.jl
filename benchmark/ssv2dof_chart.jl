# =============================================================================
# 2-DOF SSV milling stability + quality chart (journal figure, final model).
# Symmetric-tool X–Y model with full directional cross-coupling (up-milling,
# cut time = 25% of the tooth pitch, z = 2):
#   q̈ + 2ζ q̇ + q = −w H̄(t) [q − q(t−τ(t))] (1 + σc Ẇ1) + σa [Ẇ2; Ẇ3]
# with the standard directional matrix built from the tangential/normal force
# decomposition (chip thickness h_c = Δx sinφ + Δy cosφ; F_t = K_t a h_c,
# F_r = K_r F_t):
#   H(φ) = g(φ) · [ (cosφ + K_r sinφ) sinφ   (cosφ + K_r sinφ) cosφ ;
#                  −(sinφ − K_r cosφ) sinφ  −(sinφ − K_r cosφ) cosφ ]
# (the (1,1) entry reduces to the 1-DOF factor used previously).
# SSV: Ω(t) = Ω0(1 + RVA sin(2π t/T_SSV)), RVA = 0.25, T_SSV = 10 revolutions.
# Axes: Ω0 ∈ [0.125, 1.5] log axis (plain labels), w ∈ [0, 4].
# Layers: BF colormap of log10 Var(x) + 4 MDBM boundary curves + shading.
# Run modes:  julia ssv2dof_chart.jl bf     → BF colormap grid only
#             julia ssv2dof_chart.jl mdbm   → MDBM curves only (needs nothing)
#             julia ssv2dof_chart.jl        → both
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, DelimitedFiles
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX = π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10
const VARLIM=0.25
const R_RES=24; const NAT_RES=30
const ΩLO=0.125; const ΩHI=1.5; const WHI=4.0
const BF_NX=96; const BF_NW=64
const N_ITER=4                 # validated smoothness at 4 iters (1-DOF test)
const CSVBF = joinpath(@__DIR__, "ssv2_chart_bf.csv")
const CSVM  = joinpath(@__DIR__, "ssv2_chart_mdbm.csv")

φ0fun(t, Ω0, Tssv, rva) = rva==0 ? Ω0*t :
    Ω0*t - (Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)

function Hdir(t, Ω0, Tssv, rva)
    φ0 = φ0fun(t, Ω0, Tssv, rva)
    H = @MMatrix zeros(2,2)
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        φ ≤ PHI_EX || continue
        s,c = sincos(φ)
        a1 = (c + Kr*s); a2 = (s - Kr*c)
        H[1,1] +=  a1*s;  H[1,2] +=  a1*c
        H[2,1] += -a2*s;  H[2,2] += -a2*c
    end
    SMatrix{2,2}(H)
end

# d=4 state [x, y, ẋ, ẏ]; returns (ρ, Var(x)-or-NaN).
# Type-stable kernels: 4×4 SMatrix built column-major from scalars (no runtime
# block hvcat → no per-call heap Array), single always-Function delay closure.
function solve_point(Ω0::Float64, w::Float64; rva::Float64=RVA, σcv::Float64=σc,
                     σav::Float64=σa, want_var::Bool=false)
    Tssv = NT * 2π/Ω0
    w ≤ 0 && return (exp(-2ζ*Tssv), σav^2/(4ζ))
    τf(t) = (2π/N_TEETH)/(Ω0*(1+rva*sin(2π*t/Tssv)))
    τmax  = (2π/N_TEETH)/(Ω0*(1-rva))
    Hf(t) = Hdir(t, Ω0, Tssv, rva)
    Af(t) = (H = Hf(t); SMatrix{4,4,Float64}(
        0.0, 0.0, -1-w*H[1,1], -w*H[2,1],
        0.0, 0.0, -w*H[1,2], -1-w*H[2,2],
        1.0, 0.0, -2ζ, 0.0,
        0.0, 1.0, 0.0, -2ζ))
    Bf(t) = (H = Hf(t); SMatrix{4,4,Float64}(
        0.0, 0.0, w*H[1,1], w*H[2,1],
        0.0, 0.0, w*H[1,2], w*H[2,2],
        0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0))
    af(t) = (H = Hf(t); SMatrix{4,4,Float64}(
        0.0, 0.0, -σcv*w*H[1,1], -σcv*w*H[2,1],
        0.0, 0.0, -σcv*w*H[1,2], -σcv*w*H[2,2],
        0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0))
    bf(t) = (H = Hf(t); SMatrix{4,4,Float64}(
        0.0, 0.0, σcv*w*H[1,1], σcv*w*H[2,1],
        0.0, 0.0, σcv*w*H[1,2], σcv*w*H[2,2],
        0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0))
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
        [stCoeffMX(1,ProportionalMX(af))],
        [stCoeffMX(1,DelayMX(τf, bf))],
        Additive(4),
        [stAdditive(2,Additive(@SVector [0.,0.,σav,0.]))])   # broadband force noise, feed direction
    Δt = min(τmax/R_RES, 2π/NAT_RES); nst=max(1,Int(round(Tssv/Δt))); Δt=Tssv/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                                n_steps=nst, calculate_additive=want_var)
    ρ = spectralRadiusOfMapping_MF_factored(rst)
    v = NaN
    if want_var && ρ < 0.999
        try; v = fixPointOfMapping_MF_factored(rst)[1]; catch; end
    end
    (ρ, v)
end

mode = isempty(ARGS) ? "both" : ARGS[1]

if mode in ("bf","both")
    ξs  = collect(range(log10(ΩLO), log10(ΩHI), length=BF_NX))
    Ω0s = 10 .^ ξs
    ws  = collect(range(0.0, WHI, length=BF_NW))
    t0=time()
    open(CSVBF,"w") do io
        println(io,"Omega0,w,rho,var")
        for (j,Ω) in enumerate(Ω0s)
            rr=zeros(length(ws)); vv=fill(NaN,length(ws))
            t=@elapsed Threads.@threads for i in eachindex(ws)
                ρ,v = solve_point(Ω, ws[i]; want_var=true); rr[i]=ρ; vv[i]=v
            end
            for (i,w) in enumerate(ws); @printf(io,"%.6f,%.5f,%.8f,%.8e\n",Ω,w,rr[i],vv[i]); end
            flush(io); @printf("BF Ω0=%.3f (%.1fs) [%d/%d]\n",Ω,t,j,length(Ω0s)); flush(stdout)
        end
    end
    @printf("2DOF BF TOTAL: %.1f s\n", time()-t0); flush(stdout)
end

CURVEDEFS = Dict{String,Function}(
    "cs_det"  => (ξ,w) -> solve_point(10.0^ξ, w; rva=0.0, σcv=0.0, σav=0.0)[1] - 1.0,
    "ssv_det" => (ξ,w) -> solve_point(10.0^ξ, w; σcv=0.0, σav=0.0)[1] - 1.0,
    "ssv_sto" => (ξ,w) -> solve_point(10.0^ξ, w)[1] - 1.0,
    "ssv_var" => (ξ,w) -> begin
        ρ,v = solve_point(10.0^ξ, w; want_var=true)
        (ρ ≥ 0.999 || isnan(v)) ? 1.0 : (v - VARLIM)
    end)

if haskey(CURVEDEFS, mode)          # single-curve mode: run curves as parallel processes
    using MDBM
    name = mode; f = CURVEDEFS[name]
    niter = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : N_ITER
    local pts
    t=@elapsed begin
        ax = [Axis(range(log10(ΩLO), log10(ΩHI), length=10), :ξ),
              Axis(range(0.0, WHI, length=6), :w)]
        prob = MDBM_Problem(f, ax)
        # two-phase: cheap zeroth-order bracketing search, then one linear-
        # interpolation recompute of the solution points on the final cubes
        solve!(prob, niter, verbosity=1, doThreadprecomp=true,
               interpolationorder=0, checkneighbourNum=5)
        solve!(prob, 0, verbosity=1, doThreadprecomp=true,
               interpolationorder=1, checkneighbourNum=0)
        pts = getinterpolatedsolution(prob)
        open(joinpath(@__DIR__,"ssv2_mdbm_$(name).csv"),"w") do io
            println(io,"curve,xi,w")
            for k in eachindex(pts[1])
                @printf(io,"%s,%.6f,%.6f\n", name, pts[1][k], pts[2][k])
            end
        end
    end
    @printf("MDBM %-8s: %d pts (%.0fs)\n", name, length(pts[1]), t); flush(stdout)
end
println("done")
