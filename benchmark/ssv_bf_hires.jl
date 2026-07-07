# High-resolution brute-force colormap grid for the final SSV chart
# (96 log-spaced Ω0 columns × 64 w rows; same model as ssv_milling_chart_mdbm.jl).
# Runs independently of the MDBM boundary-curve computation.
# Output: benchmark/ssv_chart_bf_hires.csv
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX = π/4; const KtKn=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10
const R_RES=24; const NAT_RES=30
const ΩLO=0.125; const ΩHI=1.5; const WHI=4.0
const NX=96; const NW=64

function hfun(t, Ω0, Tssv, rva)
    φ0 = rva==0 ? Ω0*t : Ω0*t - (Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
    hsum = 0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        (φ ≤ PHI_EX) && (hsum += sin(φ)*(cos(φ) + KtKn*sin(φ)))
    end
    hsum
end

function solve_point(Ω0, w)
    Tssv = NT * 2π/Ω0
    w ≤ 0 && return (exp(-2ζ*Tssv), σa^2/(4ζ))
    τf(t) = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
    τmax  = (2π/N_TEETH)/(Ω0*(1-RVA))
    Af(t) = @SMatrix [0. 1.; -(1.0 + w*hfun(t,Ω0,Tssv,RVA)) -2ζ]
    Bf(t) = @SMatrix [0. 0.; w*hfun(t,Ω0,Tssv,RVA) 0.]
    af(t) = @SMatrix [0. 0.; σc*w*hfun(t,Ω0,Tssv,RVA) 0.]
    bf(t) = @SMatrix [0. 0.; -σc*w*hfun(t,Ω0,Tssv,RVA) 0.]
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
        [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τf, bf))],
        Additive(2), [stAdditive(1,Additive(@SVector [0., σa]))])
    Δt = min(τmax/R_RES, 2π/NAT_RES); nst=max(1,Int(round(Tssv/Δt))); Δt=Tssv/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                                n_steps=nst, calculate_additive=true)
    ρ = spectralRadiusOfMapping_MF_factored(rst)
    v = NaN
    if ρ < 0.999
        try; v = fixPointOfMapping_MF_factored(rst)[1]; catch; end
    end
    (ρ, v)
end

ξs  = collect(range(log10(ΩLO), log10(ΩHI), length=NX))
Ω0s = 10 .^ ξs
ws  = collect(range(0.0, WHI, length=NW))
t0=time()
open(joinpath(@__DIR__,"ssv_chart_bf_hires.csv"),"w") do io
    println(io,"Omega0,w,rho,var")
    for (j,Ω) in enumerate(Ω0s)
        rr=zeros(length(ws)); vv=fill(NaN,length(ws))
        t=@elapsed Threads.@threads for i in eachindex(ws)
            ρ,v = solve_point(Ω, ws[i]); rr[i]=ρ; vv[i]=v
        end
        for (i,w) in enumerate(ws); @printf(io,"%.6f,%.5f,%.8f,%.8e\n",Ω,w,rr[i],vv[i]); end
        flush(io); @printf("Ω0=%.3f (%.1fs) [%d/%d]\n",Ω,t,j,length(Ω0s)); flush(stdout)
    end
end
@printf("HIRES BF TOTAL: %.1f s (%d threads)\n", time()-t0, Threads.nthreads())
