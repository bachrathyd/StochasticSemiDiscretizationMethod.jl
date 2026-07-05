# RVA=0 (constant spindle speed) stability boundary on the same (Ω0,w) grid as
# ssv_milling_chart.jl — overlay baseline requested in review (Reviewer A8).
# Only ρ(H) is needed (no Var). Output: benchmark/ssv_rva0.csv
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(1)

const N_TEETH=2; const aD=0.5; const KtKn=0.3
const NT=10; const ζ=0.02; const σc=0.20; const σa=0.10; const R_RES=24; const NAT_RES=30

function hfun0(t, Ω0)                       # RVA=0: φ(t)=Ω0 t
    φen = acos(2aD - 1); φex = float(π)
    hsum = 0.0
    for j in 0:N_TEETH-1
        φ = mod(Ω0*t + 2π*j/N_TEETH, 2π)
        (φen ≤ φ ≤ φex) && (hsum += sin(φ)*(cos(φ) + KtKn*sin(φ)))
    end
    hsum
end

function rho0(Ω0, w)
    T = NT * 2π/Ω0                           # same principal period as SSV case
    τ0 = (2π/N_TEETH)/Ω0
    Af(t) = @SMatrix [0. 1.; -(1.0 + w*hfun0(t,Ω0)) -2ζ]
    Bf(t) = @SMatrix [0. 0.; w*hfun0(t,Ω0) 0.]
    af(t) = @SMatrix [0. 0.; σc*w*hfun0(t,Ω0) 0.]
    bf(t) = @SMatrix [0. 0.; -σc*w*hfun0(t,Ω0) 0.]
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τ0, Bf)],
        [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τ0, bf))],
        Additive(2), [stAdditive(1,Additive(@SVector [0., σa]))])
    Δt = min(τ0/R_RES, 2π/NAT_RES); nst = max(1, Int(round(T/Δt))); Δt = T/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τ0; n_steps=nst)
    spectralRadiusOfMapping_MF_factored(rst)
end

Ω0s = collect(range(0.16, 1.50, length=80))
ws  = collect(range(0.01, 2.40, length=56))
open(joinpath(@__DIR__,"ssv_rva0.csv"),"w") do io
    println(io,"Omega0,w,rho")
    for (j,Ω) in enumerate(Ω0s)
        vals = zeros(length(ws))
        Threads.@threads for i in eachindex(ws)
            vals[i] = rho0(Ω, ws[i])
        end
        for (i,w) in enumerate(ws); @printf(io,"%.5f,%.5f,%.8f\n",Ω,w,vals[i]); end
        flush(io); @printf("Ω0=%.3f [%d/%d]\n",Ω,j,length(Ω0s)); flush(stdout)
    end
end
println("done — ssv_rva0.csv")
