# Diagnostic: who is biased at (Ω0=1, w=0.30)? SSDM Var vs resolution, and
# MC Var vs dt. Includes RVA=0 variant to isolate the time-varying-delay path.
import Pkg; Pkg.activate(raw"D:\BD\StochasticSemiDiscretizationMethod.jl")
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, Random

const N_TEETH=2; const aD=0.5; const KtKn=0.3
const RVA=0.10; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10
const Ω0=1.0; const wdc=0.30
const Tssv = NT*2π/Ω0

function hfun(t, rva)
    φ0 = rva==0 ? Ω0*t : Ω0*t - (Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
    φen = acos(2aD-1); φex = float(π)
    hs=0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        (φen ≤ φ ≤ φex) && (hs += sin(φ)*(cos(φ)+KtKn*sin(φ)))
    end
    hs
end

function ssdm_var(rres, natres; rva=RVA)
    τf(t) = (2π/N_TEETH)/(Ω0*(1+rva*sin(2π*t/Tssv)))
    τmax  = (2π/N_TEETH)/(Ω0*(1-rva))
    Af(t) = @SMatrix [0. 1.; -(1.0+wdc*hfun(t,rva)) -2ζ]
    Bf(t) = @SMatrix [0. 0.; wdc*hfun(t,rva) 0.]
    af(t) = @SMatrix [0. 0.; σc*wdc*hfun(t,rva) 0.]
    bf(t) = @SMatrix [0. 0.; -σc*wdc*hfun(t,rva) 0.]
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(rva==0 ? τmax : τf, Bf)],
        [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(rva==0 ? τmax : τf, bf))],
        Additive(2), [stAdditive(1,Additive(@SVector [0., σa]))])
    Δt = min(τmax/rres, 2π/natres); nst=Int(round(Tssv/Δt)); Δt=Tssv/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                                n_steps=nst, calculate_additive=true)
    fixPointOfMapping_MF_factored(rst)[1]
end

function mc_var(dtdiv, npath; rva=RVA, ntrans=40, navg=20)
    dtm0 = min(((2π/N_TEETH)/(Ω0*(1-rva)))/24, 2π/30)/dtdiv
    nsub = Int(round(Tssv/dtm0)); dtm = Tssv/nsub
    τmax = (2π/N_TEETH)/(Ω0*(1-rva))
    nb = Int(ceil(τmax/dtm)) + 2
    per = nsub
    nchunk=64
    chunks=[i:min(i+cld(npath,nchunk)-1,npath) for i in 1:cld(npath,nchunk):npath]
    accs=zeros(length(chunks)); cnts=zeros(Int,length(chunks))
    Threads.@threads for ci in eachindex(chunks)
        a=0.0; c=0
        for pth in chunks[ci]
            rng=MersenneTwister(77_000+pth)
            xs=zeros(nb); x=0.0; v=0.0; head=0
            nstep=(ntrans+navg)*per
            for k in 0:nstep-1
                t=k*dtm
                τ = rva==0 ? τmax : (2π/N_TEETH)/(Ω0*(1+rva*sin(2π*t/Tssv)))
                back=τ/dtm; ib=floor(Int,back); fr=back-ib
                i1=mod(head-ib,nb)+1; i2=mod(head-ib-1,nb)+1
                xd=(1-fr)*xs[i1]+fr*xs[i2]
                h=hfun(t,rva)
                ξ1=randn(rng)*sqrt(dtm); ξ2=randn(rng)*sqrt(dtm)
                dv=(-2ζ*v - x - wdc*h*(x-xd))*dtm - σc*wdc*h*(x-xd)*ξ1 + σa*ξ2
                x+=v*dtm; v+=dv
                head=mod(head+1,nb); xs[head+1]=x
                if k ≥ ntrans*per && k % per == 0
                    a+=x^2; c+=1
                end
            end
        end
        accs[ci]=a; cnts[ci]=c
    end
    var=sum(accs)/sum(cnts); (var, var*sqrt(2/sum(cnts)))
end

println("── SSDM resolution sweep (SSV) ──")
for (rr,nr) in ((24,30),(48,60),(96,120))
    t=@elapsed v=ssdm_var(rr,nr)
    @printf("r=%3d nat=%3d  Var=%.6f  (%.0fs)\n", rr, nr, v, t)
end
println("── MC dt sweep (SSV) ──")
for (div,np) in ((1,4000),(2,4000),(4,4000))
    t=@elapsed (v,se)=mc_var(div,np)
    @printf("dt/%d  Var=%.6f ± %.4f  (%.0fs)\n", div, v, se, t)
end
println("── RVA=0 cross-check (constant delay) ──")
t=@elapsed v=ssdm_var(48,60; rva=0.0); @printf("SSDM r=48: Var=%.6f (%.0fs)\n", v, t)
t=@elapsed (v,se)=mc_var(2,4000; rva=0.0); @printf("MC dt/2:   Var=%.6f ± %.4f (%.0fs)\n", v, se, t)
