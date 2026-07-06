# Timing split for one BF grid point: coefficients / ρ(H) / M2 fixpoint.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10; const R_RES=24; const NAT_RES=30

φ0fun(t,Ω0,Tssv) = Ω0*t - (Ω0*RVA*Tssv/(2π))*(cos(2π*t/Tssv)-1.0)
function HS(t,Ω0,Tssv)
    φ0=φ0fun(t,Ω0,Tssv); h11=0.0;h12=0.0;h21=0.0;h22=0.0
    for j in 0:N_TEETH-1
        φ=mod(φ0+2π*j/N_TEETH,2π); φ ≤ PHI_EX || continue
        s,c=sincos(φ); a1=(c+Kr*s); a2=(s-Kr*c)
        h11+=a1*s;h12+=a1*c;h21+=-a2*s;h22+=-a2*c
    end
    @SMatrix [h11 h12; h21 h22]
end

function point(Ω0,w; warm=false)
    Tssv=NT*2π/Ω0
    τf(t)=(2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
    τmax=(2π/N_TEETH)/(Ω0*(1-RVA))
    Z2=@SMatrix zeros(2,2); I2=SMatrix{2,2}(1.0I)
    Af(t)=SMatrix{4,4}([Z2 I2; (-I2 .- w.*HS(t,Ω0,Tssv)) (-2ζ).*I2])
    Bf(t)=SMatrix{4,4}([Z2 Z2; (w.*HS(t,Ω0,Tssv)) Z2])
    af(t)=SMatrix{4,4}([Z2 Z2; ((-σc*w).*HS(t,Ω0,Tssv)) Z2])
    bf(t)=SMatrix{4,4}([Z2 Z2; ((σc*w).*HS(t,Ω0,Tssv)) Z2])
    lddep=LDDEProblem(ProportionalMX(Af),[DelayMX(τf,Bf)],
        [stCoeffMX(1,ProportionalMX(af))],[stCoeffMX(1,DelayMX(τf,bf))],
        Additive(4),[stAdditive(2,Additive(@SVector [0.,0.,σa,0.]))])
    Δt=min(τmax/R_RES,2π/NAT_RES); nst=Int(round(Tssv/Δt)); Δt=Tssv/nst
    t_coef=@elapsed rst=SSDM.calculateResults(lddep,SemiDiscretization(2,Δt),τmax;
                                              n_steps=nst,calculate_additive=true)
    t_rho=@elapsed ρ=spectralRadiusOfMapping_MF_factored(rst)
    t_fix=@elapsed v=fixPointOfMapping_MF_factored(rst)[1]
    warm || @printf("Ω0=%.3f w=%.2f (p=%d, r≈%d): coeffs %.2fs | ρ(H) %.2fs | M2 fixpoint %.2fs  → ρ=%.4f Var=%.4f\n",
                    Ω0,w,nst,Int(ceil(τmax/Δt)),t_coef,t_rho,t_fix,ρ,v)
end

point(1.0,0.30; warm=true)          # JIT warmup
point(1.0,0.30)                     # representative (fast end, p=360)
point(0.3,0.30)
point(0.125,0.30)                   # worst case (p=2400)
println("done")
