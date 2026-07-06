# Diagnose deterministic ρ(Φ): KrylovKit(LR) vs dense eigvals(LR) vs dense
# monodromy, along a w-slice. A milling lobe should give a smooth ρ(w).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using SemiDiscretizationMethod
using StaticArrays, LinearAlgebra, SparseArrays, KrylovKit, Printf
BLAS.set_num_threads(1)

const N_TEETH=2; const PHI_EX=π/4; const Kr=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const R_RES=24; const NAT_RES=30

φ0fun(t,Ω0,Tssv,rva) = rva==0 ? Ω0*t : Ω0*t-(Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv)-1.0)
function Hdir(t,Ω0,Tssv,rva)
    φ0=φ0fun(t,Ω0,Tssv,rva); h11=0.0;h12=0.0;h21=0.0;h22=0.0
    for j in 0:N_TEETH-1
        φ=mod(φ0+2π*j/N_TEETH,2π); φ ≤ PHI_EX || continue
        s,c=sincos(φ); a1=(c+Kr*s); a2=(s-Kr*c)
        h11+=a1*s;h12+=a1*c;h21+=-a2*s;h22+=-a2*c
    end
    @SMatrix [h11 h12; h21 h22]
end
const Z2=@SMatrix zeros(2,2); const I2=SMatrix{2,2}(1.0I)

function mapping_LR(Ω0,w;rva)
    Tssv=NT*2π/Ω0
    if rva==0
        T=(2π/N_TEETH)/Ω0; τmax=T; delay=τmax
    else
        T=Tssv; τmax=(2π/N_TEETH)/(Ω0*(1-rva))
        delay=t->(2π/N_TEETH)/(Ω0*(1+rva*sin(2π*t/Tssv)))
    end
    Hf(t)=Hdir(t,Ω0,Tssv,rva)
    Af(t)=SMatrix{4,4}([Z2 I2; (-I2 .- w.*Hf(t)) (-2ζ).*I2])
    Bf(t)=SMatrix{4,4}([Z2 Z2; (w.*Hf(t)) Z2])
    lddep=LDDEProblem(ProportionalMX(Af),[DelayMX(delay,Bf)],Additive(zeros(4)))
    Δt=min(τmax/R_RES,2π/NAT_RES); nst=max(1,Int(round(T/Δt))); Δt=T/nst
    (DiscreteMapping_LR(lddep,SemiDiscretization(2,Δt),τmax;n_steps=nst), nst)
end

function rhos(Ω0,w;rva)
    m,nst = mapping_LR(Ω0,w;rva=rva)
    L=Matrix(m.LmappingMX); R=Matrix(m.RmappingMX)
    # dense generalized eigenvalues
    ev = eigvals(R, L)
    ρ_dense = maximum(x->isfinite(x) ? abs(x) : 0.0, ev)
    # KrylovKit on ΦL⁻¹ΦR
    F=lu(m.LmappingMX)
    vals,_,info = KrylovKit.eigsolve(x->F\(m.RmappingMX*x), ones(size(R,1)),1,:LM;
                                     tol=1e-10,maxiter=400,krylovdim=30)
    (ρ_dense, abs(vals[1]), info.converged, size(R,1), nst)
end

for (Ω0,rva,tag) in ((1.0,0.0,"cs"),(0.2,0.0,"cs"),(1.0,RVA,"ssv"),(0.2,RVA,"ssv"))
    println("── Ω0=$Ω0  $tag ──")
    for w in 0.1:0.15:1.6
        ρd,ρk,conv,D,nst = rhos(Ω0,w;rva=rva)
        @printf("  w=%.2f: dense ρ=%.4f | Krylov ρ=%.4f conv=%d | D=%d nst=%d\n",
                w,ρd,ρk,conv,D,nst)
    end
end
println("done")
