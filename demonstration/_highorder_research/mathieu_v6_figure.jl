# =============================================================================
# FIGURE: stochastic delay Mathieu 2nd-moment ρ(H) — high-order moment-collocation (v6) vs SDM.
# v6 (cov_colloc_v6.jl): per-step window-covariance collocation with the noise INSIDE the implicit
# stage solve (Σ0=0, full Egg source, α⊗α self-feedback operator) → noise-off exact (1e-17),
# present-noise superconvergent to the exact exp value (GL3 O(h⁶)), Hayes delayed-noise correct.
# Periodic Mathieu (present+delayed multiplicative noise): converges toward ≈0.1559.
# SDM (trusted package): O(h³) ceiling, ρ≈0.156228.
# Output: demonstration/mathieu_v6_figure.png (+ .csv). Two panels: ρ vs p, and |Δρ| self-conv.
# =============================================================================
include(joinpath(@__DIR__,"cov_colloc_v6.jl"))
using Printf
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using Plots; gr()
using StochasticSemiDiscretizationMethod; const SSDM=StochasticSemiDiscretizationMethod
using StaticArrays

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1

# v6 Mathieu problem
Amat(t)=[0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA]; Bmat(t)=[0.0 0.0; Bval 0.0]
αmat(t)=[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA]; βmat(t)=[0.0 0.0; ALPHA*Bval 0.0]
pb=Prob(2,PER,TAU, Amat,Bmat,αmat,βmat)
v6rho(S,p)=rho_H_dense(build_v6(pb,S,p))

# trusted SDM
function sdm_rho(q,p)
    AMx=ProportionalMX(t->@SMatrix [0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA])
    BMx=DelayMX(TAU,@SMatrix [0.0 0.0; Bval 0.0])
    αMx=stCoeffMX(1,ProportionalMX(t->@SMatrix [0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA]))
    βMx=stCoeffMX(1,DelayMX(TAU,t->@SMatrix [0.0 0.0; ALPHA*Bval 0.0]))
    cV=Additive(@SVector [0.0,0.0]); σV=stAdditive(1,Additive(@SVector [0.0,0.0]))
    lddep=LDDEProblem(AMx,[BMx],[αMx],[βMx],cV,[σV])
    rst=SSDM.calculateResults(lddep,SemiDiscretization(q,PER/p),TAU;n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst);tol=1e-12)
end

println("computing SDM reference (q=2, p=512)..."); flush(stdout)
const REF = sdm_rho(2,512)
@printf("SDM ρ_ref (q2 p512) = %.10f\n\n", REF); flush(stdout)

rows=Tuple{String,Int,Int,Float64,Float64}[]   # method,order,p,rho,err
println("v6 high-order:"); flush(stdout)
for S in [2,3]
    for p in [8,10,12,14,16,20]
        v=v6rho(S,p); e=abs(v-REF); push!(rows,("v6GL$S",2S,p,v,e))
        @printf("  v6 GL%d p=%2d ρ=%.10f err=%.2e\n",S,p,v,e); flush(stdout)
    end
end
println("SDM:"); flush(stdout)
for q in [0,2,4]
    for p in [16,32,64,128,256]
        v=sdm_rho(q,p); e=abs(v-REF); push!(rows,("SDM$q",q,p,v,e))
        @printf("  SDM q=%d p=%3d ρ=%.10f err=%.2e\n",q,p,v,e); flush(stdout)
    end
end

open(joinpath(@__DIR__,"mathieu_v6_figure.csv"),"w") do io
    println(io,"# stoch delay Mathieu 2nd moment ρ(H); v6 high-order moment-collocation vs SDM")
    println(io,"# A=$Aval B=$Bval α=$ALPHA ε=$EPS ζ=$ZETA T=4π τ=2π; present+delayed mult noise; ref=SDM q2 p512=$REF")
    println(io,"method,order,p,rho,abserr")
    for (m,o,p,r,e) in rows; @printf(io,"%s,%d,%d,%.12g,%.12g\n",m,o,p,r,e); end
end

col=Dict("v6GL2"=>:seagreen,"v6GL3"=>:purple,"SDM0"=>:gray,"SDM2"=>:black,"SDM4"=>:orange)
plt=plot(xlabel="p (lépés/periódus)",ylabel="|ρ(H) − ρ_SDM,ref|",
    title="Stoch. delay Mathieu 2nd moment — magasrendű moment-kollokáció (v6) vs SDM\n(jelen+múlt multiplikatív zaj; ρ_ref≈$(round(REF,sigdigits=7)))",
    xscale=:log10,yscale=:log10,legend=:bottomleft,lw=2,size=(950,700),titlefontsize=9)
for m in ["SDM0","SDM2","SDM4","v6GL2","v6GL3"]
    sel=filter(r->r[1]==m,rows); isempty(sel)&&continue
    xs=[r[3] for r in sel]; es=[max(r[5],1e-16) for r in sel]; pm=sortperm(xs)
    lbl = startswith(m,"SDM") ? "SDM q=$(m[4])" : "$(m[3:end]) (rend $(2*parse(Int,m[5])))"
    ls = startswith(m,"SDM") ? :dash : :solid; mk= startswith(m,"SDM") ? :diamond : :circle
    plot!(plt,xs[pm],es[pm],label=lbl,marker=mk,linestyle=ls,color=col[m],ms=5,markerstrokewidth=0)
end
savefig(plt,joinpath(@__DIR__,"mathieu_v6_figure.png"))
println("\nsaved mathieu_v6_figure.png + .csv"); flush(stdout)
println("FIGURE DONE"); flush(stdout)
