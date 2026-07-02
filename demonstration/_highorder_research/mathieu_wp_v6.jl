# =============================================================================
# WORK-PRECISION DIAGRAM — stochastic delay Mathieu 2nd-moment ρ(H).
# High-order moment-collocation engine v6 (cov_colloc_v6.jl, Krylov ρ) at GL(1)..GL(5), vs the
# trusted SDM (O(h³) ceiling). Present + delayed multiplicative noise (the user's target problem).
#
# Two panels (log-log):
#   (A) |ρ(H) − ρ_ref| vs p (steps/period)
#   (B) |ρ(H) − ρ_ref| vs CPU time  ← the true work-precision view
# Reference ρ_ref = SDM q=2, p=512 (trusted, monotone O(h²)).
# Krylov (eigsolve) for v6 ρ ⇒ scales to high p (dense eigen blows up at W²). Per-GL p-grids capped
# so high-GL × high-p (most expensive: W≈(p/2+1)(S+1)·2, per-step S·d² Sylvester) stays tractable.
# Output: demonstration/mathieu_wp_v6.png, mathieu_wp_v6_time.png, mathieu_wp_v6.csv
# =============================================================================
include(joinpath(@__DIR__,"cov_colloc_v6.jl"))
using Printf
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using Plots; gr()
using StochasticSemiDiscretizationMethod; const SSDM=StochasticSemiDiscretizationMethod
using StaticArrays
FILL_OFFDIAG[]=false; CROSS_ON[]=true   # correct defaults

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1
Amat(t)=[0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA]; Bmat(t)=[0.0 0.0; Bval 0.0]
αmat(t)=[0.0 0.0; -ALPHA*(Aval+EPS*cos(0.5*t)) -ALPHA*2ZETA]; βmat(t)=[0.0 0.0; ALPHA*Bval 0.0]
pb=Prob(2,PER,TAU, Amat,Bmat,αmat,βmat)
v6rho(S,p)=rho_H_krylov(build_v6(pb,S,p))

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
timed(f)=(v=f(); t0=time(); v=f(); (v, time()-t0))   # discard 1st (compile), time 2nd

# CONVERGED reference: v6 GL4 p=120 (= v6 GL3 p120 to 1.4e-8, = SDM-q2-Richardson to 1.2e-6).
# The old SDM q2 p512 ref was WRONG (~1e-5, non-monotone/unconverged) and made the GL curves look
# capped. Use the converged value so the true superconvergence shows.
println("computing CONVERGED reference (v6 GL4 p=120)..."); flush(stdout)
const REF = rho_H_krylov(build_v6(pb,4,120))
@printf("ρ_ref (v6 GL4 p120, converged) = %.10f\n\n", REF); flush(stdout)

# per-GL p grids (log-spaced-ish; capped so high GL doesn't run huge p)
pgrid = Dict(
  1=>round.(Int, exp10.(range(log10(6), log10(300), length=12))),
  2=>round.(Int, exp10.(range(log10(6), log10(200), length=12))),
  3=>round.(Int, exp10.(range(log10(6), log10(140), length=11))),
  4=>round.(Int, exp10.(range(log10(6), log10(90),  length=10))),
  5=>round.(Int, exp10.(range(log10(6), log10(60),  length=9))),
)
for S in 1:5; pgrid[S]=sort(unique(pgrid[S])); end

rows=Tuple{String,Int,Int,Float64,Float64,Float64}[]   # method,order,p,rho,time,err
println("v6 high-order (Krylov):"); flush(stdout)
for S in 1:5
    for p in pgrid[S]
        try
            v,t = timed(()->v6rho(S,p)); e=max(abs(v-REF),1e-16)
            push!(rows,("v6GL$S",2S,p,v,t,e))
            @printf("  v6 GL%d p=%3d ρ=%.9f t=%.2fs err=%.2e\n",S,p,v,t,e); flush(stdout)
        catch err; @warn "v6 GL$S p=$p failed" err; flush(stdout); end
    end
end
println("SDM (trusted):"); flush(stdout)
for q in [0,2,4]
    for p in round.(Int, exp10.(range(log10(8), log10(320), length=10)))
        try
            v,t = timed(()->sdm_rho(q,p)); e=max(abs(v-REF),1e-16)
            push!(rows,("SDM$q",q,p,v,t,e))
            @printf("  SDM q=%d p=%3d ρ=%.9f t=%.2fs err=%.2e\n",q,p,v,t,e); flush(stdout)
        catch err; @warn "SDM$q p=$p failed" err; flush(stdout); end
    end
end

open(joinpath(@__DIR__,"mathieu_wp_v6.csv"),"w") do io
    println(io,"# stoch delay Mathieu 2nd moment ρ(H); v6 high-order moment-collocation (Krylov) vs SDM")
    println(io,"# A=$Aval B=$Bval α=$ALPHA ε=$EPS ζ=$ZETA T=4π τ=2π; present+delayed mult noise; ref=v6 GL4 p120 converged=$REF")
    println(io,"method,order,p,rho,cputime,abserr")
    for (m,o,p,r,t,e) in rows; @printf(io,"%s,%d,%d,%.12g,%.12g,%.12g\n",m,o,p,r,t,e); end
end

col=Dict("v6GL1"=>:dodgerblue,"v6GL2"=>:seagreen,"v6GL3"=>:purple,"v6GL4"=>:darkorange,"v6GL5"=>:red,
         "SDM0"=>:gray70,"SDM2"=>:black,"SDM4"=>:brown)
ordr=["SDM0","SDM2","SDM4","v6GL1","v6GL2","v6GL3","v6GL4","v6GL5"]
function mkplot(xof,xlab,fname,ttl)
    plt=plot(xlabel=xlab,ylabel="|ρ(H) − ρ_ref|",title=ttl,xscale=:log10,yscale=:log10,
             legend=:bottomleft,lw=2,size=(980,720),titlefontsize=9)
    for m in ordr
        sel=filter(r->r[1]==m,rows); isempty(sel)&&continue
        xs=[xof(r) for r in sel]; es=[r[6] for r in sel]; pm=sortperm(xs)
        lbl = startswith(m,"SDM") ? "SDM q=$(m[4])" : "GL$(m[5]) (rend $(2*parse(Int,m[5])))"
        ls = startswith(m,"SDM") ? :dash : :solid; mk= startswith(m,"SDM") ? :diamond : :circle
        plot!(plt,xs[pm],es[pm],label=lbl,marker=mk,linestyle=ls,color=col[m],ms=4,markerstrokewidth=0)
    end
    savefig(plt,joinpath(@__DIR__,fname)); println("saved ",fname); flush(stdout)
end
mkplot(r->r[3],"p (lépés/periódus)","mathieu_wp_v6.png",
    "Stoch. delay Mathieu 2nd moment — magasrendű moment-kollokáció GL(1)..GL(5) vs SDM\n(jelen+múlt multiplikatív zaj; KONVERGÁLT ρ_ref≈$(round(REF,sigdigits=8)))")
mkplot(r->r[5],"CPU idő [s]","mathieu_wp_v6_time.png",
    "Work-precision: stoch. delay Mathieu 2nd moment — GL(1)..GL(5) vs SDM\n(|ρ−ρ_ref| vs CPU idő; jelen+múlt multiplikatív zaj)")
println("WP_V6 DONE"); flush(stdout)
