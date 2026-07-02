# =============================================================================
# WORK-PRECISION DIAGRAM — scalar stochastic Hayes equation, TRUSTED SDM only.
#   dx = (A x + B x(t-1)) dt + (β x(t-1)) dW,   A=-1, B=-0.4, β=0.3, τ=T=1.
# Shows the q-order convergence of the trusted SDM 2nd moment ρ(H):
#   q=0 → O(h¹),  q=1..3 → O(h²),  q=4,5 → O(h³)   (Sykora 2020).
# Reference: q=5 at very fine p (Richardson-clean). Correct, validated values.
# Outputs: demonstration/wp_hayes.csv, wp_hayes.png (err vs CPU), wp_hayes_order.png (err vs p)
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, Plots
gr()

const A=-1.0; const B=-0.4; const BETA=0.3; const TAU=1.0

function sdm_rho(q,p)
    AMx=ProportionalMX(t->SMatrix{1,1}(A)); BMx=DelayMX(TAU,SMatrix{1,1}(B))
    αMx=stCoeffMX(1,ProportionalMX(t->SMatrix{1,1}(0.0)))
    βMx=stCoeffMX(1,DelayMX(TAU,SMatrix{1,1}(BETA)))
    cV=Additive(SVector{1}(0.0)); σV=stAdditive(1,Additive(SVector{1}(0.0)))
    lddep=LDDEProblem(AMx,[BMx],[αMx],[βMx],cV,[σV])
    rst=SSDM.calculateResults(lddep,SemiDiscretization(q,TAU/p),TAU;n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst); tol=1e-13)
end
timed(f)=(f(); t0=time_ns(); v=f(); (v,(time_ns()-t0)/1e9))

# Richardson reference from two fine q=3 (O(h²)) runs: ρ_exact ≈ (4 ρ(2p) − ρ(p))/3.
r1 = sdm_rho(3, 256); r2 = sdm_rho(3, 512)
ρref = (4*r2 - r1)/3
@printf("ρ_ref (Richardson q=3, p=256/512) = %.13f  (raw r2=%.13f)\n\n", ρref, r2)

rows=Tuple{String,Int,Int,Float64,Float64,Float64}[]
for q in 0:5
    @printf("SDM q=%d:\n", q)
    for p in [8,16,32,64,128,256]
        v,t=timed(()->sdm_rho(q,p)); e=abs(v-ρref)
        push!(rows,("q$q",q,p,v,t,e))
        @printf("  p=%3d  ρ=%.10f  t=%.4fs  err=%.2e\n",p,v,t,e)
    end
end

open(joinpath(@__DIR__,"wp_hayes.csv"),"w") do io
    println(io,"# Stochastic Hayes dx=(Ax+Bx(t-1))dt+βx(t-1)dW; A=$A,B=$B,β=$BETA,τ=1; trusted SDM")
    println(io,"q,order,p,rho,cputime,abserr")
    for (m,o,p,r,t,e) in rows; @printf(io,"%s,%d,%d,%.12g,%.12g,%.12g\n",m,o,p,r,t,e); end
    @printf(io,"# rho_ref=%.13g\n",ρref)
end

palette=[:gray,:steelblue,:seagreen,:darkgreen,:orange,:crimson]
function mkplot(xs_of, xlab, fname, title)
    plt=plot(xlabel=xlab, ylabel="|ρ(H) − ρ_ref|", title=title,
        xscale=:log10, yscale=:log10, legend=:bottomleft, lw=2, size=(880,650), titlefontsize=9)
    for q in 0:5
        sel=filter(r->r[2]==q,rows); isempty(sel)&&continue
        xs=[xs_of(r) for r in sel]; es=[max(r[6],1e-16) for r in sel]; pm=sortperm(xs)
        ord = q==0 ? 1 : (q<=3 ? 2 : 3)
        plot!(plt,xs[pm],es[pm],label="q=$q (rend≈$ord)",marker=:circle,color=palette[q+1],ms=5,markerstrokewidth=0)
    end
    savefig(plt,joinpath(@__DIR__,fname)); println("Saved ",fname)
end
mkplot(r->r[5], "CPU idő [s]", "wp_hayes.png",
    "Work-precision: sztochasztikus Hayes egyenlet (SDM)\nmagasabb q-rend → nagyobb pontosság/költség (ρ_ref≈$(round(ρref,sigdigits=7)))")
mkplot(r->r[3], "p (lépésszám / periódus)", "wp_hayes_order.png",
    "Konvergencia rend: Hayes (SDM)\nq=0→O(h¹), q=1-3→O(h²), q=4-5→O(h³)")
