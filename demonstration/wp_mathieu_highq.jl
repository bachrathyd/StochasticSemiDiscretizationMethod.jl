# =============================================================================
# High-q convergence test on the stochastic Mathieu equation.
# Question: does q=6, q=8 raise the convergence order of ρ(H) beyond q=4's O(h³)?
# Uses the TRUSTED Multiplication-Free SDM (spectralRadiusOfMapping_MF), many p-points.
# Self-convergence rate (reference-free) + Richardson reference for the work-precision plot.
# Outputs: wp_mathieu_highq.csv, wp_mathieu_highq.png (err vs CPU), wp_mathieu_highq_order.png
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, Plots
gr()

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1

function mathieu_lddep()
    AMxfun(t) = @SMatrix [0.0 1.0; -(Aval + EPS*cos(0.5*t)) -2ZETA]
    AMx  = ProportionalMX(AMxfun)
    BMx  = DelayMX(TAU, @SMatrix [0.0 0.0; Bval 0.0])
    af(t) = @SMatrix [0.0 0.0; -ALPHA*(Aval + EPS*cos(0.5*t)) -ALPHA*2ZETA]
    bf(t) = @SMatrix [0.0 0.0; ALPHA*Bval 0.0]
    αMx = stCoeffMX(1, ProportionalMX(af))
    βMx = stCoeffMX(1, DelayMX(TAU, bf))
    cV  = Additive(@SVector [0.0, 0.0])
    σV  = stAdditive(1, Additive(@SVector [0.0, 0.0]))
    LDDEProblem(AMx, [BMx], [αMx], [βMx], cV, [σV])
end

# Multiplication-Free 2nd-moment spectral radius
function mf_rho(q,p)
    lddep=mathieu_lddep()
    method=SemiDiscretization(q, PER/p)
    rst=SSDM.calculateResults(lddep, method, TAU; n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst); tol=1e-13)
end
timed(f)=(f(); t0=time_ns(); v=f(); (v,(time_ns()-t0)/1e9))

const QS = [0,2,4,6,8]
# many points: dense p-grid (delay τ=2π=PER/2 ⇒ r=p/2, always well resolved)
const PS = [12,16,24,32,48,64,96,128,192,256]

# Reference from the STABLEST order: q=2 converges monotonically (O(h²)); high-q has NOT
# converged at feasible p (self-convergence study + Monte-Carlo confirm this), so a high-q
# reference is BIASED and causes the spurious 'dip-then-rise'. Use q=2 Richardson instead:
#   q=2 is O(h²) ⇒ ρ_exact ≈ (4 ρ(2p) − ρ(p))/3, taken at very fine p.
println("Reference: q=2 Richardson from p=512/1024 (q=2 is the monotone-convergent order) ...")
ra = mf_rho(2, 512); rb = mf_rho(2, 1024)
ρref = (4*rb - ra)/3
@printf("ρ_ref = %.13f   (raw q2 p1024=%.13f, p512=%.13f)\n\n", ρref, rb, ra)

rows = Tuple{Int,Int,Float64,Float64,Float64}[]   # q,p,rho,time,err
for q in QS
    @printf("q=%d:\n", q)
    prevρ=nothing; prevp=nothing
    for p in PS
        v,t = timed(()->mf_rho(q,p)); e=max(abs(v-ρref),1e-16)
        push!(rows,(q,p,v,t,e))
        # self-convergence rate between successive p
        rate = (prevρ===nothing) ? NaN : log(abs(prevρ-v)/max(abs(v-ρref),1e-16))/log(p/prevp)
        @printf("  p=%3d  ρ=%.11f  t=%.4fs  err=%.2e\n", p, v, t, e)
        prevρ=v; prevp=p
    end
end

# fitted order (slope of log err vs log p, using the pre-floor points)
println("\nFitted convergence orders (log-log slope, err>1e-11):")
for q in QS
    sel = filter(r->r[1]==q && r[5]>1e-11, rows)
    length(sel)<3 && continue
    lp=log.([r[2] for r in sel]); le=log.([r[5] for r in sel])
    slope = ([lp ones(length(lp))]\le)[1]
    @printf("  q=%d :  order ≈ %.2f\n", q, -slope)
end

# CSV
open(joinpath(@__DIR__,"wp_mathieu_highq.csv"),"w") do io
    println(io,"# Stochastic Mathieu high-q convergence (MF SDM); A=$Aval,B=$Bval,α=$ALPHA,ε=$EPS,P=4π,τ=2π")
    println(io,"q,p,rho,cputime,abserr")
    for (q,p,r,t,e) in rows; @printf(io,"%d,%d,%.12g,%.12g,%.12g\n",q,p,r,t,e); end
    @printf(io,"# rho_ref=%.13g\n",ρref)
end

pal=Dict(0=>:gray,2=>:seagreen,4=>:crimson,6=>:purple,8=>:black)
function mk(xs_of,xlab,fname,ttl)
    plt=plot(xlabel=xlab,ylabel="|ρ(H) − ρ_ref|",title=ttl,
        xscale=:log10,yscale=:log10,legend=:bottomleft,lw=2,size=(900,680),titlefontsize=9)
    for q in QS
        sel=filter(r->r[1]==q,rows); isempty(sel)&&continue
        xs=[xs_of(r) for r in sel]; es=[r[5] for r in sel]; pm=sortperm(xs)
        plot!(plt,xs[pm],es[pm],label="q=$q",marker=:circle,color=pal[q],ms=4,markerstrokewidth=0)
    end
    savefig(plt,joinpath(@__DIR__,fname)); println("Saved ",fname)
end
mk(r->r[4],"CPU idő [s]","wp_mathieu_highq.png",
   "Work-precision: sztochasztikus Mathieu, magas q (MF SDM)\nρ_ref≈$(round(ρref,sigdigits=8)), P=4π, τ=2π")
mk(r->r[2],"p (lépés/periódus)","wp_mathieu_highq_order.png",
   "Konvergencia rend: Mathieu q=0..8 (MF SDM)\nnő-e a rend q=6,8-nál? ρ_ref≈$(round(ρref,sigdigits=8))")
