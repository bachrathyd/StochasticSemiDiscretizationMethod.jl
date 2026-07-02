# =============================================================================
# Work-precision diagrams for ALL examples, using the TRUSTED SDM (package).
# For each example: ρ(H) vs CPU time and vs p, at SDM orders q=0,2,4
#   (q=0→O(h¹), q=2→O(h²), q=4→O(h³); the Sykora-2020 convergence ladder).
# Reference: Richardson extrapolation from fine q=4 runs.
# Handles: single & multiple delays, time-varying delays, periodic coeffs,
#          additive & multiplicative noise, dimension d=1..4.
# Outputs per example:  demonstration/<name>.csv, <name>.png, <name>_order.png
#
# Run:  julia --project=. demonstration/wp_all.jl [index|all]
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, Plots
gr()
include(joinpath(@__DIR__,"sdde_types.jl"))      # SDDEProblem struct + maxdelay
include(joinpath(@__DIR__,"examples.jl"))

# ---- convert SDDEProblem → package LDDEProblem (multi-delay, time-varying, n-dim) ----
_mkdelay(τf, Bf, d, T) = (abs(τf(0.0)-τf(0.37*T)) < 1e-12) ?
    DelayMX(τf(0.0), t -> SMatrix{d,d}(Bf(t))) :
    DelayMX(t->τf(t), t -> SMatrix{d,d}(Bf(t)))

function to_package(prob::SDDEProblem)
    d = prob.d; T = prob.T
    AMx = ProportionalMX(t -> SMatrix{d,d}(prob.A(t)))
    BMxs = [_mkdelay(τf, Bf, d, T) for (τf,Bf) in prob.delays]
    # NOTE: package M2 supports single-delay multiplicative noise. We attach the
    # delayed multiplicative noise to the FIRST delay only (others → deterministic).
    αs = [stCoeffMX(j, ProportionalMX(t -> SMatrix{d,d}(αf(t))))
          for (j,(αf,βfs,σf)) in enumerate(prob.noise)]
    βs = [stCoeffMX(j, _mkdelay(prob.delays[1][1], βfs[1], d, T))
          for (j,(αf,βfs,σf)) in enumerate(prob.noise)]
    σs = [stAdditive(j, Additive(t -> SVector{d}(σf(t))))
          for (j,(αf,βfs,σf)) in enumerate(prob.noise)]
    cV = Additive(SVector{d}(zeros(d)))
    return LDDEProblem(AMx, BMxs, αs, βs, cV, σs)
end

# longest delay for the n_steps/τ argument
maxdelay_t(prob) = maximum(maximum(τf(t) for t in range(0,prob.T,length=33)) for (τf,_) in prob.delays)

function sdm_rho(prob::SDDEProblem, q, p)
    lddep = to_package(prob)
    # Δt = T/p (p steps per period); n_steps=p maps over the full period.
    # (Matches the validated stoch_mathieu_wp_sdm.jl convention: DiscretizationLength
    #  arg is overridden by n_steps; the delay is resolved via r=round(τ/Δt).)
    method = SemiDiscretization(q, prob.T/p)
    rst = SSDM.calculateResults(lddep, method, maxdelay_t(prob); n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst); tol=1e-12)
end
timed(f)=(f(); t0=time_ns(); v=f(); (v,(time_ns()-t0)/1e9))

function run_one(ex, outdir; qs=[0,2,4])
    prob = ex.prob
    # choose p-grid so the SMALLEST delay is resolved by r≥4 steps even at the coarsest p
    # (needed so the q=4 five-point delay stencil never indexes < 1).
    τmin = minimum(minimum(τf(t) for t in range(0,prob.T,length=33)) for (τf,_) in prob.delays)
    p0 = max(8, ceil(Int, 5*prob.T/τmin))           # r(τmin) ≥ ~5 at p0
    ps   = [p0, 2p0, 4p0, 8p0]
    pref = (12p0, 16p0)
    # Richardson reference from q=4 (O(h³)): (8 r2 - r1)/7
    r1 = sdm_rho(prob,4,pref[1]); r2 = sdm_rho(prob,4,pref[2])
    ρref = (8*r2 - r1)/7
    rows = Tuple{Int,Int,Float64,Float64,Float64}[]   # q,p,rho,time,err
    for q in qs, p in ps
        v,t = timed(()->sdm_rho(prob,q,p)); e=max(abs(v-ρref),1e-16)
        push!(rows,(q,p,v,t,e))
    end
    # CSV
    open(joinpath(outdir,ex.name*".csv"),"w") do io
        println(io,"# ",ex.notes)
        println(io,"# category: ",ex.cat,"  d=",prob.d,"  T=",round(prob.T,sigdigits=6),
                "  maxτ=",round(maxdelay_t(prob),sigdigits=6))
        println(io,"q,order,p,rho,cputime,abserr")
        for (q,p,r,t,e) in rows
            ord = q==0 ? 1 : (q<=3 ? 2 : 3)
            @printf(io,"%d,%d,%d,%.12g,%.12g,%.12g\n",q,ord,p,r,t,e)
        end
        @printf(io,"# rho_ref=%.13g\n",ρref)
    end
    # PNGs
    pal=Dict(0=>:gray,2=>:seagreen,4=>:crimson)
    function mk(xs_of,xlab,fname,ttl)
        plt=plot(xlabel=xlab,ylabel="|ρ(H) − ρ_ref|",title=ttl,
            xscale=:log10,yscale=:log10,legend=:bottomleft,lw=2,size=(820,600),titlefontsize=8)
        for q in qs
            sel=filter(r->r[1]==q,rows); isempty(sel)&&continue
            xs=[xs_of(r) for r in sel]; es=[r[5] for r in sel]; pm=sortperm(xs)
            ord = q==0 ? 1 : (q<=3 ? 2 : 3)
            plot!(plt,xs[pm],es[pm],label="SDM q=$q (rend≈$ord)",marker=:circle,
                  color=pal[q],ms=5,markerstrokewidth=0)
        end
        savefig(plt,joinpath(outdir,fname))
    end
    base="$(ex.name): $(ex.cat)\nρ_ref≈$(round(ρref,sigdigits=7)), d=$(prob.d), T=$(round(prob.T,sigdigits=4)), maxτ=$(round(maxdelay_t(prob),sigdigits=4))"
    mk(r->r[4],"CPU idő [s]",ex.name*".png","Work-precision — "*base)
    mk(r->r[2],"p (lépés/periódus)",ex.name*"_order.png","Konvergencia rend — "*base)
    return ρref, length(rows)
end

function main(which="all")
    outdir=@__DIR__; exs=build_examples()
    sel = which=="all" ? eachindex(exs) : [parse(Int,which)]
    for i in sel
        ex=exs[i]
        print(@sprintf("[%2d/%2d] %-30s ... ", i, length(exs), ex.name))
        try
            ρref,n = run_one(ex,outdir)
            @printf("OK  ρ_ref=%.7g (%d pts)\n", ρref, n)
        catch e
            println("FAILED: ", sprint(showerror,e)[1:min(end,160)])
        end
        flush(stdout)
    end
end

main(length(ARGS)>=1 ? ARGS[1] : "all")
