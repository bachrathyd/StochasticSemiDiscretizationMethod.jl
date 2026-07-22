# Work-precision (error vs CPU time) for the TIME-VARYING-delay collocation engine
# on the SSV turning model. After the NoiseOpVT + ring-buffer + D-mask optimization
# the vT engine is chart-scale, so accuracy-per-second is meaningful: the high-order
# collocation dominates the classical scheme by orders of magnitude at tight
# tolerance. Incremental CSV + re-render after every point.
#   benchmark/ssv_vt_wp.csv
#   assets/TimeVaryingDelayWorkPrecision.png              (README)
#   journal_paper/images/ssv_vt_wp.{png,pdf}              (paper; skipped if absent)
using Pkg; Pkg.activate(@__DIR__)
using LinearAlgebra
BLAS.set_num_threads(1)
using StochasticSemiDiscretizationMethod, StaticArrays, Plots, Printf, DelimitedFiles

const CSV=joinpath(@__DIR__,"ssv_vt_wp.csv")
const ASSET=joinpath(@__DIR__,"..","assets","TimeVaryingDelayWorkPrecision.png")
const PAPER_IMG=joinpath(@__DIR__,"..","journal_paper","images")

# SSV turning WITH regenerative multiplicative (cutting-force) noise σc: β ≢ 0, so
# the classical scheme is genuinely FIRST order in the second moment — the regime
# where the high-order collocation earns its keep (additive-only problems let the
# classical scheme ride its deterministic order 2 and win, see the paper).
# const ⇒ the coefficient closures below are type-stable; the collocation build
# evaluates them at every Gauss node of every stage, so non-const globals here
# would box every call and dominate the (parallelised) build time. Run this script
# with `julia -t auto` so the embarrassingly-parallel engine build uses all cores.
const Ω0=0.87; const RVA=0.1; const RVF=0.1; const ζ=0.05; const w=0.4; const σ=0.1; const σc=0.25
const T=2π/RVF
τf(t)=(2π)/(Ω0*(1.0+RVA*sin(RVF*t)))
Af(t)=@SMatrix [0.0 1.0; -(1.0+w) -2ζ]
Bf(t)=@SMatrix [0.0 0.0; w 0.0]
αf(t)=@SMatrix [0.0 0.0; -σc*(1.0+w) 0.0]     # present cutting-force coefficient noise
βf(t)=@SMatrix [0.0 0.0; σc*w 0.0]            # delayed (regenerative) coefficient noise
prob=LDDEProblem(ProportionalMX(Af),[DelayMX(τf,Bf)],
                 [stCoeffMX(1,ProportionalMX(αf))],[stCoeffMX(1,DelayMX(τf,βf))],
                 Additive(2),[stAdditive(1,Additive(@SVector [0.0,σ]))],1)

# min-of-3 warm wall-clock
function timed(f)
    f()                               # warm
    minimum(@elapsed(f()) for _ in 1:3)
end

have=Dict{Tuple{String,Int},NTuple{3,Float64}}()  # (case,p) -> (rho, cputime, var)
if isfile(CSV)
    raw,_=readdlm(CSV,',';header=true)
    for k in axes(raw,1); have[(String(raw[k,1]),Int(raw[k,2]))]=(Float64(raw[k,3]),Float64(raw[k,4]),Float64(raw[k,5])); end
end
save()=open(CSV,"w") do io
    println(io,"case,p,rho,cpu,var")
    for ((c,p),(r,t,v)) in sort(collect(have);by=x->(x[1][1],x[1][2]))
        println(io,"$c,$p,$r,$t,$v")
    end
end
richardson(v)=(q2=(v[2]-v[1])/(v[3]-v[2]); v[3]+(v[3]-v[2])/(q2-1))

function point!(case,p;S=0,q=2)
    haskey(have,(case,p)) && return
    ρ = case=="classical" ? spectralRadiusOfMoment(prob,T,p;method=ClassicalSD(q)) :
                            spectralRadiusOfMoment(prob,T,p;method=Collocation(S),verbosity=0)
    v = case=="classical" ? stationaryVariance(prob,T,p;method=ClassicalSD(q)) :
                            stationaryVariance(prob,T,p;method=Collocation(S),verbosity=0)
    t = case=="classical" ? timed(()->spectralRadiusOfMoment(prob,T,p;method=ClassicalSD(q))) :
                            timed(()->spectralRadiusOfMoment(prob,T,p;method=Collocation(S),verbosity=0))
    have[(case,p)]=(ρ,t,v); save()
    @printf("  %-10s p=%4d  ρ=%.10f  cpu=%.3fs\n",case,p,ρ,t); flush(stdout)
end

function render()
    all(haskey(have,("S3ref",p)) for p in (120,240,480)) || return
    ρref=richardson([have[("S3ref",p)][1] for p in (120,240,480)])
    plt=plot(size=(760,560),dpi=300,framestyle=:box,legend=:topright,
             xscale=:log10,yscale=:log10,xlabel="CPU time  [s]",
             ylabel="error in ρ(H)",guidefontsize=12,tickfontsize=10,
             title="SSV turning τ(t), β≢0: accuracy per second (GL build ∥ $(Threads.nthreads()) thr)",
             left_margin=5Plots.mm,bottom_margin=5Plots.mm)
    series=[("classical","classical SD (q=2)",:black,:circle),
            ("S1","GL collocation S=1",:seagreen,:utriangle),
            ("S2","GL collocation S=2",:royalblue,:diamond),
            ("S3","GL collocation S=3",:firebrick,:square)]
    for (case,name,col,mk) in series
        pts=sort([(t,abs(r-ρref)) for ((c,p),(r,t,v)) in have if c==case])
        isempty(pts) && continue
        xs=[t for (t,_) in pts]; es=[e for (_,e) in pts]; keep=es.>0
        plot!(plt,xs[keep],es[keep];marker=mk,color=col,label=name,markersize=5)
    end
    png(plt,ASSET)
    if isdir(PAPER_IMG)
        try; png(plt,joinpath(PAPER_IMG,"ssv_vt_wp.png")); savefig(plt,joinpath(PAPER_IMG,"ssv_vt_wp.pdf")); catch e; @warn e; end
    end
    println("  figure updated (ρ*=$ρref)")
end

# high-accuracy reference ladder (S=3)
for p in (120,240,480); point!("S3ref",p;S=3); render(); end
# work-precision sweeps
for p in (20,40,80,160);        point!("S3",p;S=3); render(); end
for p in (20,40,80,160,320);    point!("S2",p;S=2); render(); end
for p in (40,80,160,320,640);   point!("S1",p;S=1); render(); end
for p in (100,200,400,800,1600,3200); point!("classical",p); render(); end
println("VT WORK-PRECISION DONE")
