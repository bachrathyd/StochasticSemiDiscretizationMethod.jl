# =============================================================================
# Work-precision & complexity: CLASSICAL explicit period-product SSDM vs MF-SSDM
# on the fully periodic stochastic delayed Mathieu equation (q=2 for both — the
# two paths evaluate the SAME discrete operator; they differ only in HOW).
#   classical: DiscreteMapping_M2 → explicit sparse product prodl(M2_MXs) → eigs
#   MF:        spectralRadiusOfMapping_MF_factored (matrix-free Krylov)
# Measures wall-clock time and allocated memory per solve, and |ρ−ρ_ref|.
# Outer loop = resolution; CSV+PNG re-saved after every point (live viewing).
# Outputs: benchmark/wp_ultra.csv, benchmark/wp_ultra.png/pdf (journal style)
# =============================================================================
using Pkg; Pkg.activate(@__DIR__)
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Plots, Printf, DelimitedFiles
BLAS.set_num_threads(1)

const TCAP = 300.0
const CSV  = joinpath(@__DIR__, "wp_ultra.csv")
const PNG  = joinpath(@__DIR__, "wp_ultra.png")
const PDF  = joinpath(@__DIR__, "wp_ultra.pdf")
const PAPER_IMG = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"

# fully periodic stochastic delayed Mathieu (all matrices time-periodic)
function lddep()
    AM(t)=@SMatrix [0. 1.; -(1.0+0.8cos(2π*t)) -(0.10+0.02cos(2π*t))]
    BM(t)=@SMatrix [0. 0.; 0.35*(1+0.3cos(2π*t)) 0.]
    aM(t)=@SMatrix [0. 0.; 0.45*(1+0.2cos(2π*t)) 0.]
    bM(t)=@SMatrix [0. 0.; 0.30*(1+0.4cos(2π*t)) 0.]
    LDDEProblem(ProportionalMX(AM), [DelayMX(1.0,BM)],
        [stCoeffMX(1,ProportionalMX(aM))], [stCoeffMX(1,DelayMX(1.0,bM))],
        Additive(2), [stAdditive(1,Additive(@SVector [0.,0.]))])
end
rst_at(p) = SSDM.calculateResults(lddep(), SemiDiscretization(2, 1.0/p), 1.0, n_steps=p)

# reference: MF at fine resolution (first-order in ρ, Richardson-extrapolated);
# modest krylovdim keeps the Krylov basis memory-bounded at large p.
println("computing reference ..."); flush(stdout)
ρ1 = spectralRadiusOfMapping_MF_factored(rst_at(1024); krylovdim=15)
ρ2 = spectralRadiusOfMapping_MF_factored(rst_at(2048); krylovdim=15)
const ρref = 2ρ2 - ρ1                                   # h-extrapolation (order 1)
@printf("ρ(1024)=%.10f ρ(2048)=%.10f → ρref=%.10f\n", ρ1, ρ2, ρref); flush(stdout)

rows = NamedTuple[]                                     # (method,p,D,t,mem,err)
function redraw()
    open(CSV,"w") do io
        println(io,"method,p,D,time_s,mem_MB,err")
        for r in rows
            @printf(io,"%s,%d,%d,%.4f,%.2f,%.6e\n", r.method,r.p,r.D,r.t,r.mem,r.err)
        end
    end
    cl = [r for r in rows if r.method=="classical"]; mf = [r for r in rows if r.method=="MF"]
    isempty(mf) && return
    kw = (framestyle=:box, guidefontsize=11, tickfontsize=9, legendfontsize=9,
          xscale=:log10, minorgrid=true, gridalpha=0.25, minorgridalpha=0.10)
    p1 = plot(; xlabel="", ylabel="wall-clock time [s]", yscale=:log10,
              title="(a) cost vs resolution", titleloc=:left, titlefontsize=11,
              legend=:topleft, kw...)
    p2 = plot(; xlabel="", ylabel="allocated memory [MB]", yscale=:log10,
              title="(b) memory vs resolution", titleloc=:left, titlefontsize=11,
              legend=false, kw...)
    p3 = plot(; xlabel="p (steps per period)", ylabel="|ρ − ρ_ref|", yscale=:log10,
              title="(c) error vs resolution", titleloc=:left, titlefontsize=11,
              legend=false, kw...)
    p4 = plot(; xlabel="wall-clock time [s]", ylabel="|ρ − ρ_ref|", yscale=:log10,
              title="(d) error vs cost", titleloc=:left, titlefontsize=11,
              legend=false, kw...)
    for (dat,c,mk,lab) in ((cl,:black,:circle,"classical (explicit product)"),
                           (mf,:crimson,:utriangle,"multiplication-free"))
        isempty(dat) && continue
        ps=[r.p for r in dat]; ts=[r.t for r in dat]; ms=[r.mem for r in dat]
        es=max.([r.err for r in dat],1e-14)
        plot!(p1, ps, ts, color=c, marker=mk, ms=4, lw=1.8, label=lab)
        plot!(p2, ps, ms, color=c, marker=mk, ms=4, lw=1.8, label="")
        plot!(p3, ps, es, color=c, marker=mk, ms=4, lw=1.8, label="")
        plot!(p4, ts, es, color=c, marker=mk, ms=4, lw=1.8, xscale=:log10, label="")
    end
    # slope guides on the cost panel
    if length(mf) ≥ 3
        pg=[mf[1].p, mf[end].p]
        plot!(p1, pg, mf[1].t .* (pg ./ pg[1]).^2, ls=:dot, color=:gray60, label="∝ p²")
    end
    if length(cl) ≥ 3
        pg=[cl[1].p, cl[end].p]
        plot!(p1, pg, cl[1].t .* (pg ./ pg[1]).^4, ls=:dot, color=:gray30, label="∝ p⁴")
    end
    plt = plot(p1,p2,p3,p4, layout=(2,2), size=(1150,860), dpi=300,
               left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    savefig(plt, PNG); savefig(plt, PDF)
end

stopped_cl = Ref(false)
ps_all = [8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048]
for p in ps_all
    rst = rst_at(p)
    D = SSDM.CovVecIdx(rst.n).sectionStarts[end]
    # MF (factored, matrix-free)
    GC.gc()
    st = @timed spectralRadiusOfMapping_MF_factored(rst; krylovdim=15)
    push!(rows, (method="MF", p=p, D=D, t=st.time, mem=st.bytes/1e6, err=abs(st.value-ρref)))
    @printf("p=%5d MF        t=%8.2fs mem=%9.1fMB err=%.3e\n", p, st.time, st.bytes/1e6,
            abs(st.value-ρref)); flush(stdout)
    redraw()
    # classical explicit product (until it exceeds the time cap)
    if !stopped_cl[]
        GC.gc()
        st2 = try
            @timed spectralRadiusOfMapping(DiscreteMapping_M2(rst))
        catch e
            @warn "classical failed at p=$p" e; stopped_cl[]=true; nothing
        end
        if st2 !== nothing
            push!(rows, (method="classical", p=p, D=D, t=st2.time, mem=st2.bytes/1e6,
                         err=abs(st2.value-ρref)))
            @printf("p=%5d classical t=%8.2fs mem=%9.1fMB err=%.3e%s\n", p, st2.time,
                    st2.bytes/1e6, abs(st2.value-ρref), st2.time>TCAP ? " → cap" : "")
            flush(stdout)
            st2.time > TCAP && (stopped_cl[]=true)
            redraw()
        end
    end
end
for f in ("wp_ultra.png","wp_ultra.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG,f); force=true) catch e; @warn e end
end
println("done — $PNG")
