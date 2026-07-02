# =============================================================================
# Work-precision driver: for each example compute ρ(H) vs CPU time for
#   • the high-order MOMENT-COLLOCATION method (GL1, GL2, GL3)  [this work]
#   • the trusted SDM baseline (q=0, q=2)                        [package]
# Reference ρ_ref = highest-order/finest run. Emits <name>.csv and <name>.png.
#
# Run:  julia --project=. demonstration/workprecision.jl [example_index|all]
# (uses only julia + file ops — all permitted)
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Printf, DelimitedFiles
using Plots
gr()

include(joinpath(@__DIR__,"moment_engine.jl"))
include(joinpath(@__DIR__,"moment_engine2.jl"))
include(joinpath(@__DIR__,"examples.jl"))

# ---- convert a (single-delay) SDDEProblem to the package LDDEProblem for SDM baseline ----
# package wants StaticArrays; coefficients as functions returning SMatrix/SVector.
function to_package(prob::SDDEProblem)
    d = prob.d
    @assert length(prob.delays)==1 "SDM baseline path: single delay only"
    τf, Bf = prob.delays[1]
    τconst = τf(0.0)   # baseline uses constant delay (package handles τ(t) via quadgk too)
    AMx = ProportionalMX(t -> SMatrix{d,d}(prob.A(t)))
    BMx = DelayMX(τconst, SMatrix{d,d}(Bf(0.0)))   # constant B for baseline simplicity
    # for time-varying B, wrap as function:
    BMxf = DelayMX(τconst, t -> SMatrix{d,d}(Bf(t)))
    αs = stCoeffMX[]; βs = stCoeffMX[]; σs = stAdditive[]
    for (j,(αf,βfs,σf)) in enumerate(prob.noise)
        push!(αs, stCoeffMX(j, ProportionalMX(t -> SMatrix{d,d}(αf(t)))))
        push!(βs, stCoeffMX(j, DelayMX(τconst, t -> SMatrix{d,d}(βfs[1](t)))))
        push!(σs, stAdditive(j, Additive(t -> SVector{d}(σf(t)))))
    end
    cV = Additive(SVector{d}(zeros(d)))
    return LDDEProblem(AMx, [BMxf], αs, βs, cV, σs), τconst
end

function sdm_rho(prob::SDDEProblem, q, p)
    lddep, τ = to_package(prob)
    method = SemiDiscretization(q, prob.T/p)
    rst = SSDM.calculateResults(lddep, method, τ; n_steps=p)
    spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst); tol=1e-12)
end

timed(f) = (f(); t0=time_ns(); v=f(); (v, (time_ns()-t0)/1e9))

# ---- run one example: sweep p for each method, collect (method,order,p,rho,time) ----
function run_example(ex; ps_hi=[8,16,32,48,64], ps_sdm=[20,40,80,160,320])
    prob = ex.prob
    rows = Tuple{String,Int,Int,Float64,Float64}[]   # method, order, p, rho, time
    # high-order moment-collocation
    for S in 1:3
        for p in ps_hi
            try
                v,t = timed(()->second_moment_rho(prob,S,p))
                push!(rows, ("GL$S", 2S, p, v, t))
            catch e
                @warn "GL$S p=$p failed" exception=e
            end
        end
    end
    # SDM baseline (single-delay only)
    if length(prob.delays)==1
        for q in [0,2]
            for p in ps_sdm
                try
                    v,t = timed(()->sdm_rho(prob,q,p))
                    push!(rows, ("SDM$q", q==0 ? 1 : 2, p, v, t))
                catch e
                    @warn "SDM$q p=$p failed" exception=e
                end
            end
        end
    end
    return rows
end

# reference: finest GL3
function reference_rho(ex)
    return second_moment_rho(ex.prob, 3, 96)
end

function emit(ex, rows, ρref, outdir)
    # CSV
    open(joinpath(outdir, ex.name*".csv"), "w") do io
        println(io, "# ", ex.notes)
        println(io, "method,order,p,rho,cputime,abserr")
        for (m,o,p,r,t) in rows
            @printf(io, "%s,%d,%d,%.12g,%.12g,%.12g\n", m,o,p,r,t,abs(r-ρref))
        end
        @printf(io, "# rho_ref=%.12g\n", ρref)
    end
    # PNG: work-precision (abs err vs CPU time, log-log)
    plt = plot(xlabel="CPU time [s]", ylabel="|ρ(H) − ρ_ref|",
        title="$(ex.name)\n$(ex.cat) — ρ_ref≈$(round(ρref,sigdigits=6))",
        xscale=:log10, yscale=:log10, legend=:bottomleft, lw=2, size=(820,600),
        titlefontsize=9)
    colors = Dict("GL1"=>:dodgerblue,"GL2"=>:seagreen,"GL3"=>:purple,
                  "SDM0"=>:gray,"SDM2"=>:orange)
    for m in ["SDM0","SDM2","GL1","GL2","GL3"]
        sel = filter(r->r[1]==m, rows)
        isempty(sel) && continue
        ts=[r[5] for r in sel]; es=[max(abs(r[4]-ρref),1e-16) for r in sel]
        pm=sortperm(ts)
        lbl = startswith(m,"SDM") ? "SDM q=$(m[4]) (order $(m=="SDM0" ? 1 : 2))" : "$m (order $(2*parse(Int,m[3])))"
        ls = startswith(m,"SDM") ? :dash : :solid
        mk = startswith(m,"SDM") ? :diamond : :circle
        plot!(plt, ts[pm], es[pm], label=lbl, marker=mk, linestyle=ls,
              color=colors[m], markersize=4, markerstrokewidth=0)
    end
    savefig(plt, joinpath(outdir, ex.name*".png"))
end

# ---- main ----
function main(which="all")
    outdir = @__DIR__
    exs = build_examples()
    sel = which=="all" ? exs : [exs[parse(Int,which)]]
    for ex in sel
        print("== $(ex.name) ($(ex.cat)) ... ")
        try
            ρref = reference_rho(ex)
            rows = run_example(ex)
            emit(ex, rows, ρref, outdir)
            println("done, ρ_ref=", round(ρref,sigdigits=8), ", ", length(rows), " points")
        catch e
            println("FAILED: ", e)
        end
    end
end

main(length(ARGS)>=1 ? ARGS[1] : "all")
