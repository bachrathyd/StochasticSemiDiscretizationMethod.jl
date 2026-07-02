# =============================================================================
# v7 validation ladder — stage 4: HONEST work-precision vs SDM (mirror problem)
#
# Protocol mirrors the archived benchmark_mf_v6.jl (identical problem, BLAS 1
# thread, log-spaced p, median-of-3 timing, same log_slope), with two fixes:
#  * REFERENCE: v7-causal GL3 converged value, independently confirmed by the
#    fine-grid arbiter to 4e-10 (0.7389661254). The old raw SDM q=2 p=800
#    reference is ~1.3e-5 biased (SDM q2's measured order here is ≈1).
#  * LOOP ORDER: outermost loop = resolution p; CSV + PNG are re-saved after
#    EVERY resolution level, so the figure can be watched in real time.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
using BenchmarkTools, StaticArrays, LinearAlgebra, Plots, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))

const Ac=1.0; const εc=0.5; const Bc=0.2; const ζc=0.1
const τc=1.0; const σc=0.1; const αc=0.2; const Pc=1.0

function createStochMathieuProblem()
    AMxfun(t) = @SMatrix [0. 1.; -(Ac + εc*cos(2π*t/Pc)) -2ζc]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τc, @SMatrix [0. 0.; Bc 0.])
    cVec = Additive(2)
    αMx1  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; αc 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τc, @SMatrix [0. 0.; αc 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σc]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end
lddep = createStochMathieuProblem()

pb_v7 = Prob(2, Pc, τc,
    t->[0.0 1.0; -(Ac + εc*cos(2π*t/Pc)) -2ζc],
    t->[0.0 0.0; Bc 0.0],
    t->[0.0 0.0; αc 0.0],
    t->[0.0 0.0; αc 0.0])

# arbiter-confirmed reference (highorder/README.md)
const ρ_ref = 0.7389661254
ρ_check = rho_H_krylov(build_v7(pb_v7, 3, 24); offdiag=:causal)
@printf("ρ_ref = %.10f  (fresh GL3 p=24 check: %.10f, Δ=%.1e)\n",
        ρ_ref, ρ_check, abs(ρ_check-ρ_ref))

ps_all = unique(round.(Int, exp10.(range(log10(4), log10(500), length=25))))
const TIME_LIMIT = 3.0

function log_slope(xs, ys)
    valid = (xs .> 0) .& (ys .> 1e-15)
    lx = log10.(xs[valid]); ly = log10.(ys[valid])
    length(lx) < 3 && return NaN
    n = length(lx); i0 = max(1, n÷2)
    lx = lx[i0:end]; ly = ly[i0:end]
    slopes = diff(ly) ./ diff(lx)
    return round(sum(slopes)/length(slopes), digits=2)
end

struct BenchRow
    method::String; p::Int; D::Int
    time_s::Float64; rho::Float64; err::Float64
end
rows = BenchRow[]

function bench_sdm(order, p)
    method = SemiDiscretization(order, Pc/p)
    rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τc)
    dm = DiscreteMapping_M2_MF(rst)
    d=2; r = div(rst.n, d) - 1
    D = StochasticSemiDiscretizationMethod.CovVecIdx((r+1)*d).sectionStarts[end]
    ρ = spectralRadiusOfMapping_MF(dm)
    bm = @benchmark spectralRadiusOfMapping_MF($dm) samples=3 evals=1 seconds=1.0
    return D, median(bm).time/1e9, ρ
end
function bench_v7(S, p)
    eng = build_v7(pb_v7, S, p)
    D = (eng.W*(eng.W+1))÷2
    ρ = rho_H_krylov(eng; offdiag=:causal)
    bm = @benchmark rho_H_krylov($eng; offdiag=:causal) samples=3 evals=1 seconds=1.0
    return D, median(bm).time/1e9, ρ
end

methcfg = [
    ("SDM0", :blue,      :circle),
    ("SDM1", :red,       :rect),
    ("SDM2", :green,     :diamond),
    ("GL1",  :dodgerblue,:utriangle),
    ("GL2",  :seagreen,  :utriangle),
    ("GL3",  :purple,    :utriangle),
]

csv_path = joinpath(@__DIR__, "out_wp.csv")
png_path = joinpath(@__DIR__, "out_wp.png")

function _write(rows)
    open(csv_path,"w") do io
        println(io,"method,p,D,time_s,rho,err")
        for r in rows
            @printf(io,"%s,%d,%d,%.6f,%.10f,%.6e\n",r.method,r.p,r.D,r.time_s,r.rho,r.err)
        end
    end
    p2 = plot(title="ρ(H) error vs p — mirror Mathieu (arbiter-confirmed reference)",
              xlabel="p", ylabel="|ρ − ρ_ref|",
              xscale=:log10, yscale=:log10, legend=:bottomleft)
    p4 = plot(title="Work-precision (error vs CPU time)",
              xlabel="CPU time (s)", ylabel="|ρ − ρ_ref|",
              xscale=:log10, yscale=:log10, legend=:bottomleft)
    for (name,col,mk) in methcfg
        sel = filter(r->r.method==name, rows); isempty(sel)&&continue
        pv=[r.p for r in sel]; tv=[max(r.time_s,1e-4) for r in sel]
        ev=[max(r.err,1e-12) for r in sel]
        se=log_slope(pv,ev)
        plot!(p2,pv,ev,marker=mk,color=col,label="$name (sl≈$se)")
        plot!(p4,tv,ev,marker=mk,color=col,label="$name")
    end
    savefig(plot(p2,p4,layout=(1,2),size=(1400,550)), png_path)
end

# OUTERMOST loop: resolution. All methods computed at each p, then CSV+PNG
# saved immediately → the figure is watchable in real time.
stopped = Set{String}()
for p in ps_all
    for (name,_,_) in methcfg
        name in stopped && continue
        try
            D,t,ρ = startswith(name,"SDM") ? bench_sdm(parse(Int,name[4:end]), p) :
                                             bench_v7(parse(Int,name[3:end]), p)
            err = abs(ρ - ρ_ref)
            @printf("p=%4d %-5s D=%8d t=%8.4fs ρ=%.10f err=%.2e\n", p,name,D,t,ρ,err)
            push!(rows, BenchRow(name,p,D,t,ρ,err))
            t > TIME_LIMIT && (push!(stopped,name); println("  → $name hit time limit"))
        catch e
            @warn "$name p=$p" e; push!(stopped,name)
        end
        flush(stdout)
    end
    _write(rows)                          # save figure after EVERY resolution
    length(stopped) == length(methcfg) && break
end

println("\n══ slopes (err vs p) ══")
for (name,_,_) in methcfg
    sel=filter(r->r.method==name, rows); isempty(sel)&&continue
    @printf("  %-5s slope %.2f   best err %.2e at t=%.2fs\n", name,
            log_slope([r.p for r in sel],[r.err for r in sel]),
            minimum(r->r.err, sel), sel[argmin([r.err for r in sel])].time_s)
end
println("done — $csv_path, $png_path")
