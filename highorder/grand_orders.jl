# =============================================================================
# GRAND error–resolution diagram on the HARD PD-Mathieu (strong noise):
#   * original Stoch-SDM (MF, q=2 and q=4 — the best available classic method)
#   * new engine WITHOUT IBP: v8-direct, GL1..GL5
#   * new engine WITH IBP:    v8-IBP,   GL1..GL5
# Reference: 1.324866438112 (v8-IBP GL4 self-converged Δ=4.6e-11, certified to
# 9.4e-13 against the independent fine-grid arbiter — v8ibp_hard.jl run).
# Live figure: highorder/out_grand_orders.png re-saved after EVERY point.
# Stop rules per method: last 3 errors below the noise floor (5e-12), or the
# last solve took > 5 s.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Plots, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))

const ρref  = 1.324866438112
const FLOOR = 5e-12
const TCAP  = 5.0
const PNG   = joinpath(@__DIR__, "out_grand_orders.png")
const CSV   = joinpath(@__DIR__, "out_grand_orders.csv")

# hard PD-Mathieu (identical to v8ibp_hard.jl)
Afun(t)=[0.0 1.0; -(1.0+0.8cos(2π*t)) -0.1]
Bfun(t)=[0.0 0.0; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
αfun(t)=[0.0 0.0; 0.5 0.0]
βfun(t)=[0.0 0.0; 0.35 0.0]
pb = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun)
const ROUGH=[2]; const POSMAP=Dict(2=>1)

# same problem for the SDM package path
function lddep_pd()
    AM(t)=@SMatrix [0. 1.; -(1.0+0.8cos(2π*t)) -0.1]
    BM(t)=@SMatrix [0. 0.; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
    aM(t)=@SMatrix [0. 0.; 0.5 0.]
    bM(t)=@SMatrix [0. 0.; 0.35 0.]
    LDDEProblem(ProportionalMX(AM), [DelayMX(1.0,BM)],
        [stCoeffMX(1,ProportionalMX(aM))], [stCoeffMX(1,DelayMX(1.0,bM))],
        Additive(2), [stAdditive(1,Additive(@SVector [0.,0.]))])
end
ρ_sdm(q,p) = spectralRadiusOfMapping_MF_factored(
    SSDM.calculateResults(lddep_pd(), SemiDiscretization(q, 1.0/p), 1.0, n_steps=p))

# sanity: reference reproducibility
let ρchk = rho_H_krylov_v8m(build_v8ibp(pb,4,32,ROUGH,POSMAP))
    @printf("reference check: GL4-IBP p=32 = %.12f (expect ≈1.324866438066)\n", ρchk)
    @assert abs(ρchk-ρref) < 1e-9
end

methods = Vector{Tuple{String,Function,Symbol,Int}}()
push!(methods, ("SDM q=2", p->ρ_sdm(2,p), :black, 0))
push!(methods, ("SDM q=4", p->ρ_sdm(4,p), :gray, 0))
for S in 1:5
    push!(methods, ("v8 GL$S",  p->rho_H_krylov_v8m(build_v8m(pb,S,p)), :none, S))
    push!(methods, ("IBP GL$S", p->rho_H_krylov_v8m(build_v8ibp(pb,S,p,ROUGH,POSMAP)), :none, S))
end
scol = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)

data = Dict(m[1]=>(ps=Int[], errs=Float64[], ts=Float64[]) for m in methods)
stopped = Set{String}()

function redraw()
    plt = plot(title="ρ(H) error vs resolution — hard PD-Mathieu (strong noise)\nref certified 9.4e-13 vs independent arbiter",
               xlabel="p (steps per period)", ylabel="|ρ − ρ_ref|",
               xscale=:log10, yscale=:log10, legend=:outerright,
               size=(1150,700), framestyle=:box)
    hline!(plt, [FLOOR], color=:gray, ls=:dot, label="noise floor")
    for (name, _, col, S) in methods
        d = data[name]; isempty(d.ps) && continue
        c   = S==0 ? col : scol[S]
        ls  = startswith(name,"IBP") ? :dash : :solid
        mk  = startswith(name,"SDM") ? :circle : (startswith(name,"IBP") ? :diamond : :utriangle)
        plot!(plt, d.ps, max.(d.errs, 1e-14), marker=mk, ls=ls, color=c, label=name)
    end
    savefig(plt, PNG)
    open(CSV,"w") do io
        println(io,"method,p,err,t")
        for (name,_,_,_) in methods, i in eachindex(data[name].ps)
            @printf(io,"%s,%d,%.6e,%.3f\n", name, data[name].ps[i], data[name].errs[i], data[name].ts[i])
        end
    end
end

ps_all = [4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048]
for p in ps_all
    for (name, f, _, _) in methods
        name in stopped && continue
        t = @elapsed ρ = try f(p) catch e; @warn "$name p=$p" e; push!(stopped,name); continue; end
        err = abs(ρ-ρref)
        d = data[name]; push!(d.ps,p); push!(d.errs,err); push!(d.ts,t)
        @printf("p=%5d  %-9s err=%.3e  (%.2fs)%s\n", p, name, err, t,
                t>TCAP ? "  → time cap" : "")
        flush(stdout)
        redraw()                                     # live update after EVERY point
        t > TCAP && push!(stopped, name)
        if length(d.errs) ≥ 3 && all(e -> e < FLOOR, d.errs[end-2:end])
            push!(stopped, name)
            println("  → $name reached the noise floor"); flush(stdout)
        end
    end
    length(stopped) == length(methods) && break
end

println("\n══ final slopes (last 3 resolved points above floor) ══")
for (name,_,_,_) in methods
    d = data[name]
    idx = findall(e -> e > FLOOR, d.errs)
    length(idx) < 3 && continue
    i3 = idx[end-2:end]
    sl = log(d.errs[i3[1]]/d.errs[i3[3]]) / log(d.ps[i3[3]]/d.ps[i3[1]])
    @printf("  %-9s slope ≈ %.2f\n", name, sl)
end
println("done — $PNG")
