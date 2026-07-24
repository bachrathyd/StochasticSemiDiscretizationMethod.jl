# =============================================================================
# PUBLICATION-QUALITY error–resolution diagram (journal figure) on the HARD
# PD-Mathieu with strong noise. Same 12 methods as grand_orders.jl:
#   * original Stoch-SDM (MF factored, q=2 and q=4)
#   * new engine WITHOUT IBP: v8-direct, GL1..GL5
#   * new engine WITH IBP:    v8-IBP,   GL1..GL5
# Differences vs the exploratory run:
#   * dense p grid (~10 points/decade, up to 4096)
#   * time cap 60 s per solve (journal run — we can wait)
#   * order-guide lines, journal styling, 300 dpi PNG + vector PDF
# Live figure: highorder/out_grand_orders_pub.png re-saved after EVERY point.
# Final copies go to the Drive journal_paper/images folder.
# =============================================================================
using Pkg; Pkg.activate("D:/BD/StochasticSemiDiscretizationMethod.jl/benchmark")
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StochasticSemiDiscretizationMethod: StepV8, _lagr_coefs, _lint, _G8, gl_tab,
      Prob, build_v8m, rho_H_krylov_v8m, fixPoint_v8m
using StaticArrays, LinearAlgebra, Plots, Printf
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "cov_colloc_v8_ibp.jl"))

const ρref  = 1.324866438112
const FLOOR = 5e-12
const TCAP  = 60.0
const PNG   = joinpath(@__DIR__, "out_grand_orders_pub.png")
const PDF   = joinpath(@__DIR__, "out_grand_orders_pub.pdf")
const CSV   = joinpath(@__DIR__, "grand_orders_pub.csv")
const PAPER_IMG = raw"D:\BD\ssdm-ps\journal_paper\images"

# hard PD-Mathieu (identical to v8ibp_hard.jl / grand_orders.jl)
Afun(t)=[0.0 1.0; -(1.0+0.8cos(2π*t)) -0.1]
Bfun(t)=[0.0 0.0; 0.40*(1+0.3cos(2π*t)) 0.45*(1+0.4cos(2π*t))]
αfun(t)=[0.0 0.0; 0.5 0.0]
βfun(t)=[0.0 0.0; 0.35 0.0]
pb = Prob(2,1.0,1.0, Afun, Bfun, αfun, βfun)
const ROUGH=[2]; const POSMAP=Dict(2=>1)

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
push!(methods, ("SDM q=4", p->ρ_sdm(4,p), :gray40, 0))
for S in 1:5
    push!(methods, ("v8 GL$S",  p->rho_H_krylov_v8m(build_v8m(pb,S,p)), :none, S))
    push!(methods, ("IBP GL$S", p->rho_H_krylov_v8m(build_v8ibp(pb,S,p,ROUGH,POSMAP)), :none, S))
end
scol = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)

data = Dict(m[1]=>(ps=Int[], errs=Float64[], ts=Float64[]) for m in methods)
stopped = Set{String}()

function redraw(; final=false)
    plt = plot(xlabel="p  (steps per principal period)",
               ylabel="|ρ − ρ_ref|",
               xscale=:log10, yscale=:log10, legend=:outerright,
               size=(1200,760), framestyle=:box, dpi=300,
               guidefontsize=13, tickfontsize=11, legendfontsize=10,
               left_margin=6Plots.mm, bottom_margin=5Plots.mm,
               minorgrid=true, gridalpha=0.25, minorgridalpha=0.10,
               xticks=10.0 .^ (0:4), yticks=10.0 .^ (-13:2:1))
    # order-guide lines anchored at (p=8, err chosen per order)
    for (ord, anch) in ((1,3e-2),(2,1e-2),(4,3e-3),(6,1e-3),(8,3e-4))
        pg = [8.0, 4096.0]
        eg = anch .* (8.0 ./ pg).^ord
        eg[2] < 1e-14 && (pg[2] = 8.0*(anch/1e-14)^(1/ord); eg[2]=1e-14)
        plot!(plt, pg, eg, color=:gray70, ls=:dot, lw=1, label="",
              annotations=(pg[1]*1.35, eg[1]*1.6,
                           Plots.text("O(p⁻$ord)", 8, :gray50, :left)))
    end
    hline!(plt, [FLOOR], color=:gray, ls=:dashdot, lw=1, label="noise floor")
    for (name, col, S) in [(m[1],m[3],m[4]) for m in methods]
        d = data[name]; isempty(d.ps) && continue
        c   = S==0 ? col : scol[S]
        ls  = startswith(name,"IBP") ? :dash : :solid
        mk  = startswith(name,"SDM") ? :circle : (startswith(name,"IBP") ? :diamond : :utriangle)
        plot!(plt, d.ps, max.(d.errs, 1e-14), marker=mk, markersize=4.5,
              markerstrokewidth=0.4, lw=1.8, ls=ls, color=c, label=name)
    end
    savefig(plt, PNG)
    final && savefig(plt, PDF)
    open(CSV,"w") do io
        println(io,"method,p,err,t")
        for (name,_,_,_) in methods, i in eachindex(data[name].ps)
            @printf(io,"%s,%d,%.6e,%.3f\n", name, data[name].ps[i], data[name].errs[i], data[name].ts[i])
        end
    end
    plt
end

# dense grid: start low (p=2,3) so the high-order slopes are visible before
# they hit the floor; ~10 points per decade above.
ps_all = [2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128,
          160,192,224,256,320,384,448,512,640,768,896,1024,
          1280,1536,1792,2048,2560,3072,3584,4096]
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
        # keep ~5 points past the floor so the saturation plateau is visible
        if count(e -> e < FLOOR, d.errs) ≥ 5
            push!(stopped, name)
            println("  → $name saturated on the noise floor"); flush(stdout)
        end
    end
    length(stopped) == length(methods) && break
end

redraw(final=true)
for f in (basename(PNG), basename(PDF), basename(CSV))
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG, replace(f,"out_"=>"")); force=true)
    catch e; @warn "copy to paper images failed" f e end
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
println("done — $PNG / $PDF")
