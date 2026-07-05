# =============================================================================
# Triple work-precision diagram (journal figure), 2 rows × 3 columns:
#   row 1 = spectral radius ρ(H)      (data: out_grand_orders_pub.csv)
#   row 2 = stationary 2nd moment Var (data: out_grand_orders_fix.csv)
#   col 1 = error vs p                (accuracy per resolution)
#   col 2 = CPU time vs p             (cost per resolution)
#   col 3 = error vs CPU time         (accuracy per cost — the practical tradeoff)
# Re-plots from the CSVs; no recomputation. Error curves truncated a couple of
# points past the solver floor; timing curves shown in full.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf, DelimitedFiles

const ROWS = [
    (csv="out_grand_orders_pub.csv", floor=2e-11, ylab="|ρ − ρ_ref|",
     tag="(spectral radius)"),
    (csv="out_grand_orders_fix.csv", floor=5e-11, ylab="|Var(q) − Var_ref|",
     tag="(stationary 2nd moment)"),
]
const ORDER = ["SDM q=2","SDM q=4","v8 GL1","IBP GL1","v8 GL2","IBP GL2",
               "v8 GL3","IBP GL3","v8 GL4","IBP GL4","v8 GL5","IBP GL5"]
scol = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)
colof(n) = n=="SDM q=2" ? :black : n=="SDM q=4" ? :gray40 : scol[parse(Int,n[end])]
lsof(n)  = startswith(n,"IBP") ? :dash : :solid
disp(n) = replace(replace(n, "v8 GL"=>"GL-"), "IBP GL"=>"IBP GL-")
mkof(n)  = startswith(n,"SDM") ? :circle : (startswith(n,"IBP") ? :diamond : :utriangle)

function load(csv)
    raw,_ = readdlm(joinpath(@__DIR__,csv), ','; header=true)
    names=String.(raw[:,1]); ps=Int.(raw[:,2]); err=Float64.(raw[:,3]); t=Float64.(raw[:,4])
    Dict(n => (p=ps[names.==n], e=err[names.==n], t=t[names.==n]) for n in unique(names))
end
# show all collected points (runs stop ~5 past the floor) so saturation shows
trunc_floor(p,e,t,flr) = (p,e,t)

panels = Plots.Plot[]
for (ri,row) in enumerate(ROWS)
    D = load(row.csv)
    showleg = ri==1              # legend only once (top-right panel)
    # --- col 1: error vs p ---
    p1 = plot(xscale=:log10, yscale=:log10, framestyle=:box,
              xlabel = ri==2 ? "p (steps/period)" : "", ylabel=row.ylab,
              title = ri==1 ? "error vs resolution" : "", titlefontsize=11,
              legend=false, guidefontsize=10, tickfontsize=8)
    hline!(p1,[row.floor],color=:gray,ls=:dashdot,lw=1,label="")
    for n in ORDER
        haskey(D,n) || continue
        p,e,t = trunc_floor(D[n].p, D[n].e, D[n].t, row.floor)
        plot!(p1, p, max.(e,1e-13), color=colof(n), ls=lsof(n), marker=mkof(n),
              ms=3, markerstrokewidth=0.3, lw=1.4, label="")
    end
    # --- col 2: CPU time vs p ---
    p2 = plot(xscale=:log10, yscale=:log10, framestyle=:box,
              xlabel = ri==2 ? "p (steps/period)" : "", ylabel="CPU time [s]",
              title = ri==1 ? "cost vs resolution" : "", titlefontsize=11,
              legend=false, guidefontsize=10, tickfontsize=8)
    for n in ORDER
        haskey(D,n) || continue
        plot!(p2, D[n].p, max.(D[n].t,1e-4), color=colof(n), ls=lsof(n),
              marker=mkof(n), ms=3, markerstrokewidth=0.3, lw=1.4, label="")
    end
    # --- col 3: error vs CPU time ---
    p3 = plot(xscale=:log10, yscale=:log10, framestyle=:box,
              xlabel = ri==2 ? "CPU time [s]" : "", ylabel=row.ylab,
              title = ri==1 ? "accuracy vs cost" : "", titlefontsize=11,
              legend = showleg ? :outertopright : false,
              guidefontsize=10, tickfontsize=8, legendfontsize=7)
    hline!(p3,[row.floor],color=:gray,ls=:dashdot,lw=1,label=showleg ? "floor" : "")
    for n in ORDER
        haskey(D,n) || continue
        p,e,t = trunc_floor(D[n].p, D[n].e, D[n].t, row.floor)
        plot!(p3, max.(t,1e-4), max.(e,1e-13), color=colof(n), ls=lsof(n),
              marker=mkof(n), ms=3, markerstrokewidth=0.3, lw=1.4,
              label = showleg ? disp(n) : "")
    end
    push!(panels, p1, p2, p3)
end

plt = plot(panels..., layout=(2,3), size=(1500,860), dpi=300,
           left_margin=6Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm,
           plot_title="Work-precision: spectral radius (top) and stationary 2nd moment (bottom)",
           plot_titlefontsize=13)
savefig(plt, joinpath(@__DIR__,"out_grand_triple.png"))
savefig(plt, joinpath(@__DIR__,"out_grand_triple.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
for f in ("out_grand_triple.png","out_grand_triple.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(dst, replace(f,"out_"=>"")); force=true) catch e; @warn e end
end
println("done — out_grand_triple.png/pdf (+ paper images/grand_triple.*)")
