# =============================================================================
# Work-precision replotted against STORED HISTORY POINTS, not just p.
# A GL-S collocation step stores (2S+2) sub-blocks of history (endpoint + S
# stage values + S+1 integrated-history DOFs); classical SDM stores 1 (the
# nodal value) regardless of its interpolation order q. The fair "resolution"
# axis is therefore N_stored = p·mult, mult = 2S+2 for GL-S, mult = 1 for SDM.
# We show BOTH axes side by side for each quantity:
#   left  column: error vs p            (steps per period — the naive axis)
#   right column: error vs N_stored     (stored history sub-points per period)
# Rows: spectral radius (top), stationary 2nd moment (bottom).
# Re-plots from the CSVs; no recomputation.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf, DelimitedFiles

const ROWS = [
    (csv="out_grand_orders_pub.csv", floor=2e-11, ylab="|ρ − ρ_ref|",     tag="spectral radius"),
    (csv="out_grand_orders_fix.csv", floor=5e-11, ylab="|Var(q) − Var_ref|", tag="stationary 2nd moment"),
]
const ORDER = ["SDM q=2","SDM q=4","v8 GL1","IBP GL1","v8 GL2","IBP GL2",
               "v8 GL3","IBP GL3","v8 GL4","IBP GL4","v8 GL5","IBP GL5"]
scol = Dict(1=>:dodgerblue, 2=>:seagreen, 3=>:purple, 4=>:darkorange, 5=>:crimson)
colof(n) = n=="SDM q=2" ? :black : n=="SDM q=4" ? :gray40 : scol[parse(Int,n[end])]
lsof(n)  = startswith(n,"IBP") ? :dash : :solid
mkof(n)  = startswith(n,"SDM") ? :circle : (startswith(n,"IBP") ? :diamond : :utriangle)
# stored sub-blocks per step: SDM stores 1 nodal block; GL-S stores 2S+2.
multof(n) = startswith(n,"SDM") ? 1 : (2*parse(Int,n[end]) + 2)

function load(csv)
    raw,_ = readdlm(joinpath(@__DIR__,csv), ','; header=true)
    names=String.(raw[:,1]); ps=Int.(raw[:,2]); err=Float64.(raw[:,3])
    Dict(n => (p=ps[names.==n], e=err[names.==n]) for n in unique(names))
end
function trunc_floor(x,e,flr)
    ic=findfirst(<(flr), e)
    ic===nothing ? (x,e) : (x[1:min(end,ic+2)], e[1:min(end,ic+2)])
end

panels = Plots.Plot[]
for (ri,row) in enumerate(ROWS)
    D = load(row.csv)
    for (ci, xmode) in enumerate((:p, :stored))
        showleg = (ri==1 && ci==2)
        xlab = xmode==:p ? "p  (steps per period)" : "N = p·(2S+2)  (stored history sub-points)"
        ttl  = ri==1 ? (xmode==:p ? "vs resolution p" : "vs stored points N") : ""
        pl = plot(xscale=:log10, yscale=:log10, framestyle=:box,
                  xlabel = ri==2 ? xlab : "", ylabel = ci==1 ? row.ylab : "",
                  title=ttl, titlefontsize=11,
                  legend = showleg ? :outertopright : false,
                  guidefontsize=10, tickfontsize=8, legendfontsize=7,
                  minorgrid=true, gridalpha=0.22, minorgridalpha=0.08)
        hline!(pl,[row.floor],color=:gray,ls=:dashdot,lw=1,label= showleg ? "floor" : "")
        for n in ORDER
            haskey(D,n) || continue
            x = xmode==:p ? float.(D[n].p) : float.(D[n].p) .* multof(n)
            xx,ee = trunc_floor(x, D[n].e, row.floor)
            plot!(pl, xx, max.(ee,1e-13), color=colof(n), ls=lsof(n), marker=mkof(n),
                  ms=3.2, markerstrokewidth=0.3, lw=1.5, label= showleg ? n : "")
        end
        push!(panels, pl)
    end
end

plt = plot(panels..., layout=(2,2), size=(1350,900), dpi=300,
           left_margin=7Plots.mm, bottom_margin=6Plots.mm, top_margin=3Plots.mm,
           plot_title="Work-precision: resolution p (left) vs stored history points N=p·(2S+2) (right)",
           plot_titlefontsize=12)
savefig(plt, joinpath(@__DIR__,"out_grand_stored.png"))
savefig(plt, joinpath(@__DIR__,"out_grand_stored.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
for f in ("out_grand_stored.png","out_grand_stored.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(dst, replace(f,"out_"=>"")); force=true) catch e; @warn e end
end
println("done — out_grand_stored.png/pdf (+ paper images/grand_stored.*)")
