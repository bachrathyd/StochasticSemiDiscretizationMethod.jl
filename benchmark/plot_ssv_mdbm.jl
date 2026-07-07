# Final render of the SSV chart: BF colormap (hires CSV if present, else test
# CSV) + MDBM boundary curves + beyond-validity shading, log Ω axis with plain
# labels. Re-plots from CSVs only.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, Printf, DelimitedFiles

const VARLIM=0.25
const ΩLO=0.125; const ΩHI=1.5; const WHI=4.0
const PAPER_IMG = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"

bffile = "ssv_chart_bf.csv"
if isfile(joinpath(@__DIR__,"ssv_chart_bf_hires.csv"))
    nh = length(unique(readdlm(joinpath(@__DIR__,"ssv_chart_bf_hires.csv"), ','; header=true)[1][:,1]))
    nh ≥ 90 && (bffile = "ssv_chart_bf_hires.csv")   # use hires only when complete
end
raw,_ = readdlm(joinpath(@__DIR__,bffile), ','; header=true)
Ωv=Float64.(raw[:,1]); wv=Float64.(raw[:,2]); ρv=Float64.(raw[:,3]); Vv=Float64.(raw[:,4])
Ω0s=sort(unique(Ωv)); ws=sort(unique(wv))
iΩ=Dict(round(x,digits=6)=>i for (i,x) in enumerate(Ω0s))
iw=Dict(round(x,digits=5)=>i for (i,x) in enumerate(ws))
Rho=fill(NaN,length(ws),length(Ω0s)); Var=fill(NaN,length(ws),length(Ω0s))
for k in eachindex(Ωv)
    Rho[iw[round(wv[k],digits=5)],iΩ[round(Ωv[k],digits=6)]]=ρv[k]
    Var[iw[round(wv[k],digits=5)],iΩ[round(Ωv[k],digits=6)]]=Vv[k]
end
ξs = log10.(Ω0s)
println("BF source: $bffile ($(length(Ω0s))×$(length(ws)))")

Ωticks = [0.125, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
plt = plot(xlabel="dimensionless spindle speed  Ω₀", ylabel="depth of cut  w",
           size=(1500,520), framestyle=:box, dpi=300,
           guidefontsize=13, tickfontsize=11, legendfontsize=9,
           left_margin=5Plots.mm, bottom_margin=6Plots.mm,
           xlim=(log10(ΩLO), log10(ΩHI)), ylim=(0, WHI),
           xticks=(log10.(Ωticks), string.(Ωticks)),
           legend=:topleft)
L = map(eachindex(Var)) do k
    (isnan(Var[k]) || Rho[k] ≥ 1.0) ? NaN : log10(max(Var[k],1e-6))
end
heatmap!(plt, ξs, ws, reshape(L,size(Var)), color=:viridis,
         colorbar_title="log₁₀ Var(x)   (stable region)", clims=(-2.5, 1.5))
# beyond-validity shading as translucent rectangles (GR cannot stack heatmaps)
let dx=ξs[2]-ξs[1], dy=ws[2]-ws[1], Sm=reshape(map(eachindex(Var)) do k
        (!isnan(Var[k]) && Rho[k] < 1.0 && Var[k] > VARLIM) ? 1.0 : NaN
    end, size(Var))
    rects=Plots.Shape[]
    for j in eachindex(ξs), i in eachindex(ws)
        isnan(Sm[i,j]) && continue
        x0=ξs[j]-dx/2; y0=ws[i]-dy/2
        push!(rects, Plots.Shape([x0,x0+dx,x0+dx,x0],[y0,y0,y0+dy,y0+dy]))
    end
    isempty(rects) || plot!(plt, rects, fillcolor=RGBA(1,1,1,0.62), linewidth=0, label="")
end

mfile = joinpath(@__DIR__,"ssv_chart_mdbm.csv")
if isfile(mfile)
    rawm,_ = readdlm(mfile, ','; header=true)
    names=String.(rawm[:,1]); xs=Float64.(rawm[:,2]); ys=Float64.(rawm[:,3])
    style = Dict("cs_det"=>(:gray35,2.0,"1. constant speed, deterministic (classic lobes)"),
                 "ssv_det"=>(:black,2.4,"2. SSV, deterministic  ρ(Φ)=1"),
                 "ssv_sto"=>(:blue3,2.4,"3. SSV, 2nd-moment  ρ(H)=1"),
                 "ssv_var"=>(:red3,2.4,"4. SSV, quality limit  Var(x)=$(VARLIM)"))
    for name in ("cs_det","ssv_det","ssv_sto","ssv_var")
        sel = names .== name
        any(sel) || continue
        (c,ms,lab) = style[name]
        scatter!(plt, xs[sel], ys[sel], color=c, marker=:circle, markersize=ms,
                 markerstrokewidth=0, label=lab)
    end
else
    println("(no MDBM CSV yet — colormap-only preview)")
end
savefig(plt, joinpath(@__DIR__,"ssv_chart.png"))
savefig(plt, joinpath(@__DIR__,"ssv_chart.pdf"))
for f in ("ssv_chart.png","ssv_chart.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG,f); force=true) catch e; @warn e end
end
println("done — ssv_chart.png")
