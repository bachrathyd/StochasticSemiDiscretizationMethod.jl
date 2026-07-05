# Final journal rendering of the SSV milling chart from benchmark/ssv_chart.csv.
# Three zones: unstable (white), 2nd-moment stable but variance above the
# quality limit (light blue), and the safe process window Var ≤ VARLIM (green).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Plots, DelimitedFiles, Printf

const VARLIM = 0.25
raw,_ = readdlm(joinpath(@__DIR__,"ssv_chart.csv"), ','; header=true)
Ωv=Float64.(raw[:,1]); wv=Float64.(raw[:,2]); ρv=Float64.(raw[:,3]); Vv=Float64.(raw[:,4])
Ω0s=sort(unique(Ωv)); ws=sort(unique(wv))
iΩ=Dict(x=>i for (i,x) in enumerate(Ω0s)); iw=Dict(x=>i for (i,x) in enumerate(ws))
Rho=fill(NaN,length(ws),length(Ω0s)); Var=fill(NaN,length(ws),length(Ω0s))
for k in eachindex(Ωv)
    Rho[iw[wv[k]],iΩ[Ωv[k]]]=ρv[k]; Var[iw[wv[k]],iΩ[Ωv[k]]]=Vv[k]
end
# zone: 0 unstable, 1 stable but Var>lim (or Var failed), 2 quality window
Z = map(eachindex(Rho)) do k
    Rho[k] ≥ 1.0 ? 0.0 : ((!isnan(Var[k]) && Var[k] ≤ VARLIM) ? 2.0 : 1.0)
end
Z = reshape(Z, size(Rho))

plt = plot(xlabel="dimensionless spindle speed  Ω₀", ylabel="depth of cut  w",
           size=(1000,640), framestyle=:box, dpi=300,
           guidefontsize=13, tickfontsize=11, legendfontsize=10,
           left_margin=5Plots.mm, bottom_margin=5Plots.mm,
           xlim=(Ω0s[1],Ω0s[end]), ylim=(0,ws[end]), legend=:topleft)
contourf!(plt, Ω0s, ws, Z, levels=[-0.5,0.5,1.5,2.5],
          color=cgrad([:white, RGBA(0.68,0.85,1.0,1), RGBA(0.60,0.90,0.60,1)], 3,
          categorical=true), lw=0, colorbar=false)
contour!(plt, Ω0s, ws, Rho, levels=[1.0], lw=2.5, color=:blue3)
Vc = map(x -> isnan(x) ? 1e6 : x, Var)
contour!(plt, Ω0s, ws, Vc, levels=[VARLIM], lw=2.5, color=:red3, ls=:dash)
# RVA=0 (constant speed) stability baseline, if computed
rva0file = joinpath(@__DIR__,"ssv_rva0.csv")
if isfile(rva0file)
    raw0,_ = readdlm(rva0file, ','; header=true)
    R0 = fill(NaN,length(ws),length(Ω0s))
    for k in 1:size(raw0,1)
        R0[iw[raw0[k,2]],iΩ[raw0[k,1]]] = raw0[k,3]
    end
    contour!(plt, Ω0s, ws, R0, levels=[1.0], lw=1.8, color=:gray40, ls=:dot)
    plot!(plt, [NaN],[NaN], color=:gray40, lw=1.8, ls=:dot,
          label="ρ(H) = 1 at RVA = 0  (constant speed)")
end
plot!(plt, [NaN],[NaN], color=:blue3, lw=2.5, label="ρ(H) = 1  (2nd-moment stability limit)")
plot!(plt, [NaN],[NaN], color=:red3, lw=2.5, ls=:dash, label="Var(x) = $(VARLIM)  (surface-quality limit)")
scatter!(plt, [NaN],[NaN], marker=:square, ms=8, color=RGBA(0.68,0.85,1.0,1),
         markerstrokewidth=0, label="stable but quality-violating")
scatter!(plt, [NaN],[NaN], marker=:square, ms=8, color=RGBA(0.60,0.90,0.60,1),
         markerstrokewidth=0, label="safe process window")
savefig(plt, joinpath(@__DIR__,"ssv_chart.png"))
savefig(plt, joinpath(@__DIR__,"ssv_chart.pdf"))
dst = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"
cp(joinpath(@__DIR__,"ssv_chart.png"), joinpath(dst,"ssv_chart.png"); force=true)
cp(joinpath(@__DIR__,"ssv_chart.pdf"), joinpath(dst,"ssv_chart.pdf"); force=true)
println("done — ssv chart final render")
