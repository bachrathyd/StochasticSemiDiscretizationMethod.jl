# Arbitration of the critical stochastic Mathieu, capped at N=2048, with a
# CONTINUOUSLY UPDATED figure: highorder/out_mathieu_fine.png is re-saved
# after every computed point (arbiter N-sequence, then v7 GL3/GL4 p-sequence),
# with the candidate reference values as horizontal lines.
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))
using Printf, Plots

const PNG = joinpath(@__DIR__, "out_mathieu_fine.png")

pb_fg = FGProb(2, 4π, 2π,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])
pb_v7 = Prob(2, 4π, 2π,
    t->[0.0 1.0; -(3.0+2.0*cos(0.5*t)) -0.2],
    t->[0.0 0.0; 0.5 0.0],
    t->[0.0 0.0; -0.1*(3.0+2.0*cos(0.5*t)) -0.1*0.2],
    t->[0.0 0.0; 0.1*0.5 0.0])

arbN = Int[]; arbρ = Float64[]
v7p  = Int[]; v7ρ  = Float64[]; v7lbl = String[]
ρ_rich = Ref(NaN)

function redraw()
    plt = plot(title="critical stoch. Mathieu ρ(H) — arbiter vs v7 vs archived refs",
               xlabel="resolution (arbiter N/16  |  v7 p)", ylabel="ρ(H)",
               legend=:bottomright, size=(950,600))
    hline!(plt, [0.15622747], ls=:dash,  color=:gray,   label="archived v6GL4 p120 (0.15622747)")
    hline!(plt, [0.15622870], ls=:dot,   color=:gray,   label="archived SDM-q2-Rich (0.15622870)")
    isnan(ρ_rich[]) || hline!(plt, [ρ_rich[]], ls=:dashdot, color=:red,
                              label=@sprintf("arbiter Richardson (%.7f)", ρ_rich[]))
    isempty(arbN) || plot!(plt, arbN .÷ 16, arbρ, marker=:circle, color=:red,
                           label="arbiter (h² fine grid), x=N/16")
    for lbl in unique(v7lbl)
        idx = findall(==(lbl), v7lbl)
        isempty(idx) || plot!(plt, v7p[idx], v7ρ[idx], marker=:utriangle,
                              label="v7 causal $lbl")
    end
    ylims!(plt, 0.1555, 0.1567)
    savefig(plt, PNG)
end

println("── arbiter, N up to 2048 (figure: $PNG) ──")
for N in (256, 512, 1024, 2048)
    t0=time(); ρ=fg_rho_H(pb_fg, N)
    push!(arbN, N); push!(arbρ, ρ)
    if length(arbρ) ≥ 2
        ρ_rich[] = arbρ[end] + (arbρ[end]-arbρ[end-1])/((arbN[end]/arbN[end-1])^2-1)
    end
    @printf("  fg N=%5d ρ=%.10f  Richardson→%.10f  (%.0fs)\n", N, ρ, ρ_rich[], time()-t0)
    flush(stdout); redraw()
end

println("── v7 causal GL3/GL4 ──")
for S in (3, 4), p in (32, 48, 64, 96)
    t0=time()
    ρ = rho_H_krylov(build_v7(pb_v7, S, p); offdiag=:causal)
    push!(v7p, p); push!(v7ρ, ρ); push!(v7lbl, "GL$S")
    @printf("  GL%d p=%3d ρ=%.10f  |ρ−Rich|=%.2e  (%.0fs)\n", S, p, ρ, abs(ρ-ρ_rich[]), time()-t0)
    flush(stdout); redraw()
end
@printf("\nFINAL: arbiter Richardson %.10f | archived candidates 0.15622747 / 0.15622870\n", ρ_rich[])
