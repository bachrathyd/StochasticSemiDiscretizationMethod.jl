# =============================================================================
# SSV milling stability + quality chart (journal figure, fully reproducible).
# 1-DOF milling with spindle speed variation (dimensionless, Insperger–Stépán):
#   x'' + 2ζ x' + x = -w h(t) [x(t) − x(t−τ(t))] (1 + σc Ẇ1) + σa Ẇ2
# Per grid point (Ω0, w) FOUR quantities are computed with the MF factored
# solver:
#   ρ_stoch : second-moment spectral radius of the SSV process
#   ρ_det   : noise-off spectral radius (= ρ(Φ)², deterministic stability)
#   Var(x)  : stationary variance at period start (where ρ_stoch < 1)
#   ρ_cs,det: deterministic radius of the constant-speed process (classic lobes)
# Chart layers: log10 Var colormap over the stochastically stable region,
# deterministic limit (black dash-dot), stochastic limit (blue), variance
# quality limit (red dashed), constant-speed stochastic limit (gray dotted).
# T_SSV = NT(=10) revolutions; Δt resolves both the delay (τmax/24) and the
# natural period (2π/30). Incremental CSV+PNG per Ω0 column.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Plots, Printf, DelimitedFiles
BLAS.set_num_threads(1)

const N_TEETH=2; const aD=0.5; const KtKn=0.3
const RVA=0.10; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10
const VARLIM=0.25
const R_RES=24; const NAT_RES=30
const CSV = joinpath(@__DIR__, "ssv_chart.csv")
const PNG = joinpath(@__DIR__, "ssv_chart.png")
const PDF = joinpath(@__DIR__, "ssv_chart.pdf")
const PAPER_IMG = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"

function hfun(t, Ω0, Tssv, rva)
    φ0 = rva==0 ? Ω0*t : Ω0*t - (Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
    φen = acos(2aD - 1); φex = float(π)
    hsum = 0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        (φen ≤ φ ≤ φex) && (hsum += sin(φ)*(cos(φ) + KtKn*sin(φ)))
    end
    hsum
end

# generic solve: returns (ρ, Var-or-NaN)
function solve_point(Ω0, w; rva=RVA, σcv=σc, σav=σa, want_var::Bool=false)
    Tssv = NT * 2π/Ω0
    τf(t) = (2π/N_TEETH)/(Ω0*(1+rva*sin(2π*t/Tssv)))
    τmax  = (2π/N_TEETH)/(Ω0*(1-rva))
    Af(t) = @SMatrix [0. 1.; -(1.0 + w*hfun(t,Ω0,Tssv,rva)) -2ζ]
    Bf(t) = @SMatrix [0. 0.; w*hfun(t,Ω0,Tssv,rva) 0.]
    af(t) = @SMatrix [0. 0.; σcv*w*hfun(t,Ω0,Tssv,rva) 0.]
    bf(t) = @SMatrix [0. 0.; -σcv*w*hfun(t,Ω0,Tssv,rva) 0.]
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
        [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τf, bf))],
        Additive(2), [stAdditive(1,Additive(@SVector [0., σav]))])
    Δt  = min(τmax/R_RES, 2π/NAT_RES)
    nst = max(1, Int(round(Tssv/Δt)))
    Δt  = Tssv/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                                n_steps=nst, calculate_additive=want_var)
    ρ = spectralRadiusOfMapping_MF_factored(rst)
    v = NaN
    if want_var && ρ < 0.999
        try; v = fixPointOfMapping_MF_factored(rst)[1]; catch; end
    end
    (ρ, v)
end

Ω0s = collect(range(0.16, 1.50, length=80))
ws  = collect(range(0.01, 2.40, length=56))
Rho  = fill(NaN, length(ws), length(Ω0s))   # stochastic SSV
Rdet = fill(NaN, length(ws), length(Ω0s))   # deterministic SSV
Rcs  = fill(NaN, length(ws), length(Ω0s))   # deterministic constant-speed (classic lobes)
Var  = fill(NaN, length(ws), length(Ω0s))
t_start = time()

function redraw()
    open(CSV,"w") do io
        println(io,"Omega0,w,rho,rho_det,rho_cs_det,var")
        for (j,Ω) in enumerate(Ω0s), (i,w) in enumerate(ws)
            isnan(Rho[i,j]) && continue
            @printf(io,"%.5f,%.5f,%.8f,%.8f,%.8f,%.8e\n",
                    Ω, w, Rho[i,j], Rdet[i,j], Rcs[i,j], Var[i,j])
        end
    end
    done = findall(j -> !isnan(Rho[1,j]), eachindex(Ω0s))
    length(done) < 2 && return
    jj = done
    plt = plot(xlabel="dimensionless spindle speed  Ω₀", ylabel="depth of cut  w",
               size=(1050,640), framestyle=:box, dpi=300,
               guidefontsize=13, tickfontsize=11, legendfontsize=9,
               left_margin=5Plots.mm, bottom_margin=5Plots.mm,
               xlim=(Ω0s[1],Ω0s[end]), ylim=(0,ws[end]), legend=:topleft)
    # colormap: log10 stationary variance over the stochastically stable region
    L = map(eachindex(Var)) do k
        (isnan(Var[k]) || Rho[k] ≥ 1.0) ? NaN : log10(max(Var[k],1e-6))
    end
    L = reshape(L, size(Var))
    heatmap!(plt, Ω0s[jj], ws, L[:,jj], color=:viridis,
             colorbar_title="log₁₀ Var(x)   (stable region)", clims=(-2.5, 1.5))
    contour!(plt, Ω0s[jj], ws, Rdet[:,jj], levels=[1.0], lw=2.2, color=:black, ls=:dashdot)
    contour!(plt, Ω0s[jj], ws, Rho[:,jj],  levels=[1.0], lw=2.6, color=:blue3)
    Vc = map(x -> isnan(x) ? 1e6 : x, Var)
    contour!(plt, Ω0s[jj], ws, Vc[:,jj], levels=[VARLIM], lw=2.6, color=:red3, ls=:dash)
    contour!(plt, Ω0s[jj], ws, Rcs[:,jj], levels=[1.0], lw=1.8, color=:gray35, ls=:dot)
    plot!(plt, [NaN],[NaN], color=:gray35, lw=1.8, ls=:dot, label="1. constant speed, deterministic (classic lobes)")
    plot!(plt, [NaN],[NaN], color=:black, lw=2.2, ls=:dashdot, label="2. SSV, deterministic  ρ(Φ)=1")
    plot!(plt, [NaN],[NaN], color=:blue3, lw=2.6, label="3. SSV, 2nd-moment  ρ(H)=1")
    plot!(plt, [NaN],[NaN], color=:red3, lw=2.6, ls=:dash, label="4. SSV, quality limit  Var(x)=$(VARLIM)")
    try savefig(plt, PNG); savefig(plt, PDF)
    catch e; @warn "savefig skipped (transient lock)" e end
end

for (j,Ω) in enumerate(Ω0s)
    t=@elapsed Threads.@threads for i in eachindex(ws)
        ρ,v  = solve_point(Ω, ws[i]; want_var=true)
        ρd,_ = solve_point(Ω, ws[i]; σcv=0.0, σav=0.0)
        ρ0,_ = solve_point(Ω, ws[i]; rva=0.0, σcv=0.0, σav=0.0)
        Rho[i,j]=ρ; Var[i,j]=v; Rdet[i,j]=ρd; Rcs[i,j]=ρ0
    end
    @printf("Ω0=%.3f  column done (%.1fs)  [%d/%d]\n", Ω, t, j, length(Ω0s)); flush(stdout)
    redraw()
end
@printf("TOTAL chart time: %.1f s (%d points × 3 solves, %d threads)\n",
        time()-t_start, length(Ω0s)*length(ws), Threads.nthreads())
for Ω in (Ω0s[1], 1.0, Ω0s[end])
    Tssv = NT*2π/Ω; τmax=(2π/N_TEETH)/(Ω*(1-RVA))
    Δt=min(τmax/R_RES, 2π/NAT_RES); nst=Int(round(Tssv/Δt))
    @printf("Ω0=%.2f: T_SSV=%.1f, τmax=%.2f, Δt=%.4f, n_steps/period=%d\n",
            Ω, Tssv, τmax, Δt, nst)
end
for f in ("ssv_chart.png","ssv_chart.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG,f); force=true) catch e; @warn e end
end
println("done — $PNG")
