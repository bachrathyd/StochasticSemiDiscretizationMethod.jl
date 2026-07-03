# =============================================================================
# SSV milling stability + quality chart (journal figure, fully reproducible).
# 1-DOF milling with spindle speed variation (dimensionless, Insperger–Stépán):
#   x'' + 2ζ x' + x = -w h(t) [x(t) − x(t−τ(t))] (1 + σc Ẇ) + σa Ẇ
# h(t): directional factor (down-milling, N teeth, radial immersion aD),
# SSV:  Ω(t) = Ω0 (1 + RVA sin(2π t/T_ssv)), τ(t) = (2π/N)/Ω(t),
#       T_ssv = NT revolutions; principal period = T_ssv.
# Multiplicative noise enters the cutting coefficient (present AND delayed
# reads, β ≠ 0), plus weak additive noise — the full-featured problem class.
# Chart: (Ω0, w) grid; ρ(H)=1 stability boundary + stationary-variance quality
# boundary Var(x) = VARLIM. Incremental CSV+PNG per Ω0 column (live viewing).
# Gate (probed): noise-off ρ(H) vs deterministic-package ρ(Φ)², rel ≈ 2e-5 @r=32.
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Plots, Printf, DelimitedFiles
BLAS.set_num_threads(1)

const N_TEETH=2; const aD=0.5; const KtKn=0.3
const RVA=0.10; const NT=2; const ζ=0.02
const σc=0.30; const σa=0.10
const VARLIM=0.25
const R_RES=24
const CSV = joinpath(@__DIR__, "ssv_chart.csv")
const PNG = joinpath(@__DIR__, "ssv_chart.png")
const PDF = joinpath(@__DIR__, "ssv_chart.pdf")
const PAPER_IMG = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"

function hfun(t, Ω0, Tssv)
    φ0 = Ω0*t - (Ω0*RVA*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
    φen = acos(2aD - 1); φex = float(π)
    hsum = 0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        (φen ≤ φ ≤ φex) && (hsum += sin(φ)*(cos(φ) + KtKn*sin(φ)))
    end
    hsum
end

function point(Ω0, w; want_var::Bool)
    Tssv = NT * 2π/Ω0
    τf(t) = (2π/N_TEETH)/(Ω0*(1+RVA*sin(2π*t/Tssv)))
    τmax  = (2π/N_TEETH)/(Ω0*(1-RVA))
    Af(t) = @SMatrix [0. 1.; -(1.0 + w*hfun(t,Ω0,Tssv)) -2ζ]
    Bf(t) = @SMatrix [0. 0.; w*hfun(t,Ω0,Tssv) 0.]
    af(t) = @SMatrix [0. 0.; σc*w*hfun(t,Ω0,Tssv) 0.]
    bf(t) = @SMatrix [0. 0.; -σc*w*hfun(t,Ω0,Tssv) 0.]
    lddep = LDDEProblem(ProportionalMX(Af), [DelayMX(τf, Bf)],
        [stCoeffMX(1,ProportionalMX(af))], [stCoeffMX(1,DelayMX(τf, bf))],
        Additive(2), [stAdditive(1,Additive(@SVector [0., σa]))])
    Δt  = τmax/R_RES
    nst = max(1, Int(round(Tssv/Δt)))
    Δt  = Tssv/nst
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, Δt), τmax;
                                n_steps=nst, calculate_additive=want_var)
    ρ = spectralRadiusOfMapping_MF_factored(rst)
    v = NaN
    if want_var && ρ < 0.999
        try
            m = fixPointOfMapping_MF_factored(rst)
            v = m[1]                      # Var of x at the newest node
        catch e
        end
    end
    (ρ, v)
end

Ω0s = collect(range(0.16, 1.50, length=80))
ws  = collect(range(0.005, 1.20, length=56))
Rho = fill(NaN, length(ws), length(Ω0s))
Var = fill(NaN, length(ws), length(Ω0s))

function redraw()
    open(CSV,"w") do io
        println(io,"Omega0,w,rho,var")
        for (j,Ω) in enumerate(Ω0s), (i,w) in enumerate(ws)
            isnan(Rho[i,j]) && continue
            @printf(io,"%.5f,%.5f,%.8f,%.8e\n", Ω, w, Rho[i,j], Var[i,j])
        end
    end
    done = findall(j -> !isnan(Rho[1,j]), eachindex(Ω0s))
    length(done) < 2 && return
    jj = done
    plt = plot(xlabel="dimensionless spindle speed  Ω₀", ylabel="depth of cut  w",
               size=(1000,640), framestyle=:box, dpi=300,
               guidefontsize=13, tickfontsize=11, legendfontsize=10,
               left_margin=5Plots.mm, bottom_margin=5Plots.mm,
               xlim=(Ω0s[1],Ω0s[end]), ylim=(0,ws[end]), legend=:topleft)
    Z = Rho[:,jj]
    contourf!(plt, Ω0s[jj], ws, log10.(max.(Z,1e-3)), levels=24, alpha=0.25,
              color=:balance, colorbar=false)
    contour!(plt, Ω0s[jj], ws, Z, levels=[1.0], lw=2.5, color=:blue3)
    V = Var[:,jj]
    if any(x -> !isnan(x), V)
        Vc = map(x -> isnan(x) ? 1e6 : x, V)   # unstable/failed = above limit
        contour!(plt, Ω0s[jj], ws, Vc, levels=[VARLIM], lw=2.5, color=:red3, ls=:dash)
    end
    plot!(plt, [NaN],[NaN], color=:blue3, lw=2.5, label="ρ(H) = 1 (2nd-moment stability)")
    plot!(plt, [NaN],[NaN], color=:red3, lw=2.5, ls=:dash, label="Var(x) = $(VARLIM) (quality)")
    savefig(plt, PNG); savefig(plt, PDF)
end

for (j,Ω) in enumerate(Ω0s)
    t=@elapsed Threads.@threads for i in eachindex(ws)
        ρ,v = point(Ω, ws[i]; want_var=true)
        Rho[i,j]=ρ; Var[i,j]=v
    end
    @printf("Ω0=%.3f  column done (%.1fs)  [%d/%d]\n", Ω, t, j, length(Ω0s)); flush(stdout)
    redraw()
end
for f in ("ssv_chart.png","ssv_chart.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG, replace(f,"ssv_chart"=>"ssv_linear_iter29")); force=true)
    catch e; @warn e end
end
println("done — $PNG")
