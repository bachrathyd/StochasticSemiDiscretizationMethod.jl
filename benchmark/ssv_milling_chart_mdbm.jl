# =============================================================================
# SSV milling stability + quality chart — MDBM edition (journal figure).
# Model: 1-DOF up-milling, cut time = 25% of the tooth-passing period
# (φ_en = 0, φ_ex = π/4 per tooth, z = 2), sinusoidal SSV with RVA = 0.25,
# T_SSV = 10 nominal revolutions, ζ = 0.02, σc = 0.20 (mult., cutting coeff.,
# present+delayed reads), σa = 0.10 (additive).
#
# Axes: Ω0 ∈ [0.125, 1.5] on a LOG axis (plain labels), w ∈ [0, 4].
# Internally everything lives in ξ = log10(Ω0), so the brute-force grid, the
# MDBM initial grid, and the plot share the same low-speed densification.
#
# Layers:
#   * BF colormap: log10 Var(x) over the mean-square stable region
#     (BF_NX log-spaced columns × BF_NW rows; w = 0 row filled analytically:
#      ρ = exp(−2ζ T_SSV), Var = σa²/(4ζ))
#   * four boundary curves via MDBM (10×6 initial grid, N_ITER refinements):
#     1. constant-speed deterministic (classic lobes)      ρ_cs,det = 1
#     2. SSV deterministic                                  ρ_det   = 1
#     3. SSV second-moment                                  ρ(H)    = 1
#     4. SSV variance quality limit                         Var(x)  = VARLIM
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using StaticArrays, LinearAlgebra, Plots, Printf, DelimitedFiles, MDBM
BLAS.set_num_threads(1)

const N_TEETH=2
const PHI_EX = π/4                       # cut arc per tooth: 25% of tooth pitch
const KtKn=0.3
const RVA=0.25; const NT=10; const ζ=0.02
const σc=0.20; const σa=0.10
const VARLIM=0.25
const R_RES=24; const NAT_RES=30
const ΩLO=0.125; const ΩHI=1.5; const WHI=4.0
const BF_NX=48; const BF_NW=32           # brute-force colormap grid (test size)
const N_ITER=4                           # MDBM refinement iterations (test; 5 final)
const CSV = joinpath(@__DIR__, "ssv_chart_bf.csv")
const CSVM= joinpath(@__DIR__, "ssv_chart_mdbm.csv")
const PNG = joinpath(@__DIR__, "ssv_chart.png")
const PDF = joinpath(@__DIR__, "ssv_chart.pdf")
const PAPER_IMG = raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images"

function hfun(t, Ω0, Tssv, rva)
    φ0 = rva==0 ? Ω0*t : Ω0*t - (Ω0*rva*Tssv/(2π))*(cos(2π*t/Tssv) - 1.0)
    hsum = 0.0
    for j in 0:N_TEETH-1
        φ = mod(φ0 + 2π*j/N_TEETH, 2π)
        (φ ≤ PHI_EX) && (hsum += sin(φ)*(cos(φ) + KtKn*sin(φ)))   # up-milling, φen=0
    end
    hsum
end

function solve_point(Ω0, w; rva=RVA, σcv=σc, σav=σa, want_var::Bool=false)
    Tssv = NT * 2π/Ω0
    if w ≤ 0                                     # trivial fill: plain damped oscillator
        return (exp(-2ζ*Tssv), σav^2/(4ζ))
    end
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

# ── brute-force colormap grid (log-spaced Ω columns) ─────────────────────────
ξs  = collect(range(log10(ΩLO), log10(ΩHI), length=BF_NX))
Ω0s = 10 .^ ξs
ws  = collect(range(0.0, WHI, length=BF_NW))
Rho  = fill(NaN, length(ws), length(Ω0s))
Var  = fill(NaN, length(ws), length(Ω0s))
t_start = time()
println("── brute-force colormap grid ($(BF_NX)×$(BF_NW)) ──"); flush(stdout)
for (j,Ω) in enumerate(Ω0s)
    t=@elapsed Threads.@threads for i in eachindex(ws)
        ρ,v = solve_point(Ω, ws[i]; want_var=true)
        Rho[i,j]=ρ; Var[i,j]=v
    end
    @printf("Ω0=%.3f col done (%.1fs) [%d/%d]\n", Ω, t, j, length(Ω0s)); flush(stdout)
end
open(CSV,"w") do io
    println(io,"Omega0,w,rho,var")
    for (j,Ω) in enumerate(Ω0s), (i,w) in enumerate(ws)
        @printf(io,"%.5f,%.5f,%.8f,%.8e\n", Ω, w, Rho[i,j], Var[i,j])
    end
end
@printf("BF grid: %.1f s\n", time()-t_start); flush(stdout)

# ── MDBM boundary curves in (ξ, w) ───────────────────────────────────────────
# sign-change functions (Float64 return; positive in the "beyond" region)
f_cs(ξ,w)  = solve_point(10.0^ξ, w; rva=0.0, σcv=0.0, σav=0.0)[1] - 1.0
f_det(ξ,w) = solve_point(10.0^ξ, w; σcv=0.0, σav=0.0)[1] - 1.0
f_sto(ξ,w) = solve_point(10.0^ξ, w)[1] - 1.0
function f_var(ξ,w)
    ρ,v = solve_point(10.0^ξ, w; want_var=true)
    (ρ ≥ 0.999 || isnan(v)) ? 1.0 : (v - VARLIM)     # positive beyond stability too
end

curves = Dict{String,Any}()
for (name, f) in (("cs_det",f_cs), ("ssv_det",f_det), ("ssv_sto",f_sto), ("ssv_var",f_var))
    t=@elapsed begin
        ax = [Axis(range(log10(ΩLO), log10(ΩHI), length=10), :ξ),
              Axis(range(0.0, WHI, length=6), :w)]
        prob = MDBM_Problem(f, ax)
        solve!(prob, N_ITER, verbosity=0, doThreadprecomp=false)
        curves[name] = getinterpolatedsolution(prob)
    end
    @printf("MDBM %-8s: %d pts (%.0fs)\n", name, length(curves[name][1]), t); flush(stdout)
end
open(CSVM,"w") do io
    println(io,"curve,xi,w")
    for (name,pts) in curves, k in eachindex(pts[1])
        @printf(io,"%s,%.6f,%.6f\n", name, pts[1][k], pts[2][k])
    end
end

# ── plot (ξ coordinate, plain Ω labels) ──────────────────────────────────────
Ωticks = [0.125, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
plt = plot(xlabel="dimensionless spindle speed  Ω₀", ylabel="depth of cut  w",
           size=(1500,520), framestyle=:box, dpi=300,
           guidefontsize=13, tickfontsize=11, legendfontsize=9,
           left_margin=5Plots.mm, bottom_margin=6Plots.mm,
           xlim=(log10(ΩLO), log10(ΩHI)), ylim=(0, WHI),
           xticks=(log10.(Ωticks), string.(Ωticks)),
           legend=:topleft, legend_column=1)
L = map(eachindex(Var)) do k
    (isnan(Var[k]) || Rho[k] ≥ 1.0) ? NaN : log10(max(Var[k],1e-6))
end
L = reshape(L, size(Var))
heatmap!(plt, ξs, ws, L, color=:viridis,
         colorbar_title="log₁₀ Var(x)   (stable region)", clims=(-2.5, 1.5))
# beyond-validity shading (see plot_ssv_mdbm.jl for the final render)
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
style = Dict("cs_det"=>(:gray35,:circle,2.0,"1. constant speed, deterministic (classic lobes)"),
             "ssv_det"=>(:black,:circle,2.4,"2. SSV, deterministic  ρ(Φ)=1"),
             "ssv_sto"=>(:blue3,:circle,2.4,"3. SSV, 2nd-moment  ρ(H)=1"),
             "ssv_var"=>(:red3,:circle,2.4,"4. SSV, quality limit  Var(x)=$(VARLIM)"))
for name in ("cs_det","ssv_det","ssv_sto","ssv_var")
    pts = curves[name]; (c,mk,ms,lab) = style[name]
    scatter!(plt, pts[1], pts[2], color=c, marker=mk, markersize=ms,
             markerstrokewidth=0, label=lab)
end
try savefig(plt, PNG); savefig(plt, PDF)
catch e; @warn "savefig failed" e end
@printf("TOTAL: %.1f s\n", time()-t_start)
for f in ("ssv_chart.png","ssv_chart.pdf")
    try cp(joinpath(@__DIR__,f), joinpath(PAPER_IMG,f); force=true) catch e; @warn e end
end
println("done — $PNG")
