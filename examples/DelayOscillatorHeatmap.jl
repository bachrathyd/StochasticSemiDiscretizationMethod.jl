using StochasticSemiDiscretizationMethod
using StaticArrays
using MDBM
using Plots
using LaTeXStrings
using Base.Threads
using DelaunayTriangulation

BLAS.set_num_threads(1)   # avoid contention with Julia threads
gr();

# ── Problem definition ────────────────────────────────────────────────────────
function createSLDOProblem(A, B, ζ, α, β, σ)
    AMx = ProportionalMX(@SMatrix [0. 1.; -A -2ζ])
    τ1 = 2π
    BMx1 = DelayMX(τ1, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    αMx1 = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; α 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τ1, @SMatrix [0. 0.; β 0.]))
    σVec = stAdditive(1, Additive(@SVector [0., σ]))
    LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

const method = SemiDiscretization(5, 2π / 30)
const τmax = 2π + 100eps()
const _ζ = 0.05
const _σ = 0.5

# ── Step 1: high-res stability boundary via MDBM ─────────────────────────────
function foo_stab(A::Float64, B::Float64)::Float64
    lddep = createSLDOProblem(A, B, _ζ, 0.3A, 0.3B, 0.)
    rst = StochasticSemiDiscretizationMethod.calculateResults(
        lddep, method, τmax, n_steps=30)
    return log(spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst)))
end

println("Step 1: MDBM stability boundary (high resolution)...")
axis_stab = [Axis(-0.5:0.125:4.0, :A), Axis(LinRange(-1.0, 1.0, 24), :B)]
mdbm_stab = MDBM_Problem(foo_stab, axis_stab)
solve!(mdbm_stab, 3, verbosity=2)
stab_pts = getinterpolatedsolution(mdbm_stab)
println("Equivlen resolutison $(length(mdbm_stab.axes[1])) x  $(length(mdbm_stab.axes[2]))")
# Extract all MDBM-evaluated points and their log(ρ) values
eval_pts = getevaluatedpoints(mdbm_stab)        # Vector of (A, B) tuples
eval_vals = getevaluatedfunctionvalues(mdbm_stab) # Vector of log(ρ)
println("  $(length(eval_pts[1])) points evaluated by MDBM")

# # ── Step 2: brute-force heatmap on coarse grid, only stable points ────────────
# # Coarse grid covering the same domain
# nA, nB = 70, 40
# Av = range(-0.5, 4.0, length=nA)
# Bv = range(-1.0, 1.0, length=nB)
# 
# # Use MDBM evaluations to build a fast log(ρ) interpolation check:
# # We classify (A,B) as "stable" if the nearest MDBM evaluation has log(ρ)<0.
# # Fast approach: for each grid point, compute log(ρ) directly (cheap if σ=0).
# # But to avoid double work, use the MDBM stability function (no additive, fast).
# 
# pos_std = fill(NaN, nA, nB)   # sqrt(E[q²])
# 
# println("\nStep 2: brute-force heatmap on $(nA)×$(nB) grid ($(nthreads()) threads)...")
# 
# # Pre-check stability on the coarse grid (parallel, no additive → fast)
# log_rho = zeros(nA, nB)
# println("Time of Bruto force grid spectral radiuse:")
# @time @threads for ia in 1:nA
#     A = Av[ia]
#     for ib in 1:nB
#         B = Bv[ib]
#         lddep = createSLDOProblem(A, B, _ζ, 0.3A, 0.3B, 0.)
#         rst = StochasticSemiDiscretizationMethod.calculateResults(
#             lddep, method, τmax, n_steps=30)
#         log_rho[ia, ib] = log(spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst)))
#     end
# end
# 
# n_stable = count(log_rho .< 0)
# println("  $n_stable / $(nA*nB) grid points are stable")
# 
# println("Time ofsecont moment fix potin on stalbe grid poitns:")
# # Compute stationary moment only at stable points (parallel)
# @time @threads for ia in 1:nA
#     A = Av[ia]
#     for ib in 1:nB
#         log_rho[ia, ib] >= 0 && continue   # skip unstable
#         B = Bv[ib]
#         lddep = createSLDOProblem(A, B, _ζ, 0.3A, 0.3B, _σ)
#         rst = StochasticSemiDiscretizationMethod.calculateResults(
#             lddep, method, τmax, n_steps=30, calculate_additive=true)
#         dm = DiscreteMapping_M2_MF(rst)
#         r2 = div(rst.n, 2) - 1
#         fp = fixPointOfMapping_MF(dm)
#         M2 = VecToCovMx(fp, (r2 + 1) * 2)
#         pos_std[ia, ib] = (M2[1, 1])# sqrt(M2[1, 1])
#     end
# end
# 
Astab = eval_pts[1][eval_vals.<0]
Bstab = eval_pts[2][eval_vals.<0]
pos_std = zeros(length(Astab))
@time @threads for i in 1:length(Astab)

    A = Astab[i]
    B = Bstab[i]

    lddep = createSLDOProblem(A, B, _ζ, 0.3A, 0.3B, _σ)
    rst = StochasticSemiDiscretizationMethod.calculateResults(
        lddep, method, τmax, n_steps=30, calculate_additive=true)
    dm = DiscreteMapping_M2_MF(rst)
    r2 = div(rst.n, 2) - 1
    fp = fixPointOfMapping_MF(dm)
    M2 = VecToCovMx(fp, (r2 + 1) * 2)
    pos_std[i] = (M2[1, 1])# sqrt(M2[1, 1])scatter!(eval_pts[1][eval_vals.<0],eval_pts[2][eval_vals.<0])F

end

# ── Step 3: Delaunay triangulation of stable MDBM points ──────────────────────
println("\nBuilding Delaunay triangulation...")
z_vals = atan.(pos_std)   # colour values: atan(E[q²])

# All MDBM-evaluated points (stable + unstable) for centroid stability check
All_A   = eval_pts[1]
All_B   = eval_pts[2]

pts_for_tri = collect(zip(Float64.(Astab), Float64.(Bstab)))
tri = triangulate(pts_for_tri)

# Fix colour scale: cmax = π/2 so the colourbar top = "unstable" yellow
cmin = minimum(z_vals)
cmax = π / 2
cg   = cgrad(:viridis)

# ── Step 4: plot (yellow background = unstable level π/2) ─────────────────────
println("\nPlotting...")

# yellow = cg[1.0] in viridis is actually bright yellow; use it as background
bg_col = cg[1.0]

p = plot(
    xlabel          = L"A",
    ylabel          = L"B",
    title           = "Delay Oscillator — stationary " * L"E[q^2]" *
                      "  (σ=$(_σ), ζ=$(_ζ))\n" *
                      "colour: " * L"\arctan(E[q^2])" * "  (yellow = unstable)",
    guidefontsize   = 13,
    tickfont        = font(8),
    legend          = :topright,
    right_margin    = 10Plots.mm,
    background_color_inside = bg_col)

# Fill each triangle only if its centroid is inside the stable region.
# Fast check: nearest MDBM-evaluated point determines stability.
for (i, j, k) in each_solid_triangle(tri)
    cA = (Astab[i] + Astab[j] + Astab[k]) / 3
    cB = (Bstab[i] + Bstab[j] + Bstab[k]) / 3
    # Find nearest MDBM point by squared distance
    dists = @. (All_A - cA)^2 + (All_B - cB)^2
    nearest = argmin(dists)
    eval_vals[nearest] >= 0 && continue   # centroid in unstable region → skip

    c_mean = (z_vals[i] + z_vals[j] + z_vals[k]) / 3
    cn = (c_mean - cmin) / (cmax - cmin)
    fc = cg[clamp(cn, 0.0, 1.0)]
    plot!(p,
        [Astab[i], Astab[j], Astab[k], Astab[i]],
        [Bstab[i], Bstab[j], Bstab[k], Bstab[i]],
        fill        = true,
        fillcolor   = fc,
        fillalpha   = 1.0,
        linewidth   = 0,
        label       = "")
end

# Colourbar via invisible scatter (cmin → cmax=π/2)
scatter!(p, [NaN], [NaN],
    marker_z          = [cmin],
    color             = :viridis,
    colorbar          = true,
    colorbar_title    = L"\arctan(E[q^2])",
    clim              = (cmin, cmax),
    markersize        = 0,
    label             = "")

# Overlay stability boundary (high-res MDBM)
scatter!(p, stab_pts...,
    label             = "2nd moment stab. boundary",
    color             = :red,
    markersize        = 2,
    markerstrokewidth = 0)

display(p)
savefig(p, "assets/DelayOscillatorHeatmap.png")
println("Saved assets/DelayOscillatorHeatmap.png")
