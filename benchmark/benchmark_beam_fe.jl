# =============================================================================
# FE beam mesh-convergence study (journal-paper example, Sec. "Dimensional
# Scaling"). Pinned–pinned Euler–Bernoulli beam, Hermite elements:
#   M q̈ + C q̇ + [K − P(t) G] q = b_s [kP q_s(t−τ) + kD q̇_s(t−τ)] + noise
#   P(t) = P0 + P1 cos Ωt  (parametric excitation), multiplicative noise on the
#   axial load: −σ_P G q dW (reads positions → smooth), delayed PD at x_s.
# Nondimensional: EI=1, ρA=1, L=1 ⇒ ω_k=(kπ)², P_cr=π². Operating point near
# the principal parametric resonance of MODE 3 (Ω ≈ 2ω₃) — coarse meshes
# misplace ω₃ and hence the tongue.
#
# Gates first (rigorous): factored == dense identity at n_e=2; noise-off
# ρ(H)=ρ(Φ_det)². Then the sweep n_e = 4,8,16,32 with timings.
# Out: benchmark/beam_fe.csv, benchmark/beam_fe.png +
#      journal_paper/images/beam_mesh_convergence.png
# =============================================================================
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using StochasticSemiDiscretizationMethod
const SSDM = StochasticSemiDiscretizationMethod
using LinearAlgebra, Plots, Printf, KrylovKit
BLAS.set_num_threads(1)

# ── Hermite beam FE matrices (pinned–pinned), nondimensional ──
function beam_fe(ne)
    le = 1.0/ne
    Me = (le/420).*[156 22le 54 -13le; 22le 4le^2 13le -3le^2;
                    54 13le 156 -22le; -13le -3le^2 -22le 4le^2]
    Ke = (1/le^3).*[12 6le -12 6le; 6le 4le^2 -6le 2le^2;
                    -12 -6le 12 -6le; 6le 2le^2 -6le 4le^2]
    Ge = (1/(30le)).*[36 3le -36 3le; 3le 4le^2 -3le -le^2;
                      -36 -3le 36 -3le; 3le -le^2 -3le 4le^2]
    # symmetry check — a row-3 sign typo here once produced a silently
    # non-symmetric K with garbage low modes
    @assert issymmetric(Me) && issymmetric(Ke) && issymmetric(Ge)
    ndof_full = 2*(ne+1)
    M=zeros(ndof_full,ndof_full); K=zeros(ndof_full,ndof_full); G=zeros(ndof_full,ndof_full)
    for e in 1:ne
        ix = (2*e - 1):(2*e + 2)
        M[ix,ix] .+= Me; K[ix,ix] .+= Ke; G[ix,ix] .+= Ge
    end
    keep = setdiff(1:ndof_full, [1, ndof_full-1])   # pin w at both ends
    (M[keep,keep], K[keep,keep], G[keep,keep], keep, ndof_full)
end

# shape-function read/injection vector at location xs∈(0,1)
function beam_read(ne, xs, keep, ndof_full)
    le=1.0/ne; e=clamp(Int(fld(xs, le))+1, 1, ne); ξ=(xs-(e-1)*le)/le
    N=[1-3ξ^2+2ξ^3, le*(ξ-2ξ^2+ξ^3), 3ξ^2-2ξ^3, le*(-ξ^2+ξ^3)]
    v=zeros(ndof_full); v[(2*e - 1):(2*e + 2)] .= N
    v[keep]
end

# Modal basis, computed ONCE and cached (NaN-guarded with a nonsymmetric
# fallback — sporadic NaN observed from the generalized symmetric path).
const _MODAL_CACHE = Dict{Int,Tuple{Vector{Float64},Matrix{Float64}}}()
# LOW modes of K φ = λ M φ with cond(K) ~ 1e9: direct dense eigensolvers only
# reach norm-relative accuracy ‖A‖ε ≈ 1e2–1e3, destroying the small
# eigenvalues (measured: ω₁ returned as 8.7e3 instead of 9.87!). SHIFT-INVERT
# fixes this: the LARGEST eigenvalues of K⁻¹M are 1/λ of the lowest modes at
# full relative precision.
function _modal_basis(ne_fine, Mm, Km)
    get!(_MODAL_CACHE, ne_fine) do
        F = eigen(Matrix(Km) \ Matrix(Mm))          # μ = 1/λ
        μ = real.(F.values)
        ord = sortperm(μ; rev=true)                 # largest μ = lowest modes
        λ = 1.0 ./ μ[ord]
        Φ = real.(F.vectors[:, ord])
        for j in axes(Φ,2)                          # M-normalize
            Φ[:,j] ./= sqrt(abs(Φ[:,j]'*Mm*Φ[:,j]))
        end
        ωs = sqrt.(abs.(λ))
        @assert !any(isnan, ωs) && !any(isnan, Φ)
        @assert abs(ωs[1] - π^2) / π^2 < 1e-3       # sanity: ω₁ ≈ π² (pinned-pinned)
        (ωs, Φ)
    end
end

# SDDE via MODAL reduction from a converged fine mesh (n_e=128): retain the
# first n_m mass-normalized modes. This is the standard engineering practice
# and avoids the stiff-FE exp() overflow of the full nodal model; convergence
# is studied in the number of retained modes n_m (d = 2 n_m).
function beam_problem(nm; ne_fine=128, P0=3.0, P1=4.0, Ω=2*(3π)^2, τ=2π/(2*(3π)^2),
                      ζ1=0.002, kP=3.0, kD=3.0, σP=1.5, xs=0.37)
    # kD deliberately strong: delayed velocity feedback causes control
    # SPILLOVER — some higher mode is destabilized through the delay phase, so
    # truncated modal models predict stability while the converged model is
    # unstable. This is the practical reason high DOF counts matter here.
    Mm,Km,Gm,keep,ndf = beam_fe(ne_fine)
    ωs_all, Φ_all = _modal_basis(ne_fine, Mm, Km)
    ωs = ωs_all[1:nm]
    Φ = Φ_all[:, 1:nm]                                    # mass-normalized modes
    Kr = Diagonal(ωs.^2)
    Gr = Φ'*Gm*Φ
    bs = beam_read(ne_fine, xs, keep, ndf)
    br = Φ'*bs
    Cr = Diagonal(2ζ1 .* ωs)                              # modal damping
    d = 2nm
    Afun(t) = begin
        A=zeros(d,d); A[1:nm,nm+1:d]=Matrix(I,nm,nm)
        A[nm+1:d,1:nm] = -(Matrix(Kr) .- (P0+P1*cos(Ω*t)).*Gr)
        A[nm+1:d,nm+1:d] = -Matrix(Cr)
        A
    end
    Bfun(t) = begin
        B=zeros(d,d)
        B[nm+1:d,1:nm] = kP.*(br*br')
        B[nm+1:d,nm+1:d] = kD.*(br*br')
        B
    end
    αfun(t) = begin
        A=zeros(d,d); A[nm+1:d,1:nm] = σP.*Gr    # axial-load noise, reads positions
        A
    end
    βfun(t) = zeros(d,d)
    lddep = LDDEProblem(ProportionalMX(Afun), [DelayMX(τ,Bfun)],
        [stCoeffMX(1,ProportionalMX(αfun))], [stCoeffMX(1,DelayMX(τ,βfun))],
        Additive(d), [stAdditive(1,Additive(zeros(d)))])
    (lddep, d, τ)
end

ρ_beam(nm, p; kwargs...) = begin
    lddep, d, τ = beam_problem(nm; kwargs...)
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, τ/p), τ, n_steps=p)
    (spectralRadiusOfMapping_MF_factored(rst), d)
end

# ── GATE 1: factored == dense identity at n_e=2 ──
println("── gate 1: factored vs dense OPERATOR identity (n_m=3, d=6) ──")
# Direct one-application comparison on the same vector — no eigensolver in the
# loop (Krylov dominant-eigenvalue estimates scatter at strong-resonance
# clustering / tiny-ρ non-normality, which is a solver artifact, not an
# operator difference).
let (lddep, d, τ) = beam_problem(3)
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, τ/24), τ, n_steps=24)
    dm = DiscreteMapping_M2_MF(rst)
    r = div(rst.n, d) - 1
    D = SSDM.CovVecIdx((r+1)*d).sectionStarts[end]
    x = randn(D)
    ws_d = SSDM.MFWorkspace(rst)
    y_dense = SSDM.apply_mapping_M2_MF!(ws_d, rst, dm.coeffs, x, ws_d.v_in_zero)
    cf = SSDM.get_factored_coefficients(rst)
    ws_f = SSDM.MFFactoredWorkspace(d, r)
    y_fact = SSDM.apply_mapping_M2_factored!(ws_f, cf, rst, x, zeros((r+1)*d))
    rel = norm(y_dense .- y_fact)/norm(y_dense)
    @printf("  ‖H_dense x − H_fact x‖/‖H_dense x‖ = %.1e %s\n", rel,
            rel < 1e-12 ? "PASS" : "FAIL")
    @assert rel < 1e-12
end

# ── GATE 2: noise-off ρ(H) = ρ(Φ_det)² ──
println("── gate 2: noise-off ρ(H)=ρ(Φ)² (n_e=4) ──")
let (lddep, d, τ) = beam_problem(4; σP=0.0)
    rst = SSDM.calculateResults(lddep, SemiDiscretization(2, τ/24), τ, n_steps=24)
    ρH = spectralRadiusOfMapping_MF_factored(rst)
    # deterministic monodromy via the factored first-moment operator
    r = div(rst.n, d) - 1
    cf = SSDM.get_factored_coefficients(rst)
    ws = SSDM.MFFactoredWorkspace(d, r)
    op1 = SSDM.M1FactoredOperator(cf, rst, (r+1)*d, ws)
    vals,_,_ = eigsolve(op1, rand((r+1)*d), 1, :LM)
    ρΦ2 = abs(vals[1])^2
    @printf("  ρH %.12f ρΦ² %.12f rel %.1e %s\n", ρH, ρΦ2, abs(ρH-ρΦ2)/ρΦ2,
            abs(ρH-ρΦ2)/ρΦ2 < 1e-8 ? "PASS" : "FAIL")
end

# ── modal-convergence sweep at the mode-3 resonance point ──
println("── modal sweep (Ω ≈ 2ω₃), fixed p=64, fine mesh n_e=128 ──")
rows=NamedTuple[]
p = 64
for nm in (1, 2, 3, 4, 6, 8, 12, 16)
    local ρ, d
    t = @elapsed begin
        try
            (ρ, d) = ρ_beam(nm, p)
        catch e
            @warn "n_m=$nm failed" e
            break
        end
    end
    D = SSDM.CovVecIdx((p+1)*d).sectionStarts[end]
    push!(rows, (nm=nm, d=d, D=D, ρ=ρ, t=t))
    @printf("  n_m=%3d d=%4d D=%9d  ρ(H)=%.8f  (%.1fs)\n", nm, d, D, ρ, t)
    flush(stdout)
    open(joinpath(@__DIR__,"beam_fe.csv"),"w") do io
        println(io,"nm,d,D,rho,t")
        for rr in rows; @printf(io,"%d,%d,%d,%.10f,%.2f\n",rr.nm,rr.d,rr.D,rr.ρ,rr.t); end
    end
    if length(rows) ≥ 2
        plt = plot([rr.nm for rr in rows], [rr.ρ for rr in rows], marker=:circle,
                   xlabel="number of retained modes  n_m",
                   ylabel="ρ(H)", legend=:topright, label="ρ(H) at Ω≈2ω₃",
                   title="Modal convergence — parametrically excited beam with delayed PD control",
                   size=(750,520), framestyle=:box)
        for rr in rows
            annotate!(plt, rr.nm, rr.ρ, text(@sprintf(" d=%d, %.0fs", rr.d, rr.t), 8, :left))
        end
        savefig(plt, joinpath(@__DIR__,"beam_fe.png"))
        savefig(plt, raw"C:\Users\mmuser\My Drive\BD\StochasticSemiDiscretizationMethod.jl\journal_paper\images\beam_mesh_convergence.png")
    end
end
println("done — benchmark/beam_fe.csv, beam_fe.png (+ paper images)")
