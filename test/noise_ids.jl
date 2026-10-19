# Regression test: several noise sources, additive terms on only some of them.
# Before the fix, `Result` allocated `stsubVs` with one slot per noise source (LDDEP.w) and filled
# it by broadcasting the per-additive-term results: one additive term with w = 2 noise sources was
# copied into both slots (both with the same nID), its noise vector summed twice -> 2σ, variance 4σ²;
# two additive terms with w = 3 threw a DimensionMismatch.
using StochasticSemiDiscretizationMethod, StaticArrays, QuadGK, Test

# x'' + 2ζx' + x = -(P x(t-τ) + D x'(t-τ))(1 + σm ξ_m) + σa ξ_a   (Itô)
function pd_problem(P, D, τ, ζ, σm, σa; idm = 1, ida = 2, extra = 0)
    AMx = ProportionalMX(@SMatrix [0.0 1.0; -1.0 -2ζ])
    BMx = DelayMX(τ, @SMatrix [0.0 0.0; -P -D])
    α = [stCoeffMX(idm, ProportionalMX(@SMatrix [0.0 0.0; 0.0 0.0]))]
    β = [stCoeffMX(idm, DelayMX(τ, @SMatrix [0.0 0.0; -σm*P -σm*D]))]
    σs = [stAdditive(ida, Additive(@SVector [0.0, σa]))]
    # optionally a further independent additive force noise on yet another source
    extra > 0 && push!(σs, stAdditive(extra, Additive(@SVector [0.0, σa])))
    LDDEProblem(AMx, [BMx], α, β, Additive(2), σs)
end
function var_ssd(prob, τ; r = 20)
    mp = DiscreteMapping_M2(prob, SemiDiscretization(1, τ / r), τ, n_steps = 1, calculate_additive = true)
    VecToCovMx(fixPointOfMapping(mp), length(mp.M1_Vs[1]))[1, 1]
end
# exact: Var = (Σ σa²) (1/2π)∫|1/D|² / (1 - (1/2π)∫|σm (P + iωD) e^{-iωτ}/D|²)
function var_exact(P, D, τ, ζ, σm, σa; nadd = 1)
    Df(w) = -w^2 + 2im * ζ * w + 1 + (P + im * w * D) * exp(-im * w * τ)
    iy = quadgk(w -> abs2(1 / Df(w)), 0, 5, 20, Inf; rtol = 1e-11)[1] / π
    ig = quadgk(w -> abs2(σm * (P + im * w * D) * exp(-im * w * τ) / Df(w)), 0, 5, 20, Inf; rtol = 1e-11)[1] / π
    return nadd * σa^2 * iy / (1 - ig)
end

p = (0.5, 0.4, 0.5, 0.05, 0.6, 0.1)
ve = var_exact(p...)
@testset "noise IDs" begin
    for (lab, kw, nadd) in [("shared ID", (idm = 1, ida = 1), 1),
                           ("additive on ID 2", (idm = 1, ida = 2), 1),
                           ("additive on ID 1, multiplicative on ID 2", (idm = 2, ida = 1), 1),
                           ("two additive sources (IDs 2, 3)", (idm = 1, ida = 2, extra = 3), 2)]
        prob = pd_problem(p...; kw...)
        v = try var_ssd(prob, p[3]) catch e; e isa DimensionMismatch ? NaN : rethrow() end
        vx = var_exact(p...; nadd)
        println(rpad(lab, 44), " w = ", prob.w, "  SSD ", round(v, sigdigits = 6), "  exact ", round(vx, sigdigits = 6),
                "  ratio ", round(v / vx, sigdigits = 5))
        @test isapprox(v, vx; rtol = 2e-3)          # first-order SSD, r = 20: ~4e-4
    end
end
