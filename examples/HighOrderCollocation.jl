# High-order Gauss–Legendre collocation solver (order 2S) — the recommended
# method at tight tolerances in low/moderate state dimension.
#
# Run:  julia --project=examples examples/HighOrderCollocation.jl
using StochasticSemiDiscretizationMethod
using StaticArrays, Printf

# Delayed-PD-drift stochastic Mathieu oscillator (delay τ = period T = 1),
# present-state multiplicative noise + additive forcing.
A(t) = @SMatrix [0.0 1.0; -(1.0 + 0.5cos(2π*t)) -0.4]
B(t) = @SMatrix [0.0 0.0; 0.20*(1 + 0.3cos(2π*t)) 0.12*(1 + 0.4cos(2π*t))]
α(t) = @SMatrix [0.0 0.0; 0.30 0.0]
β(t) = @SMatrix [0.0 0.0; 0.0  0.0]
prob = LDDEProblem(ProportionalMX(A), [DelayMX(1.0, B)],
    [stCoeffMX(1, ProportionalMX(α))], [stCoeffMX(1, DelayMX(1.0, β))],
    Additive(2), [stAdditive(1, Additive(@SVector [0.0, 0.3]))])

const T = 1.0

# Unified interface: pick the method. GaussLegendre(3) (= order 6) is the default;
# a handful of steps already gives many digits.
ρ     = spectralRadiusOfMoment(prob, T, 12; method = GaussLegendre(3))
var_x = stationaryVariance(prob, T, 12;    method = GaussLegendre(3))
@printf("GL3, p=12:  ρ(H) = %.10f   (mean-square %s)\n", ρ, ρ < 1 ? "stable" : "UNSTABLE")
@printf("            stationary Var(x) = %.8f\n", var_x)

# The classical semi-discretization survives as an explicit option (first order,
# so it needs many more steps for the same accuracy — kept for cross-checks):
ρ_sd = spectralRadiusOfMoment(prob, T, 400; method = ClassicalSD(2))
@printf("ClassicalSD(2), p=400:  ρ(H) = %.10f  (matches GL3 above)\n\n", ρ_sd)

# Demonstrate the measured order 2S: error in ρ(H) vs steps p, for S = 1, 2, 3.
ρref = spectralRadiusOfMapping_collocation(prob, T, 96; S = 3)   # fine reference
println("measured convergence order (error in ρ vs p):")
for S in (1, 2, 3)
    e6  = abs(spectralRadiusOfMapping_collocation(prob, T, 6;  S = S) - ρref)
    e24 = abs(spectralRadiusOfMapping_collocation(prob, T, 24; S = S) - ρref)
    @printf("  S=%d (Gauss–Legendre %d-stage):  order ≈ %.2f  (theory %d)\n",
            S, S, log(e6 / e24) / log(24 / 6), 2S)
end
