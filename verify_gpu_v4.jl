using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, CUDA, Printf

BLAS.set_num_threads(1)

function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε*cos(2π*t/P)) -2ζ]
    AMx   = ProportionalMX(AMxfun)
    BMx1  = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec  = Additive(2)
    αMx1  = stCoeffMX(1, ProportionalMX(@SMatrix [0. 0.; α_val 0.]))
    βMx11 = stCoeffMX(1, DelayMX(τ, @SMatrix [0. 0.; 0. 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

A=1.0; ε=0.5; B=0.2; ζ=0.1; τ=1.0; σ=0.1; α_val=0.2; P=1.0
lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val, P)
order  = 1
ρ_ref  = 0.6817666221

println("GPU: ", CUDA.name(CUDA.device()))
println()

# Pre-warm all three paths
println("Pre-warming...")
_pw = SemiDiscretization(order, P/20)
_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, _pw, τ)
_dm  = DiscreteMapping_M2_MF(_rst)
spectralRadiusOfMapping_MF(_dm)
spectralRadiusOfMapping_GPU_v3(_dm); CUDA.synchronize()
spectralRadiusOfMapping_GPU_v4(_dm); CUDA.synchronize()
println("done\n")

test_ps = [10, 25, 50, 100, 225, 500]

println("="^85)
@printf("%-6s  %-12s  %-12s  %-12s  %-10s  %-10s\n",
        "p", "ρ_CPU", "ρ_GPU_v3", "ρ_GPU_v4", "err_v3", "err_v4")
println("-"^85)

for p in test_ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    ρ_cpu = spectralRadiusOfMapping_MF(dm)
    ρ_v3  = spectralRadiusOfMapping_GPU_v3(dm); CUDA.synchronize()
    ρ_v4  = spectralRadiusOfMapping_GPU_v4(dm); CUDA.synchronize()

    e3 = abs(ρ_v3 - ρ_ref) / ρ_ref
    e4 = abs(ρ_v4 - ρ_ref) / ρ_ref

    @printf("%-6d  %-12.8f  %-12.8f  %-12.8f  %-10.2e  %-10.2e\n",
            p, ρ_cpu, ρ_v3, ρ_v4, e3, e4)

    if abs(ρ_v4 - ρ_cpu) / ρ_cpu > 1e-5
        @printf("  *** WARNING: v4 deviates from CPU by %.2e ***\n", abs(ρ_v4 - ρ_cpu)/ρ_cpu)
    end
end
println("="^85)
println()

# Timing comparison
println("Timing (single shot after JIT):")
println("-"^65)
@printf("%-6s  %-12s  %-12s  %-12s\n", "p", "t_CPU(s)", "t_v3(s)", "t_v4(s)")
println("-"^65)

for p in test_ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    t_cpu = @elapsed spectralRadiusOfMapping_MF(dm)
    t_v3  = @elapsed begin spectralRadiusOfMapping_GPU_v3(dm); CUDA.synchronize() end
    t_v4  = @elapsed begin spectralRadiusOfMapping_GPU_v4(dm); CUDA.synchronize() end

    @printf("%-6d  %-12.4f  %-12.4f  %-12.4f\n", p, t_cpu, t_v3, t_v4)
end
println("-"^65)

# Test auto dispatch
println()
println("Testing spectralRadiusOfMapping_auto:")
for p in [10, 100, 500]
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)
    ρ_auto = spectralRadiusOfMapping_auto(dm)
    @printf("  p=%4d  ρ_auto=%.8f  err=%.2e\n", p, ρ_auto, abs(ρ_auto-ρ_ref)/ρ_ref)
end
