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

# Pre-warm all paths (JIT compilation)
println("Pre-warming all GPU versions...")
_pw = SemiDiscretization(order, P/20)
_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, _pw, τ)
_dm  = DiscreteMapping_M2_MF(_rst)
spectralRadiusOfMapping_MF(_dm)
spectralRadiusOfMapping_GPU_v3(_dm); CUDA.synchronize()
spectralRadiusOfMapping_GPU_v4(_dm); CUDA.synchronize()
spectralRadiusOfMapping_GPU_v4g(_dm); CUDA.synchronize()
println("done\n")

test_ps = [10, 25, 50, 100, 225, 500, 1000]

println("="^100)
@printf("%-6s  %-12s  %-12s  %-12s  %-10s  %-10s\n",
        "p", "ρ_v3", "ρ_v4", "ρ_v4g", "err_v3", "err_v4g")
println("-"^100)
for p in test_ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    ρ_v3  = spectralRadiusOfMapping_GPU_v3(dm);  CUDA.synchronize()
    ρ_v4  = spectralRadiusOfMapping_GPU_v4(dm);  CUDA.synchronize()
    ρ_v4g = spectralRadiusOfMapping_GPU_v4g(dm); CUDA.synchronize()

    e3  = abs(ρ_v3  - ρ_ref) / ρ_ref
    e4g = abs(ρ_v4g - ρ_ref) / ρ_ref

    match = abs(ρ_v4g - ρ_v3) / ρ_ref < 1e-8 ? "✓" : "MISMATCH"
    @printf("%-6d  %-12.8f  %-12.8f  %-12.8f  %-10.2e  %-10.2e  %s\n",
            p, ρ_v3, ρ_v4, ρ_v4g, e3, e4g, match)
end
println("="^100)
println()

println("Timing comparison (single shot, JIT already done):")
println("-"^72)
@printf("%-6s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
        "p", "t_CPU", "t_v3", "t_v4", "t_v4g", "v3/v4g")
println("-"^72)

for p in test_ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    t_cpu = @elapsed spectralRadiusOfMapping_MF(dm)
    t_v3  = @elapsed begin spectralRadiusOfMapping_GPU_v3(dm);  CUDA.synchronize() end
    t_v4  = @elapsed begin spectralRadiusOfMapping_GPU_v4(dm);  CUDA.synchronize() end
    t_v4g = @elapsed begin spectralRadiusOfMapping_GPU_v4g(dm); CUDA.synchronize() end

    speedup = t_v3 / t_v4g
    @printf("%-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.2fx\n",
            p, t_cpu, t_v3, t_v4, t_v4g, speedup)
end
println("-"^72)
println()
println("Note: t_v4g includes one-time graph build+capture cost (shown here).")
println("In repeated calls (e.g. parameter sweep) the build cost is paid once.")
