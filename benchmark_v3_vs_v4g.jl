using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, CUDA, Printf, KrylovKit

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

# ── Helpers to build operators directly so we can time eigsolve separately ──

import StochasticSemiDiscretizationMethod: MFGPUWorkspace, MFGPUMappingOperator_v3,
    extract_gpu_coeffs, CovVecIdx, _build_v4_graph, MFGPUMappingOperator_v4,
    MFGPUMappingOperator_v4g

function make_v3_op(dm)
    d = size(dm.coeffs.det[1][1][1], 1)
    p = dm.rst.n_steps
    r = div(dm.rst.n, d) - 1
    D = CovVecIdx((r+1)*d).sectionStarts[end]
    gpu_coeffs = extract_gpu_coeffs(dm.coeffs, p, d, true)
    ws = MFGPUWorkspace(dm.rst)
    n_sm = Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    op = MFGPUMappingOperator_v3(gpu_coeffs, D, r, d, p, n_sm, ws)
    return op, D
end

function make_v4g_op(dm)
    d = size(dm.coeffs.det[1][1][1], 1)
    p = dm.rst.n_steps
    r = div(dm.rst.n, d) - 1
    D = CovVecIdx((r+1)*d).sectionStarts[end]
    gpu_coeffs = extract_gpu_coeffs(dm.coeffs, p, d, false)
    ws = MFGPUWorkspace(dm.rst)
    n_sm = Int(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    x_buf = CUDA.zeros(Float64, D)
    y_buf = CUDA.zeros(Float64, D)
    op_v4 = MFGPUMappingOperator_v4(gpu_coeffs, D, r, d, p, n_sm, ws)
    # warm up all kernels (JIT)
    mul!(y_buf, op_v4, x_buf); CUDA.synchronize()
    CUDA.fill!(op_v4.ws.C, 0.0)
    exec = _build_v4_graph(op_v4, x_buf, y_buf)
    op = MFGPUMappingOperator_v4g(gpu_coeffs, D, r, d, p, n_sm, ws, x_buf, y_buf, exec)
    return op, D
end

function kd_for(D, C0)
    avail    = Int(CUDA.available_memory())
    reserved = Int(sizeof(C0)) + 64*1024^2
    mem_kd   = max(4, (avail - reserved) ÷ max(1, D * 8))
    dim_kd   = max(4, D ÷ 500)
    min(30, min(mem_kd, dim_kd))
end

function run_eigsolve(op, D, kd)
    CUDA.seed!(42)
    C0 = CUDA.rand(Float64, D)
    vals, _, _ = eigsolve(op, C0, 1, :LM; ishermitian=false, krylovdim=kd)
    CUDA.synchronize()
    return abs(vals[1])
end

# ── Phase 1: JIT warm-up of ALL kernels ─────────────────────────────────────
println("Phase 1: JIT warm-up (p=20)...")
_pw  = SemiDiscretization(order, P/20)
_rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, _pw, τ)
_dm  = DiscreteMapping_M2_MF(_rst)

# warm up v3
_op3, _D3 = make_v3_op(_dm)
CUDA.seed!(42); _c0 = CUDA.rand(Float64, _D3)
eigsolve(_op3, _c0, 1, :LM; ishermitian=false, krylovdim=4); CUDA.synchronize()

# warm up v4g (also warms v4 kernels inside make_v4g_op)
_op4g, _D4g = make_v4g_op(_dm)
CUDA.seed!(42); _c0 = CUDA.rand(Float64, _D4g)
mul!(_c0, _op4g, _c0); CUDA.synchronize()   # warm graph replay path

println("done\n")

# ── Phase 2: Accuracy check ──────────────────────────────────────────────────
test_ps = [10, 25, 50, 100, 225, 500]

println("="^90)
@printf("%-6s  %-8s  %-4s  %-14s  %-14s  %-10s  %-10s\n",
        "p", "D", "kd", "ρ_v3", "ρ_v4g", "err_v3", "err_v4g")
println("-"^90)

for p in test_ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    op3,  D3  = make_v3_op(dm)
    op4g, D4g = make_v4g_op(dm)
    CUDA.seed!(42); C0 = CUDA.rand(Float64, D3)
    kd = kd_for(D3, C0)

    ρ_v3  = run_eigsolve(op3,  D3,  kd)
    ρ_v4g = run_eigsolve(op4g, D4g, kd)

    e3  = abs(ρ_v3  - ρ_ref) / ρ_ref
    e4g = abs(ρ_v4g - ρ_ref) / ρ_ref
    match = abs(ρ_v3 - ρ_v4g) / ρ_ref < 1e-7 ? "✓" : "MISMATCH"

    @printf("%-6d  %-8d  %-4d  %-14.8f  %-14.8f  %-10.2e  %-10.2e  %s\n",
            p, D3, kd, ρ_v3, ρ_v4g, e3, e4g, match)
end
println("="^90)
println()

# ── Phase 3: Timing (both ops pre-built, graph pre-captured) ─────────────────
println("Timing (operators + graph pre-built, eigsolve only):")
println("-"^80)
@printf("%-6s  %-8s  %-4s  %-12s  %-12s  %-8s\n",
        "p", "D", "kd", "t_v3 (s)", "t_v4g (s)", "v3/v4g")
println("-"^80)

for p in test_ps
    method = SemiDiscretization(order, P/p)
    rst    = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ)
    dm     = DiscreteMapping_M2_MF(rst)

    op3,  D3  = make_v3_op(dm)
    op4g, D4g = make_v4g_op(dm)
    CUDA.seed!(42); C0 = CUDA.rand(Float64, D3)
    kd = kd_for(D3, C0)

    # one ignored run to let GPU settle
    run_eigsolve(op3,  D3,  kd); CUDA.synchronize()
    run_eigsolve(op4g, D4g, kd); CUDA.synchronize()

    t_v3  = @elapsed begin run_eigsolve(op3,  D3,  kd); CUDA.synchronize() end
    t_v4g = @elapsed begin run_eigsolve(op4g, D4g, kd); CUDA.synchronize() end

    @printf("%-6d  %-8d  %-4d  %-12.4f  %-12.4f  %-8.2fx\n",
            p, D3, kd, t_v3, t_v4g, t_v3/t_v4g)
end
println("-"^80)
println()
println("Note: ratio > 1 means v4g faster than v3, < 1 means v3 faster.")
