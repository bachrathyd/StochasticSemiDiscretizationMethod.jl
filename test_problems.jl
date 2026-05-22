using Pkg; Pkg.activate(".")
using StochasticSemiDiscretizationMethod
using LinearAlgebra, StaticArrays, CUDA

BLAS.set_num_threads(1)

println("=== Test Problem 1: Mathieu 2-delay ===")
τ1 = 2π; A_m=1.0; ε_m=0.5; B1_m=0.15; B2_m=0.1; ζ_m=0.1; σ_m=0.05; α_m=0.1; P_math=2π
T_period = 2π
τ2fun(t) = τ1 * (1.0 + 0.5*sin(2π/T_period * t))
AMxfun(t) = @SMatrix [0.0 1.0; -(A_m + ε_m*cos(2π*t/P_math)) -2ζ_m]
AMx  = ProportionalMX(AMxfun)
B1m  = @SMatrix [0.0 0.0; B1_m 0.0]
B2m  = @SMatrix [0.0 0.0; B2_m 0.0]
BMx1 = DelayMX(τ1,    B1m)
BMx2 = DelayMX(τ2fun, B2m)
cVec = Additive(2)
am   = @SMatrix [0.0 0.0; α_m 0.0]
zm   = @SMatrix [0.0 0.0; 0.0 0.0]
αMx1 = stCoeffMX(1, ProportionalMX(am))
βMx1 = stCoeffMX(1, DelayMX(τ1, zm))
sv   = @SVector [0.0, σ_m]
σVec = stAdditive(1, Additive(sv))
lddep1 = LDDEProblem(AMx, [BMx1, BMx2], [αMx1], [βMx1], cVec, [σVec])

# DiscretizationLength = max possible delay = τ1 * 1.5
τ_max = τ1 * 1.5
m = SemiDiscretization(1, P_math/20)
rst = StochasticSemiDiscretizationMethod.calculateResults(lddep1, m, τ_max)
dm  = DiscreteMapping_M2_MF(rst)
rho = spectralRadiusOfMapping_MF(dm)
println("Mathieu 2-delay rho = $rho  ✓")

println("\n=== Test Problem 2: 10-DOF chain ===")
n_dof = 10
d_chain = 2 * n_dof
m_val = 1.0; k_val = 10.0; c_val = 0.3; kp = 5.0; kd = 1.0

M_mat = m_val * Matrix{Float64}(I, n_dof, n_dof)
K_mat = diagm(0 => fill(2k_val, n_dof), 1 => fill(-k_val, n_dof-1), -1 => fill(-k_val, n_dof-1))
K_mat[1,1] = k_val          # pinned left: only 1 spring
K_mat[n_dof, n_dof] = k_val # free right: only 1 spring
C_mat = (c_val/k_val) .* K_mat

M_inv = inv(M_mat)
A_top = zeros(n_dof, n_dof)
A_bot_q = -M_inv * K_mat
A_bot_v = -M_inv * C_mat
A_sys_arr = [A_top I; A_bot_q A_bot_v]
A_sys = SMatrix{d_chain, d_chain}(A_sys_arr)

B_delay_arr = zeros(d_chain, d_chain)
B_delay_arr[2*n_dof, n_dof]   = -M_inv[n_dof,n_dof] * kp
B_delay_arr[2*n_dof, 2*n_dof] = -M_inv[n_dof,n_dof] * kd
B_delay_s = SMatrix{d_chain, d_chain}(B_delay_arr)

σ_chain = 0.02
sv_chain = SVector{d_chain}([zeros(2*n_dof-1); σ_chain])
σVec_chain = stAdditive(1, Additive(sv_chain))
zero_s = SMatrix{d_chain,d_chain}(zeros(d_chain,d_chain))
αMx_chain = stCoeffMX(1, ProportionalMX(zero_s))
βMx_chain = stCoeffMX(1, DelayMX(1.0, zero_s))

τ_delay = 0.3
AMx_c  = ProportionalMX(A_sys)
BMx_c  = DelayMX(τ_delay, B_delay_s)
cVec_c = Additive(d_chain)
lddep2 = LDDEProblem(AMx_c, [BMx_c], [αMx_chain], [βMx_chain], cVec_c, [σVec_chain])

m2 = SemiDiscretization(1, τ_delay/20)
rst2 = StochasticSemiDiscretizationMethod.calculateResults(lddep2, m2, τ_delay)
dm2  = DiscreteMapping_M2_MF(rst2)
rho2 = spectralRadiusOfMapping_MF(dm2)
println("10-DOF chain rho = $rho2  ✓")

println("\nAll tests passed.")
