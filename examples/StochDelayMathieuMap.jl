using StochasticSemiDiscretizationMethod
using StaticArrays
using MDBM
using Plots
using LaTeXStrings
using LinearAlgebra; BLAS.set_num_threads(1)   # small/thin solves: single-thread BLAS is faster

gr();

# Stochastic Delayed Mathieu Equation — 2nd moment stability map
# Ref: Sykora (2020) Eq. (41-45)
# x''(t) + 2ζ x'(t) + (A + ε cos(t/2))x(t) = B x(t-τ) + noise
#
# Period P = 4π (half-frequency excitation)
# Noise: multiplicative with strength α (scales system matrices), additive σ

function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε * cos(0.5 * t)) -2ζ]
    AMx  = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    noiseID = 1
    αMxfun(t) = @SMatrix [0. 0.; -α_val*(A + ε*cos(0.5*t)) -α_val*2ζ]
    αMx1  = stCoeffMX(noiseID, ProportionalMX(αMxfun))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ, @SMatrix [0. 0.; α_val*B 0.]))
    σVec  = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end

const ε     = 2.0
const ζ     = 0.1
const τ     = 2π
const P     = 4π
const p_res = 40   # discretisation steps per period

# α_vals: sweep of noise strengths to overlay on one plot
const α_vals = 0.05:0.05:0.5

axis = [Axis(0.0:0.2:5.0, :A), Axis(-1.5:0.2:1.5, :B)]

p = plot(xlabel=L"A", ylabel=L"B",
         title="Stochastic Delay Mathieu — 2nd moment stability\n(ε=$ε, ζ=$ζ, τ=2π, P=4π)",
         xlim=(0, 5), ylim=(-1.5, 1.5),
         legend=false)

for α_val in α_vals
    method = SemiDiscretization(2, P / p_res)

    function foo_stab(A::Float64, B::Float64)::Float64
        lddep = createStochMathieuProblem(A, ε, B, ζ, τ, 0.0, α_val)
        rst   = StochasticSemiDiscretizationMethod.calculateResults(
                    lddep, method, τ, n_steps=p_res)
        return log(spectralRadiusOfMapping_MF(DiscreteMapping_M2_MF(rst)))
    end

    println("Solving stability boundary for α=$α_val ...")
    mdbm_stab    = MDBM_Problem(foo_stab, axis)
    solve!(mdbm_stab, 3, verbosity=0, doThreadprecomp=false)
    stab_points  = getinterpolatedsolution(mdbm_stab)

    scatter!(p, stab_points..., markersize=2, markerstrokewidth=0)
end

savefig(p, "assets/StochDelayMathieuMap.png")
println("Saved assets/StochDelayMathieuMap.png")
display(p)
