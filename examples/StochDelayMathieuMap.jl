using StochasticSemiDiscretizationMethod
using StaticArrays
using MDBM
using Plots
using LaTeXStrings

gr();

# Define the Stochastic Delayed Mathieu Equation
# Ref: Sykora (2020) Eq. (41-45)
# x''(t) + 2ζ x'(t) + (A + ε cos(t))x(t) = B x(t-τ) + integrated noise
function createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
    # T = 2π period
    AMxfun(t) = @SMatrix [0. 1.; -(A + ε * cos(0.5 * t)) -2ζ]
    AMx = ProportionalMX(AMxfun)
    BMx1 = DelayMX(τ, @SMatrix [0. 0.; B 0.])
    cVec = Additive(2)
    noiseID = 1
    # Noise terms
    #αMx1 = stCoeffMX(noiseID, ProportionalMX(@SMatrix [0. 0.; α_val 0.]))
    αMxfun(t) = @SMatrix [0. 0.; -α_val*(A+ε*cos(0.5 * t)) -α_val*2ζ]
    αMx1 = stCoeffMX(noiseID, ProportionalMX(αMxfun))
    βMx11 = stCoeffMX(noiseID, DelayMX(τ, @SMatrix [0. 0.; α_val*B 0.]))
    σVec = stAdditive(1, Additive(@SVector [0., σ]))
    return LDDEProblem(AMx, [BMx1], [αMx1], [βMx11], cVec, [σVec])
end


const STATIONARY_THRESHOLD = 15.0 # Example threshold for "norm - const = 0"
for α_val in 0.05:0.05:0.5

    # Parameters from Fig 4 (bottom right)
    #ε = 1.0; ζ = 0.05; τ = 2π; σ = 0.25; α_val = 0.1; P = 2π;
    ε = 2.0
    ζ = 0.1
    τ = 2π
    σ = 0.0
    P = 4π#; α_val = 0.0
    p_res = 40 # steps per period
    method = SemiDiscretization(2, P / p_res)


    # 1. Stability Map (Spectral Radius)
    function foo_stab(A::Float64, B::Float64)::Float64
        lddep = createStochMathieuProblem(A, ε, B, ζ, τ, 0.0, 0.0) # Stability depends on α,β but here we follow fig
        # Actually fig 4 has α=0.1. Re-read: "σ=0.25, α=0.1"
        lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
        rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ, n_steps=p_res)
        dm = DiscreteMapping_M2_MF(rst)
        return log(spectralRadiusOfMapping_MF(dm))
        # dm = DiscreteMapping_M2(rst)
        # return log(spectralRadiusOfMapping(dm))
    end

    # 2. Stationary Moment Map (norm - const = 0)
    function foo_stat(A::Float64, B::Float64)::Float64
        lddep = createStochMathieuProblem(A, ε, B, ζ, τ, σ, α_val)
        rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τ, n_steps=p_res, calculate_additive=true)
        dm = DiscreteMapping_M2_MF(rst)
        fp = fixPointOfMapping_MF(dm)
        # Length of first moment part is D1 = (r+1)*d
        r = div(rst.n, 2) - 1
        D1 = (r + 1) * 2
        statM2 = VecToCovMx(fp, D1)
        # We look for norm(stationary_moment) - constant = 0
        if log(spectralRadiusOfMapping_MF(dm)) > 0
            return 1e30
        else
            return norm(statM2) - STATIONARY_THRESHOLD
        end
    end

    axis = [Axis(0.0:0.2:5.0, :A), Axis(-1.5:0.2:1.5, :B)]

    println("Solving stability boundary...")
    mdbm_stab = MDBM_Problem(foo_stab, axis)
    solve!(mdbm_stab, 3, verbosity=2,doThreadprecomp=false)
    stab_points = getinterpolatedsolution(mdbm_stab)

    # println("Solving stationary moment boundary...")
    # mdbm_stat = MDBM_Problem(foo_stat, axis)
    # solve!(mdbm_stat, 3, verbosity=2)
    # stat_points = getinterpolatedsolution(mdbm_stat)

   # p = scatter!(stab_points..., label="2nd Moment Stability", markersize=2, markerstrokewidth=0)
    p = scatter!(stab_points..., label="", markersize=2, markerstrokewidth=0)
    #p = scatter(stab_points..., label="2nd Moment Stability", color=:blue, markersize=2, markerstrokewidth=0)
    #scatter!(p, stat_points..., label="Stationary Moment Norm = $STATIONARY_THRESHOLD", color=:red, markersize=2, markerstrokewidth=0)
    #plot!(p, title="Stochastic Delay Mathieu Eq (Sykora 2020 Fig 4)", xlabel="A", ylabel="B", xlim=(0, 5), ylim=(-1.5, 1.5))

    savefig(p, "assets/StochDelayMathieuMap.png")
    println("Saved assets/StochDelayMathieuMap.png")
    display(p)

end
plot!(legend=false)