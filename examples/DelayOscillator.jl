using StochasticSemiDiscretizationMethod
using StaticArrays
using MDBM
using Plots
using LaTeXStrings

gr();

function createSLDOProblem(A,B,ζ,α,β,σ)
    AMx =  ProportionalMX(@SMatrix [0. 1.;-A -2ζ]);
    τ1=2π
    BMx1 = DelayMX(τ1,@SMatrix [0. 0.; B 0.]);
    cVec = Additive(2)
    noiseID = 1
    αMx1 = stCoeffMX(noiseID,ProportionalMX(@SMatrix [0. 0.; α 0.]))
    βMx11 = stCoeffMX(noiseID,DelayMX(τ1,@SMatrix [0. 0.; β 0.]))
    σVec = stAdditive(1,Additive(@SVector [0., σ]))
    LDDEProblem(AMx,[BMx1],[αMx1],[βMx11],cVec,[σVec])
end

method=SemiDiscretization(5, 2π/30)
τmax=2π + 100eps()

# ── Point calculation ────────────────────────────────────────────────────────
sldo_lddep=createSLDOProblem(1.,0.1,0.1,0.1,0.1,0.5);
mapping=DiscreteMapping_M2_MF(sldo_lddep,method,τmax,n_steps=30,calculate_additive=true);

@show spectralRadiusOfMapping_MF(mapping);
r = div(mapping.rst.n, 2) - 1
D1 = (r+1)*2
statM2=VecToCovMx(fixPointOfMapping_MF(mapping), D1);
@show statM2[1,1] |> sqrt;

# ── Map functions ─────────────────────────────────────────────────────────────
# Shared noise parameters: α=0.3A, β=0.3B, σ=0.5, ζ=0.05
const _ζ  = 0.05
const _σ  = 0.5

# Stability boundary: log(ρ) = 0
function foo_stab(A::Float64, B::Float64)::Float64
    lddep = createSLDOProblem(A, B, _ζ, 0.3*A, 0.3*B, 0.)
    rst = StochasticSemiDiscretizationMethod.calculateResults(lddep, method, τmax, n_steps=30)
    dm  = DiscreteMapping_M2_MF(rst)
    return log(spectralRadiusOfMapping_MF(dm))
end

# Stationary position std contour: sqrt(E[q²]) - limit = 0
# Only meaningful inside the stable region (ρ < 1).
# Returns +Inf outside stable region so MDBM only finds crossings inside.
const POS_LIMIT = 1.0   # position std threshold [same units as q]

function foo_stat(A::Float64, B::Float64)::Float64
    lddep = createSLDOProblem(A, B, _ζ, 0.3*A, 0.3*B, _σ)
    rst = StochasticSemiDiscretizationMethod.calculateResults(
              lddep, method, τmax, n_steps=30, calculate_additive=true)
    dm  = DiscreteMapping_M2_MF(rst)
    log(spectralRadiusOfMapping_MF(dm)) >= 0 && return 1e0   # unstable → skip
    r2  = div(rst.n, 2) - 1
    fp  = fixPointOfMapping_MF(dm)
    M2  = VecToCovMx(fp, (r2+1)*2)
    return sqrt(M2[1,1]) - POS_LIMIT
end

# ── Solve stability boundary ─────────────────────────────────────────────────
axis = [Axis(-1.0:0.25:5.0, :A), Axis(LinRange(-1.5, 1.5, 12), :B)]
iteration = 4

println("Solving 2nd moment stability boundary...")
mdbm_stab = MDBM_Problem(foo_stab, axis)
solve!(mdbm_stab, iteration, verbosity=2)
stab_pts = getinterpolatedsolution(mdbm_stab)

# ── Solve stationary moment contour ─────────────────────────────────────────
println("\nSolving stationary position std = $POS_LIMIT contour...")
mdbm_stat = MDBM_Problem(foo_stat, axis)
solve!(mdbm_stat, iteration, verbosity=2)
stat_pts = getinterpolatedsolution(mdbm_stat)

