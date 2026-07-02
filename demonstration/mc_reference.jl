# =============================================================================
# INDEPENDENT Monte-Carlo reference for ρ(H) of the stochastic Mathieu equation.
# No SDM, no IRK — direct Euler–Maruyama simulation of the SDDE, many realisations.
# The 2nd moment E[‖x(t)‖²] grows like ρ(H)^(t/P) (P = period). We estimate ρ(H) from
# the per-period growth factor of the ensemble mean square, in the linear (transient-free)
# regime, averaged over many periods and many sample paths.
#
# This is the unbiased ground truth to judge whether SDM (and IRK) are correct.
# =============================================================================
using Random, Statistics, Printf, LinearAlgebra

const EPS=2.0; const ZETA=0.1; const TAU=2π; const PER=4π
const Aval=3.0; const Bval=0.5; const ALPHA=0.1

# SDDE:  x'' + 2ζx' + (A+ε cos(t/2)) x = B x(t-τ) + noise
# state u=(x, x'). drift A(t)u + B u(t-τ); noise: multiplicative on the 2nd row.
#   dW couples α-scaled present (A+εcos)·x, 2ζ·x'  and β-scaled delayed B·x(t-τ).
Amat(t) = @inline [0.0 1.0; -(Aval+EPS*cos(0.5*t)) -2ZETA]
Bmat()  = [0.0 0.0; Bval 0.0]
# noise coefficient vector g(u, uτ, t) for the single scalar Wiener process (Itô):
#   2nd-row only:  -α(A+εcos)·x - α·2ζ·x' + αB·xτ   (matches the SDM α,β structure)
gvec(u, uτ, t) = [0.0,
    -ALPHA*(Aval+EPS*cos(0.5*t))*u[1] - ALPHA*2ZETA*u[2] + ALPHA*Bval*uτ[1]]

# Euler–Maruyama over [0, Nper·P] with step dt; history buffer for delay τ.
# Returns the per-period mean-square growth factor (geometric), averaged.
function mc_growth(; Nper=40, nsub=400, npath=4000, seed=20240523, burn=8)
    rng = MersenneTwister(seed)
    P = PER; dt = P/nsub; rstep = round(Int, TAU/dt)
    nsteps = Nper*nsub
    # ensemble mean-square at each period boundary, accumulated over paths
    ms_at_period = zeros(Nper+1)
    sqrtdt = sqrt(dt)
    for ip in 1:npath
        # full trajectory buffer: indices 1..rstep+1 = constant initial history,
        # then rstep+1+n holds the state after n EM steps. Delayed state at step n
        # (current index cur) is buf[cur-rstep].
        buf = Vector{Vector{Float64}}(undef, nsteps+rstep+1)
        u0 = [1.0, 0.0]                              # IC; overall scale cancels in the ratio
        for k in 1:rstep+1; buf[k] = copy(u0); end   # constant history on [-τ,0]
        ms_at_period[1] += dot(u0,u0)
        cur = rstep+1
        pidx = 1
        for n in 1:nsteps
            t = (n-1)*dt
            uτ = buf[cur-rstep]
            un = buf[cur]
            dW = sqrtdt*randn(rng)
            unew = un .+ dt.*(Amat(t)*un .+ Bmat()*uτ) .+ gvec(un,uτ,t).*dW
            cur += 1; buf[cur] = unew
            if n % nsub == 0
                pidx += 1; ms_at_period[pidx] += dot(unew,unew)
            end
        end
    end
    ms_at_period ./= npath
    # per-period growth factors after burn-in; ρ(H) = mean geometric ratio
    ratios = [ms_at_period[k+1]/ms_at_period[k] for k in (burn+1):Nper]
    return exp(mean(log.(ratios))), ms_at_period
end

println("Monte-Carlo ρ(H) reference for stochastic Mathieu (independent of SDM/IRK)\n")
for (np,ns,nsub) in [(2000,40,300),(8000,40,400),(20000,50,500)]
    ρmc, _ = mc_growth(npath=np, Nper=ns, nsub=nsub)
    @printf("  npath=%5d Nper=%2d nsub=%3d  →  ρ_MC ≈ %.5f\n", np, ns, nsub, ρmc)
end
println("\nCompare to: SDM q=2 Aitken-limit ≈ 0.15622 ; high-q raw ≈ 0.1559")
