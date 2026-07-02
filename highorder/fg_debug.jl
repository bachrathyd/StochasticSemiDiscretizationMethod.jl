# Debug the arbiter's deterministic (delay-drift) path.
# Scalar Hayes x' = a x + b x(t−1), a=−1, b=−0.5, T=τ=1:
# exact dominant multiplier 0.3319869969 (verified in HIGHORDER_M2_PLAN).
# Also mirror-Mathieu deterministic vs v7 ρU. Both via the explicit U built
# from the SAME FGOps the covariance step uses.
include(joinpath(@__DIR__, "arbiter_finegrid.jl"))
include(joinpath(@__DIR__, "cov_colloc_v7.jl"))
using Printf

# build the fg deterministic monodromy U by applying the row recursion to unit histories
function fg_U(pb::FGProb, N::Int)
    ops, r, h = fg_ops(pb, N)
    d=pb.d; W=(r+1)*d
    U=zeros(W,W)
    for col in 1:W
        hist=zeros(d, r+1)                      # hist[:,1+i] = x(t−i h)
        hist[mod1(col,d), 1+div(col-1,d)] = 1.0
        for op in ops
            xnew = op.R0*hist[:,1] .+ op.Rr*hist[:,r+1] .+ op.Rr1*hist[:,r]
            hist[:,2:end] .= hist[:,1:end-1]
            hist[:,1] .= xnew
        end
        for i in 0:r, a in 1:d
            U[i*d+a, col] = hist[a, 1+i]
        end
    end
    return U
end

println("── D-1 scalar Hayes deterministic, exact μ = 0.3319869969 ──")
pbH = FGProb(1, 1.0, 1.0, t->fill(-1.0,1,1), t->fill(-0.5,1,1),
             t->zeros(1,1), t->zeros(1,1))
for N in (16, 32, 64, 128)
    ρ = maximum(abs.(eigen(fg_U(pbH,N)).values))
    @printf("  N=%4d ρ(U_fg)=%.10f err=%.2e\n", N, ρ, abs(ρ-0.3319869969))
end

println("── D-2 same but τ=0.5 < T=1 (r=N/2) ──")
# reference from v7 deterministic (GL3 p=64, trusted superconvergent)
pbH2_v7 = Prob(1, 1.0, 0.5, t->fill(-1.0,1,1), t->fill(-0.5,1,1),
               t->zeros(1,1), t->zeros(1,1))
ρref2 = rho_U_v7(build_v7(pbH2_v7, 3, 64))
@printf("  v7 ρU ref = %.10f\n", ρref2)
pbH2 = FGProb(1, 1.0, 0.5, t->fill(-1.0,1,1), t->fill(-0.5,1,1),
              t->zeros(1,1), t->zeros(1,1))
for N in (16, 32, 64, 128)
    ρ = maximum(abs.(eigen(fg_U(pbH2,N)).values))
    @printf("  N=%4d ρ(U_fg)=%.10f err=%.2e\n", N, ρ, abs(ρ-ρref2))
end

println("── D-3 mirror Mathieu deterministic ──")
pbM_v7 = Prob(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0], t->zeros(2,2), t->zeros(2,2))
ρref3 = rho_U_v7(build_v7(pbM_v7, 3, 48))
@printf("  v7 ρU ref = %.10f (ρU² = %.10f)\n", ρref3, ρref3^2)
pbM = FGProb(2, 1.0, 1.0,
    t->[0.0 1.0; -(1.0+0.5*cos(2π*t)) -0.2],
    t->[0.0 0.0; 0.2 0.0], t->zeros(2,2), t->zeros(2,2))
for N in (16, 32, 64)
    ρ = maximum(abs.(eigen(fg_U(pbM,N)).values))
    @printf("  N=%4d ρ(U_fg)=%.10f err=%.2e\n", N, ρ, abs(ρ-ρref3))
end

println("── D-4 covariance path vs U congruence (noise-off, small N) ──")
# With α=β=0 the fg covariance period map MUST equal C ↦ U C Uᵀ exactly.
for N in (8, 16)
    ops, r, h = fg_ops(pbM, N)
    d=2; W=(r+1)*d
    U = fg_U(pbM, N)
    # random symmetric C
    X = randn(W,W); C0 = X*X'
    C1 = copy(C0); Ctmp=similar(C1); RC=zeros(d,W)
    base = fg_period!(C1, ops, r, d, RC)
    base != 0 && fg_unpermute!(C1, Ctmp, base, r, d)
    Cref = U*C0*U'
    @printf("  N=%3d ‖fg(C) − U C Uᵀ‖/‖·‖ = %.2e\n", N, norm(C1-Cref)/norm(Cref))
end
