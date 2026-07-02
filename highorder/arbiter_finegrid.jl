# =============================================================================
# arbiter_finegrid.jl — INDEPENDENT reference for ρ(H) of linear SDDEs
#
# Purpose: arbitrate value disagreements between the SDM family and the
# moment-collocation family. This solver shares NO code or discretization
# philosophy with either: plain method-of-steps on the two-time window
# covariance kernel, Heun (2nd-order) drift row + trapezoidal Itô increment,
# equidistant fine grids + h² Richardson.
#
# Kernel state: C[node i, node j] (d×d blocks) = E[x(t−ih) x(t−jh)ᵀ], i,j=0..r,
# τ = r·h exactly. Stored with a CIRCULAR block index so a step costs O(d·W),
# not a full-kernel shift.
#
# One step t0 → t1 = t0+h:
#   row map (Heun):  x⁺ = R·[x(t0); x(t0−τ); x(t1−τ)]
#   cross:  C⁺[0,j] = (R·C)[:,j]        (this step's noise ⊥ all window nodes)
#   diag:   C⁺[0,0] = (R C Rᵀ) + Q,  Q = h/2 (Φ Egg(t0) Φᵀ + Egg(t1)_pred)
#           Egg = α M αᵀ + α K βᵀ + β Kᵀ αᵀ + β Mdd βᵀ  (all read from C)
#   oldest node (t0−τ) falls out of the window.
# ρ(H): PSD-cone power iteration (the exact map preserves PSD; Krein–Rutman ⇒
# dominant eigenvector PSD) with Aitken finish; optionally cross-checked
# against a dense vech eigendecomposition at tiny N (fg_rho_H_dense).
# =============================================================================
using LinearAlgebra, Printf

struct FGProb
    d::Int; T::Float64; τ::Float64
    A::Function; B::Function; α::Function; β::Function
end

struct FGOps
    R0::Matrix{Float64}   # coefficient of x(t0)      (d×d)
    Rr::Matrix{Float64}   # coefficient of x(t0−τ)    (node r)
    Rr1::Matrix{Float64}  # coefficient of x(t1−τ)    (node r−1)
    Φ::Matrix{Float64}
    α0::Matrix{Float64}; α1::Matrix{Float64}
    β0::Matrix{Float64}; β1::Matrix{Float64}
    h::Float64
end

function fg_ops(pb::FGProb, N::Int)
    d=pb.d; h=pb.T/N; r=round(Int,pb.τ/h)
    abs(r*h-pb.τ) < 1e-9*max(pb.τ,1.0) || error("τ/h=$(pb.τ/h) not integer")
    r ≥ 2 || error("need r ≥ 2 (τ must span ≥ 2 grid steps)")
    Id=Matrix{Float64}(I,d,d)
    ops=Vector{FGOps}(undef,N)
    for n in 1:N
        t0=(n-1)*h; t1=n*h
        A0=pb.A(t0); A1=pb.A(t1); B0=pb.B(t0); B1=pb.B(t1)
        Am=pb.A(t0+h/2)
        ops[n]=FGOps(
            Id + h/2*(A0 + A1*(Id + h*A0)),
            h/2*(B0 + h*A1*B0),
            h/2*B1,
            Id + h*Am + h^2/2*(Am*Am),
            pb.α(t0), pb.α(t1), pb.β(t0), pb.β(t1), h)
    end
    return ops, r, h
end

# circular slot of logical node i (0=newest). Convention: slot = base+i (mod rp1),
# so at base=0 the storage IS the logical layout (node i at block i). A step
# decrements base: the new node 0 lands on the outgoing node-r slot
# (mod(base−1) = mod(base+r)), and every surviving node keeps its slot.
@inline slot(base,i,rp1) = mod(base + i, rp1)

blk(C,si,sj,d) = @view C[si*d+1:(si+1)*d, sj*d+1:(sj+1)*d]

# one period, in place on C (W×W, W=(r+1)d); base returned for chaining
function fg_period!(C, ops, r, d, RC)
    rp1=r+1
    base=0                       # logical node i lives at slot mod(base−i,rp1)
    for op in ops
        h=op.h
        s0=slot(base,0,rp1); sr=slot(base,r,rp1); sr1=slot(base,r-1,rp1)
        # Egg(t0)
        M   = Matrix(blk(C,s0,s0,d))
        K   = Matrix(blk(C,s0,sr,d))          # E[x(t0) x(t0−τ)ᵀ]
        Mdd = Matrix(blk(C,sr,sr,d))
        Egg0 = op.α0*M*op.α0' + op.α0*K*op.β0' + op.β0*K'*op.α0' + op.β0*Mdd*op.β0'
        # new row vs all nodes j=0..r−1 (node r falls out): RC[:, slot(j)]
        for j in 0:r-1
            sj=slot(base,j,rp1)
            RC[:,sj*d+1:(sj+1)*d] .= op.R0*blk(C,s0,sj,d) .+ op.Rr*blk(C,sr,sj,d) .+
                                     op.Rr1*blk(C,sr1,sj,d)
        end
        # deterministic diag: R C Rᵀ from the 3×3 support blocks
        Mdet1 = op.R0*M*op.R0' + op.Rr*Mdd*op.Rr' +
                op.Rr1*Matrix(blk(C,sr1,sr1,d))*op.Rr1' +
                op.R0*K*op.Rr' + (op.R0*K*op.Rr')' +
                op.R0*Matrix(blk(C,s0,sr1,d))*op.Rr1' + (op.R0*Matrix(blk(C,s0,sr1,d))*op.Rr1')' +
                op.Rr*Matrix(blk(C,sr,sr1,d))*op.Rr1' + (op.Rr*Matrix(blk(C,sr,sr1,d))*op.Rr1')'
        # Egg(t1) predictor
        K1   = Matrix(RC[:,sr1*d+1:sr1*d+d])   # E[x(t1) x(t1−τ)ᵀ], t1−τ = node r−1
        Mdd1 = Matrix(blk(C,sr1,sr1,d))
        Mp   = Mdet1 + h*(op.Φ*Egg0*op.Φ')
        Egg1 = op.α1*Mp*op.α1' + op.α1*K1*op.β1' + op.β1*K1'*op.α1' + op.β1*Mdd1*op.β1'
        Q = h/2*(op.Φ*Egg0*op.Φ' + Egg1)
        # write the new block into the outgoing slot (old node r == new base)
        snew=sr
        for j in 0:r-1
            sj=slot(base,j,rp1)
            C[snew*d+1:(snew+1)*d, sj*d+1:(sj+1)*d] .= RC[:,sj*d+1:(sj+1)*d]
            C[sj*d+1:(sj+1)*d, snew*d+1:(snew+1)*d] .= transpose(RC[:,sj*d+1:(sj+1)*d])
        end
        C[snew*d+1:(snew+1)*d, snew*d+1:(snew+1)*d] .= Mdet1 .+ Q
        base = mod(base-1, rp1)
    end
    return base
end

# undo the circular permutation so successive period applies are consistent:
# after one period base advanced by N mod (r+1); we don't unpermute — instead
# ρ from norm ratios is permutation-invariant, and successive applies continue
# from the returned base. For power iteration we just keep applying periods
# with a persistent base offset folded into the ops order — simplest correct
# approach: since N steps advance base by N mod rp1 and the PROBLEM is
# T-periodic, the (i,j) labels rotate by a fixed permutation each period. The
# spectral radius of P∘H (permutation ∘ map) equals that of H in modulus only
# if P is trivial. To avoid any subtlety we UNPERMUTE the physical storage
# after each period so the map is the true H.
function fg_unpermute!(C, Ctmp, base, r, d)
    rp1=r+1
    for i in 0:r, j in 0:r
        si=slot(base,i,rp1); sj=slot(base,j,rp1)
        Ctmp[i*d+1:(i+1)*d, j*d+1:(j+1)*d] .= blk(C,si,sj,d)
    end
    C .= Ctmp
    return nothing
end

function fg_rho_H(pb::FGProb, N::Int; tol=1e-10, maxit=5000, verbose=false)
    ops, r, h = fg_ops(pb, N)
    d=pb.d; W=(r+1)*d
    C = Matrix{Float64}(I, W, W)
    Ctmp = similar(C)
    RC = zeros(d, W)
    ρ_old=0.0; ρ=0.0; hist=Float64[]
    for it in 1:maxit
        base=fg_period!(C, ops, r, d, RC)
        base != 0 && fg_unpermute!(C, Ctmp, base, r, d)
        nrm=norm(C)
        ρ=nrm; C ./= nrm
        push!(hist,ρ)
        if it>5 && abs(ρ-ρ_old) < tol*ρ
            r1,r2,r3=hist[end-2],hist[end-1],hist[end]
            den=(r3-r2)-(r2-r1)
            abs(den)>1e-15 && (ρ = r3-(r3-r2)^2/den)
            verbose && @printf("    fg N=%d it=%d ρ=%.12f\n",N,it,ρ)
            return ρ
        end
        ρ_old=ρ
    end
    @warn "fg power iteration hit maxit" ρ
    return ρ
end

# dense vech eigendecomposition (tiny N only) — validates the power iteration
function fg_rho_H_dense(pb::FGProb, N::Int)
    ops, r, h = fg_ops(pb, N)
    d=pb.d; W=(r+1)*d
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx); H=zeros(Nv,Nv)
    Ctmp=zeros(W,W); RC=zeros(d,W)
    for k in 1:Nv
        (i,j)=idx[k]; C=zeros(W,W); C[i,j]=1.0; C[j,i]=1.0
        base=fg_period!(C, ops, r, d, RC)
        base != 0 && fg_unpermute!(C, Ctmp, base, r, d)
        for m in 1:Nv; (p2,q2)=idx[m]; H[m,k]=C[p2,q2]; end
    end
    maximum(abs.(eigen(H).values))
end

# Richardson-extrapolated arbiter value (h² scheme)
function fg_arbiter(pb::FGProb, Ns::Vector{Int}; kwargs...)
    ρs=Float64[]
    for N in Ns
        t0=time(); ρv=fg_rho_H(pb,N;kwargs...)
        @printf("    fg N=%5d ρ=%.12f  (%.0fs)\n", N, ρv, time()-t0); flush(stdout)
        push!(ρs,ρv)
    end
    ρR = ρs[end] + (ρs[end]-ρs[end-1])/((Ns[end]/Ns[end-1])^2-1)
    if length(ρs) ≥ 3
        ρR2 = ρs[end-1] + (ρs[end-1]-ρs[end-2])/((Ns[end-1]/Ns[end-2])^2-1)
        @printf("    Richardson: %.12f (prev pair %.12f, drift %.2e)\n", ρR, ρR2, abs(ρR-ρR2))
    else
        @printf("    Richardson: %.12f\n", ρR)
    end
    return ρR
end
println("arbiter_finegrid loaded")
