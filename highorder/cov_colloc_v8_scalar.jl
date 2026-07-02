# =============================================================================
# cov_colloc_v8_scalar.jl — v8 prototype (scalar d=1, constant B):
# window blocks carry INTEGRATED-HISTORY DOFs so the delayed drift term uses
# the EXACT integral of the (rough) delayed path instead of Gauss sampling.
# See V8_DESIGN.md. Targets the O(h²) cap of v7 on rough delayed-drift reads.
#
# Block layout (BSIZE = 2S+2 scalars, newest→oldest blocks as in v7):
#   [ x_e ; Y_1..Y_S ; P_1..P_S ; P_e ]
#   x_e  = endpoint value, Y_k = stage values (times c_k h into the block step)
#   P_k  = ∫_0^{c_k h} x(t_blk+s) ds ,  P_e = ∫_0^{h} x(t_blk+s) ds
#
# Deterministic step (stage eqs use the STORED integrals of block r — exact
# for the rough path; τ = r h, same tableau ⇒ the requested partial integrals
# are exactly the stored ones):
#   Y_i = x_n + h Σ_j a_ij A_j Y_j + B P_i^{(r)}
#   x_e = x_n + h Σ_j b_j  A_j Y_j + B P_e^{(r)}
#   new integrals from the continuous output through the stage values:
#   K = Atab⁻¹ (Y − 1 x_n)/h ,  P_i = c_i h x_n + h² Σ_j w2_{ij} K_j
#
# Noise increment ΔB ((2S+2)²): node–node exactly as v7 (Σ_noise stage solve,
# operator drift+α², endpoint quadrature, causal fill); node–integral and
# integral–integral from the causal kernel
#   Δ(s,v) = Σn(min) · φ(max)/φ(min)
# with Σn, φ the collocation polynomials of the noise-variance and propagator
# — integrated EXACTLY (Gauss on the smooth pieces, split at the kink).
# =============================================================================
using LinearAlgebra, Printf, KrylovKit

include_gl_defined = isdefined(Main, :gl_tab)
include_gl_defined || include(joinpath(@__DIR__, "cov_colloc_v7.jl"))  # gl_tab, Prob, ...

# ℓint_j(θ) = ∫_0^θ ℓ_j, for the S-stage Lagrange basis on the c-nodes
function lint_weights(c, θ)
    S=length(c)
    # ℓ_j = Lagrange poly on nodes c (degree S−1); integrate monomial form
    w=zeros(S)
    for j in 1:S
        # coefficients of ℓ_j via polynomial division (small S — direct)
        coef=[1.0]
        for m in 1:S
            m==j && continue
            coef=vcat(coef,0.0).*1.0
            newc=zeros(length(coef))
            for (k,ck) in enumerate(coef[1:end-1]); newc[k+1]+=ck; end   # ·θ
            for (k,ck) in enumerate(coef[1:end-1]); newc[k]-=c[m]*ck; end # ·(−c_m)
            coef=newc ./ (c[j]-c[m])
        end
        # ∫_0^θ Σ coef_k θ^{k-1} dθ  (coef[1] is the highest degree? — built as
        # ascending? verify: we built by convolution starting [1]; ordering is
        # ascending powers with the shifts above.)
        s=0.0
        for (k,ck) in enumerate(coef); s += ck*θ^k/k; end
        w[j]=s
    end
    return w
end

# 8-point Gauss–Legendre on [0,1]
const GN = 8
function gauss01()
    a,b,c = gl_tab(GN)
    return c, b
end
const GX, GW = gauss01()

struct Step8
    Pblock::Matrix{Float64}     # BSIZE×W new-block rows
    Yrows::Matrix{Float64}      # S×W (stage-value rows, for Egg)
    Dk::Vector{Vector{Float64}} # node reads of the delayed stage values (for Egg)
    As::Vector{Float64}; αs::Vector{Float64}; βs::Vector{Float64}
    a::Matrix{Float64}; b::Vector{Float64}; c::Vector{Float64}
    h::Float64; S::Int; W::Int; BSIZE::Int; r::Int
    φstage::Vector{Float64}     # drift propagator at stages
    lint_at::Matrix{Float64}    # ℓint_j(c_i) matrix (S×S)
    w2::Matrix{Float64}         # w2[i,j] = ∫_0^{c_i} ℓint_j dθ  (+ row S+1 for θ=1)
end

function step_v8(pb::Prob, a, b, c, h, t_n, r)
    S=length(c); BSIZE=2S+2; W=(r+1)*BSIZE
    As=[pb.A(t_n+c[i]*h)[1,1] for i in 1:S]
    αs=[pb.α(t_n+c[i]*h)[1,1] for i in 1:S]
    βs=[pb.β(t_n+c[i]*h)[1,1] for i in 1:S]
    Bv = pb.B(t_n)[1,1]                     # constant-B prototype
    # window row indices (scalar): block k offset k*BSIZE, layout
    # [x_e(1); Y(2..S+1); P(S+2..2S+1); P_e(2S+2)]
    xn_col   = 1                             # newest block endpoint = x(t_n)
    # Delayed interval [t_n−τ, t_n−τ+c_i h] = [t_{n−r}, t_{n−r}+c_i h] lies in
    # window block r−1 (block k covers [t_{n−k−1}, t_{n−k}]).
    del_node(k) = (r-1)*BSIZE + 1 + k        # Y_k of block r−1
    del_P(k)    = (r-1)*BSIZE + S + 1 + k    # P_k of block r−1
    del_Pe      = (r-1)*BSIZE + 2S + 2
    # stage solve: (I − h a·A) Y = 1·x_n + B·P_del
    M=Matrix{Float64}(I,S,S)
    for i in 1:S, j in 1:S; M[i,j]-= h*a[i,j]*As[j]; end
    Minv=inv(M)
    RHS=zeros(S,W)
    for i in 1:S
        RHS[i,xn_col]=1.0
        RHS[i,del_P(i)]+=Bv
    end
    Yrows=Minv*RHS
    # endpoint
    erow=zeros(1,W); erow[1,xn_col]=1.0; erow[1,del_Pe]+=Bv
    for j in 1:S; erow[1,:] .+= h*b[j]*As[j].*Yrows[j,:]; end
    # continuous output K = Atab⁻¹(Y − 1 x_n)/h ; P_i = c_i h x_n + h² Σ_j w2_ij K_j
    Ainv=inv(a)
    Krows=zeros(S,W)
    for j in 1:S
        for m in 1:S; Krows[j,:] .+= Ainv[j,m].*Yrows[m,:]; end
        Krows[j,xn_col] -= sum(Ainv[j,:])
    end
    Krows ./= h
    lint_at=zeros(S,S); for i in 1:S; lint_at[i,:]=lint_weights(c, c[i]); end
    w2=zeros(S+1,S)
    for i in 1:S
        # ∫_0^{c_i} ℓint_j(θ)dθ via Gauss on [0,c_i]
        for (gx,gw) in zip(GX,GW)
            θ=c[i]*gx; lw=lint_weights(c,θ)
            for j in 1:S; w2[i,j]+= c[i]*gw*lw[j]; end
        end
    end
    for (gx,gw) in zip(GX,GW)
        lw=lint_weights(c,gx)
        for j in 1:S; w2[S+1,j]+= gw*lw[j]; end
    end
    Prows=zeros(S+1,W)
    for i in 1:S+1
        θi = i<=S ? c[i] : 1.0
        Prows[i,xn_col]=θi*h
        for j in 1:S; Prows[i,:] .+= h^2*w2[i,j].*Krows[j,:]; end
    end
    Pblock=vcat(erow, Yrows, Prows)          # BSIZE×W  [x_e;Y;P_1..P_S;P_e]
    # delayed node reads for Egg (point values, exact alignment)
    Dk=[begin v=zeros(W); v[del_node(k)]=1.0; v end for k in 1:S]
    # drift propagator stage values (φ' = A φ, φ(0)=1)
    φstage = Minv*ones(S)
    return Step8(Pblock, Yrows, Dk, As, αs, βs, a, b, c, h, S, W, BSIZE, r,
                 φstage, lint_at, w2)
end

# polynomial evaluations at arbitrary θ∈[0,1] (continuous output of collocations)
φ_at(st::Step8, θ) = 1.0 + st.h*sum(lint_weights(st.c,θ)[j]*st.As[j]*st.φstage[j] for j in 1:st.S)
function Σn_at(st::Step8, θ, vΣ, Egg)
    lw=lint_weights(st.c,θ)
    s=0.0
    for j in 1:st.S
        Lj=2st.As[j]+st.αs[j]^2
        s += lw[j]*(Lj*vΣ[j]+Egg[j])
    end
    return st.h*s
end
# causal kernel Δ(s,v), s,v ∈ [0,1] step-local (in units of h)
function Δker(st::Step8, θs, θv, vΣ, Egg)
    θm=min(θs,θv); θM=max(θs,θv)
    return Σn_at(st,θm,vΣ,Egg)*φ_at(st,θM)/φ_at(st,θm)
end

function noise_block_v8(st::Step8, C)
    S=st.S; h=st.h; a=st.a; b=st.b; c=st.c; BSIZE=st.BSIZE
    # Egg at stages (scalar): (α Yk + β Dk)-squared reads
    Egg=zeros(S)
    for k in 1:S
        Yk=@view st.Yrows[k,:]; Dk=st.Dk[k]
        Mxx=dot(Yk,C*Yk); Mxd=dot(Yk,C*Dk); Mdd=dot(Dk,C*Dk)
        Egg[k]=st.αs[k]^2*Mxx + 2st.αs[k]*st.βs[k]*Mxd + st.βs[k]^2*Mdd
    end
    # Σ_noise stage solve (scalar): (I − h a·L) σ = h a·Egg,  L_j=2A_j+α_j²
    Mop=Matrix{Float64}(I,S,S)
    for i in 1:S, j in 1:S; Mop[i,j]-=h*a[i,j]*(2st.As[j]+st.αs[j]^2); end
    vΣ=Mop\(h*(a*Egg))
    endv=0.0
    for j in 1:S; endv += h*b[j]*((2st.As[j]+st.αs[j]^2)*vΣ[j]+Egg[j]); end
    # node times (θ units): stages c_k, endpoint 1
    θnode=vcat(c, 1.0); Δdiag=vcat(vΣ, endv)
    idx_node(k)= k<=S ? 1+k : 1              # ΔB row of node k (stage k → 1+k, endpoint → 1)
    ΔB=zeros(BSIZE,BSIZE)
    # node–node (causal fill, v7)
    for i in 1:S+1, j in 1:S+1
        θi=θnode[i]; θj=θnode[j]
        v = θi==θj ? Δdiag[i] :
            (θi<θj ? Δdiag[i]*φ_at(st,θj)/φ_at(st,θi) : Δdiag[j]*φ_at(st,θi)/φ_at(st,θj))
        ΔB[idx_node(i),idx_node(j)]=v
    end
    # node–integral: E[η(u_k) ∫_0^{θa h} η] = h∫_0^{θa} Δ(θ_k, θ)dθ (split at θ_k)
    θint=vcat(c, 1.0)
    idx_int(k)= S+1+k                        # P_k → S+1+k, P_e → 2S+2
    for k in 1:S+1, ii in 1:S+1
        θk=θnode[k]; θa=θint[ii]
        s=0.0
        segs = θk<θa ? [(0.0,θk),(θk,θa)] : [(0.0,θa)]
        for (lo,hi) in segs
            hi<=lo && continue
            for (gx,gw) in zip(GX,GW)
                θ=lo+(hi-lo)*gx
                s += (hi-lo)*gw*Δker(st,θk,θ,vΣ,Egg)
            end
        end
        v=h*s
        ΔB[idx_node(k), idx_int(ii)]=v
        ΔB[idx_int(ii), idx_node(k)]=v
    end
    # integral–integral: E[∫_0^{a}η ∫_0^{b}η] = h²∬ Δ(θ,ϑ), triangle split
    for ii in 1:S+1, jj in ii:S+1
        θa=θint[ii]; θb=θint[jj]
        s=0.0
        for (gx,gw) in zip(GX,GW)          # ϑ over [0,θb]
            ϑ=θb*gx; wϑ=θb*gw
            # ∫_0^{θa} Δ(θ,ϑ)dθ — split at ϑ if inside
            segs = ϑ<θa ? [(0.0,ϑ),(ϑ,θa)] : [(0.0,θa)]
            for (lo,hi) in segs
                hi<=lo && continue
                for (gx2,gw2) in zip(GX,GW)
                    θ=lo+(hi-lo)*gx2
                    s += wϑ*(hi-lo)*gw2*Δker(st,θ,ϑ,vΣ,Egg)
                end
            end
        end
        v=h^2*s
        ΔB[idx_int(ii), idx_int(jj)]=v
        ΔB[idx_int(jj), idx_int(ii)]=v
    end
    return ΔB
end

function build_v8(pb::Prob, S, p)
    a,b,c=gl_tab(S); h=pb.T/p; r=max(round(Int,pb.τ/h),1)
    steps=[step_v8(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W)
    for st in steps
        Td=zeros(W,W); Td[1:BSIZE,:]=st.Pblock
        for k in 1:r; Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
        U=Td*U
    end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p)
end

function applyH_v8(eng,C)
    Ck=copy(C)
    for st in eng.steps
        W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock
        PC=P*Ck
        newdiag=PC*P' + noise_block_v8(st,Ck)
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

rho_U_v8(eng)=maximum(abs.(eigen(eng.U).values))

function rho_H_krylov_v8(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    function vec2sym(v)
        C=zeros(W,W); @inbounds for k in 1:Nv; (i,j)=idx[k]; C[i,j]=v[k]; C[j,i]=v[k]; end; C
    end
    function sym2vec(C)
        v=zeros(Nv); @inbounds for k in 1:Nv; (i,j)=idx[k]; v[k]=C[i,j]; end; v
    end
    op(v)= sym2vec(applyH_v8(eng, vec2sym(v)))
    x0=sym2vec(Matrix{Float64}(I,W,W))
    vals,_,_ = KrylovKit.eigsolve(op, x0, 1, :LM; tol=tol,
                                  krylovdim=min(krylovdim,Nv), maxiter=300)
    return maximum(abs.(vals))
end
println("cov_colloc_v8_scalar loaded")
