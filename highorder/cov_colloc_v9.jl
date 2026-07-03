# =============================================================================
# cov_colloc_v9.jl — DOF-pruned integrated-history engine.
#
# v8 persists (2S+2) sub-blocks per delay slot: [x_e; Y_1..Y_S; J_1..J_S; J_e].
# The covariance therefore costs (2S+2)^2 relative to classical SDM (1 block).
# But the S stage-value blocks Y_i are read ONLY by the delayed MULTIPLICATIVE
# noise term (β·x(t−τ) sampled at the stage nodes). When the problem has no
# delayed multiplicative noise (β ≡ 0 over the whole period — the common case
# for delayed feedback control, turning/milling, and any problem whose noise
# reads only the present state), the persistent Y_i are never read and can be
# dropped. The block shrinks to [x_e; J_1..J_S; J_e] of size (S+2)d, cutting the
# covariance factor from (2S+2)^2 to (S+2)^2 (e.g. S=4: 100 → 36, ≈2.8×).
#
# The stage values are STILL computed inside each step (they build the J
# integrals, the endpoint, and the present-noise contraction) — they are simply
# not carried in the persistent covariance window. All order-critical machinery
# (integrated history J, causal intra-block noise fill) is unchanged, so v9
# reproduces v8 to solver tolerance whenever β ≡ 0.
#
# When β ≢ 0 the reduction is unsafe (the delayed noise genuinely reads the
# stage values); build_v9m then transparently falls back to build_v8m.
#
# Block layout (BSIZE9 = (S+2)d): [x_e; J_1..J_S; J_e]. Requires τ = r·h, r ≥ 1.
# =============================================================================
isdefined(Main, :StepV8) || include(joinpath(@__DIR__, "cov_colloc_v8.jl"))

struct StepV9
    Pblock::Matrix{Float64}        # BSIZE9×W (deterministic, reduced block)
    Yrows::Matrix{Float64}         # Sd×W stage-value rows (computed, NOT persisted)
    As::Vector{Matrix{Float64}}; αs::Vector{Matrix{Float64}}
    σs::Vector{Matrix{Float64}}
    Bf::Function
    a::Matrix{Float64}; b::Vector{Float64}; c::Vector{Float64}
    lcoef::Vector{Vector{Float64}}
    φstage::Vector{Matrix{Float64}}
    h::Float64; d::Int; S::Int; W::Int; BSIZE::Int; r::Int
end

# β ≡ 0 test over a fine sample of the period
function _no_delay_noise(pb::Prob; nt=64)
    for k in 0:nt-1
        maximum(abs, pb.β((k+0.5)/nt * pb.T)) > 1e-14 && return false
    end
    true
end

function step_v9m(pb::Prob, a, b, c, h, t_n, r)
    d=pb.d; S=length(c); BSIZE=(S+2)*d; W=(r+1)*BSIZE
    As=[Matrix(pb.A(t_n+c[i]*h)) for i in 1:S]
    αs=[Matrix(pb.α(t_n+c[i]*h)) for i in 1:S]
    σs=[Matrix(pb.σ(t_n+c[i]*h)) for i in 1:S]
    Bf = s -> Matrix(pb.B(t_n + r*h + s))
    Id=Matrix{Float64}(I,d,d)
    lcoef=_lagr_coefs(c)
    xn_rng = 1:d
    # reduced-block delayed offsets (J's start at offset d, no Y block)
    delJ(k) = (r-1)*BSIZE + (k)*d              # J_k of block r−1 (k=1..S): offset (k)*d
    delJe   = (r-1)*BSIZE + (S+1)*d            # J_e of block r−1
    M=Matrix{Float64}(I,S*d,S*d)
    for i in 1:S, j in 1:S; M[(i-1)*d+1:i*d,(j-1)*d+1:j*d] .-= h*a[i,j].*As[j]; end
    Minv=inv(M)
    RHS=zeros(S*d, W)
    for i in 1:S
        RHS[(i-1)*d+1:i*d, xn_rng] .= Id
        for q in 1:d; RHS[(i-1)*d+q, delJ(i)+q] += 1.0; end
    end
    Yrows=Minv*RHS
    erow=zeros(d, W); erow[:, xn_rng] .= Id
    for q in 1:d; erow[q, delJe+q] += 1.0; end
    for j in 1:S; erow .+= h*b[j].*(As[j]*Yrows[(j-1)*d+1:j*d, :]); end
    Ainv=inv(a)
    Krows=zeros(S*d, W)
    for j in 1:S, m in 1:S
        Krows[(j-1)*d+1:j*d, :] .+= Ainv[j,m].*Yrows[(m-1)*d+1:m*d, :]
        Krows[(j-1)*d+1:j*d, xn_rng] .-= Ainv[j,m].*Id
    end
    Krows ./= h
    θs=vcat(c, 1.0)
    Jrows=zeros((S+1)*d, W)
    for (i,θi) in enumerate(θs)
        Wx = zeros(d,d); Wk = [zeros(d,d) for _ in 1:S]
        for (gx,gw) in zip(_G8.x, _G8.w)
            s=θi*h*gx; wq=θi*h*gw; Bs=Bf(s)
            Wx .+= wq.*Bs
            for j in 1:S; Wk[j] .+= (wq*h*_lint(lcoef[j], s/h)).*Bs; end
        end
        Jrows[(i-1)*d+1:i*d, xn_rng] .= Wx
        for j in 1:S; Jrows[(i-1)*d+1:i*d, :] .+= Wk[j]*Krows[(j-1)*d+1:j*d, :]; end
    end
    Pblock=vcat(erow, Jrows)                    # [x_e; J_1..J_S; J_e], no Y
    RHSΦ=zeros(S*d, d); for i in 1:S; RHSΦ[(i-1)*d+1:i*d, :] .= Id; end
    Φstack=Minv*RHSΦ
    φstage=[Φstack[(k-1)*d+1:k*d, :] for k in 1:S]
    return StepV9(Pblock, Yrows, As, αs, σs, Bf, a, b, c, lcoef, φstage,
                  h, d, S, W, BSIZE, r)
end

# helpers mirroring the v8 versions but taking a StepV9
_φ_at9(st::StepV9, θ) = begin
    Φ=Matrix{Float64}(I, st.d, st.d)
    for j in 1:st.S; Φ .+= (st.h*_lint(st.lcoef[j], θ)).*(st.As[j]*st.φstage[j]); end
    Φ
end
_Σn_at9(st::StepV9, θ, Σs, Egg) = begin
    out=zeros(st.d,st.d)
    for j in 1:st.S
        rhs = st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j]
        out .+= (st.h*_lint(st.lcoef[j], θ)).*rhs
    end
    out
end
_Δker9(st::StepV9, θa, θb, Σs, Egg, φc) =
    θa<=θb ? _Σn_at9(st,θa,Σs,Egg)*(φc(θb)/φc(θa))' :
             (φc(θa)/φc(θb))*_Σn_at9(st,θb,Σs,Egg)

function noise_block_v9m(st::StepV9, C)
    d=st.d; S=st.S; h=st.h; a=st.a; b=st.b; c=st.c; BSIZE=st.BSIZE
    Id=Matrix{Float64}(I,d,d)
    # present-state contraction only (β ≡ 0 ⇒ no delayed-node reads)
    Egg=Vector{Matrix{Float64}}(undef,S)
    for k in 1:S
        Yk=@view st.Yrows[(k-1)*d+1:k*d, :]
        Mxx=Yk*C*Yk'
        e = st.αs[k]*Mxx*st.αs[k]' .+ st.σs[k]*st.σs[k]'
        Egg[k]=(e.+e')./2
    end
    d2=d*d
    Mop=Matrix{Float64}(I,S*d2,S*d2)
    for i in 1:S, j in 1:S
        Lj=kron(Id,st.As[j]) .+ kron(st.As[j],Id) .+ kron(st.αs[j],st.αs[j])
        Mop[(i-1)*d2+1:i*d2, (j-1)*d2+1:j*d2] .-= h*a[i,j].*Lj
    end
    rhs=zeros(S*d2)
    for j in 1:S, k in 1:S; rhs[(j-1)*d2+1:j*d2] .+= h*a[j,k].*vec(Egg[k]); end
    vΣ=Mop\rhs
    Σs=[reshape(vΣ[(k-1)*d2+1:k*d2],d,d) for k in 1:S]
    endm=zeros(d,d)
    for j in 1:S
        endm .+= h*b[j].*(st.As[j]*Σs[j] .+ Σs[j]*st.As[j]' .+ st.αs[j]*Σs[j]*st.αs[j]' .+ Egg[j])
    end
    cache=Dict{Float64,Matrix{Float64}}(); φc(θ)=get!(()->_φ_at9(st,θ),cache,θ)
    # reduced block DOFs: x_e (rows 1:d) then J_1..J_S, J_e
    rng_J(i) = (i*d+1 : (i+1)*d)               # i=1..S+1 (i=S+1 → J_e)
    ΔB=zeros(BSIZE,BSIZE)
    ΔB[1:d, 1:d] .= endm                        # x_e – x_e (endpoint noise)
    θJ=vcat(c,1.0)
    # E[Jη_i η(x_e)]  — x_e is the endpoint node θ=1 (θk=1 ≥ θa ⇒ single segment)
    for i in 1:S+1
        θa=θJ[i]; acc=zeros(d,d)
        for (gx,gw) in zip(_G8.x,_G8.w)
            θ=θa*gx
            acc .+= (θa*gw).*(st.Bf(θ*h)*_Δker9(st,θ,1.0,Σs,Egg,φc))
        end
        V=h.*acc
        ΔB[rng_J(i), 1:d] .= V; ΔB[1:d, rng_J(i)] .= V'
    end
    # E[Jη_i Jη_j]
    for i in 1:S+1, j in 1:S+1
        j < i && continue
        θa=θJ[i]; θb=θJ[j]; acc=zeros(d,d)
        for (gx,gw) in zip(_G8.x,_G8.w)
            ϑ=θb*gx; wϑ=θb*gw; Bv=st.Bf(ϑ*h)
            segs = ϑ<θa ? ((0.0,ϑ),(ϑ,θa)) : ((0.0,θa),)
            for (lo,hi) in segs
                hi<=lo && continue
                for (gx2,gw2) in zip(_G8.x,_G8.w)
                    θ=lo+(hi-lo)*gx2
                    acc .+= (wϑ*(hi-lo)*gw2).*(st.Bf(θ*h)*_Δker9(st,θ,ϑ,Σs,Egg,φc)*Bv')
                end
            end
        end
        V=(h^2).*acc
        ΔB[rng_J(i), rng_J(j)] .= V
        i != j && (ΔB[rng_J(j), rng_J(i)] .= V')
    end
    return ΔB
end

function build_v9m(pb::Prob, S, p; force=false)
    if !force && !_no_delay_noise(pb)
        return build_v8m(pb, S, p)               # unsafe to prune ⇒ fall back to v8
    end
    a,b,c=gl_tab(S); h=pb.T/p; r=round(Int,pb.τ/h)
    abs(r*h-pb.τ) < 1e-9*max(pb.τ,1.0) || error("τ/h not integer")
    r ≥ 1 || error("need r ≥ 1")
    steps=[step_v9m(pb,a,b,c,h,(n-1)*h,r) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W)
    for st in steps
        Td=zeros(W,W); Td[1:BSIZE,:]=st.Pblock
        for k in 1:r; Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
        U=Td*U
    end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p,engine=:v9)
end

function applyH_v9m(eng,C)
    Ck=copy(C)
    for st in eng.steps
        W=st.W; BSIZE=st.BSIZE; keep=W-BSIZE
        P=st.Pblock; PC=P*Ck
        newdiag=PC*P' + noise_block_v9m(st,Ck)
        Cnew=similar(Ck)
        Cnew[1:BSIZE,1:BSIZE]=newdiag
        Cnew[1:BSIZE,BSIZE+1:end]=PC[:,1:keep]
        Cnew[BSIZE+1:end,1:BSIZE]=transpose(PC[:,1:keep])
        Cnew[BSIZE+1:end,BSIZE+1:end]=Ck[1:keep,1:keep]
        Ck=Cnew
    end
    return Ck
end

# dispatch helpers: v9 eng (has :engine=:v9) vs v8 fallback (a plain NamedTuple)
_applyH(eng,C) = haskey(eng,:engine) && eng.engine==:v9 ? applyH_v9m(eng,C) : applyH_v8m(eng,C)

function rho_H_krylov_v9m(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    vec2sym(v)=(C=zeros(W,W); @inbounds for k in 1:Nv;(i,j)=idx[k];C[i,j]=v[k];C[j,i]=v[k];end; C)
    sym2vec(C)=(v=zeros(Nv); @inbounds for k in 1:Nv;(i,j)=idx[k];v[k]=C[i,j];end; v)
    D=sym2vec(_applyH(eng, zeros(W,W)))
    op(v)= sym2vec(_applyH(eng, vec2sym(v))) .- D
    x0=sym2vec(Matrix{Float64}(I,W,W))
    vals,_,_=KrylovKit.eigsolve(op,x0,1,:LM;tol=tol,krylovdim=min(krylovdim,Nv),maxiter=300)
    maximum(abs.(vals))
end

function fixPoint_v9m(eng; tol=1e-11, krylovdim=30)
    W=eng.W
    idx=Tuple{Int,Int}[]; for i in 1:W,j in i:W; push!(idx,(i,j)); end
    Nv=length(idx)
    vec2sym(v)=(C=zeros(W,W); @inbounds for k in 1:Nv;(i,j)=idx[k];C[i,j]=v[k];C[j,i]=v[k];end; C)
    sym2vec(C)=(v=zeros(Nv); @inbounds for k in 1:Nv;(i,j)=idx[k];v[k]=C[i,j];end; v)
    D=_applyH(eng, zeros(W,W)); dvec=sym2vec(D)
    Hlin(v)=sym2vec(_applyH(eng, vec2sym(v))) .- dvec
    sol,info=KrylovKit.linsolve(v->v .- Hlin(v), dvec, dvec; tol=tol,
                                krylovdim=min(krylovdim,Nv), maxiter=300)
    info.converged==0 && @warn "fixPoint_v9m: not fully converged" info
    vec2sym(sol)
end

println("cov_colloc_v9 (DOF-pruned, β≡0) loaded")
