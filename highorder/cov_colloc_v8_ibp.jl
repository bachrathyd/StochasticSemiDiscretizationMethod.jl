# =============================================================================
# cov_colloc_v8_ibp.jl — integration-by-parts for delayed VELOCITY (rough) reads
#
# v8 caps at O(h⁴) when the delayed drift reads a Brownian-rough component
# (delayed velocity feedback, the "D" of PD control): the J-integral DOF stores
# ∫ B_read(s)·v(s) ds, whose integrand v(s) is C^{1/2}-rough ⇒ its 2nd-moment
# Gauss quadrature caps at O(h⁴).
#
# IBP fix (localized to the J-DOF construction): for each column c of B_read
# that reads a rough component v_c = q̇_{p(c)} (p = its position antiderivative),
#     ∫ B[:,c](s) v_c(s) ds = [B[:,c](s) q_{p(c)}(s)]₀^{θh} − ∫ B'[:,c](s) q_{p(c)}(s) ds
# Every RHS term now reads the POSITION component q_{p(c)}(s) (C^{3/2}-smooth,
# and q̇=v holds inside the collocation block, so q is the exact antiderivative
# of the velocity polynomial). "Moving column c onto column p(c)" builds Bibp.
# Position-reading columns of B stay direct. Noise machinery is UNCHANGED
# (v8's β noise reads position already), so all v8 gates carry over.
#
# ibp spec: rough_cols (state indices read as velocity) → posmap (their position
# antiderivative index). Mechanical [q₁..qₙ, v₁..vₙ]: rough_cols=n+1..2n,
# posmap[n+i]=i.
# =============================================================================
isdefined(Main, :StepV8) || include(joinpath(@__DIR__, "cov_colloc_v8.jl"))

# Move rough (velocity) columns of B onto their position-antiderivative columns.
function _ibp_mat(B::Matrix{Float64}, rough_cols, posmap, d)
    R = zeros(d, d)
    for c in 1:d
        if c in rough_cols
            R[:, posmap[c]] .+= @view B[:, c]      # read position p(c) instead of v_c
        end
    end
    return R
end
# zero out the rough columns (the direct part keeps only position reads)
function _pos_only(B::Matrix{Float64}, rough_cols, d)
    R = copy(B)
    for c in rough_cols; R[:, c] .= 0.0; end
    return R
end

function step_v8ibp(pb::Prob, a, b, c, h, t_n, r, rough_cols, posmap)
    d=pb.d; S=length(c); BSIZE=(2S+2)*d; W=(r+1)*BSIZE
    As=[Matrix(pb.A(t_n+c[i]*h)) for i in 1:S]
    αs=[Matrix(pb.α(t_n+c[i]*h)) for i in 1:S]
    βs=[Matrix(pb.β(t_n+c[i]*h)) for i in 1:S]
    Bf = s -> Matrix(pb.B(t_n + r*h + s))
    εfd = 1e-6*h
    Bder = s -> (Matrix(pb.B(t_n+r*h+s+εfd)) .- Matrix(pb.B(t_n+r*h+s-εfd))) ./ (2εfd)
    Id=Matrix{Float64}(I,d,d)
    lcoef=_lagr_coefs(c)
    xn_rng = 1:d
    delJ(k)  = (r-1)*BSIZE + (S+1)*d + (k-1)*d
    delJe    = (r-1)*BSIZE + (2S+1)*d
    delY(k)  = (r-1)*BSIZE + d + (k-1)*d
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
    # continuous-output reader at within-block time θ (units of h): returns d×W
    xread(θ) = begin
        R = zeros(d, W); R[:, xn_rng] .= Id
        for j in 1:S; R .+= (h*_lint(lcoef[j], θ)).*Krows[(j-1)*d+1:j*d, :]; end
        R
    end
    θs=vcat(c, 1.0)
    Jrows=zeros((S+1)*d, W)
    for (i,θi) in enumerate(θs)
        Ji = zeros(d, W)
        # (1) direct part: ∫ Bpos(s) x(s) ds  (position-reading columns only)
        for (gx,gw) in zip(_G8.x, _G8.w)
            s=θi*h*gx; wq=θi*h*gw
            Bpos = _pos_only(Bf(s), rough_cols, d)
            Ji .+= wq.*(Bpos*xread(gx*θi))
        end
        # (2) IBP boundary: Bibp(θi h) x(θi h) − Bibp(0) x_n
        Ji .+= _ibp_mat(Bf(θi*h), rough_cols, posmap, d) * xread(θi)
        Rxn = zeros(d, W); Rxn[:, xn_rng] .= Id
        Ji .-= _ibp_mat(Bf(0.0), rough_cols, posmap, d) * Rxn
        # (3) IBP integral: − ∫_0^{θi h} Bibp'(s) x(s) ds
        for (gx,gw) in zip(_G8.x, _G8.w)
            s=θi*h*gx; wq=θi*h*gw
            Bibpd = _ibp_mat(Bder(s), rough_cols, posmap, d)
            Ji .-= wq.*(Bibpd*xread(gx*θi))
        end
        Jrows[(i-1)*d+1:i*d, :] .= Ji
    end
    Pblock=vcat(erow, Yrows, Jrows)
    Dk=[begin R=zeros(d,W); for q in 1:d; R[q, delY(k)+q]=1.0; end; R end for k in 1:S]
    RHSΦ=zeros(S*d, d); for i in 1:S; RHSΦ[(i-1)*d+1:i*d, :] .= Id; end
    Φstack=Minv*RHSΦ
    φstage=[Φstack[(k-1)*d+1:k*d, :] for k in 1:S]
    return StepV8(Pblock, Yrows, Dk, As, αs, βs, Bf, a, b, c, lcoef, φstage,
                  h, d, S, W, BSIZE, r)
end

function build_v8ibp(pb::Prob, S, p, rough_cols, posmap)
    a,b,c=gl_tab(S); h=pb.T/p; r=round(Int,pb.τ/h)
    abs(r*h-pb.τ) < 1e-9*max(pb.τ,1.0) || error("τ/h not integer")
    steps=[step_v8ibp(pb,a,b,c,h,(n-1)*h,r,rough_cols,posmap) for n in 1:p]
    W=steps[1].W; BSIZE=steps[1].BSIZE
    U=Matrix{Float64}(I,W,W)
    for st in steps
        Td=zeros(W,W); Td[1:BSIZE,:]=st.Pblock
        for k in 1:r; Td[k*BSIZE+1:(k+1)*BSIZE,(k-1)*BSIZE+1:k*BSIZE]=Matrix(I,BSIZE,BSIZE); end
        U=Td*U
    end
    return (steps=steps,U=U,W=W,BSIZE=BSIZE,p=p)
end
# reuse applyH_v8m / noise_block_v8m / rho_H_krylov_v8m / rho_U_v8m unchanged
println("cov_colloc_v8_ibp loaded")
