# =============================================================================
# CORRECT general second-moment engine: window-covariance propagation.
#
# The augmented window state y_{n} evolves y_{n+1} = F̂ₙ y_n + (noise increment).
# Covariance C_n = E[y_n y_nᵀ] (size W×W) evolves:
#     C_{n+1} = F̂ₙ C_n F̂ₙᵀ + Qₙ[C_n]
# Qₙ[C] = Σ_{noise j} Σ_{Gauss stage i} (b_i h) · (Eⱼᵢ C Eⱼᵢᵀ) embedded in the new
#         endpoint block, where Eⱼᵢ :: d×W reads the noise coefficient
#         αⱼ(t_n+c_i h)·x(t_n) + Σₖ βₖⱼ(·)·x(t-τₖ)  off the window (collocation routing).
# With noise OFF, Qₙ=0 ⇒ C→F̂CF̂ᵀ ⇒ ρ(H)=ρ(Φ)² exactly.
#
# ρ(H) = dominant eigenvalue of the linear map vech(C) ↦ vech(period map).
# requires moment_engine.jl loaded first.
# =============================================================================
using LinearAlgebra

vech3(C) = (W=size(C,1); v=zeros(W*(W+1)÷2); k=0; for i in 1:W, j in i:W; k+=1; v[k]=C[i,j]; end; v)
function unvech3(v,W)
    C=zeros(W,W); k=0; for i in 1:W, j in i:W; k+=1; C[i,j]=v[k]; C[j,i]=v[k]; end; C
end

# Per-step window first-moment transitions F̂ₙ and noise-coefficient operators Eⱼᵢ (d×W).
function window_steps(prob::SDDEProblem, S, p)
    a,b,c = gl_tableau(S)
    h = prob.T/p
    ts = range(0, prob.T, length=p+1)
    r = max(round(Int, maxdelay(prob, ts)/h), 1)
    d=prob.d; BSIZE=(S+1)*d; W=(r+1)*BSIZE
    Fs=Vector{Matrix{Float64}}(undef,p)
    Es=Vector{Vector{Tuple{Matrix{Float64},Float64}}}(undef,p)  # [(E::d×W, weight)]
    for n in 1:p
        t_n=(n-1)*h
        Mprop, deldata, _ = step_blocks(prob,a,b,c,h,t_n,0.0,r)
        F=zeros(W,W)
        for β in 1:r
            F[β*BSIZE+1:(β+1)*BSIZE,(β-1)*BSIZE+1:β*BSIZE]=Matrix(I,BSIZE,BSIZE)
        end
        for di in 1:d, rb in 1:BSIZE; F[rb,di]+=Mprop[rb,di]; end
        pre_newest=(n-1)+r
        # window-block index helper for a buffer block 'blk' at this step
        winblk(blk) = pre_newest-blk+1
        for perstage in deldata, (Md,mi,w) in perstage
            for (blk,sel) in ((mi,:x0),(mi+1,:se))
                wb=winblk(blk); (1<=wb<=r+1) || continue; base=(wb-1)*BSIZE
                if sel==:x0
                    for dj in 1:d, rb in 1:BSIZE; F[rb,base+dj]+=Md[rb,dj]*w[1]; end
                else
                    for dj in 1:d, rb in 1:BSIZE
                        F[rb,base+dj]+=Md[rb,dj]*w[S+2]
                        for ss in 1:S; F[rb,base+d+(ss-1)*d+dj]+=Md[rb,dj]*w[ss+1]; end
                    end
                end
            end
        end
        Fs[n]=F
        # noise operators
        el=Tuple{Matrix{Float64},Float64}[]
        for (αf,βfs,σf) in prob.noise
            for i in 1:S
                ti=t_n+c[i]*h
                E=zeros(d,W)
                α=αf(ti)
                # present-state x(t_n) ≈ newest window endpoint = block1 cols 1:d
                for di in 1:d, dj in 1:d; E[di,dj]+=α[di,dj]; end
                for (k,(τf,_)) in enumerate(prob.delays)
                    β=βfs[k](ti); all(iszero,β) && continue
                    τval=τf(ti); rel=(ti-τval-0.0)/h+r+1; mi=floor(Int,rel); ww=colloc_weights(c,rel-mi)
                    for (blk,sel) in ((mi,:x0),(mi+1,:se))
                        wb=winblk(blk); (1<=wb<=r+1) || continue; base=(wb-1)*BSIZE
                        if sel==:x0
                            for di in 1:d, dj in 1:d; E[di,base+dj]+=β[di,dj]*ww[1]; end
                        else
                            for di in 1:d, dj in 1:d
                                E[di,base+dj]+=β[di,dj]*ww[S+2]
                                for ss in 1:S; E[di,base+d+(ss-1)*d+dj]+=β[di,dj]*ww[ss+1]; end
                            end
                        end
                    end
                end
                push!(el,(E, b[i]*h))
            end
        end
        Es[n]=el
    end
    return Fs, Es, W, BSIZE, d, r
end

function second_moment_rho_cov(prob::SDDEProblem, S, p)
    Fs, Es, W, BSIZE, d, r = window_steps(prob,S,p)
    Nv=W*(W+1)÷2
    Hop=zeros(Nv,Nv); e=zeros(Nv)
    for kk in 1:Nv
        fill!(e,0.0); e[kk]=1.0; C=unvech3(e,W)
        for n in 1:p
            Cn = Fs[n]*C*Fs[n]'
            # noise covariance lands in NEW endpoint block (post-shift rows/cols 1:d)
            for (E,wt) in Es[n]
                ECE = E*C*E'              # d×d
                @views Cn[1:d,1:d] .+= wt .* ECE
            end
            C=Cn
        end
        Hop[:,kk]=vech3(C)
    end
    return maximum(abs.(eigen(Hop).values))
end
