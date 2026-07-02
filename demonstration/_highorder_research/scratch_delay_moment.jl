# DELAY-CASE theory test: stochastic Hayes  dx=(A x+B x_{t-1})dt+(β x_{t-1}+σ)dW.
# The 2nd moment satisfies a DETERMINISTIC delay system. Its dominant multiplier over
# one delay τ=1 is the exact ρ(H) rate. We show GL(S) collocation on the MOMENT system
# converges at O(h^{2S}) — beating SDM's O(h³) ceiling.
#
# Moment variables (τ=1):  M(t)=E[x_t²], P(t)=E[x_t x_{t-1}].  By Itô:
#   dM/dt = 2A M + 2B P + β² M(t-1) + σ²        (β² M(t-1) = Itô corr. of delayed mult. noise)
#   dP/dt = E[d(x_t) x_{t-1}] = A P + B M(t-1) + (cross terms = 0, no common noise at t,t-1)
# This is a linear delay system in (M,P) with delay 1. Dominant multiplier over τ=1 → ρ(H).
#
# We compare GL(S) collocation monodromy of this MOMENT DDE to a high-p reference.
using LinearAlgebra, SparseArrays, Printf

const A_t=-1.0; const B_t=-0.4; const beta=0.3; const TAU=1.0
# (σ only shifts the fixed point, not the spectral radius — set 0 for the homogeneous part.)

# Moment DDE as a 2D linear DDE:  u=(M,P),  u'(t)=A0 u(t)+A1 u(t-1)
# dM/dt = 2A M + 2B P + β² M(t-1)
# dP/dt = A P            + B M(t-1)
const A0 = [2A_t  2B_t;
            0.0    A_t]
const A1 = [beta^2  0.0;
            B_t      0.0]

# GL collocation monodromy of a 2D linear DDE over one delay window (reuse port logic, D=2).
function gl_tableau(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5]
    elseif S==2; s3=sqrt(3); return [0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S==3; s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    end
end
colloc_weights(c,θ)=(nodes=vcat(0.0,c,1.0);n=length(nodes);[prod(j->j==i ? 1.0 : (θ-nodes[j])/(nodes[i]-nodes[j]),1:n) for i in 1:n])

# D=2 augmented monodromy: build per-step F̂ via L,R like before but for matrices A0,A1.
function window_phi_2d(S,p)
    a,b,c=gl_tableau(S); D=2; BSIZE=(S+1)*D; h=TAU/p; r=p; t_start=0.0
    SD=S*D
    IL=Int[];JL=Int[];VL=Float64[];IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE
        # M = I_SD - h*(a ⊗ A0)
        M=Matrix{Float64}(I,SD,SD)
        for i in 1:S, j in 1:S, di in 1:D, dj in 1:D
            M[(i-1)*D+di,(j-1)*D+dj]-= h*a[i,j]*A0[di,dj]
        end
        Minv=inv(M)
        # M_prop: stages from prev endpoint x (RHS = I_D stacked into each stage's A0-driven eq)
        RHSy=zeros(SD,D); for i in 1:S, d in 1:D; RHSy[(i-1)*D+d,d]=1.0; end
        Yy=Minv*RHSy   # SD×D
        ynext=Matrix{Float64}(I,D,D)
        for j in 1:S; ynext+= h*b[j]*(A0*Yy[(j-1)*D+1:j*D,:]); end
        Mprop=vcat(ynext,Yy)   # BSIZE×D
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        for di in 1:D, rb in 1:BSIZE
            v=-Mprop[rb,di]
            if v!=0
                if n==1; push!(IR,roff+rb);push!(JR,r*BSIZE+di);push!(VR,v)
                else; push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+di);push!(VL,v); end
            end
        end
        # delay: single delay τ=1, A1. routing at each stage.
        for st in 1:S
            RHSd=zeros(SD,D)
            for i in 1:S, di in 1:D, dj in 1:D; RHSd[(i-1)*D+di,dj]=h*a[i,st]*A1[di,dj]; end
            Yd=Minv*RHSd
            ynd=zeros(D,D)
            for j in 1:S
                term=A0*Yd[(j-1)*D+1:j*D,:]; if j==st; term+=A1; end
                ynd+= h*b[j]*term
            end
            Md=vcat(ynd,Yd)  # BSIZE×D
            t_ni=t_start+(n-1)*h+c[st]*h; rel=(t_ni-TAU-t_start)/h+r+1
            mi=floor(Int,rel); w=colloc_weights(c,rel-mi)
            # block mi x-part w[1]
            bx=mi-(r+1)
            for dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[1]
                if val!=0
                    if bx<=0; push!(IR,roff+rb);push!(JR,(bx+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(bx-1)*BSIZE+dj);push!(VL,val); end
                end
            end
            # block mi+1 stages w[2..S+1] + endpoint w[S+2]
            be=(mi+1)-(r+1)
            for ss in 1:S, dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[ss+1]
                if val!=0
                    col=D+(ss-1)*D+dj
                    if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+col);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+col);push!(VL,val); end
                end
            end
            for dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[S+2]
                if val!=0
                    if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+dj);push!(VL,val); end
                end
            end
        end
    end
    L=sparse(IL,JL,VL,p*BSIZE,p*BSIZE); R=sparse(IR,JR,VR,p*BSIZE,(r+1)*BSIZE)
    Lf=lu(Matrix(L)); Rm=Matrix(R); W=(r+1)*BSIZE
    Phi=zeros(W,W)
    for k in 1:W
        x=zeros(W); x[k]=1.0
        vh=zeros((r+1)*BSIZE)
        for i in 0:r; vh[(r-i)*BSIZE+1:(r-i+1)*BSIZE]=x[i*BSIZE+1:(i+1)*BSIZE]; end
        vper=Lf\(Rm*vh); y=zeros(W)
        for i in 0:r
            kk=p-i
            if kk>=1; y[i*BSIZE+1:(i+1)*BSIZE]=vper[(kk-1)*BSIZE+1:kk*BSIZE]
            else; y[i*BSIZE+1:(i+1)*BSIZE]=vh[(kk+r)*BSIZE+1:(kk+r+1)*BSIZE]; end
        end
        Phi[:,k]=y
    end
    Phi
end

rho_moment(S,p)=maximum(abs.(eigen(window_phi_2d(S,p)).values))

# high-p reference
ref = rho_moment(3,256)
@printf("Reference dominant 2nd-moment multiplier (GL3 p=256) = %.12f\n\n", ref)

for S in 1:3
    @printf("GL(%d) collocation on the MOMENT DDE:\n", S)
    prev=nothing
    for p in [4,8,16,32,64]
        ρ=rho_moment(S,p); err=abs(ρ-ref)
        rate=prev===nothing ? NaN : log2(prev/err)
        @printf("  p=%3d  ρ=%.11f  err=%.2e  rate=%.2f\n",p,ρ,err,rate); prev=err
    end
    println()
end
