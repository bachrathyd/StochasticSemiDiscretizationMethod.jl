# CONCEPT: high-order 2nd moment via collocation of the window-covariance Lyapunov ODE,
# in vec form. Validate on scalar dx=αx dW (no delay) → exp((2a+α²)T), expect O(h^2S).
#
# The augmented window state y∈R^W evolves (deterministic + noise): the covariance
# vec(C)∈R^{W²} obeys d vec(C)/dt = 𝓛(t) vec(C), 𝓛 = I⊗𝓐 + 𝓐⊗I + Σ_j 𝓖_j⊗𝓖_j.
# We collocate THIS linear ODE with GL(S) over one period; ρ(monodromy)=ρ(H), order 2S.
#
# For the SCALAR NO-DELAY case the window is just x (W=1): 𝓐=a, 𝓖=α ⇒ 𝓛=2a+α². Then the
# collocation monodromy of d vec(C)/dt=(2a+α²)vec(C) over T is exactly scratch_theory's
# s1_step^p. This file confirms the vec-Lyapunov lifting reproduces that, as the template
# for the windowed case.
using LinearAlgebra, Printf

function gl_tableau(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5]
    elseif S==2; s3=sqrt(3); return [0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S==3; s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    end
end

# Generic GL(S) collocation monodromy over [0,T] for d vec/dt = 𝓛(t) vec, NO delay (ODE).
# Returns the one-period propagator matrix (D×D where D=length of vec state).
function colloc_monodromy_ode(𝓛, T, D, S, p)
    a,b,c = gl_tableau(S); h=T/p
    Φ = Matrix{Float64}(I, D, D)
    for n in 1:p
        t_n=(n-1)*h
        Ls=[𝓛(t_n+c[i]*h) for i in 1:S]            # D×D at each stage
        # stage system: Yi = y + h Σ_j a_ij Ls[j] Yj ; solve [I - h a⊗?]... block form
        SD=S*D; M=Matrix{Float64}(I,SD,SD)
        for i in 1:S, j in 1:S
            M[(i-1)*D+1:i*D,(j-1)*D+1:j*D] -= h*a[i,j]*Ls[j]
        end
        # Y from y: RHS = [I;I;...;I] (each stage gets +y)
        RHS=zeros(SD,D); for i in 1:S; RHS[(i-1)*D+1:i*D,:]=Matrix(I,D,D); end
        Y=M\RHS
        ynext=Matrix{Float64}(I,D,D)
        for j in 1:S; ynext += h*b[j]*(Ls[j]*Y[(j-1)*D+1:j*D,:]); end
        Φ = ynext*Φ
    end
    return Φ
end

# scalar no-delay: 𝓛 = 2a+α² (1×1). vec(C)=C scalar.
const a_t=-0.7; const α=0.5; const T=1.0
const exact=exp((2a_t+α^2)*T)
𝓛scalar(t)=reshape([2a_t+α^2],1,1)

println("Scalar dx=αx dW, exact=", exact)
for S in 1:3
    @printf("GL%d:\n",S); prev=nothing
    for p in [4,8,16,32]
        Φ=colloc_monodromy_ode(𝓛scalar,T,1,S,p)
        ρ=abs(Φ[1,1]); err=abs(ρ-exact)
        rate=prev===nothing ? NaN : log2(prev/err)
        @printf("  p=%2d ρ=%.11f err=%.2e rate=%.2f\n",p,ρ,err,rate); prev=err
    end
end
