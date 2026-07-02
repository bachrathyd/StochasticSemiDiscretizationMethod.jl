# THEORY TEST: is the multiplicative-noise 2nd-moment O(h) wall fundamental, or an
# artifact of (a) frozen within-step covariance + (b) low-order ∫ds quadrature?
#
# Scalar dx = a x dt + σ x dW.  Exact 2nd-moment map over h: m_{n+1}=exp((2a+σ²)h) m_n.
# The 2nd moment m=E[x²] obeys the DETERMINISTIC ODE  dm/ds = (2a+σ²) m.
#
# Claim: if we integrate the MOMENT EQUATION with a high-order method (treating the
# within-step covariance evolution exactly, not frozen), order = method order, NOT O(h).
# Compare three per-step 2nd-moment schemes:
#   (S0) SDM-like:    m_{n+1} = F²·m + G2·m,  F=exp(ah)-split, G2=∫exp(2a(h-s))σ²ds, C frozen
#   (S1) GL(S) collocation applied to the MOMENT ODE dm/ds=(2a+σ²)m  (within-step exact)
using LinearAlgebra, Printf

const a_t=-0.7; const sig=0.5; const T=1.0
const exact = exp((2a_t+sig^2)*T)
const μ = 2a_t + sig^2          # moment-equation rate

function gl_tableau(S)
    if S==1; return reshape([0.5],1,1),[1.0],[0.5]
    elseif S==2; s3=sqrt(3); return [0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S==3; s15=sqrt(15)
        return [5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    end
end

# (S0) SDM-like split, frozen covariance, exact ∫ds for the gain (best-case SDM)
function s0_step(h)
    F = exp(a_t*h)              # use exact deterministic factor (favor SDM)
    G2 = sig^2 * (exp(2a_t*h)-1)/(2a_t)   # ∫₀ʰ exp(2a(h-s))σ² ds, exact
    return F^2 + G2
end

# (S1) GL(S) collocation factor for scalar linear ODE dm/ds = μ m  → m_{n+1}=R(μh) m
#   R = 1 + μh bᵀ(I-μh A)⁻¹ 1  (the RK stability function = Padé ≈ exp to order 2S)
function s1_step(S,h)
    a,b,c = gl_tableau(S)
    M = I - μ*h*a
    Y = M \ ones(S)
    return 1 + μ*h*(b'*Y)
end

function report(label, stepfun, ps)
    println(label)
    prev=nothing
    for p in ps
        h=T/p; mult=stepfun(h)^p; err=abs(mult-exact)
        rate = prev===nothing ? NaN : log2(prev/err)
        @printf("  p=%3d err=%.2e rate=%.2f\n",p,err,rate); prev=err
    end
end

println("exact 2nd-moment multiplier exp((2a+σ²)T) = ", exact, "\n")
report("(S0) SDM-like split (frozen C, exact gain):", s0_step, [4,8,16,32,64,128])
for S in 1:3
    report("\n(S1) GL($S) collocation on the MOMENT ODE (within-step exact):",
           h->s1_step(S,h), [4,8,16,32,64])
end
