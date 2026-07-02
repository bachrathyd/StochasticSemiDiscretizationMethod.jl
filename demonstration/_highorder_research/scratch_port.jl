# Self-contained port (no MFCM, no Drive): GL tableau + L/R window monodromy Φ for a
# scalar DDE x'=a x + b x(t-τ).  Must reproduce the trusted ρ(Φ)=0.331986996893.
# Stays in SSDM project; dominant eig of Φ via dense eigen (Φ is small, (p+1)*BSIZE).
using LinearAlgebra, SparseArrays, Printf

# ---- GL(S) Butcher tableaux ----
function gl_tableau(S::Int)
    if S == 1
        return reshape([0.5],1,1), [1.0], [0.5]
    elseif S == 2
        s3=sqrt(3)
        a=[0.25 0.25-s3/6; 0.25+s3/6 0.25]; return a,[0.5,0.5],[0.5-s3/6,0.5+s3/6]
    elseif S == 3
        s15=sqrt(15)
        a=[5/36 2/9-s15/15 5/36-s15/30; 5/36+s15/24 2/9 5/36-s15/24; 5/36+s15/30 2/9+s15/15 5/36]
        return a,[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]
    end
    error("S=1,2,3 only")
end

# collocation interp weights on nodes {0,c...,1} (length S+2): [w0, w_c1..w_cS, w1]
function colloc_weights(c, theta)
    nodes = vcat(0.0, c, 1.0); n=length(nodes); w=zeros(n)
    for i in 1:n
        wi=1.0
        for j in 1:n; i!=j && (wi *= (theta-nodes[j])/(nodes[i]-nodes[j])); end
        w[i]=wi
    end
    return w
end

# ---- per-step system matrices (D=1), mirroring MFCM build_system_matrices ----
function step_sys(a,b,c,A,B,h,t_n,tau,t_start,r)
    S=length(c); BSIZE=S+1
    M = Matrix{Float64}(I,S,S) .- h .* a .* A
    Minv = inv(M)
    Y_from_y = Minv*ones(S)
    y_next = 1.0 + h*sum(b[j]*A*Y_from_y[j] for j in 1:S)
    M_prop = vcat(y_next, Y_from_y)
    M_del = Vector{Vector{Float64}}(undef,S); didx=zeros(Int,S); wts=Vector{Vector{Float64}}(undef,S)
    for st in 1:S
        RHS = [h*a[i,st]*B for i in 1:S]
        Yd = Minv*RHS
        ynd = 0.0
        for j in 1:S
            term = A*Yd[j]; j==st && (term += B); ynd += h*b[j]*term
        end
        M_del[st]=vcat(ynd,Yd)
        t_ni=t_n+c[st]*h; rel=(t_ni-tau-t_start)/h + r + 1
        mi=floor(Int,rel); didx[st]=mi; wts[st]=colloc_weights(c, rel-mi)
    end
    return M_prop, M_del, didx, wts, BSIZE
end

# ---- L (p*BSIZE square), R (p*BSIZE × (r+1)*BSIZE) — collocation routing (D=1) ----
function build_LR(a,b,c,A,B,p,h,tau,t_start,r)
    S=length(c); BSIZE=S+1
    IL=Int[];JL=Int[];VL=Float64[]; IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE; t_n=t_start+(n-1)*h
        Mp,Md,didx,wts,_=step_sys(a,b,c,A,B,h,t_n,tau,t_start,r)
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        if n==1
            for rb in 1:BSIZE
                v=-Mp[rb]; v!=0 && (push!(IR,roff+rb);push!(JR,r*BSIZE+1);push!(VR,v))
            end
        else
            for rb in 1:BSIZE
                v=-Mp[rb]; v!=0 && (push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+1);push!(VL,v))
            end
        end
        for st in 1:S
            mi=didx[st]; w=wts[st]; Mkj=Md[st]
            # block mi: x-part, weight w[1], endpoint column
            bx=mi-(r+1)
            if w[1]!=0
                for rb in 1:BSIZE
                    val=-Mkj[rb]*w[1]
                    if val!=0
                        if bx<=0; push!(IR,roff+rb);push!(JR,(bx+r)*BSIZE+1);push!(VR,val)
                        else; push!(IL,roff+rb);push!(JL,(bx-1)*BSIZE+1);push!(VL,val); end
                    end
                end
            end
            # block mi+1: stages w[2..S+1] and endpoint w[S+2]
            be=(mi+1)-(r+1)
            for ss in 1:S
                ws=w[ss+1]; ws==0 && continue
                for rb in 1:BSIZE
                    val=-Mkj[rb]*ws
                    if val!=0
                        col_in_block = 1 + ss          # stage ss column (D=1: endpoint=1, stages=2..S+1)
                        if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+col_in_block);push!(VR,val)
                        else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+col_in_block);push!(VL,val); end
                    end
                end
            end
            we=w[S+2]
            if we!=0
                for rb in 1:BSIZE
                    val=-Mkj[rb]*we
                    if val!=0
                        if be<=0; push!(IR,roff+rb);push!(JR,(be+r)*BSIZE+1);push!(VR,val)
                        else; push!(IL,roff+rb);push!(JL,(be-1)*BSIZE+1);push!(VL,val); end
                    end
                end
            end
        end
    end
    L=sparse(IL,JL,VL,p*BSIZE,p*BSIZE)
    R=sparse(IR,JR,VR,p*BSIZE,(r+1)*BSIZE)
    return L,R,BSIZE
end

# window monodromy Φ on the (r+1)*BSIZE window, replicating base_sweep load/store.
# x_in window: block 0=newest..block r=oldest. Build Φ column by column.
function window_phi(a,b,c,A,B,p,tau; r=p)
    h = tau/p   # for r=p, grid=[0,τ], so h=τ/p
    t_start=0.0
    L,R,BSIZE = build_LR(a,b,c,A,B,p,h,tau,t_start,r)
    Lf = lu(Matrix(L)); Rm = Matrix(R)
    W=(r+1)*BSIZE
    Phi=zeros(W,W)
    for k in 1:W
        x=zeros(W); x[k]=1.0
        # load: v_hist[(r-i)*BSIZE..] = x_in[i*BSIZE..]
        vh=zeros((r+1)*BSIZE)
        for i in 0:r
            vh[(r-i)*BSIZE+1:(r-i+1)*BSIZE] = x[i*BSIZE+1:(i+1)*BSIZE]
        end
        vper = Lf \ (Rm*vh)         # [v_1..v_p]
        # store: y_out[i*BSIZE..] = v_period block (p-i)
        y=zeros(W)
        for i in 0:r
            kk=p-i
            if kk>=1
                y[i*BSIZE+1:(i+1)*BSIZE]=vper[(kk-1)*BSIZE+1:kk*BSIZE]
            else
                y[i*BSIZE+1:(i+1)*BSIZE]=vh[(kk+r)*BSIZE+1:(kk+r+1)*BSIZE]
            end
        end
        Phi[:,k]=y
    end
    return Phi
end

const A_t=-1.0; const B_t=-0.5; const TAU=1.0
const REF=0.331986996893
println("Self-contained port — ρ(Φ) must → $REF")
for S in 1:3
    @printf("GL(%d): ", S)
    for p in [4,8,16,32]
        a,b,c = gl_tableau(S)
        Phi=window_phi(a,b,c,A_t,B_t,p,TAU)
        ρ=maximum(abs.(eigen(Phi).values))
        @printf("p=%d ρ=%.10f(err %.1e)  ", p, ρ, abs(ρ-REF))
    end
    println()
end
