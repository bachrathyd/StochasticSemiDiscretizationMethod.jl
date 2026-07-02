# Reusable GL(S) collocation window-monodromy for a linear constant-coefficient DDE
#   u'(t) = A0 u(t) + A1 u(t-τ),   u ∈ R^D.
# rho_moment(A0,A1,S,p,τ) = dominant multiplier of the monodromy over one delay window.
# (D inferred from size(A0,1). r=p, grid=[0,τ]. Builds L,R like build_explicit_matrices.)
using LinearAlgebra, SparseArrays

_gl_tab(S) = S==1 ? (reshape([0.5],1,1),[1.0],[0.5]) :
             S==2 ? (let s3=sqrt(3); ([0.25 0.25-s3/6;0.25+s3/6 0.25],[0.5,0.5],[0.5-s3/6,0.5+s3/6]); end) :
             S==3 ? (let s15=sqrt(15); ([5/36 2/9-s15/15 5/36-s15/30;5/36+s15/24 2/9 5/36-s15/24;5/36+s15/30 2/9+s15/15 5/36],[5/18,4/9,5/18],[0.5-s15/10,0.5,0.5+s15/10]); end) :
             error("S=1,2,3")

_cw(c,θ)=(nodes=vcat(0.0,c,1.0);n=length(nodes);[prod(j->j==i ? 1.0 : (θ-nodes[j])/(nodes[i]-nodes[j]),1:n) for i in 1:n])

function _window_phi(A0,A1,S,p,τ)
    a,b,c=_gl_tab(S); D=size(A0,1); BSIZE=(S+1)*D; h=τ/p; r=p; t_start=0.0; SD=S*D
    IL=Int[];JL=Int[];VL=Float64[];IR=Int[];JR=Int[];VR=Float64[]
    for n in 1:p
        roff=(n-1)*BSIZE
        M=Matrix{Float64}(I,SD,SD)
        for i in 1:S, j in 1:S, di in 1:D, dj in 1:D
            M[(i-1)*D+di,(j-1)*D+dj]-= h*a[i,j]*A0[di,dj]
        end
        Minv=inv(M)
        RHSy=zeros(SD,D); for i in 1:S, d in 1:D; RHSy[(i-1)*D+d,d]=1.0; end
        Yy=Minv*RHSy
        ynext=Matrix{Float64}(I,D,D)
        for j in 1:S; ynext+= h*b[j]*(A0*Yy[(j-1)*D+1:j*D,:]); end
        Mprop=vcat(ynext,Yy)
        for i in 1:BSIZE; push!(IL,roff+i);push!(JL,roff+i);push!(VL,1.0); end
        for di in 1:D, rb in 1:BSIZE
            v=-Mprop[rb,di]
            if v!=0
                if n==1; push!(IR,roff+rb);push!(JR,r*BSIZE+di);push!(VR,v)
                else; push!(IL,roff+rb);push!(JL,(n-2)*BSIZE+di);push!(VL,v); end
            end
        end
        for st in 1:S
            RHSd=zeros(SD,D)
            for i in 1:S, di in 1:D, dj in 1:D; RHSd[(i-1)*D+di,dj]=h*a[i,st]*A1[di,dj]; end
            Yd=Minv*RHSd
            ynd=zeros(D,D)
            for j in 1:S
                term=A0*Yd[(j-1)*D+1:j*D,:]; if j==st; term+=A1; end
                ynd+= h*b[j]*term
            end
            Md=vcat(ynd,Yd)
            t_ni=t_start+(n-1)*h+c[st]*h; rel=(t_ni-τ-t_start)/h+r+1
            mi=floor(Int,rel); w=_cw(c,rel-mi)
            bx=mi-(r+1)
            for dj in 1:D, rb in 1:BSIZE
                val=-Md[rb,dj]*w[1]
                if val!=0
                    if bx<=0; push!(IR,roff+rb);push!(JR,(bx+r)*BSIZE+dj);push!(VR,val)
                    else; push!(IL,roff+rb);push!(JL,(bx-1)*BSIZE+dj);push!(VL,val); end
                end
            end
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
        x=zeros(W); x[k]=1.0; vh=zeros((r+1)*BSIZE)
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

rho_moment(A0,A1,S,p,τ)=maximum(abs.(eigen(_window_phi(A0,A1,S,p,τ)).values))
