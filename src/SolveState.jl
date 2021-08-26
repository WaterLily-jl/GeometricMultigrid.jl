struct SolveState{matT<:Poisson, vecT<:FieldVec}
    A::matT
    iD::vecT
    x::vecT
    r::vecT
    ϵ::vecT
    function SolveState(x::FieldVec,A::Poisson,b::FieldVec,invtol=1e-8)
        iD,r = zero(x),zero(x)
        @loop iD[I] = abs(A.D[I])>invtol ? inv(A.D[I]) : zero(eltype(A))
        @loop r[I] = b[I]-mult(I,A.L,A.D,x)
        new{typeof(A),typeof(x)}(A,iD,x,r,zero(x))
    end
end
Base.show(io::IO, ::MIME"text/plain", st::SolveState) = print(io, "SolveState:\n   ", st)
Base.show(io::IO, st::SolveState) = print(io, "residual=",norm(st.r))

@fastmath function increment!(st::SolveState)
    @loop st.x[I] = st.x[I]+st.ϵ[I]
    @loop st.r[I] = st.r[I]-mult(I,st.A.L,st.A.D,st.ϵ)
end

@fastmath function GS!(st::SolveState;inner=0)
    @loop st.ϵ[I] = st.r[I]*st.iD[I]
    for i ∈ 1:inner
        @loop st.ϵ[I] = st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st)
    return st
end

function solve!(st::SolveState;iterator!::Function=GS!,mxiter=1000,tol=100eps(eltype(st.r)),reslog=false,kw...)
    reslog && (hist = Vector{eltype(st.r)}(undef,mxiter))
    res,i = norm(st.r),1
    while res>tol && i<mxiter
        reslog && (hist[i] = res)
        iterator!(st;kw...)
        res,i = norm(st.r),i+1
    end
    return reslog ? resize!(hist,i-1) : i-1
end
solve!(x::AbstractVector,A::AbstractMatrix,b::AbstractVector;kw...) = solve!(SolveState(x,A,b);kw...)
solve(A::AbstractMatrix,b::AbstractVector;kw...) = (x=zero(b); (x,solve!(x,A,b;kw...)))