mutable struct SolveState{matT<:Poisson, vecT<:FieldVec}
    A::matT
    iD::vecT
    x::vecT
    r::vecT
    ϵ::vecT
    child::Union{SolveState, Nothing}
    function SolveState(A::Poisson,x::FieldVec,r::FieldVec,invtol=1e-8)
        iD = zero(x)
        @loop iD[I] = abs(A.D[I])>invtol ? inv(A.D[I]) : zero(eltype(A))
        new{typeof(A),typeof(x)}(A,iD,x,r,zero(x),nothing)
    end
end
Base.show(io::IO, ::MIME"text/plain", st::SolveState) = print(io, "SolveState:\n   ", st)
Base.show(io::IO, st::SolveState) = print(io, "residual=",norm(st.r))

@fastmath resid!(r,A,x) = (@loop r[I] = r[I]-mult(I,A.L,A.D,x);r)
residual(A,x,b) = resid!(copy(b),A,x)
@fastmath function increment!(st) 
    @loop st.x[I] = st.x[I]+st.ϵ[I]
    resid!(st.r,st.A,st.ϵ)
    st
end

function iterate!(st::SolveState,iterator!,mxiter=1000,tol=100eps(eltype(st.r)),reslog=false,kw...)
    res,i = norm(st.r),1
    reslog && (hist = Vector{eltype(st.r)}(undef,mxiter); hist[i] = res)
    while res>tol && i<mxiter
        iterator!(st;kw...)
        res,i = norm(st.r),i+1
        reslog && (hist[i] = res)
    end
    return reslog ? resize!(hist,i) : i
end

gs!(st::SolveState;kw...) = iterate!(st,GS!;kw...)
function gs(A::AbstractMatrix,b::AbstractVector;kw...)
    x = zero(b)
    return x,gs!(SolveState(A,x,copy(b));kw...)
end

@fastmath function GS!(st;inner=floor(Int,√size(st.A,1)))
    @loop st.ϵ[I] = st.r[I]*st.iD[I]
    for i ∈ 1:inner
        @loop st.ϵ[I] = st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st)
end
