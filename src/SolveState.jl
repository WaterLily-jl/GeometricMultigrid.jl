mutable struct SolveState{T,matT<:Poisson, iDT<:FieldVec, vecT<:FieldVec}
    A::matT
    iD::iDT
    x::vecT
    r::vecT
    ϵ::vecT
    child::Union{SolveState, Nothing}
    function SolveState(A::Poisson{T},x::FieldVec,r::FieldVec,invtol=1e-8) where T
        iD = zero(x,T)
        @loop iD[I] = abs(A.D[I])>invtol ? inv(A.D[I]) : zero(T)
        new{eltype(x),typeof(A),typeof(iD),typeof(x)}(A,iD,x,r,zero(x),nothing)
    end
end
Base.show(io::IO, ::MIME"text/plain", st::SolveState{T}) where T = print(io, "SolveState{",T,"}:\n   ", st)
Base.show(io::IO, st::SolveState) = print(io, "residual=",norm(st.r),"\n   ", st.child)

@fastmath resid!(r,A,x) = (@loop r[I] = r[I]-mult(I,A.L,A.D,x);r)
function residual(A,x,b) 
    r=zero(x)
    @loop r[I] = b[I]
    resid!(r,A,x)
end
@fastmath function increment!(st;resid=true,kw...)
    @loop st.x[I] = st.x[I]+st.ϵ[I]
    resid && resid!(st.r,st.A,st.ϵ)
    st
end

epsr(st) = eps(real(eltype(st.r)))
function iterate!(st::SolveState,iterator!::Function;
                  abstol::Real = 20*epsr(st),reltol::Real = √epsr(st),
                  mxiter::Int = size(st.A,2), log::Bool = false, kw...)
    res0,i = norm(st.r),1
    log && (hist = Vector{eltype(st.r)}(undef,mxiter); hist[i] = res0)
    res = res0
    while res>max(abstol,reltol*res0) && i<mxiter
        iterator!(st;kw...)
        res,i = norm(st.r),i+1
        log && (hist[i] = res)
    end
    return log ? resize!(hist,i) : i
end

gs!(st::SolveState;kw...) = iterate!(st,GS!;kw...)
function gs(A::AbstractMatrix,b::AbstractVector;kw...)
    x = zero(b)
    return x,gs!(SolveState(A,x,copy(b));kw...)
end

@fastmath function GS!(st;inner=8,kw...)
    @loop st.ϵ[I] = st.r[I]*st.iD[I]
    for i ∈ 1:inner
        @loop st.ϵ[I] = st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st;kw...)
end

tuned!(st::SolveState{T},θ::AbstractVector{T}=T[-0.02777439,-0.06331559,-0.50800335,-0.1463638];
       inner=2,resid=true,kw...) where T = for i=1:inner
    @loop st.ϵ[I] = Mxr(I,@inbounds(st.A.D[I]),st.r,θ)
    increment!(st;resid=(i<inner||resid))
end
@fastmath @inline Mxr(I::CartesianIndex{N},D,r,θ) where {N} =
    muladd(θ[2],D,θ[3])*@inbounds(r[I])+
    muladd(θ[1],D,θ[4])*sum(@inbounds(r[I-δ(i,N)]+r[I+δ(i,N)]) for i ∈ 1:N)
