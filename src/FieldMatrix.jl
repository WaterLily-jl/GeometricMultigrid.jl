struct FieldMatrix{T,N,Lt<:AbstractArray{T},Dt<:AbstractArray{T,N}} <: AbstractMatrix{T}
    L :: Lt # Lower diagonal coefficients
    D :: Dt # Diagonal coefficients
    R :: CartesianIndices{N,NTuple{N,UnitRange{Int}}}
end
Base.size(p::FieldMatrix) = (s = length(p.R); (s,s))
Base.IndexStyle(::Type{<:FieldMatrix}) = IndexCartesian()
function Base.getindex(p::FieldMatrix{T}, i::Int, j::Int) where T
    if i == j
        return p.D[p.R[i]]
    else
        N = length(size(p.R))
        for d ∈ 1:N
            p.R[i]==p.R[j]+δ(d,N) && return p.L[p.R[i],d]
            p.R[i]==p.R[j]-δ(d,N) && return p.L[p.R[j],d]
        end
        return zero(T)
    end
end

@inline δ(i,N::Int) = CartesianIndex(ntuple(j -> j==i ? 1 : 0, N))
@fastmath @inline multL(I::CartesianIndex{N},L,x) where {N} =
    sum(@inbounds(x[I-δ(i,N)]*L[I,i]) for i ∈ 1:N)
@fastmath @inline multU(I::CartesianIndex{N},L,x) where {N} =
    sum(@inbounds(x[I+δ(i,N)]*L[I+δ(i,N),i]) for i ∈ 1:N)
@fastmath @inline mult(I::CartesianIndex{N},L,D,x) where {N} =
    @inbounds(x[I]*D[I])+multL(I,L,x)+multU(I,L,x)

import LinearAlgebra: mul!,dot,diag
mul!(b::FieldVec,p::FieldMatrix,x::FieldVec) = (@loop b[I]=mult(I,p.L,p.D,x); b)
@fastmath function dot(b::FieldVec,p::FieldMatrix,x::FieldVec)
    s = zero(eltype(x))
    @inbounds @simd for I ∈ b.R
        s+= b[I]*mult(I,p.L,p.D,x)
    end
    s
end
diag(p::FieldMatrix) = FieldVec(p.D)

import Base: *
*(p::FieldMatrix,x::FieldVec) = mul!(zero(x),p,x)

"""
    Poisson(L)

Construct a symmetric `FieldMatrix` from lower diagonal coefficients `L` with zero-sum rows.
This is a requirement for conservative Poisson equations: ∫ ∇⋅β∇ϕ dv = ∮ β ∂ϕ/∂n da.
"""
function Poisson(L::AbstractArray{T}) where T
    D = zeros(T,Base.front(size(L)))
    R = inside(D)
    for I ∈ R; D[I] = calcdiag(I,L); end
    FieldMatrix(L,D,R)
end
@fastmath @inline calcdiag(I::CartesianIndex{N},L) where {N} =
    -sum(@inbounds(L[I,i]+L[I+δ(i,N),i]) for i ∈ 1:N)
