"""
    Poisson

Matrix defined by the conservative variable coefficient Poisson equation

    Ax = [L+D+L']x = b

where `A` is symmetric, block-tridiagonal and extremely sparse. `L` are the lower
diagonal components (stored as a multi-dimensional array) and the main diagonal is
`D[I]=-∑ᵢ(L[I,i]+L'[I,i])`.
"""
struct Poisson{T,Lt<:AbstractArray{T},Dt<:AbstractArray{T},Rt<:CartesianIndices} <: AbstractMatrix{T}
    L :: Lt # Lower diagonal coefficients
    D :: Dt # Diagonal coefficients
    R :: Rt # CartesianIndices
    function Poisson(L::AbstractArray{T}) where T
        D = zeros(T,Base.front(size(L)))
        R = inside(D)
        for I ∈ R; D[I] = diag(I,L); end
        new{T,typeof(L),typeof(D),typeof(R)}(L,D,R)
    end
end
Base.size(p::Poisson) = (s = prod(size(p.R)); (s,s))
Base.IndexStyle(::Type{<:Poisson}) = IndexCartesian()
function Base.getindex(p::Poisson{T}, i::Int, j::Int) where T
    if i == j
        return p.D[p.R[i]]
    else
        N = length(size(p))
        for d ∈ 1:N
            p.R[i]==p.R[j]+δ(d,N) && return p.L[p.R[i],d]
            p.R[i]==p.R[j]-δ(d,N) && return p.L[p.R[j],d]
        end
        return zero(T)
    end
end

@inline δ(i,N::Int) = CartesianIndex(ntuple(j -> j==i ? 1 : 0, N))

@fastmath @inline diag(I::CartesianIndex{N},L) where {N} =
    -sum(@inbounds(L[I,i]+L[I+δ(i,N),i]) for i ∈ 1:N)
@fastmath @inline multL(I::CartesianIndex{N},L,x) where {N} =
    sum(@inbounds(x[I-δ(i,N)]*L[I,i]) for i ∈ 1:N)
@fastmath @inline multU(I::CartesianIndex{N},L,x) where {N} =
    sum(@inbounds(x[I+δ(i,N)]*L[I+δ(i,N),i]) for i ∈ 1:N)
@fastmath @inline mult(I::CartesianIndex{N},L,D,x) where {N} =
    @inbounds(x[I]*D[I])+multL(I,L,x)+multU(I,L,x)

import LinearAlgebra: mul!,dot
mul!(b::FieldVec,p::Poisson,x::FieldVec) = (@loop b[I]=mult(I,p.L,p.D,x); b)
@fastmath function dot(b::FieldVec,p::Poisson,x::FieldVec)
    s = zero(eltype(b))
    @inbounds @simd for I ∈ b.R
        s+= b[I]*mult(I,p.L,p.D,x)
    end
    s
end
import Base: *
*(p::Poisson,x::FieldVec) = mul!(similar(x),p,x)
