"""
    FieldVec

Simple vector wrapper around a padded multi-dimensional array. Used to
cast field/image data as a vector for linear algebra.
"""
struct FieldVec{T,Dt<:AbstractArray{T},Rt<:CartesianIndices} <: AbstractVector{T}
    data :: Dt # data array
    R :: Rt # CartesianIndices
    FieldVec(data::AbstractArray{T},R=inside(data)) where T = 
        new{T,typeof(data),typeof(R)}(data,R)
end
inside(dims::NTuple{N}) where {N} = CartesianIndices(ntuple(i-> 2:dims[i]-1,N))
inside(a::AbstractArray) = inside(size(a))

Base.size(x::FieldVec) = (prod(size(x.R)),)
Base.IndexStyle(::Type{<:FieldVec}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(x::FieldVec, i::Int) = x.data[x.R[i]]
Base.@propagate_inbounds Base.getindex(x::FieldVec, I::CartesianIndex) = x.data[I]
Base.@propagate_inbounds Base.setindex!(x::FieldVec, y , i::Int) = x.data[x.R[i]]=y
Base.@propagate_inbounds Base.setindex!(x::FieldVec, y , I::CartesianIndex) = x.data[I]=y
Base.similar(x::FieldVec) = FieldVec(similar(x.data))
Base.zero(x::FieldVec) = FieldVec(zero(x.data))

import LinearAlgebra: dot,norm
@fastmath function dot(x::FieldVec,y::FieldVec)
    s = zero(eltype(x))
    @inbounds @simd for I ∈ x.R
        s+= y[I]*x[I]
    end
    s
end
norm(x::FieldVec) = √dot(x,x)

macro loop(ex)
    a,I = ex.args[1].args
    return quote
        @inbounds @simd for $I ∈ $a.R
            $ex
        end
    end |> esc
end
