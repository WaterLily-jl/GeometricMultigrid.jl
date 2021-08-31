"""
    FieldVec

Simple vector wrapper around a padded multi-dimensional array. Used to
cast field/image data as a vector for linear algebra.
"""
struct FieldVec{T,N,Dt<:AbstractArray{T,N}} <: AbstractVector{T}
    data :: Dt # data array
    R :: CartesianIndices{N,NTuple{N,UnitRange{Int}}}
    FieldVec(data::AbstractArray{T,N},R=inside(data)) where {T,N} = 
        new{T,N,typeof(data)}(data,R)
end
inside(dims::NTuple{N}) where {N} = CartesianIndices(ntuple(i-> 2:dims[i]-1,N))
inside(a::AbstractArray) = inside(size(a))

Base.size(x::FieldVec) = (length(x.R),)
Base.IndexStyle(::Type{<:FieldVec}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(x::FieldVec, i::Int) = x.data[x.R[i]]
Base.@propagate_inbounds Base.getindex(x::FieldVec, I::CartesianIndex) = x.data[I]
Base.@propagate_inbounds Base.setindex!(x::FieldVec, y , i::Int) = x.data[x.R[i]]=y
Base.@propagate_inbounds Base.setindex!(x::FieldVec, y , I::CartesianIndex) = x.data[I]=y
Base.similar(x::FieldVec) = FieldVec(similar(x.data),x.R)
Base.zero(x::FieldVec,T=eltype(x)) = FieldVec(zeros(T,size(x.data)),x.R)

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
