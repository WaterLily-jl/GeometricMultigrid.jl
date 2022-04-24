"""
    FieldVector

Simple vector wrapper around a padded multi-dimensional array. Used to
cast field/image data as a vector for linear algebra.
"""
struct FieldVector{T,N,Dt<:AbstractArray{T,N}} <: AbstractVector{T}
    data :: Dt # data array
    R :: CartesianIndices{N,NTuple{N,UnitRange{Int}}}
    FieldVector(data::AbstractArray{T,N},R=inside(data)) where {T,N} = 
        new{T,N,typeof(data)}(data,R)
end
inside(dims::NTuple{N}) where {N} = CartesianIndices(ntuple(i-> 2:dims[i]-1,N))
inside(a::AbstractArray) = inside(size(a))

Base.size(x::FieldVector) = (length(x.R),)
Base.IndexStyle(::Type{<:FieldVector}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(x::FieldVector, i::Int) = x.data[x.R[i]]
Base.@propagate_inbounds Base.getindex(x::FieldVector, I::CartesianIndex) = x.data[I]
Base.@propagate_inbounds Base.setindex!(x::FieldVector, y , i::Int) = x.data[x.R[i]]=y
Base.@propagate_inbounds Base.setindex!(x::FieldVector, y , I::CartesianIndex) = x.data[I]=y
Base.similar(x::FieldVector) = FieldVector(similar(x.data),x.R)
Base.zero(x::FieldVector,T=eltype(x)) = FieldVector(zeros(T,size(x.data)),x.R)

@inline tasks(x) = min(4*Threads.nthreads(),isqrt(length(x))÷2^6)
@inline Rv(R,i,step,n) = @view(R[i:min(n,i+step-1)])

import LinearAlgebra: dot,norm
@fastmath function inner(x,y,R)
    s = zero(eltype(x))
    @inbounds @simd for I ∈ R
        s+= y[I]*x[I]
    end
    s
end
function dot(x::FieldVector,y::FieldVector,ntasks=tasks(x)) 
    ntasks<3 && return inner(x,y,x.R)
    # multithreaded
    acc = Threads.Atomic{eltype(x)}(0)
    n = length(x.R); step = n÷ntasks
    Threads.@sync for i = 1:step:n
        Threads.@spawn Threads.atomic_add!(acc,inner(x,y,Rv(x.R,i,step,n)))
    end
    acc[]
end
norm(x::FieldVector) = √dot(x,x)

macro loop(ex)
    a,I = ex.args[1].args
    return quote
        n = length($a.R); step = n÷tasks($a)
        Threads.@sync for i = 1:step:n
            Threads.@spawn @inbounds @simd for $I ∈ Rv($a.R,i,step,n)
                $ex
            end
        end
    end |> esc
end
