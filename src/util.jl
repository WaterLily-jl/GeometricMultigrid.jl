"""
    δ(i,N::Int)
    δ(i,I::CartesianIndex{N}) where {N}

Return a CartesianIndex of dimension `N` which is one at index `i` and zero elsewhere.
"""
@inline δ(i,N::Int) = CartesianIndex(ntuple(j -> j==i ? 1 : 0, N))
@inline δ(i,I::CartesianIndex{N}) where {N} = δ(i,N)

"""
    inside(dims)
    inside(a) = inside(size(a))

Return CartesianIndices range excluding the ghost-cells on the boundaries of
a _scalar_ array `a` with `dims=size(a)`.
"""
@inline inside(dims::NTuple{N}) where {N} = CartesianIndices(ntuple(i-> 2:dims[i]-1,N))
@inline inside(a::AbstractArray) = inside(size(a))

"""
    L₂(a)

L₂ norm of array `a` excluding ghosts.
"""
L₂(a) = sum(@inbounds(abs2(a[I])) for I ∈ inside(a))

size_u(a) = (s = size(a); return s[1:end-1],s[end])

"""
    @inside <expr>

Simple macro to automate efficient loops over cells excluding ghosts. For example

    @inside p[I] = sum(I.I)

becomes

    @inbounds @simd for I ∈ inside(p)
        p[I] = sum(I.I)
    end
"""
macro inside(ex)
    a,I = Meta.parse.(split(string(ex.args[1]),union("[",",","]")))
    return quote 
        GeometricMultigrid.@loop $ex over $I ∈ inside($a)
    end |> esc
end
macro loop(args...)
    ex,_,itr = args
    op,I,R = itr.args
    @assert op ∈ (:(∈),:(in))
    return quote
        @inbounds @simd for $I ∈ $R
            $ex
        end
    end |> esc
end