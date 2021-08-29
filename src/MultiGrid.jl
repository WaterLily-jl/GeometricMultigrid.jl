mg!(st::SolveState;kw...) = iterate!(st,Vcycle!;kw...)
function mg(A::AbstractMatrix,b::AbstractVector;kw...) 
    x=zero(b)
    return x,mg!(mg_state(A,x,b);kw...)
end
mg_state(A,x,b) = fill_children!(SolveState(A,x,residual(A,x,b)))

function Vcycle!(st::SolveState;smooth!::Function=GS!,kw...)
    fine,coarse = st,st.child
    # base case
    isnothing(coarse) && return smooth!(fine;kw...)
    # set up & solve coarse recursively
    GS!(fine;inner=0)
    restrict!(coarse.r,fine.r)
    fill!(coarse.x,0.)
    Vcycle!(coarse)
    # correct & solve fine
    prolongate!(fine.ϵ,coarse.x)
    increment!(fine)
    smooth!(fine;kw...)
end

restrict!(a,b) = @loop a[I] = sum(@inbounds(b[J]) for J ∈ up(I))
prolongate!(a,b) = @loop a[I] = b[down(I)]

@inline up(I::CartesianIndex{N},a=0) where N = (2I-2oneunit(I)):(2I-oneunit(I)-δ(a,N))
@inline down(I::CartesianIndex) = CartesianIndex((I+2oneunit(I)).I .÷2)

function fill_children!(st::SolveState)
    divisible(st) && isnothing(st.child) && fill_children!(create_child(st.A.L))
    return st
end

function create_child(b::AbstractArray{T,N}) where {T,N}
    dims = 1 .+ Base.front(size(b)) .÷ 2
    a = zeros(T,dims...,N)
    x = FieldVec(zeros(T,dims))
    for i ∈ 1:N-1, I ∈ x.R
        a[I,i] = 0.5sum(@inbounds(b[J,i]) for J ∈ up(I,i))
    end
    SolveState(Poisson(a),x,zero(x))
end

@inline divisible(N::Int) = mod(N,2)==0 && N>2
@inline divisible(st::SolveState) = all(size(st.x.R) .|> divisible)
