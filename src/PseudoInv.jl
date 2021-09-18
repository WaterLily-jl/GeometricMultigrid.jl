struct PseudoInv{T,N,Lt<:AbstractArray{T},Dt<:AbstractArray{T,N}} <: FieldMatrix{T}
    L :: Lt # Lower diagonal coefficients
    D :: Dt # Diagonal coefficients
    R :: CartesianIndices{N,NTuple{N,UnitRange{Int}}}
end
function PseudoInv(A::FieldMatrix; scale=maximum(A.L),
                   p::AbstractVector{T}=Float32[-0.1449,-0.0162,0.00734,0.3635,-0.2018],
                   models=p->(D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]+L*p[5]))) where T
    L,D,N = zeros(T,size(A.L)),zeros(T,size(A.D)),size(A.L)[end]
    Dm,Lm = models(p)
    for I ∈ A.R
        D[I] = Dm(A.D[I]/scale)
        for i ∈ 1:N
            L[I,i] = Lm(A.L[I,i]/scale)
        end
    end
    PseudoInv(L,D,A.R)
end

pseudo!(st;kw...) = pseudo!(st,st.P;kw...)
pseudo!(st,nothing;kw...) = GS!(st;kw...)
@fastmath pseudo!(st,P::PseudoInv;inner=2,resid=true,kw...) = for i=1:inner
    @loop st.ϵ[I] = st.iD[I]*mult(I,P.L,P.D,st.r)
    increment!(st;resid=(i<inner||resid))
end
