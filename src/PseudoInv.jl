struct PseudoInv{T,N,Lt<:AbstractArray{T},Dt<:AbstractArray{T,N}} <: FieldMatrix{T}
    L :: Lt # Lower diagonal coefficients
    D :: Dt # Diagonal coefficients
    R :: CartesianIndices{N,NTuple{N,UnitRange{Int}}}
end
function PseudoInv(A::Poisson{At,N},p::AbstractVector{T}=Float32[0.053,0.363,0.016,-0.203],
                   models=p->(D->D*(p[3]*D+p[1])+1,L->L*(p[4]*L+p[2]))) where {At,N,T}
    L,D = zeros(T,size(A.L)),zeros(T,size(A.D))
    Dm,Lm = models(p)
    for I ∈ A.R
        D[I] = Dm(A.D[I])
        for i ∈ 1:N
            L[I,i] = Lm(A.L[I,i])
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
