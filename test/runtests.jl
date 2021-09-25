using LinearAlgebra, GeometricMultigrid, Test, Statistics

function setup_2D(n)
    L = zeros(n+2,n+2,2); L[3:n+1,2:n+1,1] .= 1; L[2:n+1,3:n+1,2] .= 1; 
    A = Poisson(L)
    x = FieldVec(Float64[i-1 for i ∈ 1:n+2, j ∈ 1:n+2])
    A,x
end
error(x) = norm(x.-Statistics.mean(x))

@testset "FieldVec.jl" begin
    A,x = setup_2D(3)
    @test x[1] == 1
    @test x[CartesianIndex(1,1)] == 0
    @test norm(x) == √42
    b = zero(x)
    @test typeof(b) <: FieldVec
    @test b == zeros(9)
    @loop b[I] = 1/x[I]
    @test !any(isnan.(b.data))
    @test dot(b,x) == 9
end

@testset "FieldMatrix.jl" begin
    A,x = setup_2D(3)
    @test diag(A) == -Float64[2,3,2,3,4,3,2,3,2]
    @test det(A) < 2eps(eltype(A))
    b = A*x
    @test b == repeat(Float64[1,0,-1],3)
    @test dot(b,A,x) == dot(b,b)
    @test pinv(Matrix(A))*(A*x) ≈ -b

    using IterativeSolvers
    A,x = setup_2D(37)
    @test error(x.-cg(A,A*x))<1e-8
end

@testset "SolveState.jl" begin
    A,x = setup_2D(32)
    @test GeometricMultigrid.residual(A,x,A*x) == zero(x)
    y,_ = GeometricMultigrid.gs(A,A*x,reltol=1e-12)
    @test error(x.-y)<1e-8
end

@testset "MultiGrid.jl" begin
    I = CartesianIndex(4,3,2)
    @test all(GeometricMultigrid.down(J)==I for J ∈ GeometricMultigrid.up(I))
    A,x = setup_2D(4)
    st = mg_state(A,x,x)
    @test diag(st.child.A) == -2ones(4)
    @test isnothing(st.child.child)
    A,x = setup_2D(32)
    y,_ = mg(A,A*x,reltol=1e-12)
    @test error(x.-y)<1e-8    
end
