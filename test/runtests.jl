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

@testset "Poisson.jl" begin
    A,x = setup_2D(3)
    @test FieldVec(A.D) == -Float64[2,3,2,3,4,3,2,3,2]
    @test det(A) == 0
    b = A*x
    @test b == repeat(Float64[1,0,-1],3)
    @test dot(b,A,x) == dot(b,b)
    @test pinv(Matrix(A))*(A*x) ≈ -b

    using IterativeSolvers
    A,x = setup_2D(37)
    @test error(x.-cg(A,A*x))<1e-8
end

# function Poisson_test_2D(f,n)
#     c = zeros(n+2,n+2,2); c[3:n+1,:,1] .= 1; c[:,3:n+1,2] .= 1
#     p = f(c)
#     soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2]
#     b = mult(p,soln)
#     x = zeros(n+2,n+2)
#     solver!(x,p,b)
#     x .-= (x[2,2]-soln[2,2])
#     return L₂(x.-soln)/L₂(soln)
# end
# function Poisson_test_3D(f,n)
#     c = zeros(n+2,n+2,n+2,3); c[3:n+1,:,:,1] .= 1; c[:,3:n+1,:,2] .= 1; c[:,:,3:n+1,3] .= 1
#     p = f(c)
#     soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2, k ∈ 1:n+2]
#     b = mult(p,soln)
#     x = zeros(n+2,n+2,n+2)
#     solver!(x,p,b,tol=1e-5)
#     x .-= (x[2,2,2]-soln[2,2,2])
#     return L₂(x.-soln)/L₂(soln)
# end

# @testset "MultiLevelPoisson.jl" begin
#     I = CartesianIndex(4,3,2)
#     @test all(GeometricMultigrid.down(J)==I for J ∈ GeometricMultigrid.up(I))
#     @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_test_2D(MultiLevelPoisson,67)
#     @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_test_3D(MultiLevelPoisson,3^4)
#     @test Poisson_test_2D(MultiLevelPoisson,2^6) < 1e-5
#     @test Poisson_test_3D(MultiLevelPoisson,2^4) < 1e-5
# end
