using geometric-multigrid
using Test

geometric-multigrid.L₂ = L₂

@testset "util.jl" begin
    @test L₂(2ones(4,4)) == 16
end

function Poisson_test_2D(f,n)
    c = zeros(n+2,n+2,2); c[3:n+1,:,1] = 1; c[:,3:n+1,2] = 1
    p = f(c)
    soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2]
    b = mult(p,soln)
    x = zeros(n+2,n+2)
    solver!(x,p,b)
    x .-= (x[2,2]-soln[2,2])
    return L₂(x.-soln)/L₂(soln)
end
function Poisson_test_3D(f,n)
    c = zeros(n+2,n+2,2); c[3:n+1,:,:,1] = 1; c[:,3:n+1,:,2] = 1; c[:,:,3:n+1,3] = 1
    p = f(c)
    soln = Float64[ i for i ∈ 1:n+2, j ∈ 1:n+2, k ∈ 1:n+2]
    b = mult(p,soln)
    x = zeros(n+2,n+2,n+2)
    solver!(x,p,b,tol=1e-5)
    x .-= (x[2,2,2]-soln[2,2,2])
    return L₂(x.-soln)/L₂(soln)
end

@testset "Poisson.jl" begin
    @test Poisson_test_2D(Poisson,2^6) < 1e-5
    @test Poisson_test_3D(Poisson,2^4) < 1e-5
end
@testset "MultiLevelPoisson.jl" begin
    I = CartesianIndex(4,3,2)
    @test all(geometric-multigrid.down(J)==I for J ∈ geometric-multigrid.up(I))
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_test_2D(MultiLevelPoisson,67)
    @test_throws AssertionError("MultiLevelPoisson requires size=a2ⁿ, where a<31, n>2") Poisson_test_3D(MultiLevelPoisson,3^4)
    @test Poisson_test_2D(MultiLevelPoisson,2^6) < 1e-5
    @test Poisson_test_3D(MultiLevelPoisson,2^4) < 1e-5
end
