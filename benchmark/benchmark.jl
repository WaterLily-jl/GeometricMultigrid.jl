using LinearAlgebra, GeometricMultigrid, BenchmarkTools, Plots

function setup_2D(n=128,T::Type=Float64)
    L = zeros(T,n+2,n+2,2); L[3:n+1,2:n+1,1] .= 1; L[2:n+1,3:n+1,2] .= 1; 
    x = T[i-1 for i ∈ 1:n+2, j ∈ 1:n+2]
    Poisson(L),FieldVector(x)
end

begin
    suite = BenchmarkGroup()
    nlist = 2 .^ (3:10)
    for n ∈ nlist
        A,x = setup_2D(n,Float32)
        suite["mg-gs",n] = @benchmarkable mg!(st) setup=(st=mg_state($A,zero($x),$A*$x))
#         suite["pseudo",n] = @benchmarkable mg!(st) setup=(st=mg_state($A,zero($x),$A*$x,pseudo=true))
    end
end

results = run(suite)

begin
    nlogn(n) = n*log(n)
    N = nlist.^2
    time = [minimum(results["mg-gs",n]).time for n ∈ nlist]
    ptime = [minimum(results["pseudo",n]).time for n ∈ nlist]
    plot(N,time[end]/N[end] .* N, color=:grey, label="N", legend=:bottomright)
    plot!(N,time[end]/nlogn(N[end]) .* nlogn.(N) , color=:black, label="NlogN")
    scatter!(N,time, xaxis=("length x",:log10), yaxis=("time (ns)",:log10), label="mg!")
    # scatter!(N,ptime, xaxis=("length x",:log10), yaxis=("time (ns)",:log10), label="pseudo=true")
end

savefig("MGscaling.png")

begin
    A,x = setup_2D()
    b = zero(x)
    st = SolveState(A,zero(x),A*x)
#     st.P = PseudoInv(A)
end
@btime norm($x)  # 5.3 μs
@btime dot($b,$x) # 7.8 μs
@btime dot($b,$A,$x) # 29.5 μs   
@btime mul!($b,$A,$x) # 30.6 μs
@btime GeometricMultigrid.increment!($st)  # 45.4 μs
@btime GeometricMultigrid.GS!($st;inner=1) # 183 μs
@btime GeometricMultigrid.GS!($st;inner=2) # 306 μs
# @btime GeometricMultigrid.pseudo!($st)     # 150 μs
