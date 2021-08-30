using LinearAlgebra, GeometricMultigrid, BenchmarkTools, Plots

function setup_2D(n=128)
    L = zeros(n+2,n+2,2); L[3:n+1,2:n+1,1] .= 1; L[2:n+1,3:n+1,2] .= 1; 
    A = Poisson(L)
    x = FieldVec(Float64[i-1 for i ∈ 1:n+2, j ∈ 1:n+2])
    A,x
end

begin
    suite = BenchmarkGroup()
    nlist = 2 .^ (3:10)
    for n ∈ nlist
        A,x = setup_2D(n)
        # suite[n] = @benchmarkable mul!($b,$A,$x)
        suite[n] = @benchmarkable mg!(st;inner=2) setup=(st=mg_state($A,zero($x),$A*$x))
    end
end

results = run(suite)

begin
    nlogn(n) = n*log(n)
    N = nlist.^2
    time = [median(results[n]).time for n ∈ nlist]
    plot(N,time[end]/N[end] .* N, color=:grey, label="N", legend=:bottomright)
    plot!(N,time[end]/nlogn(N[end]) .* nlogn.(N) , color=:black, label="NlogN")
    scatter!(N,time, xaxis=("length x",:log10), yaxis=("time (ns)",:log10), label="mg")
end

savefig("MGscaling.png")
