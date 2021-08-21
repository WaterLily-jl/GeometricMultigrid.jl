module geometric-multigrid

include("util.jl")

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult

end
