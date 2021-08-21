module GeometricMultigrid

include("util.jl")
export L₂

include("Poisson.jl")
export AbstractPoisson,Poisson,solver!,mult

include("MultiLevelPoisson.jl")
export MultiLevelPoisson,solver!,mult

end
