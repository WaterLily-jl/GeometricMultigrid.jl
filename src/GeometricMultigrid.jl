module GeometricMultigrid

include("FieldVec.jl")
export FieldVec, @loop

include("Poisson.jl")
export Poisson

# include("SolveState.jl")
# export SolveState

# include("MultiLevelPoisson.jl")
# export MultiLevelPoisson,solver!,mult

end
