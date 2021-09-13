module GeometricMultigrid

include("FieldVec.jl")
export FieldVec, @loop

include("Poisson.jl")
export Poisson

include("SolveState.jl")
export SolveState, gs, gs!

include("PseudoInv.jl")

include("MultiGrid.jl")
export mg_state, mg, mg!

end
