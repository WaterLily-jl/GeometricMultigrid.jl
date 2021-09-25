module GeometricMultigrid

include("FieldVector.jl")
export FieldVector, @loop

include("FieldMatrix.jl")
export FieldMatrix, Poisson

include("SolveState.jl")
export SolveState

include("PseudoInv.jl")
export PseudoInv

include("MultiGrid.jl")
export mg_state, mg, mg!

end
