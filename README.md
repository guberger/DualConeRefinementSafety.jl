# PositiveConeRefinementSafety.jl
After cloning the reposotory, open a Julia REPL and type
```julia
cd([location of the repository folder])
import Pkg
Pkg.activate(".") # use the Julia environment of the project
Pkg.instantiate() # install all depedencies of the project (requires a Mosek license)
```
Run the example with
```julia
include("main.jl")
```