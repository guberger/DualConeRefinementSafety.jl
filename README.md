# DualConeRefinementSafety.jl
After cloning the reposotory, open a Julia REPL and type
```julia
cd([location of the repository folder])
import Pkg
Pkg.activate("./examples") # use the Julia environment of the project `examples`
Pkg.instantiate() # install all depedencies of the project (requires a Mosek license)
```
Run the example with
```julia
include("examples/exa_rotating.jl")
```
![GUI](https://github.com/guberger/DualConeRefinementSafety.jl/blob/main/illustration.png)