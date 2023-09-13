module Example_Ahmed2020_9

# Automated and Sound Synthesis of Lyapunov Functions with SMT Solvers

using LinearAlgebra
using Random
Random.seed!(0)
using DynamicPolynomials
using Plots
using DifferentialEquations
using CDDLib
using SumOfSquares
using MosekTools

include("utils.jl")

vars, = @polyvar x[1:4]
f = [
    -x[1] + x[2]^3 - x[3] * x[4],
    -x[1] - x[2]^3,
    x[1] * x[4] - x[3],
    x[1] * x[3] - x[4]^2,
] * (x[3]^2 + 1)
display(f)
rad = 2
funcs_init = [x' * x - rad^2]

nstep = 5
dt = 1.0
np = 10
points = generate_points(np, rad, dt, nstep, vars, f)

include("../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

tmp = DCR.Template(vars, [1, x[1]^2, x[2]^2, x[3]^2, x[4]^2])
λ = 1.0
ϵ = 1e-2
hc = DCR.hcone_from_points(tmp, f, λ, ϵ, points)
display(length(hc.halfspaces))

vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.vertices))

ϵ = 1e-2
δ = 1e-4
λ = 1.0
success = DCR.narrow_vcone!(vc, funcs_init, f, λ, ϵ, δ, Inf, solver,
                            callback_func=callback_func)
display(success)

end # module