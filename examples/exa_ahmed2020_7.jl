module Example_Ahmed2020_7

# Automated and Sound Synthesis of Lyapunov Functions with SMT Solvers

using LinearAlgebra
using DynamicPolynomials
using Plots
using DifferentialEquations

vars, = @polyvar x[1:3]
σ = x[3]^2 + 1
f = [
    -x[1]^3 + x[1] * x[3]^2,
    -x[2] - x[1]^2 * x[2],
    -x[3] - 3 * x[3] + 3 * x[1]^2 * x[3],
] * σ
display(f)
funcs_init = [x[1]^2 + x[2]^2 + x[3]^2 - 1]


function sys_map!(du, u, ::Any, ::Any)
    du[1] = f[1](vars=>u)
    du[2] = f[2](vars=>u)
end

nstep = 5
dt = 1.0
np = 10
points = Vector{Float64}[]

using Random
Random.seed!(0)

for i = 1:np
    u0 = randn(3)
    normalize!(u0)
    prob = ODEProblem(sys_map!, u0, (0, nstep*dt))
    sol = solve(prob, saveat=dt)
    append!(points, sol.u)
end

include("../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

tmp = DCR.Template(vars, [1, x[1]^2, x[2]^2, x[3]^2])
λ = 1.0
ϵ = 1e-2
hc = DCR.hcone_from_points(tmp, f, λ, ϵ, points)
display(length(hc.halfspaces))

using CDDLib

vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.vertices))
display(length(vc.generators))

using SumOfSquares
_DSOS_ = true
if !_DSOS_
    using MosekTools
    solver() = SOSModel(optimizer_with_attributes(Mosek.Optimizer, "QUIET"=>true))
else
    using Gurobi
    const GUROBI_ENV = Gurobi.Env()
    solver() = begin
        model = SOSModel(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false)
        )
        PolyJuMP.setdefault!(model, PolyJuMP.NonNegPoly, DSOSCone)
        return model
    end
end

function callback_func(iter, i, ng, r_max)
    if i < ng
        if mod(i, 10) == 0
            print("Iter $(iter): $(i) / $(ng): $(r_max) \r")
        end
    else
        print("Iter $(iter): $(i) / $(ng): $(r_max) \n")
    end
    return nothing
end

ϵ = 1e-2
δ = 1e-4
λ = 1.0
success = DCR.narrow_vcone!(vc, funcs_init, f, λ, ϵ, δ, Inf, solver,
                            callback_func=callback_func)
display(success)

end # module