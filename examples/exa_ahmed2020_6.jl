module Example_Ahmed2020_6

# Automated and Sound Synthesis of Lyapunov Functions with SMT Solvers

using LinearAlgebra
using DynamicPolynomials
using Plots
using DifferentialEquations

vars, = @polyvar x[1:2]
f = [
    -x[1]^3 + x[2],
    -x[1] - x[2],
]
display(f)
funcs_init = [x[1]^2 + x[2]^2 - 1]

x1s_ = range(-2, 2, length=10)
x2s_ = range(-2, 2, length=10)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[fi(vars=>x) for fi in f] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.4 / nx
dxs2 = getindex.(dxs, 2) * 0.4 / nx
plt = plot(xlabel="x1", ylabel="x2", aspect_ratio=:equal)
quiver!(x1s, x2s, quiver=(dxs1, dxs2))

function sys_map!(du, u, ::Any, ::Any)
    du[1] = f[1](vars=>u)
    du[2] = f[2](vars=>u)
end

x1s_ = range(-2, 2, length=20)
x2s_ = range(-2, 2, length=20)
Fplot_init(x1, x2) = maximum(g(vars=>[x1, x2]) for g in funcs_init)
z = @. Fplot_init(x1s_', x2s_)
contour!(x1s_, x2s_, z, levels=[0])

nstep = 5
dt = 1.0
np = 10
rad = 0.5
points = Vector{Float64}[]

for α in range(0, 2π, np + 1)[1:np]
    u0 = [rad*cos(α), rad*sin(α)]
    prob = ODEProblem(sys_map!, u0, (0, nstep*dt))
    sol = solve(prob, saveat=dt)
    append!(points, sol.u)
    plot!(sol[1, :], sol[2, :], label="")
end

scatter!(getindex.(points, 1), getindex.(points, 2), label="")

display(plt)

include("../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

tmp = DCR.Template(vars, [1, x[1]^2, x[2]^2])
λ = 1.0
ϵ = 1e-2
hc = DCR.hcone_from_points(tmp, f, λ, ϵ, points)
display(length(hc.halfspaces))

using CDDLib

vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.vertices))
display(length(vc.generators))

using SumOfSquares
_DSOS_ = false
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

Fplot_vc(x1, x2) = begin
    gxs = [g(vars=>[x1, x2]) for g in vc.tmp.funcs]
    maximum(c -> dot(c, gxs), vc.generators)
end
z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(x1s_, x2s_, z, levels=[0], color=:green, lw=2)

display(plt)

end # module