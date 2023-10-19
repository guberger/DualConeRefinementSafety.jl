module Example

# Automated and Sound Synthesis of Lyapunov Functions with SMT Solvers
# Example 6, modified

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

var, = @polyvar x[1:2]
flow = [
    -x[1]^3 + 5 * x[2],
    -x[1] - x[2],
]
display(flow)
rad = 0.5
dom_init = @set x' * x ≤ rad^2

x1s_ = range(-2, 2, length=15)
x2s_ = range(-2, 2, length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.4 / nx
dxs2 = getindex.(dxs, 2) * 0.4 / nx
plt = plot(xlabel="x1", ylabel="x2", aspect_ratio=:equal)
quiver!(plt, x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(-2, 2, length=500)
x2s_ = range(-2, 2, length=500)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contour!(plt, x1s_, x2s_, z, levels=[0])

nstep = 5
dt = 0.25
np = 20
rad = 0.5
vals = generate_vals_on_ball(np, rad, dt, nstep, var, flow)

scatter!(plt, getindex.(vals, 1), getindex.(vals, 2), label="")

display(plt)

include("../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

F = DCR.Field(var, flow)
points = [DCR.Point(var, val) for val in vals]
funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
λ = 1.0
ϵ = 1e-1
hc = DCR.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))

vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

δ = 1e-4
success = DCR.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                            callback_func=callback_func)
display(success)
display(vc.funcs)
display(vc.rays)
DCR.simplify_vcone!(vc, 1e-5, solver)
display(vc.rays)

Fplot_vc(x1, x2) = begin
    gxs = [g(var=>[x1, x2]) for g in vc.funcs]
    maximum(r -> dot(r.a, gxs), vc.rays)
end
z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(plt, x1s_, x2s_, z, levels=[0], color=:green, lw=2)

savefig("examples/figures/exa_rotating.png")
display(plt)

model = solver()
r = @variable(model)
dom = DCR.sos_domain_from_vcone(vc)
@constraint(model, x' * x ≤ r, domain=dom)
@objective(model, Min, r)
optimize!(model)
@assert primal_status(model) == FEASIBLE_POINT
display(sqrt(value(r)))

end # module