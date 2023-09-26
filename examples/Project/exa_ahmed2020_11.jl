module Example_Ahmed2020_6

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

var, = @polyvar x[1:2]
flow = [
    -x[1] + 4 * x[2],
    -x[1] - 10 * x[2],
]
display(flow)
rad = 1.5
dom_init = @set x' * x ≤ rad^2

x1s_ = range(-2, 2, length=10)
x2s_ = range(-2, 2, length=10)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.4 / nx
dxs2 = getindex.(dxs, 2) * 0.4 / nx
p1 = plot(xlabel="x1", ylabel="x2", aspect_ratio=:equal)
p2 = plot(xlabel="c1", ylabel="c2", zlabel="c3")
scatter3d!(p2, [(0, 0, 0)], shape=:x, c=:black, ms=4, label="")
quiver!(p1, x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(-2, 2, length=20)
x2s_ = range(-2, 2, length=20)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contour!(p1, x1s_, x2s_, z, levels=[0])

nstep = 5
dt = 1.0
np = 10
rad = 0.5
vals = generate_vals(np, rad, dt, nstep, var, flow)

scatter!(p1, getindex.(vals, 1), getindex.(vals, 2), label="")

display(plot(p1, p2, layout=2))

include("../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

F = DCR.Field(var, flow)
points = [DCR.Point(var, val) for val in vals]
funcs = [1, x[1]^2, x[2]^2]
λ = 1.0
ϵ = 1e-2
hc = DCR.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))

vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))
verts_plots = [[r.a[i] for r in vc.rays] for i = 1:3]
scatter3d!(p2, verts_plots..., ms=4, label="")

ϵ = 1e-2
δ = 1e-4
λ = 1.0
success = DCR.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                            callback_func=callback_func)
display(success)
verts_plots = [[r.a[i] for r in vc.rays] for i = 1:3]
scatter3d!(p2, verts_plots..., ms=4, label="")

Fplot_vc(x1, x2) = begin
    gxs = [g(var=>[x1, x2]) for g in vc.funcs]
    maximum(r -> dot(r.a, gxs), vc.rays)
end
z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(p1, x1s_, x2s_, z, levels=[0], color=:green, lw=2)

plt = plot(p1, p2, layout=2)
savefig("examples/figures/exa_rotating.png")
display(plt)

end # module