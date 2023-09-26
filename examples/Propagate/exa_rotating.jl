module ExampleRotating

using LinearAlgebra
using Random
Random.seed!(0)
using DynamicPolynomials
using Plots
using DifferentialEquations
using CDDLib
using SumOfSquares
using MosekTools

include("../utils.jl")

include("../../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

var, = @polyvar x[1:2]
flow = [
    x[2] / 2 + x[1] - x[1] * (4 * x[1]^2 + x[2]^2),
    -2 * x[1] + x[2] - x[2] * (4 * x[1]^2 + x[2]^2),
]
display(flow)
rad = 0.5
gens_init = [-1, x' * x - rad^2]
ϵ = 1e-2
funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
monos = monomials(var, 0:1)
mono_pairs = filter(e -> e[1] != e[2],
                    vec(collect(Iterators.product(monos, monos))))
gens_sos = filter(p -> !iszero(p), unique(vcat(
    vec(map(e -> e^2, monos)),
    vec(map(e -> (e[1] + e[2])^2, mono_pairs)),
    vec(map(e -> (e[1] - e[2])^2, mono_pairs)),
)))
display(gens_sos)
vc = DCR.closure_cone(funcs, gens_init, ϵ, gens_sos, CDDLib.Library)
display(vc)

x1s_ = range(-1, 1, length=10)
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
quiver!(p1, x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(-1, 1, length=100)
x2s_ = range(-2, 2, length=100)
Fplot_vc(x1, x2) = begin
    gxs = [g(var=>[x1, x2]) for g in vc.funcs]
    maximum(r -> dot(r.a, gxs), vc.rays)
end
z = @. Fplot_vc(x1s_', x2s_)
contour!(p1, x1s_, x2s_, z, levels=[0])

@assert all(r -> r.a[1] < 1e-5, vc.rays)
verts_plot = [r.a[2:4] / r.a[1] for r in vc.rays]
verts_plots = [getindex.(verts_plot, i) for i = 1:3]
scatter3d!(p2, verts_plots..., ms=4, label="")

display(plot(p1, p2, layout=2))

F = DCR.Field(var, flow)
λ = 1.0

vc = DCR.refine_cone(vc, F, λ, ϵ, gens_sos, CDDLib.Library)
display(vc)

@assert all(r -> r.a[1] < 1e-5, vc.rays)
verts_plot = [r.a[2:4] / r.a[1] for r in vc.rays]
verts_plots = [getindex.(verts_plot, i) for i = 1:3]
scatter3d!(p2, verts_plots..., ms=4, label="")

z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(p1, x1s_, x2s_, z, levels=[0], color=:green, lw=2)

plt = plot(p1, p2, layout=2)
savefig("examples/figures/exa_rotating.png")
display(plt)

end # module