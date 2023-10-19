module ExampleRotating

using LinearAlgebra
using DynamicPolynomials
using Plots
using CDDLib
# using LRSLib
# QHull does not support equalities

lib() = CDDLib.Library(:exact)
# lib() = LRSLib.Library(:exact)

include("../../src/DualConeRefinementSafety.jl")
const DCR = DualConeRefinementSafety

var, = @polyvar x[1:2]
flow = [
    x[2] * (1 // 2) + x[1] - x[1] * (4 * x[1]^2 + x[2]^2),
    -2 * x[1] + x[2] - x[2] * (4 * x[1]^2 + x[2]^2),
]
display(flow)
rad = 1 // 4
gens_init = [-1, x' * x - rad^2]
ϵ = 1e-2
monos = monomials(var, 0:1)
mono_pairs = filter(e -> e[1] != e[2],
                    vec(collect(Iterators.product(monos, monos))))
gens_sos = filter(p -> !iszero(p), unique(vcat(
    vec(map(e -> e^2, monos)),
    vec(map(e -> (e[1] + e[2])^2, mono_pairs)),
    vec(map(e -> (e[1] - e[2])^2, mono_pairs)),
)))
display(gens_sos)
maxdeg = 1
fc = DCR.init_cone(var, maxdeg, gens_init, gens_sos, lib)
display(fc.generators)
copy!(fc.generators, [+x[1] - rad, +x[2] - rad, -x[1] - rad, -x[2] - rad])
display(fc.generators)

x1s_ = range(-1, 1, length=10)
x2s_ = range(-2, 2, length=10)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.4 / nx
dxs2 = getindex.(dxs, 2) * 0.4 / nx
plt = plot(xlabel="x1", ylabel="x2", aspect_ratio=:equal)
quiver!(x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(-1, 1, length=100)
x2s_ = range(-2, 2, length=100)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in fc.generators)
z = @. Fplot_init(x1s_', x2s_)
contour!(x1s_, x2s_, z, levels=[0])

display(plt)

F = DCR.Field(var, flow)
λ = 1

fc = DCR.refine_cone(fc, F, λ, gens_sos, lib)
display(fc)

# does not work after first iter ...

z = @. Fplot_init(x1s_', x2s_)
display(minimum(z))
contour!(x1s_, x2s_, z, levels=[0], color=:green, lw=2)

savefig("examples/figures/exa_rotating.png")
display(plt)

end # module