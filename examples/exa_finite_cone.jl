module Example

using LinearAlgebra
using DynamicPolynomials
using Plots
using CDDLib

lib() = CDDLib.Library(:exact)

include("../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.FiniteCone

var, = @polyvar x[1:2]
flow = [
    -(1 // 2) * x[1] + x[2],
    -(1 // 2) * x[2],
] # Very simple linear dynamics, using Rational Number for exact arithmetics
display(flow)
rad = 1 // 2
gens_init = [-1, x' * x - rad^2]

monos = monomials(var, 0:1)
mono_pairs = filter(e -> e[1] != e[2],
                    vec(collect(Iterators.product(monos, monos))))
gens_pos = filter(p -> !iszero(p), unique(vcat(
    vec(map(e -> e^2, monos)),
    vec(map(e -> (e[1] + e[2])^2, mono_pairs)),
    vec(map(e -> (e[1] - e[2])^2, mono_pairs)),
)))
display(gens_pos) # Generators of DSOS_1 with two variables

# Cone generated from the init set and the cone K
maxdeg = 1
fc = MP.init_cone(maxdeg, gens_init, gens_pos, lib)
display(fc.gens)
# Override fc.gens because otherwise it is already too large for the refinement
copy!(fc.gens, [+x[1] - rad, +x[2] - rad, -x[1] - rad, -x[2] - rad])
display(fc.gens)

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
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in fc.gens)
z = @. Fplot_init(x1s_', x2s_)
contour!(plt, x1s_, x2s_, z, levels=[0])

display(plt)

F = MP.Field(var, flow)
λ = 1

fc1 = MP.refine_cone(fc, F, λ, gens_pos, lib)
display(fc1.gens)
# Takes some time (2 min)
fc2 = MP.refine_cone(fc1, F, λ, gens_pos, lib)
display(fc2.gens)
display(Set(fc1.gens) == Set(fc2.gens)) # True means convergence

fc = fc2
z = @. Fplot_init(x1s_', x2s_)
display(minimum(z))
contour!(plt, x1s_, x2s_, z, levels=[0], color=:green, lw=2)

display(plt)

end # module