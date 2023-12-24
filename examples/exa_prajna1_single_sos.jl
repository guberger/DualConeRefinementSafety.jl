module Example

# Safety Verification of Hybrid Systems Using Barrier Certificates
# Example 1

using LinearAlgebra
using Random
Random.seed!(0)
using LaTeXStrings
using DynamicPolynomials
using Plots
using DifferentialEquations
using CDDLib
using SumOfSquares
using MosekTools

include("utils.jl")

var, = @polyvar x[1:2]
p = 1
flow = [
    x[2],
    -x[1] + (p / 3) * x[1]^3 - x[2],
]
display(flow)
xc = [1.5, 0]
rad = 0.5
dom_init = @set (x - xc)' * (x - xc) ≤ rad^2

xlims = (-4, 4)
ylims = (-4, 4)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=xlims .* 1.1, ylims=ylims .* 1.1)

x1s_ = range(xlims..., length=15)
x2s_ = range(ylims..., length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.4 / nx
dxs2 = getindex.(dxs, 2) * 0.4 / nx
quiver!(x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(xlims..., length=500)
x2s_ = range(ylims..., length=500)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
            lw=5, c=:yellow, alpha=0.5, colorbar=:none)

nstep = 5
dt = 0.25
np = 20
rad = 0.5
vals = generate_vals_on_ball(np, xc, rad, dt, nstep, var, flow)
scatter!(plt, getindex.(vals, 1), getindex.(vals, 2), label="")

display(plt)

g1 = -(x[1] + 1)^2 - (x[2] + 1)^2 + 0.16
monos = monomials(x, 0:4)
model = solver()
@variable(model, B, Poly(monos))
ε = 0.001
@constraint(model, B ≥ ε, domain = @set(g1 ≥ 0))
@constraint(model, B ≤ 0, domain = dom_init)
dBdt = dot(differentiate(B, x), flow)
@constraint(model, -dBdt ≥ 0)
optimize!(model)
display(primal_status(model))

Bopt = value(B)

Fplot_vc(x1, x2) = Bopt(var=>[x1, x2])
z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(x1s_, x2s_, z, levels=[0], color=:green, lw=2)

display(plt)

end # module