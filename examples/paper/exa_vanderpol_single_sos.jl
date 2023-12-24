module Example

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

include("../utils.jl")

var, = @polyvar x[1:2]
μ = 0.5
flow = [
    x[2],
    μ * (1 - x[1]^2) * x[2] - x[1],
]
display(flow)
xc = zeros(2)
rad = 3
dom_init = @set (x - xc)' * (x - xc) ≤ rad^2
dom_unsafe = @set x' * x ≥ 10^2

xlims = (-10, 10)
ylims = (-10, 10)
# xlims = (-10.8, -9.2)
# ylims = (0, 0.5)
# xlims = (-10.1, -9.7)
# ylims = (0.19, 0.21)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
        #    aspect_ratio=:equal,
           xlims=xlims, ylims=ylims)

x1s_ = range(xlims..., length=30)
x2s_ = range(ylims..., length=30)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.04 / nx
dxs2 = getindex.(dxs, 2) * 0.04 / nx
quiver!(x1s, x2s, quiver=(dxs1, dxs2))

x1s_ = range(xlims..., length=500)
x2s_ = range(ylims..., length=500)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
          lw=2, c=:yellow, alpha=0.5, colorbar=:none)
Fplot_unsafe(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_unsafe))
z = @. Fplot_unsafe(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
          lw=1, c=:blue, alpha=0.5, colorbar=:none)

display(plt)

monos = monomials(x, 0:8)
model = solver()
@variable(model, B, Poly(monos))
ϵ = 0.001
@constraint(model, B ≥ +ϵ, domain = dom_unsafe)
@constraint(model, B ≤ 0, domain = dom_init)
dBdt = dot(differentiate(B, x), flow)
λ = 2
ϵderiv = 0
ϵderiv = 0.0001
@constraint(model, dBdt + λ * B ≤ -ϵderiv)
optimize!(model)
display(solution_summary(model))

Bopt = value(B)
dBdtopt = dot(differentiate(Bopt, x), flow)
display(Bopt)

Fplot_vc(x1, x2) = Bopt(var=>[x1, x2])
z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(x1s_, x2s_, z, levels=[0], color=:green, lw=2)

Fplot_vc(x1, x2) = dBdtopt(var=>[x1, x2]) + λ * 0 * Bopt(var=>[x1, x2])
z = @. Fplot_vc(x1s_', x2s_)
display(maximum(z))
contourf!(x1s_, x2s_, z, levels=[0, 100], color=:orange, lw=0.1, alpha=0.5)

println("test")
xtest = [-10, 0.2]
display(Bopt(var=>xtest))
display(dBdtopt(var=>xtest))

display(plt)

@polyvar x0 x1
file = open(string(@__DIR__, "/output.txt"), "w")
println(file, "Flow")
for f in flow
    str = string(f(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "Barriers")
str = string(Bopt(var=>[x0, x1]), ",")
str = replace(str, "^"=>"**")
println(file, str)
close(file)

end # module