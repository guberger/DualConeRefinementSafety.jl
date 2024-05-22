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

include("utils.jl")

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

xlims = (-4, 4)
ylims = (-4, 4)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=xlims .* 1.1, ylims=ylims .* 1.1,
           dpi=400)

x1s_ = range(xlims..., length=15)
x2s_ = range(ylims..., length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.5 / nx
dxs2 = getindex.(dxs, 2) * 0.5 / nx
quiver!(x1s, x2s, quiver=(dxs1, dxs2), arrow=:closed)

x1s_ = range(xlims..., length=500)
x2s_ = range(ylims..., length=500)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
          lw=5, c=:yellow, alpha=0.5, colorbar=:none)

F!(dx, x, _, _) = begin
    for (i, f) in enumerate(flow)
        dx[i] = f(var=>x)
    end
    nothing
end

for α in range(0, 2*pi, length=20)[1:end-1]
    x0 = [cos(α), sin(α)]
    normalize!(x0)
    lmul!(rad, x0)
    prob = ODEProblem(F!, x0, (0, 10))
    sol = solve(prob, dtmax=1e-2)
    plot!(sol[1, :], sol[2, :], lw=2, label="")
end

savefig(plt, "examples/figures/vanderpol_field.png")

#-------------------------------------------------------------------------------

nstep = 50
dt = 0.25
np = 20
vals = generate_vals_on_ball(np, xc, rad, dt, nstep, var, flow)

include("../src/main.jl")
const TK = ToolKit

F = TK.Field(var, flow)
points = [TK.Point(var, val) for val in vals]
funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
λ = 1.0
ϵ = 1e-1
hc = TK.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))

vc = TK.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

vc_list = typeof(vc)[]
function record_func(vc)
    push!(vc_list, TK.VConeSubset(vc.funcs, copy(vc.rays)))
end

δ = 1e-9
flag = @time TK.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func,
                              record_func=record_func)
@assert flag
display(length(vc.rays))
TK.simplify_vcone!(vc, 1e-9, solver)
display(length(vc.rays))

#-------------------------------------------------------------------------------

dom_unsafe = @set x' * x ≥ 10^2
monos = monomials(x, 0:8)
model = solver()
@variable(model, B, Poly(monos))
ϵ = 0.001
@constraint(model, B ≥ +ϵ, domain = dom_unsafe)
@constraint(model, B ≤ 0, domain = dom_init)
dBdt = dot(differentiate(B, x), flow)
λ = 1
ϵderiv = 1e-5
@constraint(model, dBdt + λ * B ≤ -ϵderiv)
@time optimize!(model)
display(solution_summary(model))

Bopt = value(B)
display(Bopt)

#-------------------------------------------------------------------------------

Fplot_vc(vc, x1, x2) = begin
    gxs = [g(var=>[x1, x2]) for g in vc.funcs]
    maximum(r -> dot(r.a, gxs), vc.rays)
end

#-------------------------------------------------------------------------------

xlims = (-6, 6)
ylims = (-6, 6)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=xlims .* 1.1, ylims=ylims .* 1.1,
           dpi=400)

x1s_ = range(xlims..., length=15)
x2s_ = range(ylims..., length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.5 / nx
dxs2 = getindex.(dxs, 2) * 0.5 / nx
quiver!(x1s, x2s, quiver=(dxs1, dxs2), arrow=:closed)

x1s_ = range(xlims..., length=500)
x2s_ = range(ylims..., length=500)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
          lw=5, c=:yellow, alpha=0.5, colorbar=:none)

for vc_curr in vc_list[1:8]
    z = @. Fplot_vc((vc_curr,), x1s_', x2s_)
    contour!(plt, x1s_, x2s_, z, levels=[0], c=:black, lw=2)
end

savefig(plt, "examples/figures/vanderpol_iterations.png")

#-------------------------------------------------------------------------------

xlims = (-8, 8)
ylims = (-8, 8)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=xlims .* 1.1, ylims=ylims .* 1.1,
           dpi=400)

x1s_ = range(xlims..., length=15)
x2s_ = range(ylims..., length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
dxs = [[f(var=>x) for f in flow] for x in xs]
nx = maximum(dx -> norm(dx), dxs)
dxs1 = getindex.(dxs, 1) * 0.5 / nx
dxs2 = getindex.(dxs, 2) * 0.5 / nx
quiver!(x1s, x2s, quiver=(dxs1, dxs2), arrow=:closed)

x1s_ = range(xlims..., length=500)
x2s_ = range(ylims..., length=500)
Fplot_init(x1, x2) = maximum(g(var=>[x1, x2]) for g in inequalities(dom_init))
z = @. Fplot_init(x1s_', x2s_)
contourf!(x1s_, x2s_, z, levels=[0, 100],
          lw=5, c=:yellow, alpha=0.5, colorbar=:none)

# colors = [:red, :blue, :green, :yellow, :black,
#           :gray, :purple, :brown, :pink, :magenta]
# for (i, r) in enumerate(vc.rays)
#     vc_curr = TK.VConeSubset(vc.funcs, [r])
#     z = @. Fplot_vc((vc_curr,), x1s_', x2s_)
#     contour!(plt, x1s_, x2s_, z, levels=[0], c=colors[i], lw=2)
# end

z = @. Fplot_vc((vc,), x1s_', x2s_)
contour!(plt, x1s_, x2s_, z, levels=[0], c=:green, lw=2)

vc_bis = TK.VConeSubset(vc.funcs, vc.rays[[1, 3, 6]])
z = @. Fplot_vc((vc_bis,), x1s_', x2s_)
contour!(plt, x1s_, x2s_, z, levels=[0], c=:purple, lw=2)

Fplot_B(x1, x2) = Bopt(var=>[x1, x2])
z = @. Fplot_B(x1s_', x2s_)
contour!(x1s_, x2s_, z, levels=[0], color=:blue, lw=2)

savefig(plt, "examples/figures/vanderpol_invariant.png")

#-------------------------------------------------------------------------------

file = open(string(@__DIR__, "/output.txt"), "w")
@polyvar x0 x1
println(file, "Flow")
for f in flow
    str = string(f(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "Barriers python")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    str = string(p(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "SOS python")
str = string(Bopt(var=>[x0, x1]), ",")
str = replace(str, "^"=>"**")
println(file, str)
@polyvar x1 x2
println(file, "Barriers latex")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    str = string(p(var=>[x1, x2]), ",")
    println(file, str)
end
println(file, "SOS python")
str = string(Bopt(var=>[x1, x2]), ",")
println(file, str)
close(file)

end # module