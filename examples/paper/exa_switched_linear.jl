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

function is_in(dom, var, x)
    return all(g -> g(var=>x) ≥ 0, inequalities(dom))
end

var, = @polyvar x[1:2]
β = 0
flow1 = [
    -(x[2] - β) + 0.25 * (x[1] - 1) * (2 - (x[1] - 1)^2 - (x[2] - β)^2),
    (x[1] - 1) + 0.25 * (x[2] - β) * (2 - (x[1] - 1)^2 - (x[2] - β)^2),
]
display(flow1)
dom1 = @set x[1] ≥ 0
flow2 = [
    -(x[2] + β) + 0.25 * (x[1] + 1) * (2 - (x[1] + 1)^2 - (x[2] + β)^2),
    (x[1] + 1) + 0.25 * (x[2] + β) * (2 - (x[1] + 1)^2 - (x[2] + β)^2),
]
display(flow2)
dom2 = @set x[1] ≤ 0
xc = zeros(2)
rad = 1
dom_init = @set (x - xc)' * (x - xc) ≤ rad^2
guard12 = @set x[2] == 0 && x[1] ≥ 0
guard21 = @set x[2] == 0 && x[1] ≤ 0

xlims = (-3.5, 3.5)
ylims = (-3.5, 3.5)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=xlims .* 1.1, ylims=ylims .* 1.1,
           dpi=400)

x1s_ = range(xlims..., length=15)
x2s_ = range(ylims..., length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
function compute_flow(x)
    if is_in(dom1, var, x)
        return flow1
    elseif is_in(dom2, var, x)
        return flow2
    end
    error("No domain")
end
dxs = [[f(var=>x) for f in compute_flow(x)] for x in xs]
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
    flow = compute_flow(x)
    for (i, f) in enumerate(flow)
        dx[i] = f(var=>x)
    end
    nothing
end

for _ = 1:30
    x0 = randn(2)
    normalize!(x0)
    lmul!(rad, x0)
    prob = ODEProblem(F!, x0, (0, 10))
    sol = solve(prob)
    plot!(sol[1, :], sol[2, :], label="")
end

#-------------------------------------------------------------------------------

nstep = 50
dt = 0.25
np = 20

vals1 = generate_vals_on_ball(np, xc, rad, dt, nstep, var, flow1)
vals2 = generate_vals_on_ball(np, xc, rad, dt, nstep, var, flow2)

include("../../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.Projection

funcs = [1, x[2], x[1]^2, x[2]^2]
λ = 1.0
ϵ = 1e-1
δ = 1e-8

# ------------------------------------------------------------------------------
println("--- Mode 1 ---")

F = MP.Field(var, flow1)
points = [MP.Point(var, val) for val in vals1]
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))
vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func,
                              dom_inv=dom1)
@assert flag
display(length(vc.rays))
MP.simplify_vcone!(vc, 1e-5, solver, delete=false)
display(length(vc.rays))
vc1 = vc

# ------------------------------------------------------------------------------
println("--- Mode 2 ---")

dom_init = MP.sos_domain_from_vcone(vc, init=guard12)

F = MP.Field(var, flow2)
points = [MP.Point(var, val) for val in vals2]
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))
vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func,
                              dom_inv=dom2)
@assert flag
display(length(vc.rays))
MP.simplify_vcone!(vc, 1e-5, solver, delete=false)
display(length(vc.rays))
vc2 = vc

# ------------------------------------------------------------------------------

function compute_vc(x)
    if is_in(dom1, var, x)
        return vc1
    elseif is_in(dom2, var, x)
        return vc2
    end
    error("No domain")
end

Fplot_vc(x1, x2) = begin
    vc = compute_vc([x1, x2])
    gxs = [g(var=>[x1, x2]) for g in vc.funcs]
    maximum(r -> dot(r.a, gxs), vc.rays)
end

z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(x1s_, x2s_, z, levels=[0], color=:green, lw=2)

savefig(plt, "examples/figures/switched_linear.png")

# ------------------------------------------------------------------------------

file = open(string(@__DIR__, "/output.txt"), "w")
@polyvar x0 x1
println(file, "Flow 1")
for f in flow1
    str = string(f(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "Flow 2")
for f in flow2
    str = string(f(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "Barriers python 1")
for r in vc1.rays
    p = dot(vc1.funcs, r.a)
    str = string(p(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "Barriers python 2")
for r in vc2.rays
    p = dot(vc2.funcs, r.a)
    str = string(p(var=>[x0, x1]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
@polyvar x1 x2
println(file, "Barriers latex 1")
for r in vc1.rays
    p = dot(vc1.funcs, r.a)
    str = string(p(var=>[x1, x2]), ",")
    println(file, str)
end
println(file, "Barriers latex 2")
for r in vc2.rays
    p = dot(vc2.funcs, r.a)
    str = string(p(var=>[x1, x2]), ",")
    println(file, str)
end
close(file)

end # module