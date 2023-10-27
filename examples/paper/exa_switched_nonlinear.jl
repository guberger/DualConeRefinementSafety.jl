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
flow_a = [
    - 0.1 * (x[1] - 1)^3 - x[2],
    2 * (x[1] - 1) - 0.1 * x[2]^3,
]
display(flow_a)
flow_b = [
    - 0.25 * (x[1] + 0.5) - 2 * x[2],
    (x[1] + 0.5) / 2 - 0.25 * x[2],
]
display(flow_b)
flow1 = flow_a
dom1 = @set x[1] ≥ 0 && x[2] ≤ 0
flow2 = flow_a
dom2 = @set x[1] ≥ 0 && x[2] ≥ 0
flow3 = flow_b
dom3 = @set x[1] ≤ 0
rad = 1
dom_init = @set x' * x ≤ rad^2
guard12 = @set x[2] == 0 && x[1] ≥ 0
guard23 = @set x[1] == 0 && x[2] ≥ 0

xlims = (-3.5, 3.5)
ylims = (-3.5, 3.5)
plt = plot(xlabel=L"x_1", ylabel=L"x_2",
           aspect_ratio=:equal,
           xlims=xlims .* 1.1, ylims=ylims .* 1.1)

x1s_ = range(xlims..., length=15)
x2s_ = range(ylims..., length=15)
xs = collect(Iterators.product(x1s_, x2s_))[:]
x1s = getindex.(xs, 1)
x2s = getindex.(xs, 2)
function compute_flow(x)
    if is_in(dom1, var, x) || is_in(dom2, var, x)
        return flow_a
    elseif is_in(dom3, var, x)
        return flow_b
    end
    error("No domain")
end
dxs = [[f(var=>x) for f in compute_flow(x)] for x in xs]
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

nstep = 50
dt = 0.25
np = 20

vals_a = generate_vals_on_ball(np, rad, dt, nstep, var, flow_a)
vals_b = generate_vals_on_ball(np, rad, dt, nstep, var, flow_b)

# for vals in (vals_a, vals_b)
#     scatter!(plt, getindex.(vals, 1), getindex.(vals, 2), label="", ms=1)
# end

display(plt)

include("../../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.Projection

funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
λ = 1.0
ϵ = 1e-1
δ = 1e-8

# ------------------------------------------------------------------------------
println("--- Mode 1 ---")

F = MP.Field(var, flow1)
points = [MP.Point(var, val) for val in vals_a]
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))
vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func,
                              dom_inv=dom1)
display(flag)
display(vc.funcs)
display(vc.rays)
MP.simplify_vcone!(vc, 1e-5, solver, delete=false)
display(vc.rays)
vc1 = vc

# ------------------------------------------------------------------------------
println("--- Mode 2 ---")

dom_init = MP.sos_domain_from_vcone(vc, init=guard12)

F = MP.Field(var, flow2)
points = [MP.Point(var, val) for val in vals_b]
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))
vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func,
                              dom_inv=dom2)
display(flag)
display(vc.funcs)
display(vc.rays)
MP.simplify_vcone!(vc, 1e-5, solver, delete=false)
display(vc.rays)
vc2 = vc

# ------------------------------------------------------------------------------
println("--- Mode 3 ---")

dom_init = MP.sos_domain_from_vcone(vc, init=guard23)

F = MP.Field(var, flow3)
points = [MP.Point(var, val) for val in vals_b]
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))
vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func,
                              dom_inv=dom3)
display(flag)
display(vc.funcs)
display(vc.rays)
MP.simplify_vcone!(vc, 1e-5, solver, delete=false)
display(vc.rays)
vc3 = vc

# ------------------------------------------------------------------------------
function compute_vc(x)
    if is_in(dom1, var, x)
        return vc1
    elseif is_in(dom2, var, x)
        return vc2
    elseif is_in(dom3, var, x)
        return vc3
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

display(plt)
savefig(plt, "examples/figures/switched_nonlinear.png")

vc = vc3

@polyvar x1 x2
file = open(string(@__DIR__, "/output.txt"), "w")
println(file, "Barriers")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    println(file, p(var=>[x1, x2]), ",")
end
close(file)

end # module