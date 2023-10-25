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
rad = 3
dom_init = @set x' * x ≤ rad^2

# xlims = (-4, 4)
# xlims = (-5, 5)
xlims = (-6, 6)
# ylims = (-4, 4)
# ylims = (-6, 6)
ylims = (-8, 8)
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

nstep = 50
dt = 0.25
np = 20
vals = generate_vals_on_ball(np, rad, dt, nstep, var, flow)

# scatter!(plt, getindex.(vals, 1), getindex.(vals, 2), label="", ms=1)

display(plt)
savefig(plt, "examples/figures/vanderpol_field.png")

include("../../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.Projection

F = MP.Field(var, flow)
points = [MP.Point(var, val) for val in vals]
funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
λ = 1.0
ϵ = 1e-1
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))

vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

vc_list = typeof(vc)[]
function record_func(vc)
    push!(vc_list, MP.VConeSubset(vc.funcs, copy(vc.rays)))
end

δ = 1e-9
flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                           callback_func=callback_func,
                           record_func=record_func)
display(flag)
display(vc.funcs)
display(vc.rays)
MP.simplify_vcone!(vc, 1e-9, solver)
display(vc.rays)

# ------------------------------------------------------------------------------

Fplot_vc(x1, x2) = begin
    gxs = [g(var=>[x1, x2]) for g in vc.funcs]
    maximum(r -> dot(r.a, gxs), vc.rays)
end

empty!(vc_list)

for vc_curr in vc_list
    global vc = vc_curr
    z = @. Fplot_vc(x1s_', x2s_)
    display(minimum(z))
    contour!(plt, x1s_, x2s_, z, levels=[0], c=:black, lw=2)
end

display(plt)
savefig(plt, "examples/figures/vanderpol_iterations.png")

z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(plt, x1s_, x2s_, z, levels=[0], c=:green, lw=2)

vc = MP.VConeSubset(vc.funcs, vc.rays[[2, 6, 8]])
display(vc)

z = @. Fplot_vc(x1s_', x2s_)
display(minimum(z))
contour!(plt, x1s_, x2s_, z, levels=[0], c=:red, lw=2)

display(plt)
savefig(plt, "examples/figures/vanderpol_invariant.png")

@polyvar x0 x1
file = open(string(@__DIR__, "/output.txt"), "w")
println(file, "Flow")
for f in flow
    println(file, f(var=>[x0, x1]), ",")
end
println(file, "Barriers")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    println(file, p(var=>[x0, x1]), ",")
end
close(file)

model = solver()
r = @variable(model)
dom = MP.sos_domain_from_vcone(vc)
@constraint(model, x' * x ≤ r, domain=dom)
@objective(model, Min, r)
optimize!(model)
@assert primal_status(model) == FEASIBLE_POINT
display(sqrt(value(r)))

end # module