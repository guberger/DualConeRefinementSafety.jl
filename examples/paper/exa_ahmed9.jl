module Example

# Automated and Sound Synthesis of Lyapunov Functions with SMT Solvers
# Example 9

using LinearAlgebra
using Random
Random.seed!(0)
using DynamicPolynomials
using DifferentialEquations
using CDDLib
using SumOfSquares
using MosekTools

include("../utils.jl")

var, = @polyvar x[1:4]
flow = [
    -x[1] + x[2]^3 - 3 * x[3] * x[4],
    0.5 - x[1] - x[2]^3,
    x[1] * x[4] - x[3],
    x[1] * x[3] - x[4]^3,
]
display(flow)
xc = zeros(4)
rad = 0.5
dom_init = @set (x - xc)' * (x - xc) ≤ rad^2

nstep = 5
dt = 0.25
np = 20
vals = generate_vals_on_ball(np, xc, rad, dt, nstep, var, flow)

include("../../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.Projection

F = MP.Field(var, flow)
points = [MP.Point(var, val) for val in vals]
funcs = [1, x[1]^2, x[2]^2, x[3]^2, x[4]^2]
λ = 1.0
ϵ = 1e-1
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))

vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

δ = 1e-8
flag = @time MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                              callback_func=callback_func)
@assert flag
display(length(vc.rays))
MP.simplify_vcone!(vc, 1e-5, solver, delete=false)
display(length(vc.rays))

#-------------------------------------------------------------------------------

file = open(string(@__DIR__, "/output.txt"), "w")
@polyvar x0 x1 x2 x3
println(file, "Flow")
for f in flow
    str = string(f(var=>[x0, x1, x2, x3]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
println(file, "Barriers python")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    str = string(p(var=>[x0, x1, x2, x3]), ",")
    str = replace(str, "^"=>"**")
    println(file, str)
end
@polyvar x1 x2 x3 x4
println(file, "Barriers latex")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    str = string(p(var=>[x1, x2, x3, x4]), ",")
    println(file, str)
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