module Example_Ahmed2020_7

# Automated and Sound Synthesis of Lyapunov Functions with SMT Solvers
# Example 7

using LinearAlgebra
using Random
Random.seed!(0)
using DynamicPolynomials
using Plots
using DifferentialEquations
using CDDLib
using SumOfSquares
using MosekTools

include("utils.jl")

var, = @polyvar x[1:3]
den = (x[3]^2 + 1)
flow = [
    1 - (x[1]^3 + x[1] * x[3]^2) * den,
    -(x[2] + x[1]^2 * x[2]) * den,
    -3 * x[3] + (-x[3] + 3 * x[1]^2 * x[3]) * den,
]
display(flow)
rad = 0.5
dom_init = @set x' * x ≤ rad^2

nstep = 5
dt = 0.25
np = 20
vals = generate_vals_on_ball(np, rad, dt, nstep, var, flow)

include("../src/InvariancePolynomial.jl")
const MP = InvariancePolynomial.Projection

F = MP.Field(var, flow)
points = [MP.Point(var, val) for val in vals]
funcs = [1, x[1]^2, x[2]^2, x[3]^2]
λ = 1.0
ϵ = 1e-1
hc = MP.hcone_from_points(funcs, F, λ, ϵ, points)
display(length(hc.halfspaces))

vc = MP.vcone_from_hcone(hc, () -> CDDLib.Library())
display(length(vc.rays))

δ = 1e-4
success = MP.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                            callback_func=callback_func)
display(success)
display(vc.funcs)
display(vc.rays)
MP.simplify_vcone!(vc, 1e-5, solver)
display(vc.rays)

# resize!(vc.rays, 3)

@polyvar x0 x1 x2
file = open(string(@__DIR__, "/output.txt"), "w")
println(file, "Flow")
for f in flow
    println(file, f(var=>[x0, x1, x2]), ",")
end
println(file, "Barriers")
for r in vc.rays
    p = dot(vc.funcs, r.a)
    println(file, p(var=>[x0, x1, x2]), ",")
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