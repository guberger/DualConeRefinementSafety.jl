module TestMain

using LinearAlgebra
using DynamicPolynomials
using CDDLib
using CSDP
using SumOfSquares
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/DualConeRefinementSafety.jl")
else
    using DualConeRefinementSafety
end
const DCR = DualConeRefinementSafety

solver() = SOSModel(optimizer_with_attributes(CSDP.Optimizer, "printlevel"=>0))

# Create the variables for symbolic manipulation
vars, = @polyvar x[1:2]
flow = [
    x[2] / 2 + x[1] - x[1] * (4 * x[1]^2 + x[2]^2),
    -2 * x[1] + x[2] - x[2] * (4 * x[1]^2 + x[2]^2),
]
F = DCR.Field(vars, flow)
dom_init = @set x[1]^2 + x[2]^2 ≤ 1
funcs = [1, x[1]^2, x[1]*x[2], x[2]^2]
λ = 1.0
ϵ = 1e-2
δ = 1e-4
np = 10
points = [DCR.Point(vars, [cos(α) / 2, sin(α)] * 1.1)
          for α in range(0, 2π, np + 1)[1:np]]
hc = DCR.hcone_from_points(funcs, F, λ, ϵ, points)
vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())

function callback_func(iter, i, ng, r_max)
    if i < ng
        print("Iter $(iter): $(i) / $(ng): $(r_max)                         \r")
    else
        print("Iter $(iter): $(i) / $(ng): $(r_max)                         \n")
    end
    return nothing
end

success = DCR.narrow_vcone!(vc, dom_init, F, λ, ϵ, δ, Inf, solver,
                            callback_func=callback_func)

@testset "Main" begin
    @test success
    @test all(r -> r.a[1] < 0, vc.rays)
    np = 100
    for α in range(0, 2π, np + 1)[1:np]
        v = [cos(α) / 2, sin(α)]
        fv = (v[1]^2, v[1]*v[2], v[2]^2)
        rad = maximum(r -> -dot(r.a[2:4], fv) / r.a[1], vc.rays)
        @test 1 < 1 / sqrt(rad) < 2
    end
end

end # module