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

solver() = optimizer_with_attributes(CSDP.Optimizer, "printlevel"=>0)

# Create the variables for symbolic manipulation
vars, = @polyvar x[1:2]
f = [
    +x[2] + x[1] - x[1] * (x[1]^2 + x[2]^2),
    -x[1] + x[2] - x[2] * (x[1]^2 + x[2]^2),
]
tmp = DCR.Template(vars, [1, x[1]^2, x[1]*x[2], x[2]^2])
λ = 1
maxorder = 5

np = 10
rad = 0.5
points = [[rad*cos(α), rad*sin(α)] for α in range(0, 2π, np + 1)[1:np]]
hc = DCR.hcone_from_points(tmp, f, λ, maxorder, points)
vc = DCR.vcone_from_hcone(hc, () -> CDDLib.Library())
vc_narrowed, success = DCR.narrow_vcone(vc, f, Inf, solver)

@testset "Main" begin
    @test success
    @test all(c -> c[1] < 0, vc_narrowed.gens)
    np = 100
    for α in range(0, 2π, np + 1)[1:np]
        v = [cos(α), sin(α)]
        fv = (v[1]^2, v[1]*v[2], v[2]^2)
        r = maximum(c -> -dot(c[2:4], fv) / c[1], vc_narrowed.gens)
        @test 1 < 1 / sqrt(r) < 2
    end
end

end # module