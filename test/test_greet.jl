using Test
@static if isdefined(Main, :TestLocal)
    include("../src/DualConeRefinementSafety.jl")
else
    using DualConeRefinementSafety
end

@testset "Greetings" begin
    stdout_old = Base.stdout
    Base.stdout = IOBuffer()
    DualConeRefinementSafety.greet()
    @test String(take!(stdout)) == "Hello World!"
    Base.stdout = stdout_old
end