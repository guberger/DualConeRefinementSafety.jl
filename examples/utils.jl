function generate_vals(np, rad, dt, nstep, vars, flow)
    nvar = length(vars)
    F!(du, u, ::Any, ::Any) = begin
        for (i, f) in enumerate(flow)
            du[i] = f(vars=>u)
        end
        nothing
    end
    vals = [zeros(nvar)]
    for _ = 1:np
        u0 = randn(nvar)
        normalize!(u0)
        lmul!(rad, u0)
        prob = ODEProblem(F!, u0, (0, nstep*dt))
        sol = solve(prob, saveat=dt)
        append!(vals, sol.u)
    end
    return vals
end

const opt_ = optimizer_with_attributes(Mosek.Optimizer, "QUIET"=>true)
solver() = begin
    model = SOSModel(opt_)
    @static if isdefined(Main, :DSOS) && Main.DSOS
        PolyJuMP.setdefault!(model, PolyJuMP.NonNegPoly, DSOSCone)
    end
    return model
end

function callback_func(iter, i, ng, r_max)
    if i < ng
        print("Iter $(iter): $(i) / $(ng): $(r_max)                         \r")
    else
        print("Iter $(iter): $(i) / $(ng): $(r_max)                         \n")
    end
    return nothing
end