function generate_vals_on_ball(np, xc, rad, dt, nstep, var, flow)
    nvar = length(var)
    @assert length(xc) == nvar
    F!(du, u, _, _) = begin
        for (i, f) in enumerate(flow)
            du[i] = f(var=>u)
        end
        nothing
    end
    vals = [xc]
    for _ = 1:np
        u0 = randn(nvar)
        normalize!(u0)
        u0 = rad * u0 + xc
        prob = ODEProblem(F!, u0, (0, nstep*dt))
        sol = solve(prob, saveat=dt)
        append!(vals, sol.u)
    end
    return vals
end

const opt_ = optimizer_with_attributes(Mosek.Optimizer, "QUIET"=>true)
solver() = SOSModel(opt_)

function callback_func(iter, i, ng, r_max)
    if i < ng
        print("Iter $(iter): $(i) / $(ng): $(r_max)                         \r")
    else
        print("Iter $(iter): $(i) / $(ng): $(r_max)                         \n")
    end
    return nothing
end