using Cersyve
using Test

task = Quadrotor

function constraint(x::Matrix{Float32})::Vector{Float32}
    return max.(
        max.(-task.z_con .- x[1, :], x[1, :] .- task.z_con),
        max.(-task.th_con .- x[2, :], x[2, :] .- task.th_con),
    )
end

function policy(x::Matrix{Float32})::Matrix{Float32}
    u_eq = Float32[task.T0, task.T0]
    return min.(max.(-task.K * x .+ u_eq, task.u_low), task.u_high)
end

x = Cersyve.uniform(task.x_low, task.x_high, 10)
@test isapprox(task.h_model(x)[1, :], constraint(x))
@test isapprox(task.pi_model(x), policy(x))
