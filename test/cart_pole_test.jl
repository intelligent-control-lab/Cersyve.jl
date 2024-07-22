using Cersyve
using Test

task = CartPole

function constraint(x::Matrix{Float32})::Vector{Float32}
    return max.(
        max.(-task.y_con .- x[1, :], x[1, :] .- task.y_con),
        max.(-task.th_con .- x[2, :], x[2, :] .- task.th_con),
    )
end

function policy(x::Matrix{Float32})::Matrix{Float32}
    return min.(max.(-task.K * x, task.u_low), task.u_high)
end

x = Cersyve.uniform(task.x_low, task.x_high, 10)
@test isapprox(task.h_model(x)[1, :], constraint(x))
@test isapprox(task.pi_model(x), policy(x))
