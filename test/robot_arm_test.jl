using Cersyve
using Test

task = RobotArm

function policy(x::Matrix{Float32})::Matrix{Float32}
    x_eq = Float32[pi / 2; zeros(2 * task.n - 1)]
    return min.(max.(-task.K * (x .- x_eq), task.u_low), task.u_high)
end

function closed_loop_dynamics(x::Matrix{Float32})::Matrix{Float32}
    return min.(max.(task.A * x + task.B * policy(x), task.x_low), task.x_high)
end

x = Cersyve.uniform(task.x_low, task.x_high, 10)
@test isapprox(task.pi_model(x), policy(x))
@test isapprox(task.f_pi_model(x), closed_loop_dynamics(x))
