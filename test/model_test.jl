using Cersyve
using Test
using JLD2

task = Unicycle
hidden_sizes = [32, 32]
data = JLD2.load(joinpath(@__DIR__, "../data/unicycle_data.jld2"))["data"]
f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, hidden_sizes)
f_pi_model = Cersyve.create_closed_loop_dynamics_model(
    f_model, task.pi_model, data, task.x_low, task.x_high, task.u_dim)

function closed_loop_dynamics(x::Array{Float32})::Array{Float32}
    x_normed = (x .- data["x_mean"]) ./ data["x_std"]
    u_normed = (task.pi_model(x) .- data["u_mean"]) ./ data["u_std"]
    dx = f_model(vcat(x_normed, u_normed)) .* data["dx_std"] .+ data["dx_mean"]
    return min.(max.(x .+ dx, task.x_low), task.x_high)
end

x = Cersyve.uniform(task.x_low, task.x_high, 10)
@test isapprox(f_pi_model(x), closed_loop_dynamics(x))
