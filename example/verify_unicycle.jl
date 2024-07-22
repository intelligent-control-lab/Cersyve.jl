using Cersyve
using Flux
using JLD2

task = Unicycle
value_hidden_sizes = [32, 32]
dynamics_hidden_sizes = [32, 32]
constraint_hidden_sizes = [16]
data_path = joinpath(@__DIR__, "../data/unicycle_data.jld2")
model_dir = joinpath(@__DIR__, "../model/unicycle/")
log_dir = joinpath(@__DIR__, "../log/unicycle/")

V_model = Cersyve.create_mlp(task.x_dim, 1, value_hidden_sizes)
Flux.loadmodel!(V_model, JLD2.load(joinpath(model_dir, "V_finetune.jld2"), "state"))

data = JLD2.load(data_path)["data"]
f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, dynamics_hidden_sizes)
Flux.loadmodel!(f_model, JLD2.load(joinpath(model_dir, "f.jld2"), "state"))
f_pi_model = Cersyve.create_closed_loop_dynamics_model(
    f_model, task.pi_model, data, task.x_low, task.x_high, task.u_dim)

h_model = Cersyve.create_mlp(task.x_dim, 1, constraint_hidden_sizes)
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))

V_h_model = Cersyve.create_value_constraint_model(V_model, h_model)
V_V_prime_model = Cersyve.create_value_next_value_model(V_model, f_pi_model)

verify_value(
    task.x_low,
    task.x_high,
    V_h_model,
    V_V_prime_model,
)
