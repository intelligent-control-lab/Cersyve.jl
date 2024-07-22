using Cersyve
using JLD2
using Random

task = Unicycle
hidden_sizes = [32, 32]
data_path = joinpath(@__DIR__, "../data/unicycle_data.jld2")
log_dir = joinpath(@__DIR__, "../log/unicycle/")
seed = 1

Random.seed!(seed)

data = JLD2.load(data_path)["data"]

f_model = Cersyve.create_mlp(task.x_dim + task.u_dim, task.x_dim, hidden_sizes)

train_dynamics(
    data,
    f_model;
    penalty="APA",
    space_size=[task.x_high; task.u_high] - [task.x_low; task.u_low],
    apa_coef=0.01,
    log_dir=log_dir,
)
