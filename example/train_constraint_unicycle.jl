using Cersyve
using JLD2
using Random

task = Unicycle
hidden_sizes = [16]
log_dir = joinpath(@__DIR__, "../log/unicycle/")
seed = 1

Random.seed!(seed)

h_model = Cersyve.create_mlp(task.x_dim, 1, hidden_sizes)

train_constraint(
    task.x_low,
    task.x_high,
    h_model,
    task.constraint;
    log_dir=log_dir,
)
