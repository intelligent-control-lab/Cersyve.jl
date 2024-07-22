using Cersyve
using Random

task = Unicycle
save_path = joinpath(@__DIR__, "../data/unicycle_data.jld2")
seed = 1

Random.seed!(seed)

collect_data(
    task.x_low,
    task.x_high,
    task.u_low,
    task.u_high,
    task.dynamics,
    task.terminated;
    save_path=save_path,
)
