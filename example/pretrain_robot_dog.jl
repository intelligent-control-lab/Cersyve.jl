using Cersyve
using Flux
using JLD2
using LinearAlgebra
using ModelVerification
using Random

task = RobotDog
value_hidden_sizes = [32, 32]
constraint_hidden_sizes = [16]
model_dir = joinpath(@__DIR__, "../model/robot_dog/")
log_dir = joinpath(@__DIR__, "../log/robot_dog/")
seed = 1

Random.seed!(seed)

V_model = Cersyve.create_mlp(task.x_dim, 1, value_hidden_sizes)

f_model = ModelVerification.build_flux_model(task.dynamics_path)
f_pi_model = Chain(Parallel(+,
    Dense(Matrix{Float32}(I(task.x_dim))),
    f_model,
))

h_model = Cersyve.create_mlp(task.x_dim, 1, constraint_hidden_sizes)
Flux.loadmodel!(h_model, JLD2.load(joinpath(model_dir, "h.jld2"), "state"))

pretrain_value(
    V_model,
    f_pi_model,
    h_model,
    task.x_low,
    task.x_high;
    penalty="APA",
    space_size=task.x_high - task.x_low,
    apa_coef=1e-4,
    log_dir=log_dir,
)
