module Pendulum

using Flux
using EllipsisNotation
using LinearAlgebra
using MatrixEquations

x_dim = 2
x_low = Float32[-pi / 4, -4]
x_high = Float32[pi / 4, 4]

u_dim = 1
u_low = Float32[-1]
u_high = Float32[1]

th_con = pi / 6

g = 10.0
m = 1.0
l = 1.0
dt = 0.05
tmp1 = 3 * g / (2 * l) * dt
tmp2 = 9 / (m * l ^ 2) * dt

function dynamics(x::Array{Float32}, u::Array{Float32})::Array{Float32}
    th = x[1, ..]
    thdot = x[2, ..]
    T = u[1, ..]
    newthdot = thdot + tmp1 * sin.(th) + tmp2 * T
    newthdot = clamp.(newthdot, x_low[2], x_high[2])
    newth = th + newthdot * dt
    newth = clamp.(newth, x_low[1], x_high[1])
    x_prime = Float32.(stack((newth, newthdot); dims=1))
    return x_prime
end

function terminated(x::Array{Float32})::Array{Float32}
    th = x[1, ..]
    return (th .== x_low[1]) .| (th .== x_high[1])
end

h_model = Chain(
    Dense(Float32[1 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-th_con]),
    )
)

A = Float32[1 + tmp1 * dt dt; tmp1 1]
B = Float32[tmp2 * dt; tmp2;;]
Q = diagm(Float32[1, 0.1])
R = diagm(Float32[0.1])
P, _, _ = ared(A, B, R, Q)
K = inv(R + B' * P * B) * (B' * P * A)

pi_model = Chain(
    Dense(-K),
    # max(u, u_low) = relu(x - u_low) + u_low
    Dense(Matrix{Float32}(I(u_dim)), -u_low, relu),
    Dense(Matrix{Float32}(I(u_dim)), u_low),
    # min(u, u_high) = -max(-x, -u_high) = -relu(-x + u_high) + u_high
    Dense(Matrix{Float32}(-I(u_dim)), u_high, relu),
    Dense(Matrix{Float32}(-I(u_dim)), u_high),
)

end
