module Quadrotor

using Flux
using EllipsisNotation
using LinearAlgebra
using MatrixEquations

# state: [z, th, z_dot, th_dot]
x_dim = 4
x_high = Float32[1, pi / 6, 1, 1]
x_low = -x_high

# action: [T1, T2]
u_dim = 2
u_low = Float32[2, 2]
u_high = Float32[8, 8]

z_con = 0.9
th_con = pi / 8

m = 1.0
g = 9.8
d = 0.3
Iyy = 0.01
dt = 0.1
T0 = m * g / 2

function dynamics(x::Array{Float32}, u::Array{Float32})::Array{Float32}
    z, th, z_dot, th_dot = x[1, ..], x[2, ..], x[3, ..], x[4, ..]
    T1, T2 = u[1, ..], u[2, ..]
    x_prime = Float32.(stack((
        z + z_dot * dt,
        th + th_dot * dt,
        z_dot + ((T1 + T2) .* cos.(th) / m .- g) * dt,
        th_dot + ((T2 - T1) * d / Iyy) * dt,
    ); dims=1))
    return clamp.(x_prime, x_low, x_high)
end

function terminated(x::Array{Float32})::Array{Float32}
    z, th = x[1, ..], x[2, ..]
    return (z .== x_low[1]) .| (z .== x_high[1]) .| (
        th .== x_low[2]) .| (th .== x_high[2])
end

h_z_model = Chain(
    Dense(Float32[1 0 0 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-z_con]),
    )
)
h_th_model = Chain(
    Dense(Float32[0 1 0 0]),
    # max(-x - c, x - c) = relu(-2x) + x - c
    Parallel(+,
        Dense(Float32[-2;;], Float32[0], relu),
        Dense(Float32[1;;], Float32[-th_con]),
    )
)
h_model = Parallel(+,
    Chain(
        Parallel(+,
            h_z_model,
            Chain(
                h_th_model,
                Dense(-Matrix{Float32}(I(1))),
            ),
        ),
        Dense(Matrix{Float32}(I(1)), Float32[0], relu),
    ),
    h_th_model,
)

A = Float32[
    1 0 dt 0;
    0 1 0 dt;
    0 0 1 0;
    0 0 0 1;
]
B = Float32[
    0 0;
    0 0;
    dt / m dt / m;
    -d * dt / Iyy d * dt / Iyy;
]
Q = diagm(Float32[1, 10, 0.1, 1])
R = diagm(Float32[0.1, 0.1])
P, _, _ = ared(A, B, R, Q)
K = inv(R + B' * P * B) * (B' * P * A)

pi_model = Chain(
    Dense(-K, Float32[T0, T0]),
    # max(u, u_low) = relu(x - u_low) + u_low
    Dense(Matrix{Float32}(I(u_dim)), -u_low, relu),
    Dense(Matrix{Float32}(I(u_dim)), u_low),
    # min(u, u_high) = -max(-x, -u_high) = -relu(-x + u_high) + u_high
    Dense(Matrix{Float32}(-I(u_dim)), u_high, relu),
    Dense(Matrix{Float32}(-I(u_dim)), u_high),
)

end
