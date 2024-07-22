module DoubleIntegrator

using Flux
using LinearAlgebra
using MatrixEquations

x_dim = 2
x_low = Float32[-1, -1]
x_high = Float32[1, 1]

u_dim = 1
u_low = Float32[-1]
u_high = Float32[1]

x1_con = 0.9

dt = 0.1

h_model = Parallel(+,
    Dense(Float32[-2 0], Float32[0], relu),
    Dense(Float32[1 0], Float32[-x1_con]),
)

A = Float32[1 dt; 0 1]
B = Float32[0; dt;;]
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

f_pi_model = Chain(
    # x' = Ax + Bu
    Parallel(+,
        Dense(A),
        Chain(pi_model, Dense(B)),
    ),
    # max(x, x_low) = relu(x - x_low) + x_low
    Dense(Matrix{Float32}(I(x_dim)), -x_low, relu),
    Dense(Matrix{Float32}(I(x_dim)), x_low),
    # min(x, x_high) = -max(-x, -x_high) = -relu(-x + x_high) + x_high
    Dense(-Matrix{Float32}(I(x_dim)), x_high, relu),
    Dense(-Matrix{Float32}(I(x_dim)), x_high),
)

end
