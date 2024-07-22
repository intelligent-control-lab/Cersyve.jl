module RobotArm

using Flux
using LinearAlgebra
using MatrixEquations
using EllipsisNotation

n = 3
x_dim = 2 * n
u_dim = n

x_low = Float32[pi / 3; fill(-pi / 6, n - 1); fill(-1, n)]
x_high = Float32[pi / 2; fill(pi / 6, n - 1); fill(1, n)]

u_low = Float32[fill(-1, u_dim);]
u_high = Float32[fill(1, u_dim);]

dt = 0.1

function constraint(x::Array{Float32})::Array{Float32}
    th = cumsum(x[1:n, ..], dims=1)
    x_end = sum(cos.(th), dims=1)
    x_con = 0.5 * n
    return x_end .- x_con
end

A = Matrix{Float32}(I(x_dim)) + Float32.(diagm(n => fill(dt, n)))
B = [zeros(Float32, n, n); Float32.(diagm(fill(dt, n)))]
Q = diagm(Float32[fill(1, n); fill(0.1, n)])
R = diagm(Float32[fill(0.1, n);])
P, _, _ = ared(A, B, R, Q)
K = inv(R + B' * P * B) * (B' * P * A)

pi_model = Chain(
    Dense(-K, K * Float32[pi / 2; zeros(2 * n - 1)]),
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
